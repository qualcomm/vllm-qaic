# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import gc
import json
import os
import platform
import sys
import time

try:
    from qaicrt import QStatus
    from qaicrt import Util as _QaicUtil
except ImportError:
    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
    try:
        from qaicrt import QStatus
        from qaicrt import Util as _QaicUtil
    except ImportError:
        QStatus = None
        _QaicUtil = None

# Rerankers/reward models declaring a plain "*ForCausalLM" architecture default
# to runner="generate", so they need an explicit convert= to run as pooling
# models. "classify" (not "embed") because DispatchPooler.for_seq_cls registers
# both "classify" and "token_classify".
#
# extra_hf_override is for models needing one extra build with different
# hf_overrides to reach their full task set -- e.g. bge-m3's XLMRobertaModel
# maps to embed + token_embed only, while BgeM3EmbeddingModel registers all
# four. See examples/qaic_bge_m3.py.
CONFIGS = {
    "BAAI/bge-reranker-v2-gemma": {"convert": "classify"},
    "mixedbread-ai/mxbai-rerank-base-v2": {"convert": "classify"},
    "Qwen/Qwen3-Reranker-0.6B": {"convert": "classify"},
    "peiyi9979/math-shepherd-mistral-7b-prm": {
        "convert": "classify"
    },  # needs live verification
    "BAAI/bge-m3": {"extra_hf_override": {"architectures": ["BgeM3EmbeddingModel"]}},
    "BAAI/bge-multilingual-gemma2": {"tp_size": 4},
    # config.json declares the generic trust_remote_code class name "NewModel"
    # instead of the registered "GteNewModel", so vLLM's arch lookup misses it.
    "Alibaba-NLP/gte-multilingual-base": {
        "extra_hf_override": {"architectures": ["GteNewModel"]}
    },
    # Same generic-name issue as above, but for the reranker variant: its
    # config.json says "NewForSequenceClassification" instead of the
    # registered "GteNewForSequenceClassification".
    "Alibaba-NLP/gte-multilingual-reranker-base": {
        "convert": "classify",
        "extra_hf_override": {"architectures": ["GteNewForSequenceClassification"]},
    },
    # Base Qwen2Model repo has no sentence-transformers metadata (modules.json /
    # sentence_bert_config.json), so vLLM can't auto-detect a pooling task and
    # defaults to the generate runner. Force it explicitly.
    "ssmits/Qwen2-7B-Instruct-embed-base": {"convert": "embed"},
}


def get_config(model_name):
    """
    Returns the model-specific configuration dictionary for embedding models.
    Returns an empty dict if the model is not found, allowing
    the main script to apply its own default settings.
    """
    return CONFIGS.get(model_name, {})


def get_tp_size(model_name, default):
    """
    Returns the tensor-parallel size the model will actually run with: the
    pinned tp_size from CONFIGS if there is one, else the sweep default.

    Single source of truth for the effective TP so that the runner, the log
    file names and the Excel summary can never disagree.
    """
    return int(get_config(model_name).get("tp_size", default))


def get_convert(model_name, default="auto"):
    """
    Returns the `convert=` value to build the model with: the pinned value
    from CONFIGS if there is one, else "auto".
    """
    return get_config(model_name).get("convert", default)


def get_extra_hf_override(model_name):
    """
    Returns the extra hf_overrides dict for a model that needs a second build
    to reach its full pooling task set, or None if it doesn't need one.
    """
    return get_config(model_name).get("extra_hf_override")


# --- Discovered pooling tasks (auto-managed below CONFIGS, never hand-edited) ---
#
# A model's supported pooling tasks are discovered once (see discover_tasks.py)
# and cached here so every later sweep run skips straight to running -- or to
# an instant SKIP -- instead of rebuilding a model just to re-learn something
# already known. Stored as a JSON string rather than a Python dict literal so
# that persisting a new discovery is a plain json.dumps()+splice, with no
# dict-literal formatting to get right and no risk of corrupting this file's
# Python syntax. There is no "reason" field: a cached SKIP's cause already
# lives in that model's own CI log from the run that first discovered it.
DISCOVERED_JSON = """
{
  "BAAI/bge-large-en-v1.5": {
    "discover_tasks": false,
    "tasks": [
      "embed",
      "token_embed"
    ]
  }
}
"""
DISCOVERED = json.loads(DISCOVERED_JSON)

_THIS_FILE = os.path.abspath(__file__)


def get_discover_tasks(model_name):
    """
    True (the default, for any model not yet seen) means this model still
    needs to be discovered. False means its pooling tasks are already known:
    `get_tasks` returns them directly (or an empty list, meaning skip).
    """
    return DISCOVERED.get(model_name, {}).get("discover_tasks", True)


def get_tasks(model_name):
    """Cached pooling tasks for an already-discovered model. Empty if unset,
    or if discovery previously found the model has none it can run."""
    return DISCOVERED.get(model_name, {}).get("tasks", [])


def mark_discovered(model_name, tasks):
    """Cache the discovery result for model_name and persist it to disk."""
    DISCOVERED[model_name] = {"discover_tasks": False, "tasks": tasks}
    _persist()


def _persist():
    """Rewrite this file's DISCOVERED_JSON block in place with the current
    contents of DISCOVERED. Everything else in the file -- CONFIGS, the
    getters, this function itself -- is untouched, since we only ever
    locate and replace the text between the DISCOVERED_JSON markers."""
    with open(_THIS_FILE) as f:
        src = f.read()

    start_marker = 'DISCOVERED_JSON = """'
    start = src.index(start_marker) + len(start_marker)
    end = src.index('"""', start)

    new_json = json.dumps(DISCOVERED, indent=2, sort_keys=True)
    updated = f"{src[:start]}\n{new_json}\n{src[end:]}"

    tmp_path = _THIS_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(updated)
    os.replace(tmp_path, _THIS_FILE)


# QAIC-specific eager-mode build parameters, fixed across every embed model.
_CTX_LEN = 128
_SEQ_LEN = 128
_DECODE_BSZ = 4


def build_llm_kwargs(model_name, tp_size, pooler_task=None):
    """
    Kwargs for `LLM(**kwargs)`, shared by discover_tasks.py and run_embed.py
    so discovery always builds a model exactly the way the real run would --
    same ctx/seq/batch settings, same convert=/hf_overrides= from CONFIGS.

    pooler_task=None lets vLLM auto-pick (used for discovery); a task name
    forces that one pooling task to be built (used for the actual run).
    """
    kwargs = dict(
        model=model_name,
        trust_remote_code=True,
        max_num_seqs=_DECODE_BSZ,
        max_model_len=_CTX_LEN,
        disable_log_stats=False,
        enforce_eager=True,
        async_scheduling=False,
        long_prefill_token_threshold=_SEQ_LEN,
        tensor_parallel_size=tp_size,
        convert=get_convert(model_name),
    )
    hf_override = get_extra_hf_override(model_name)
    if hf_override:
        kwargs["hf_overrides"] = hf_override
    if pooler_task:
        from vllm.config import PoolerConfig

        kwargs["pooler_config"] = PoolerConfig(task=pooler_task)
    return kwargs


# "embed&token_classify" is excluded: init_pooling_io_processors() registers no
# IO processor for that composite task, so encode() would assert out on it.
POOLING_TASKS = {"embed", "classify", "token_embed", "token_classify"}

# Seconds to wait after tearing down an LLM before building the next one for
# the SAME model, as extra margin on top of the explicit engine_core.shutdown()
# wait below. Separate from ci_fallback_ops.sh's SLEEP_BETWEEN, which only
# runs between DIFFERENT models in a sweep.
INTRA_MODEL_SLEEP = 40

# vLLM's own EngineCore subprocess teardown falls back to a hardcoded 5s
# SIGKILL grace period (vllm/v1/utils.py) when no explicit timeout is given --
# e.g. via the weakref.finalize that fires on plain `del`+gc. On QAIC, NSP
# release for eager PyTorch models has no Python-level "release device" call:
# it happens only via torch_qaic's C++ destructors running during the
# EngineCore process's *normal* exit. SIGKILL can't be caught, so if that 5s
# is too short the process gets killed mid-teardown and the NSPs are orphaned
# -- observed directly as a "[shutdown] ... force killing remaining
# processes" log line immediately followed by the next build's "QID: 0 has 0
# free nsps" crash, even after a long subsequent sleep. Explicitly calling
# shutdown() with a generous timeout (below) gives that natural exit enough
# time to happen instead.
ENGINE_SHUTDOWN_TIMEOUT = 60

# Ceiling for how long _wait_for_nsps_free() will poll before giving up and
# proceeding anyway (the next LLM() build then surfaces the real "N free
# nsps" error, but at least we've logged how long it actually took instead
# of guessing with a fixed sleep).
NSP_WAIT_TIMEOUT = 90
NSP_POLL_INTERVAL = 2


def _visible_qids() -> list:
    raw = os.environ.get("QAIC_VISIBLE_DEVICES", "")
    return [int(q) for q in raw.split(",") if q.strip()]


def _nsp_free_and_total(qid):
    """Returns (free, total) NSPs for `qid`, or None if the query itself
    failed (no qaicrt bindings, driver unreachable, no qaic-group perms)."""
    if _QaicUtil is None:
        return None
    try:
        status, device_info = _QaicUtil().getDeviceInfo(qid)
    except Exception:
        return None
    if status != QStatus.QS_SUCCESS:
        return None
    info = device_info.devData.resourceInfo
    return info.nspFree, info.nspTotal


def _wait_for_nsps_free(timeout=NSP_WAIT_TIMEOUT):
    """Poll each QID in QAIC_VISIBLE_DEVICES until it reports nspFree ==
    nspTotal, logging how long that actually took, instead of blindly
    sleeping for INTRA_MODEL_SLEEP and hoping. Falls back to that fixed
    sleep if the QIDs or the qaicrt query aren't available, so behavior
    degrades to the old approach rather than skipping the wait outright.
    """
    qids = _visible_qids()
    if not qids:
        time.sleep(INTRA_MODEL_SLEEP)
        return

    start = time.monotonic()
    deadline = start + timeout
    while True:
        statuses = {qid: _nsp_free_and_total(qid) for qid in qids}
        if any(s is None for s in statuses.values()):
            print(
                f"[teardown] NSP query unavailable for {qids}; "
                f"falling back to fixed {INTRA_MODEL_SLEEP}s sleep"
            )
            time.sleep(INTRA_MODEL_SLEEP)
            return
        if all(free == total for free, total in statuses.values()):
            print(
                f"[teardown] NSPs fully free on {qids} after "
                f"{time.monotonic() - start:.1f}s"
            )
            return
        if time.monotonic() >= deadline:
            print(
                f"[teardown] WARNING: NSPs still not fully free on {qids} "
                f"after {timeout}s (status={statuses}); proceeding anyway -- "
                "the next LLM() build may hit 'N free nsps'"
            )
            return
        time.sleep(NSP_POLL_INTERVAL)


def teardown_llm(llm):
    """Explicitly shut down `llm`'s EngineCore with a generous timeout, then
    drop it and wait for the QAIC device/model cache to clear.
    Public (no leading underscore): called from discover_tasks.py and
    run_embed.py, not just within this module.

    Calling engine_core.shutdown(timeout=...) directly -- rather than
    relying on `del`+gc.collect() to trigger it via the client's
    weakref.finalize -- matters because that finalizer calls shutdown()
    with no timeout, which falls back to vLLM's hardcoded 5s SIGKILL grace
    period (see ENGINE_SHUTDOWN_TIMEOUT above). Calling it ourselves first
    is also unaffected by how many other references to `llm` the caller
    still holds, since it doesn't depend on refcounting to fire.
    """
    engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    if engine_core is not None:
        engine_core.shutdown(timeout=ENGINE_SHUTDOWN_TIMEOUT)
    del llm
    gc.collect()
    _wait_for_nsps_free()
