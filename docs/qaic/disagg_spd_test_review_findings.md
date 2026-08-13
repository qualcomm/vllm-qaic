# Disaggregated SpD Test Review

## Summary/Status

**Status: Implemented; hardware run completed.** The disaggregated
speculative-decoding tests now have one behavior-level non-hardware unit suite, while
redundant skipped diagnostics and the standalone hardware comparator have been removed.

The full non-hardware QAIC test area passes: **22 passed** with `pytest -s
tests/test_qaic -q` on August 13, 2026.

The retained hardware suite was run on August 13, 2026 with QIDs `8,9,10,11`:
**3 passed, 5 setup errors in 3903.45 seconds**. QIDs `8–11` were `Ready` again after
cleanup.

## Objective

Review the tests introduced by the three disaggregated speculative-decoding commits
for unnecessary coverage, missing unit coverage, and consistency with the repository's
test conventions.

## Findings

### Removed or consolidated

- The former `test_disagg_spd_unit.py` was located under a `spec_decode/e2e` package
  despite being non-hardware code.
- `test_disagg_spd_guards.py` duplicated the same guard concepts with tautological
  `SimpleNamespace` conditionals rather than invoking production initialization.
- Source-text tests checked for implementation strings instead of runtime behavior.
- Acceptance-rate tests were permanently skipped because disaggregated serving disables
  the required metrics, and the acceptance-rate logic is covered by non-disaggregated
  tests.
- The AOT ngram order-independence test was a hardware debugging comparator, required a
  missing dataset, and duplicated the disaggregated consistency test. It was not needed
  as required capability coverage.
- The KV-producer basic-request test duplicated method-specific generation coverage;
  the server-startup guard was retained because it checks producer initialization.

### Added coverage

The consolidated unit suite now exercises:

- KV-producer drafter clearing through the actual `QaicModelRunnerAoT` constructor.
- Decode-specialization selection for ngram, suffix, and draft-model combinations across
  ordinary, KV-producer, and KV-consumer configurations.
- Rejection of unsupported speculative-decoding methods.
- Active-K selection for first-step, no-proposal, proposal, and single-specialization
  paths.
- Per-K decode input shapes, padding, batch indices, and optional LoRA inputs.
- Session-derived decode K values and fallback behavior when session shapes are absent.
- Extra, missing, and equal QPC K-set diagnostics through captured log output.
- Deferred target compile completion when a draft model is present.
- AOT-only, idempotent KV connector registration.

## Hardware Test Policy

The retained disaggregated E2E suite was run with:

```bash
pytest -s tests/e2e/disaggregated_serving/test_qaic_disagg_spd.py \
  --device-id '8,9,10,11' -v
```

QIDs `8–11` were free and shared board serial `62891-03-233043F0134` before and after
the run.

### Hardware results

- `suffix` consistency and non-empty generation passed.
- The KV-producer startup guard passed.
- `ngram` consistency and generation did not reach request execution because the
  disaggregated fixture failed to start after compilation; the launcher reported a
  `qaic_disagg` shutdown/cleanup failure (`AttributeError: 'dict' object has no
  attribute 'select'`) while handling the compile-only worker exit. These are startup
  errors, not the known prompt-order consistency assertion.
- `draft_model` consistency, generation, and parallel-request tests failed during
  setup because QID 9 could not activate the draft QPC:
  `QAIC_ERROR_NSP_ALLOC_FAILED: Failed to allocate NSP resources`. The failure occurred
  in `QAICInferenceSession.activate()` while creating the draft-model `ExecObj`, after
  both target and draft QPCs had compiled successfully.
- The draft-model failure is a hardware/runtime resource-allocation failure, not a
  speculative-decoding output mismatch. The resulting setup errors should be retried
  only after QID 9's NSP resources are confirmed clean.

The ngram and suffix prompt-order consistency tests have known disaggregated hardware
failures associated with QAIC/compiler behavior. Such failures should be recorded as
hardware findings and should not invalidate the non-hardware unit-test result by
themselves.

## Follow-up

- Keep the method-specific disaggregated E2E generation and consistency tests as opt-in
  integration coverage.
- Add hardware coverage only when the known consistency issue is resolved or when a run
  is explicitly intended to reproduce and document that issue.

## Investigation Follow-up

### Confirmed `qaic_disagg` cleanup bug

The editable `qaic_disagg` checkout had a selector lifecycle defect. The bug was
most visible when a worker failed during compile or startup, because the cleanup
path ran while the orchestrator was still trying to collect process output.

#### What a selector is

A selector is a standard-library I/O multiplexer. `qaic_disagg` registers the
stdout/stderr pipes for its child processes with the selector, then waits for
whichever pipe has output ready instead of blocking on one process at a time:

```text
target stdout ─┐
target stderr ─┼─> selector.select() ─> read available output
draft stdout ──┤
proxy stderr ──┘
```

The selector is therefore a live resource with two responsibilities:

1. it tracks the file descriptors used for process-output collection; and
2. it provides the `.select()` operation used by `sync_prints()` and shutdown
   diagnostics.

Closing the selector is part of teardown. After it is closed, no code may use it
to select or register file descriptors. The lifecycle should consequently be
explicit: a live selector is available during orchestration, and a cleaned-up
manager records that no selector remains.

#### Old behavior

The old cleanup path closed the selector and then assigned a plain dictionary to
`self.sel`. That dictionary was being used as an accidental sentinel for
"cleanup has happened":

```text
running:       self.sel = BaseSelector
shutdown():    self.sel.close()
               self.sel = {}
later failure: sync_prints() -> self.sel.select()
               AttributeError: 'dict' object has no attribute 'select'
```

This created two problems. First, a closed selector was represented by an object
that did not implement the selector interface, so later code failed with an
unrelated `AttributeError`. Second, that cleanup exception masked the original
compile, activation, or child-process failure that the test needed to report.
Repeated shutdown was also unsafe because later cleanup calls could operate on
the wrong sentinel or on an already-closed selector.

#### New behavior

The manager now types and manages the field as `selectors.BaseSelector | None`:

```text
startup:       self.sel = BaseSelector
normal run:    sync_prints() uses self.sel.select()
first shutdown:
               self.sel.close()
               self.sel = None
later cleanup: if self.sel is None, return safely
post-shutdown: sync_prints() returns/report failure without selecting
```

The selector is closed at most once, its absence is represented by `None`, and
all selector consumers tolerate the post-shutdown state. This makes cleanup
idempotent and preserves the original failure as the primary diagnostic. The
focused regression suite covers both calling `sync_prints()` after shutdown and
repeating shutdown after the selector has already been closed.

The focused `qaic_disagg` regression suite passes: **2 passed**. The separate
180-second draft-run shutdown timeout remains a teardown-latency follow-up, not a
selector correctness failure.

### Confirmed draft NSP allocation bug

The draft-model loader clears `vllm_config.speculative_config` before selecting the QAIC
compile configuration. That removed the information needed by the same-device target /
draft core split, so the draft QPC requested all 16 NSP cores on QID 9 after the target
QPC had already reserved its allocation. The resulting activation failure was:
`QAIC_ERROR_NSP_ALLOC_FAILED: Failed to allocate NSP resources`.

The loader now preserves the original speculative configuration only for compile-config
selection while still clearing it on the copied draft-model config. Same-device target
and draft compilation now split a 16-NSP device into 8/8, while explicit core overrides
remain authoritative. The focused vllm-qaic unit suite passes: **25 passed**.

### Remaining validation

Hardware retry completed on QIDs `8,9,10,11` after both fixes:

- `TestNgramDisagg`: **2 passed in 569.93s**. Both consistency and non-empty generation
  passed, confirming the selector cleanup fix no longer masks ngram startup.
- `TestDraftModelDisagg`: **3 passed in 727.23s**. Consistency, generation, and parallel
  requests passed. Compile logs showed target `num_cores=8` and draft `num_cores=8` on
  QID 9, and the previous NSP allocation failure did not recur.
- QIDs `8–11` returned to `Ready` after both runs with no residual test/orchestrator
  processes.

The draft run still incurred a 180-second orchestrator shutdown timeout before force
killing the proxy, although all three tests passed and device health returned to Ready.
This is teardown latency rather than a functional SpD failure; it should be investigated
separately if repeated runs continue to hit the timeout. Any remaining ngram/suffix
output consistency mismatch should continue to be recorded separately as the known
QAIC/compiler behavior.
