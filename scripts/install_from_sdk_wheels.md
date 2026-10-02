# vllm-qaic Setup Guide — Installing from SDK Wheels

How to install `vllm-qaic` from the pre-built wheels shipped in the QAIC Apps SDK at
`/opt/qti-aic/integrations/vllm_qaic/`.

This is the **wheel install** path — no source checkout, no compilation of `vllm-qaic` itself.
If you are developing `vllm-qaic` and want an install from a git clone instead, see
[`docs/installation.md`](../docs/installation.md).

---

## Contents

- [Before you start](#before-you-start)
- [Pick your mode](#pick-your-mode)
- [SDK layout](#sdk-layout)
- [PYT mode install](#pyt-mode-install)
- [AOT mode install](#aot-mode-install)
- [AOT with triton-cpu (Speculative Decoding)](#aot-with-triton-cpu-speculative-decoding)

- [Verifying the install](#verifying-the-install)
- [Environment variable reference](#environment-variable-reference)
- [Troubleshooting](#troubleshooting)

---

## Before you start

| Requirement | Value |
|---|---|
| Hardware | Qualcomm Cloud AI 100 / Cloud AI 080 |
| OS | Linux (Ubuntu 22.04+) |
| QAIC Platform SDK | >= 1.23.0 |
| QAIC Apps SDK | >= 1.23.0 — PYT mode additionally needs the installer's `--install-torch-qaic` flag |
| Python | 3.10 / 3.11 / 3.12 (both modes) |

> [!IMPORTANT]
> **AOT and PYT cannot coexist in one environment.** AOT requires `torch_qaic` to be
> *absent*; PYT requires it present, and the two modes pin different torch versions. Use a
> separate virtualenv or conda env for each. `install.sh aot` will uninstall `torch_qaic`
> if it finds it.

---

## Pick your mode

| | AOT (Ahead-of-Time) | PYT (Eager / PyTorch) |
|---|---|---|
| Inference engine | QEfficient + QAIC compiler | `torch_qaic` |
| torch version | `2.7.0+cpu` | `2.13.0+cpu` |
| `torch_qaic` required | No — must **not** be present | Yes |
| Wheel tag | `*aot*` | `*pyt*` |
| Python support | 3.10 / 3.11 / 3.12 | 3.10 / 3.11 / 3.12 |

---

## SDK layout

The installer expects this layout. The AOT wheel sits flat (it is `py3-none-any`, so one
artifact serves every Python version); PYT wheels are ABI-specific and live in per-version
subdirectories.

```text
/opt/qti-aic/integrations/vllm_qaic/
├── scripts/
│   ├── install.sh                  ← the installer you run
│   ├── utility.sh                  ← version/path constants, sourced by install.sh
│   └── install_triton_cpu.sh       ← optional, for AOT Speculative Decoding
├── requirements/
│   ├── build.txt
│   ├── vllm_dependency_aot.txt
│   └── vllm_dependency_pyt.txt
├── vllm_qaic-<ver>+aot<sdk>-py3-none-any.whl       ← AOT, flat
├── py310/vllm_qaic-<ver>+pyt<sdk>-cp310-...whl     ← PYT, per Python version
├── py311/vllm_qaic-<ver>+pyt<sdk>-cp311-...whl
└── py312/vllm_qaic-<ver>+pyt<sdk>-cp312-...whl
```

`install.sh` auto-detects wheel mode by the *absence* of a `setup.py` at the package root. It
prints `source : wheel` in its startup banner — check for that line to confirm.

---

## PYT mode install

PYT also needs the `torch_qaic` wheel, which the Apps SDK installs separately under
`/opt/qti-aic/integrations/torch_qaic/py3XX/`. Confirm your Python version is present there:

```bash
ls /opt/qti-aic/integrations/torch_qaic/
# expect: py310  py311  py312
```

If your version is missing, re-run the Apps SDK installer with `--install-torch-qaic`.

**1. Create and activate a clean environment** (3.10, 3.11 or 3.12):

```bash
conda create -y -n vllm-qaic-pyt python=3.12
conda activate vllm-qaic-pyt

# or with venv
python3.12 -m venv ~/venvs/vllm-qaic-pyt
source ~/venvs/vllm-qaic-pyt/bin/activate
```

**2. Run the installer:**

```bash
/opt/qti-aic/integrations/vllm_qaic/scripts/install.sh pyt
```

That is the whole install. The script resolves the PYT wheel matching your interpreter from
`py3XX/`, installs `torch 2.13.0+cpu` *before* `torch_qaic` (which validates at import that
torch is CPU-only), then vllm and `vllm-qaic`.

---

## AOT mode install

**1. Create and activate a clean environment** — Python 3.10, 3.11 or 3.12:

```bash
conda create -y -n vllm-qaic-aot python=3.12
conda activate vllm-qaic-aot
```

**2. Run the installer:**

```bash
/opt/qti-aic/integrations/vllm_qaic/scripts/install.sh aot
```

This installs QEfficient (which brings `torch 2.7.0+cpu`), re-pins torch to the exact AOT
version, installs vllm, then the `py3-none-any` AOT wheel.

> [!NOTE]
> On completion, pip may report a dependency conflict like
> `qefficient requires transformers==5.5.4, but you have transformers 5.18.0`.
> **This is expected.** `requirements/vllm_dependency_aot.txt` deliberately requires a newer
> `transformers`/`huggingface-hub` than QEfficient's `==` pins. Imports and plugin
> registration work normally. If a specific model needs QEfficient's exact version, pin it:
> `TRANSFORMERS_VERSION_AOT=5.5.4 ./install.sh aot`.

---

## AOT with triton-cpu (Speculative Decoding)

Speculative Decoding in AOT mode needs the `triton-cpu` backend so the rejection-sampler
Triton kernels can run on CPU. It is **opt-in** via `TRITON_CPU=1`, because it is a large C++
build (5–10 GB, tens of minutes).

> [!IMPORTANT]
> **Always set `TRITON_CPU_SRC` to a path with enough free space.**
>
> `TRITON_CPU_SRC` defaults to `${SCRIPT_DIR}/../.build/triton-cpu`. When you run the
> installer from the SDK, `SCRIPT_DIR` is `/opt/qti-aic/integrations/vllm_qaic/scripts`, so
> the default resolves to:
>
> ```text
> /opt/qti-aic/integrations/vllm_qaic/.build/triton-cpu
> ```
>
> That is a **root-owned location under `/opt`**. An ordinary user cannot write there, and
> even with privileges you do not want 5–10 GB of transient build artifacts and a git clone
> inside the installed SDK tree. Override it:

```bash
conda activate vllm-qaic-aot   # Python 3.10, 3.11 or 3.12

export TRITON_CPU=1
export TRITON_CPU_SRC=/path/with/enough/space/triton-cpu   # >= 10 GB free, writable by you

/opt/qti-aic/integrations/vllm_qaic/scripts/install.sh aot
```

Good choices for `TRITON_CPU_SRC`: a scratch or project filesystem, or `/tmp/triton-cpu`.
Avoid `$HOME` if a per-user quota applies — `df` cannot see quotas, so the pre-flight check
will pass and the build will then fail with `Disk quota exceeded`.

The installer pre-checks for 10 GB of free space and aborts early with guidance if short. To
bypass that check (for instance when free space is reported inaccurately):

```bash
export TRITON_CPU_SKIP_DISK_CHECK=1
```

To speed up the build on a machine with spare cores (default is 4):

```bash
export TRITON_CPU_COMPILE_MAX_JOBS=16
```

Installing triton-cpu later, against an environment that already has AOT, works too:

```bash
conda activate vllm-qaic-aot
export TRITON_CPU_SRC=/path/with/enough/space/triton-cpu
/opt/qti-aic/integrations/vllm_qaic/scripts/install_triton_cpu.sh
```

Note this **replaces** any PyPI `triton` in the environment — the PyPI build has no CPU
backend. When running SpD tests afterwards, set `TRITON_CPU_BACKEND=1`.

---

## Verifying the install

Run these from a **neutral directory** such as `/tmp`:

```bash
cd /tmp
```

> [!WARNING]
> Do not run the import checks from inside a `vllm-qaic` source checkout. Python puts the
> current directory on `sys.path`, so a local `vllm_qaic/` directory shadows the installed
> package and `vllm_qaic.__file__` will point at the checkout — making a correct wheel
> install look like a source install.

**Both modes:**

```bash
pip show vllm-qaic                                  # Version should carry +aot<sdk> or +pyt<sdk>
python -c "import vllm_qaic; print(vllm_qaic.__file__)"   # must be under site-packages
python -c "import vllm; print(vllm.__version__)"
```

Importing vllm should log the plugin activating:

```text
Platform plugin qaic is activated
```

**PYT only:**

```bash
python -c "import torch; print(torch.__version__)"        # expect 2.13.0+cpu
python -c "import torch_qaic; print('torch_qaic OK')"
```

**AOT only:**

```bash
python -c "import torch; print(torch.__version__)"        # expect 2.7.0+cpu
pip show torch-qaic                                       # must NOT be installed
```

**triton-cpu, if installed:**

```bash
python -c "
import os; os.environ['TRITON_CPU_BACKEND'] = '1'
from triton.backends import backends
print([k for k, v in backends.items() if v.driver and v.driver.is_active()])
"
# expect 'cpu' in the list
```

---

## Environment variable reference

All of these are read by `install.sh` / `utility.sh`; set them before invoking the installer.

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_QAIC_SDK_PATH` | `/opt/qti-aic/integrations/vllm_qaic` | Where to find the wheels. Override to install from a staged or custom SDK copy. |
| `VLLM_QAIC_INSTALL_SOURCE` | *(auto)* | Set to `wheel` to force wheel mode. Normally auto-detected from the absence of `setup.py`. |
| `TORCH_QAIC_BASE_PATH` | `/opt/qti-aic/integrations/torch_qaic` | PYT only — where the `torch_qaic` wheels live. |
| `TRITON_CPU` | `0` | AOT only — set to `1` to build and install the triton-cpu backend. |
| `TRITON_CPU_SRC` | `${SCRIPT_DIR}/../.build/triton-cpu` | **Set this explicitly.** Clone + build location; needs >= 10 GB. The default lands under `/opt` when run from the SDK. |
| `TRITON_CPU_COMPILE_MAX_JOBS` | `4` | Parallel build jobs for the triton-cpu C++ build. |
| `TRITON_CPU_SKIP_DISK_CHECK` | `0` | Set to `1` to skip the 10 GB pre-flight check. |
| `TRANSFORMERS_VERSION_AOT` | *(unset)* | Pin `transformers` after the QEfficient step. |
| `TRANSFORMERS_VERSION_PYT` | *(unset)* | Pin `transformers` after the `torch_qaic` step. |
| `VLLM_BUILD_RUST` | `0` | Build vllm's experimental Rust OpenAI frontend from source. Requires `cargo` on `PATH`. |

Review the banner `install.sh` prints before it starts — it echoes every version and path it
is about to use.

---

## Troubleshooting

**`ERROR: no python found in PATH`** — you did not activate an environment. Activate your
conda env or venv first; the installer deliberately installs into the *active* environment
and never creates one for you.

**`install.sh: line 36: .../utility.sh: No such file or directory`** — `utility.sh` is not
next to `install.sh`. Both must ship together in `scripts/`.

**`no such file or directory: .../requirements/build.txt`** — the `requirements/` directory is
missing from the SDK, or `install.sh` is not in a `scripts/` subdirectory. The installer reads
`${SCRIPT_DIR}/../requirements/`.

**`zsh: no matches found: .../py312/vllm_qaic-*pyt*.whl`** — no PYT wheel for your Python
version. Check `ls /opt/qti-aic/integrations/vllm_qaic/` for the available `py3XX/`
directories and use a matching interpreter.

**triton-cpu build fails with `Disk quota exceeded` or `No space left on device`** — point
`TRITON_CPU_SRC` at a filesystem with space and no per-user quota, then re-run.

**`import torch_qaic` prints `QAIC_WARNING: Pre-init checks for QID: 0 failed`** — a device
state warning on a shared host where another process holds NSPs, not an install problem. The
import still succeeds.

**`[torch_qaic] Triton with Hexagon backend not found`** — expected when the generic PyPI
`triton` is installed; it has no Hexagon backend. Only matters for Triton-on-Hexagon kernels.

**Every pip step stalls retrying an unreachable index** — a host-level pip
`extra-index-url` (for example `pypi.ngc.nvidia.com`) no longer resolves. Harmless but slow;
fix or remove it in `/etc/pip.conf` or `~/.config/pip/pip.conf`.
