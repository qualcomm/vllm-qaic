# Contributing

Guidelines for contributing to the vLLM QAIC plugin.

## Repository

- **URL**: [github.com/quic/vllm-qaic](https://github.com/quic/vllm-qaic)
- **Primary branch**: `main`

## Licensing

| File Type | License |
|-----------|---------|
| Files forked from vLLM upstream | Apache 2.0 (retain original license) |
| New Qualcomm-authored files | BSD 3-Clause Clear |

**Copyright notice for new Qualcomm-authored files:**

```
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
```

!!! note "External contributors"
    External (non-Qualcomm) contributors do **not** need to add the Qualcomm copyright header.
    Your contributions will be accepted under the repository's existing license terms.

**Retain markings** of any third-party code.

## Contribution Scope

Contributions are limited to **QAIC backend support** through the vLLM plugin architecture.

### Permitted

- Backend adapter code mapping vLLM inputs/outputs to QAIC-compatible APIs
- Mechanical QAIC/HVX implementations of existing vLLM primitive interfaces (attention, KV-cache, sampling, etc.)
- Backend support for existing vLLM primitives:
    - Attention and KV-cache interfaces
    - KV transfer / disaggregated serving interfaces
    - Distributed inference interfaces
    - Multimodal inputs / embeddings / encoders
    - LoRA interfaces
    - Quantization interfaces
    - Sampling/decoding interfaces
    - Chunked prefill and continuous batching
    - Speculative decoding interfaces
- Bug fixes (non-algorithmic, addressing implementation flaws)
- Sample code and test code demonstrating approved features
- Documentation of approved features
- Beautification (cosmetic formatting changes)

### Not Permitted

The following require separate approval and are excluded from standard contributions:

- New algorithms or improvements to existing algorithms
- New selection policies, heuristics, or optimization policies
- Changes to acceptance/rejection logic, cache eviction/reuse policies
- Compilation or graph scheduling algorithms
- Optimizations for parallelization (model/tensor slicing)
- Heterogeneous compute algorithms (dynamic CPU/NSP/GPU splitting)
- Any changes that alter the algorithmic behavior of upstream vLLM primitives

## Development Setup

```bash
# Clone the repository
git clone https://github.com/quic/vllm-qaic.git
cd vllm-qaic

# Create development environment (AOT mode)
./scripts/install.sh aot

# Install in editable mode
pip install -e . --no-build-isolation
```

## Code Style

- Python 3.12+
- Follow existing code formatting (enforced by pre-commit hooks)
- Type annotations where practical
- Docstrings for public APIs

### Pre-commit hooks

Formatting and lint checks are enforced with
[pre-commit](https://pre-commit.com/), aligned with the upstream vLLM project.
The following hooks run automatically on `git commit`:

- `ruff-check` / `ruff-format` — Python linting and formatting (also handles
  import sorting).
- `typos` — spell checking (config in `typos.toml`).
- `clang-format` — C/C++/CUDA formatting for `csrc/` (config in `.clang-format`).
- `actionlint` — GitHub Actions workflow linting.
- `check-qualcomm-header` — verifies the Qualcomm license header on Python files.
- `check-filenames` — rejects filenames containing spaces.
- `mypy-local` — static type checking of `vllm_qaic` and `examples`.
- `signoff-commit` — appends the DCO `Signed-off-by` trailer at commit time.

Install and enable the hooks once after cloning:

```bash
pip install -r requirements/lint.txt
pre-commit install
```

To run all commit-stage checks against the entire repository manually:

```bash
pre-commit run --all-files
```

Some hooks only run in CI (the `manual` stage) and are skipped on commit. Run
them explicitly when needed:

```bash
# Markdown linting (config in .markdownlint.yaml)
pre-commit run markdownlint --hook-stage manual --all-files
```

## Pull Request Process

1. Fork the repository and clone your fork
2. Create a feature branch from `main`
3. Make changes within the permitted scope
4. Add tests for new functionality
5. Ensure all existing tests pass: `bash ci_scripts/ci_fasttest_qaic.sh`
6. Push to your fork and create a PR against `main`
7. Provide a clear description of changes in the PR
8. Address review feedback

## Bug Fix Requirements

To qualify as a bug fix:
- Must not change stated/understood functionality or purpose
- The original implementation must have had error(s) addressed by the change
- Non-algorithmic, approximately 10 conventional lines or less
- Does not individually or in aggregate add new features

## Documentation Requirements

- Documentation must be limited to documenting approved features and bug fixes
- Must not describe features, functions, or algorithms not part of the distributed code
- Must not constitute a new feature when viewed individually or in aggregate
