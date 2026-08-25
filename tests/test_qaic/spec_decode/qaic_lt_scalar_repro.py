"""Minimal standalone repro for QAIC eager ``aten::lt.Scalar``.

This file deliberately depends only on PyTorch and ``torch_qaic``. It can be
copied to another QAIC host without cloning this repository. Set
``QAIC_VISIBLE_DEVICES`` before Python starts, for example:

    QAIC_VISIBLE_DEVICES=0 QAIC_DEBUG=1 TORCH_SHOW_CPP_STACKTRACES=1 \
      python qaic_lt_scalar_repro.py
"""

from __future__ import annotations

import os


if not os.environ.get("QAIC_VISIBLE_DEVICES"):
    raise RuntimeError("Set QAIC_VISIBLE_DEVICES before starting Python.")

import torch
import torch_qaic  # noqa: F401


SAMPLING_EPS = 1e-5


def main() -> None:
    print("torch:", torch.__version__)
    print("torch_qaic:", getattr(torch_qaic, "__version__", "unknown"))
    print("QAIC_VISIBLE_DEVICES:", os.environ["QAIC_VISIBLE_DEVICES"])
    print("QAIC_DEBUG:", os.environ.get("QAIC_DEBUG", "unset"))

    temperature = torch.tensor(
        [0.0, SAMPLING_EPS / 2, SAMPLING_EPS, 0.5],
        dtype=torch.float32,
        device="qaic",
    )
    print("input:", temperature.cpu().tolist(), temperature.device)
    print("running: torch.lt(temperature, 1e-5)")

    result = torch.lt(temperature, SAMPLING_EPS)
    torch.qaic.synchronize()

    print("result:", result.cpu().tolist(), result.device)


if __name__ == "__main__":
    main()
