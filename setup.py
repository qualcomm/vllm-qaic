# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import sys
import glob
import importlib.util
import logging
import os
import os.path as osp
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools_scm import get_version


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


VERSION = "0.15.0.dev0"
ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

# For time being we don't support single package installation for both
# pytorch and qaic compiler backend, due to QEfficient dependency issue
# Will be resolved in future
global _torch_qaic_installed
_torch_qaic_installed = (
    "1" if importlib.util.find_spec("torch_qaic") is not None else "0"
)

_torch_qaic_installed = (
    os.environ.get("TORCH_QAIC_INSTALLED", _torch_qaic_installed) == "1"
)


def get_qaic_extensions() -> list[Extension]:
    """Get the list of C++ extensions to build for qaic custom ops."""
    if not _torch_qaic_installed:
        return []
    from torch_qaic.custom_ops import HexagonKernelExtension

    debug_mode = os.getenv("DEBUG", "0") == "1"

    extra_compile_args = []
    extra_link_args = []
    if debug_mode:
        print("Building vllm_qaic in debug mode...")
        extra_compile_args += ["-O0", "-g"]
        extra_link_args += ["-O0", "-g"]
    else:
        print("Building vllm_qaic in release mode...")
        extra_compile_args += ["-O3"]

    from torch_qaic.custom_ops.build_utils import _get_device_arch

    device_arch = _get_device_arch()

    csrc_dir = osp.join(str(ROOT_DIR), "csrc")
    qaic_sources = list(glob.glob(osp.join(csrc_dir, "**", "*.cpp"), recursive=True))
    # BF16 kernel requires V81+ (AI200). Exclude it on V68 (AI100) so the linker
    # does not reference rms_norm_multi_nsp_bf16, which is guarded in dispatch.cpp.
    if device_arch != "v81":
        qaic_sources = [s for s in qaic_sources if "_bf16" not in osp.basename(s)]
    # Convert absolute paths to relative paths (required by setuptools)
    qaic_sources = [osp.relpath(src, str(ROOT_DIR)) for src in qaic_sources]
    if len(qaic_sources) == 0:
        return []
    print(f"Device arch: {device_arch}")
    print(f"QAIC extension sources: {qaic_sources}")
    ext = HexagonKernelExtension(
        qaic_sources,
        arch=device_arch,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    ext.name = "vllm_qaic.hexagon_kernels"
    ext.OUT_FILENAME = "vllm_qaic/hexagon_kernels.so"
    ext.ASM_FILENAME = "vllm_qaic/hexagon_kernels.s"

    return [ext]


def get_qaic_build_ext():
    if not _torch_qaic_installed:
        return {}
    from torch_qaic.custom_ops import QAicBuildExt, HexagonKernelExtension
    import os.path as osp

    class QAicBuildExtWithMkdir(QAicBuildExt):
        def get_ext_filename(self, ext_name):
            # Strip the package prefix before delegating so the ABI-stripping
            # check in QAicBuildExt (which compares against bare "hexagon_kernels")
            # still fires correctly.
            bare = ext_name.split(".")[-1]
            filename = super().get_ext_filename(bare)
            # Re-apply the package prefix as a directory path.
            prefix = "/".join(ext_name.split(".")[:-1])
            return f"{prefix}/{filename}" if prefix else filename

        def build_extension(self, ext):
            if isinstance(ext, HexagonKernelExtension):
                out_path = osp.join(self.build_lib, ext.OUT_FILENAME)
                os.makedirs(osp.dirname(out_path), exist_ok=True)
            super().build_extension(ext)

    return {"build_ext": QAicBuildExtWithMkdir}


def _is_qaic() -> bool:
    """Check if QAIC SDK is installed by verifying qaic-util exists on disk."""
    return osp.exists("/opt/qti-aic/tools/qaic-util")


def get_qaic_sdk_version():
    """Get the QAIC sdk version.
    """
    return "1.22"


def get_requirements(filename=None) -> list[str]:
    """Get Python package dependencies from requirements.txt.

    - filename is None              → common + mode (aot.txt or pyt.txt)
    - filename is "aot.txt"/"pyt.txt" → common + filename (mode_file skipped)
    - filename is anything else     → common + mode + filename
    """

    def _read_requirements(fname: str) -> list[str]:
        _filename = ROOT_DIR / "requirements" / fname
        with open(_filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif (
                not line.startswith("--")
                and not line.startswith("#")
                and line.strip() != ""
            ):
                resolved_requirements.append(line)
        return resolved_requirements

    mode_file = "pyt.txt" if _torch_qaic_installed else "aot.txt"

    try:
        reqs = _read_requirements("common.txt")
        if filename in ("aot.txt", "pyt.txt"):
            reqs += _read_requirements(filename)
        elif filename is None:
            reqs += _read_requirements(mode_file)
        else:
            reqs += _read_requirements(mode_file) + _read_requirements(filename)
        return reqs
    except ValueError:
        print("Failed to read requirements in vllm_qaic/requirements.")
        return []


def get_vllm_qaic_version() -> str:
    version = get_version(fallback_version=VERSION, write_to="vllm_qaic/_version.py")
    sep = "+" if "+" not in version else "."  # dev versions might contain +
    # Get the qaic sdk version
    qaic_version_str = get_qaic_sdk_version()

    if not _torch_qaic_installed:
        version += f"{sep}aot{qaic_version_str}"
    else:
        version += f"{sep}pyt{qaic_version_str}"
    return version


if not _is_qaic():
    raise SystemExit(
        "ERROR: QAic platform not found. "
        "Please ensure the QAIC SDK is installed and qaic devices are accessible."
    )

setup(
    name="vllm_qaic",
    version=get_vllm_qaic_version(),
    author="Qualcomm",
    long_description="vLLM QAIC backend plugin",
    packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
    ext_modules=get_qaic_extensions(),
    install_requires=get_requirements(),
    entry_points={
        "vllm.platform_plugins": ["qaic = vllm_qaic:register"],
        "vllm.general_plugins": ["qaic_kv_connector = vllm_qaic:register_connector"],
    },
    extras_require={
        "test": get_requirements("test.txt"),
    },
    cmdclass=get_qaic_build_ext(),
)
