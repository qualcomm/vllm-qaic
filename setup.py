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


VERSION = "0.23.0.dev0"
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


def _make_hexagon_ext(sources, device_arch, extra_compile_args, extra_link_args):
    """Build the Hexagon kernel Extension without importing torch_qaic._C.

    Replicates HexagonKernelExtension.__init__ using only stdlib + setuptools.
    Used when QAIC_DEVICE_ARCH is set (Docker build envs without live devices).
    """
    hexagon_tools = os.environ.get(
        "HEXAGON_TOOLS_DIR", "/opt/qti-aic/dev/hexagon_tools"
    )
    jit_inc = os.environ.get(
        "QAIC_PLATFORM_JIT_INCLUDE_DIR", "/opt/qti-aic/dev/inc/jit"
    )
    hex_inc = os.environ.get("QAIC_HEXAGON_INCLUDE_DIR", "/opt/qti-aic/dev/inc/jit")
    arch_ver = "v81" if device_arch == "v81" else "v68"
    compile_flags = [
        "-nostdinc++",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wglobal-constructors",
        "-Wno-pass-failed",
        "-Wnon-virtual-dtor",
        "-Wno-unused-variable",
        "-Wno-tautological-compare",
        "-Wno-unknown-pragmas",
        "-Wno-c99-designator",
        "-Wno-psabi",
        "-ffreestanding",
        "-std=c++17",
        "-fno-exceptions",
        "-fvisibility=default",
        "-B",
        osp.join(hexagon_tools, "target"),
        f"-m{arch_ver}",
        f"-mhvx={arch_ver}",
        "-mhmx",
        "-mhvx-ieee-fp",
        "-mllvm",
        "-hexagon-commgep=false",
    ] + extra_compile_args
    link_flags = ["-nostdlib++", "-nostdlib", "-fPIC", "-shared"] + extra_link_args
    ext = Extension(
        name="hexagon_kernels",
        sources=sources,
        include_dirs=[jit_inc, hex_inc],
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
        language="c++",
    )
    ext.name = "vllm_qaic.hexagon_kernels"
    ext.OUT_FILENAME = "vllm_qaic/hexagon_kernels.so"
    ext.ASM_FILENAME = "vllm_qaic/hexagon_kernels.s"
    ext.HEXAGON_COMPILER = osp.join(hexagon_tools, "bin", "hexagon-clang++")
    ext.EXT_NAME = "hexagon_kernels"
    return ext


def get_qaic_extensions() -> list[Extension]:
    """Get the list of C++ extensions to build for qaic custom ops."""
    if not _torch_qaic_installed:
        return []

    debug_mode = os.getenv("DEBUG", "0") == "1"
    extra_compile_args = ["-O0", "-g"] if debug_mode else ["-O3"]
    extra_link_args = ["-O0", "-g"] if debug_mode else []
    print(f"Building vllm_qaic in {'debug' if debug_mode else 'release'} mode...")

    # QAIC_DEVICE_ARCH: when set, bypass all torch_qaic imports (torch_qaic._C
    # triggers the QAIC driver which SIGABRTs without live devices in Docker builds).
    device_arch = os.environ.get("QAIC_DEVICE_ARCH")

    csrc_dir = osp.join(str(ROOT_DIR), "csrc")
    qaic_sources = list(glob.glob(osp.join(csrc_dir, "**", "*.cpp"), recursive=True))
    # BF16 kernel requires V81+ (AI200). Exclude it on V68 (AI100) so the linker
    # does not reference rms_norm_multi_nsp_bf16, which is guarded in dispatch.cpp.
    if device_arch != "v81":
        qaic_sources = [s for s in qaic_sources if "_bf16" not in osp.basename(s)]
    qaic_sources = [osp.relpath(src, str(ROOT_DIR)) for src in qaic_sources]
    if len(qaic_sources) == 0:
        return []

    if device_arch:
        # Verify SDK headers are present before attempting compile.
        # csrc/ headers may require a newer SDK than the current BASE_IMAGE provides.
        jit_inc = os.environ.get(
            "QAIC_PLATFORM_JIT_INCLUDE_DIR", "/opt/qti-aic/dev/inc/jit"
        )
        required_headers = ["QAicHexagonMath.h", "QAicHexagonUtils.h"]
        missing = [h for h in required_headers if not osp.exists(osp.join(jit_inc, h))]
        if missing:
            print(
                "WARNING: Skipping Hexagon C++ extension build — "
                f"missing SDK headers: {missing}"
            )
            print(f"  Expected in: {jit_inc}")
            print("  These headers require a newer SDK. Upgrade BASE_IMAGE to enable.")
            return []
        print(f"Device arch: {device_arch} (from QAIC_DEVICE_ARCH env)")
        print(f"QAIC extension sources: {qaic_sources}")
        return [
            _make_hexagon_ext(
                qaic_sources, device_arch, extra_compile_args, extra_link_args
            )
        ]

    from torch_qaic.custom_ops.build_utils import (
        _get_device_arch,
        HexagonKernelExtension,
    )

    device_arch = _get_device_arch()
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

    device_arch = os.environ.get("QAIC_DEVICE_ARCH")
    if device_arch:
        import subprocess
        from setuptools.command.build_ext import build_ext as _BuildExt

        # Inline replication of QAicBuildExt without importing torch_qaic._C.
        class QAicBuildExtNoDevice(_BuildExt):
            def get_ext_filename(self, ext_name):
                bare = ext_name.split(".")[-1]
                filename = super().get_ext_filename(bare)
                if bare == "hexagon_kernels":
                    # Mirror torch_qaic's QAicBuildExt.get_ext_filename: this
                    # extension is hand-compiled by the Hexagon toolchain, not
                    # cpython's ABI, so strip the cpython-*-linux-gnu tag that
                    # setuptools appends by default.
                    parts = filename.split(".")
                    filename = ".".join(parts[:-2] + parts[-1:])
                prefix = "/".join(ext_name.split(".")[:-1])
                return f"{prefix}/{filename}" if prefix else filename

            def build_extension(self, ext):
                if hasattr(ext, "HEXAGON_COMPILER"):
                    import os

                    cmd = [ext.HEXAGON_COMPILER]
                    cmd.extend(f"-I{d}" for d in ext.include_dirs)
                    cmd.extend(ext.extra_compile_args)
                    cmd.extend(ext.extra_link_args)
                    cmd.extend(ext.sources)
                    out = os.path.join(self.build_lib, ext.OUT_FILENAME)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    cmd.extend(["-o", out])
                    subprocess.check_call(cmd)
                else:
                    super().build_extension(ext)

        return {"build_ext": QAicBuildExtNoDevice}

    from torch_qaic.custom_ops.build_utils import QAicBuildExt, HexagonKernelExtension
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
    """Get the QAIC sdk version."""
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
    override = os.environ.get("VLLM_VERSION_OVERRIDE")
    if override:
        return override
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
