# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Build-once / cached basix tabulation shim used by the on-the-fly custom
kernel.

The generated C kernel references plain C symbols (``qugar_register_element_*``
and ``qugar_tabulate_*``) that this module's tiny ``extern "C"`` C++ wrapper
exposes. The wrapper itself links ``libbasix``. It is JIT-compiled the first
time a kernel needs it and cached on disk; downstream kernel JIT only has to
link the resulting shared library.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SHIM_SRC = _HERE / "qugar_basix_shim.cpp"
LIB_NAME = "qugar_basix_shim"
# Use the *platform* shared-library suffix (not Python's extension suffix from
# sysconfig, which is ``.so`` even on macOS): macOS's linker resolves ``-lname``
# only against ``libname.dylib`` (or .tbd), not ``.so``.
if sys.platform == "darwin":
    _SHLIB_SUFFIX = ".dylib"
elif sys.platform == "win32":
    _SHLIB_SUFFIX = ".dll"
else:
    _SHLIB_SUFFIX = ".so"
_LIB_FILENAME = f"lib{LIB_NAME}{_SHLIB_SUFFIX}"

_LOCK = threading.Lock()


def _cache_dir() -> Path:
    base = os.environ.get("QUGAR_SHIM_CACHE")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "qugar" / "shim"


def _find_cxx() -> Path:
    bindir = Path(sys.prefix) / "bin"
    for candidate in [bindir / "clang++", *sorted(bindir.glob("clang++-*"))]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no clang++ driver found in {bindir}")


def ensure_built() -> tuple[Path, str]:
    """Build the shim if missing or out of date.

    Returns:
        (library_dir, library_name) suitable for passing to cffi's
        ``library_dirs`` and ``libraries``.
    """
    with _LOCK:
        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        lib = cache / _LIB_FILENAME
        if lib.exists() and lib.stat().st_mtime >= SHIM_SRC.stat().st_mtime:
            return cache, LIB_NAME
        prefix = Path(sys.prefix)
        cmd = [
            str(_find_cxx()),
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            f"-I{prefix / 'include'}",
            f"-L{prefix / 'lib'}",
            "-lbasix",
            f"-Wl,-rpath,{prefix / 'lib'}",
            "-o",
            str(lib),
            str(SHIM_SRC),
        ]
        # Align the shim's macOS deployment target with what cffi will use
        # when it links the kernel (otherwise ld warns "built for newer
        # version X" because conda clang's default target is much newer).
        if sys.platform == "darwin":
            target = (os.environ.get("MACOSX_DEPLOYMENT_TARGET")
                      or sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET"))
            if target:
                # Insert before "-o" so the output path stays the argument
                # immediately after it.
                cmd.insert(cmd.index("-o"), f"-mmacosx-version-min={target}")
        subprocess.run(cmd, check=True)
        return cache, LIB_NAME


__all__ = ["ensure_built", "LIB_NAME"]
