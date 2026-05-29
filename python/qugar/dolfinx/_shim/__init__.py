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

import contextlib
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
    # Honour an explicit user/CI override first.
    cxx_env = os.environ.get("CXX")
    if cxx_env:
        p = Path(cxx_env)
        if p.exists():
            return p
    bindir = Path(sys.prefix) / "bin"
    for candidate in [
        bindir / "clang++",
        *sorted(bindir.glob("clang++-*")),
        bindir / "g++",
        bindir / "c++",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no C++ compiler found in {bindir}; set $CXX to override"
    )


@contextlib.contextmanager
def _file_lock(path: Path):
    """POSIX advisory exclusive file lock; no-op on platforms without fcntl.

    Serializes builds across processes (e.g. MPI ranks racing on the same
    cache directory) so only one build runs at a time. Idle waiters block
    until the holder releases.
    """
    try:
        import fcntl
    except ImportError:  # Windows -- skip; conda FEniCSx is POSIX in practice.
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def ensure_built() -> tuple[Path, str]:
    """Build the shim if missing or out of date.

    MPI/multi-process safe: cross-process serialized via a POSIX file
    lock on the cache directory (waiters block until the writer
    releases), in-process serialized via a threading lock. Double-
    checked locking inside the critical section avoids a redundant
    rebuild when another rank/process built it while we were waiting.

    Returns:
        (library_dir, library_name) suitable for passing to cffi's
        ``library_dirs`` and ``libraries``.
    """
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / _LIB_FILENAME
    # Treat a change to either the shim source OR this builder (which
    # controls the compile flags) as cache-invalidating.
    builder_mtime = Path(__file__).stat().st_mtime
    src_mtime = max(SHIM_SRC.stat().st_mtime, builder_mtime)

    # Fast path: cache hit without taking any lock.
    if lib.exists() and lib.stat().st_mtime >= src_mtime:
        return cache, LIB_NAME

    with _LOCK, _file_lock(cache / ".build.lock"):
        # Double-check under the lock — another rank/process/thread may
        # have built it while we were waiting.
        if lib.exists() and lib.stat().st_mtime >= src_mtime:
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
                cmd.insert(cmd.index("-o"), f"-mmacosx-version-min={target}")
        subprocess.run(cmd, check=True)
        return cache, LIB_NAME


__all__ = ["ensure_built", "LIB_NAME"]
