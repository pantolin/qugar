# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Stress test for the on-the-fly shim's cross-process file lock.

Spawns N subprocesses that all race on ``ensure_built()`` with an empty
cache. The lock should serialize them: exactly one compile, the rest are
cache hits, and the resulting shared library loads in every child.
Without a lock the parallel writers would corrupt each other's output.

Marked ``slow`` because it spawns 8 child Python interpreters and does a
clang build inside the lock holder. Deselect with ``-m "not slow"``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import qugar.dolfinx._shim as _shim_pkg

SHIM_INIT = Path(_shim_pkg.__file__).resolve()

_CHILD = """
import ctypes, importlib.util as u, sys, time
t = time.perf_counter()
spec = u.spec_from_file_location('s', %r)
m = u.module_from_spec(spec); spec.loader.exec_module(m)
cache, name = m.ensure_built()
elapsed = time.perf_counter() - t
suffix = {'darwin': 'dylib', 'win32': 'dll'}.get(sys.platform, 'so')
lib = cache / f'lib{name}.{suffix}'
ctypes.CDLL(str(lib))  # must load -- catches a corrupted/partial build
print(f'OK {elapsed:.3f}s')
"""


@pytest.mark.slow
def test_parallel_build_serializes(tmp_path):
    """Eight concurrent ``ensure_built`` calls on an empty cache must
    all succeed: the file lock serializes the build."""
    n = 8
    env = os.environ.copy()
    env["QUGAR_SHIM_CACHE"] = str(tmp_path)

    t0 = time.perf_counter()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD % str(SHIM_INIT)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(n)
    ]
    outs = []
    for p in procs:
        out, err = p.communicate()
        outs.append((p.returncode, out.strip(), err.strip()))
    total = time.perf_counter() - t0

    failures = [(rc, out, err) for rc, out, err in outs if rc != 0]
    assert not failures, (
        f"{len(failures)}/{n} children failed (wall {total:.2f}s):\n"
        + "\n".join(f"  rc={rc} stdout={out!r} stderr={err!r}"
                    for rc, out, err in failures)
    )
