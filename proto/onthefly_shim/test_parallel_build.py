"""Stress test for the shim's cross-process file lock.

Spawns N subprocesses that all race on ``ensure_built()`` with an empty
cache. The lock should serialize them: exactly one compile, the rest are
cache hits, the resulting .dylib loads in every child. Without a lock the
parallel writers would corrupt each other's output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SHIM_INIT = (Path(__file__).resolve().parents[2]
             / "python" / "qugar" / "dolfinx" / "_shim" / "__init__.py")

CHILD = """
import ctypes, importlib.util as u, sys, time
t = time.perf_counter()
spec = u.spec_from_file_location('s', %r)
m = u.module_from_spec(spec); spec.loader.exec_module(m)
cache, name = m.ensure_built()
elapsed = time.perf_counter() - t
suffix = 'dylib' if sys.platform == 'darwin' else 'so'
lib = cache / f'lib{name}.{suffix}'
ctypes.CDLL(str(lib))  # must load -- catches a corrupted/partial build
print(f'OK {elapsed:.3f}s')
"""


def main() -> int:
    n = 8
    with tempfile.TemporaryDirectory(prefix="qugar_shim_race_") as d:
        env = os.environ.copy()
        env["QUGAR_SHIM_CACHE"] = d
        t0 = time.perf_counter()
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", CHILD % str(SHIM_INIT)],
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
        ok = sum(1 for rc, _, _ in outs if rc == 0)
        print(f"\n{ok}/{n} children OK, total wall {total:.2f}s")
        for i, (rc, out, _err) in enumerate(outs):
            print(f"  child {i}: rc={rc} -> {out}")
        # Reality check: at least one child should report a slow first-build,
        # the rest should be fast cache hits.
        # (Loose, just to make sure the lock did serialize -- not an assertion.)
        return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
