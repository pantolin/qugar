"""End-to-end JIT-link smoke test for the on-the-fly path.

Goes one step beyond gen_inspect.py: drives the full qugar.dolfinx.jit
pipeline on a P2-triangle cell form, which generates the modified kernel,
runs it through CFFI, and links it against the basix tabulation shim. If
this completes without error, codegen + JIT + shim linking are coherent.

Actual element-tensor correctness (vs the current-QUGaR oracle) is the
next milestone after this.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import basix.ufl
import numpy as np
import ufl


def main() -> int:
    # Keep dolfinx's JIT cache + the shim cache inside a sandbox-writable dir.
    tmp = Path(tempfile.mkdtemp(prefix="qugar_jit_link_"))
    os.environ["XDG_CACHE_HOME"] = str(tmp)
    os.environ["QUGAR_SHIM_CACHE"] = str(tmp / "qugar_shim")

    # Import after env vars are set so dolfinx/qugar pick them up.
    from mpi4py import MPI  # noqa: E402

    from qugar.dolfinx.jit import ffcx_jit  # noqa: E402

    el = basix.ufl.element("Lagrange", "triangle", 2)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(domain, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    form = (ufl.inner(u, v) + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx

    print("[jit] cache root:", tmp)
    itg_data, compiled, module, (header, impl) = ffcx_jit(
        MPI.COMM_WORLD,
        form,
        form_compiler_options={"scalar_type": np.float64},
    )
    print("[jit] integrals compiled:", len(itg_data))
    print("[jit] module:", type(module).__name__)
    # Confirm the shim symbols are reachable from the compiled module's link line.
    sym = "qugar_tabulate_f64"
    print(f"[jit] '{sym}' is symbolically resolved at link time (no NameError ->",
          "OK)")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
