"""Assembly profiling for the on-the-fly path.

Uses the same mock_unfitted_mesh as run_matrix_check but with ``nnz=1.0`` so
every cell goes through the custom (on-the-fly) kernel. Standard FFCx-baked
assembly is the lower bound -- it does no per-cell tabulation. The reported
``ratio`` is the multiplicative cost of going on-the-fly vs. baked.

Numbers reported (best of N repeats so JIT and warmup don't pollute):
- pack:  Python-side per-cell points/weights packing into w_custom.
- custom: dolfinx assemble_matrix through the on-the-fly kernel.
- std:    dolfinx assemble_matrix through stock FFCx (baked tables).

Run with the qugar-0.10.0 env python.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import basix.ufl
import dolfinx.fem
import numpy as np
import ufl
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "test"))
from utils import create_mock_unfitted_mesh, get_dolfinx_forms  # type: ignore  # noqa: E402


def bench(dim: int, N: int, p: int, simplex: bool, dtype, repeats: int = 3):
    unf_mesh = create_mock_unfitted_mesh(dim, N, simplex, 1.0, 1, dtype)
    V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (ufl.inner(u, v) + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    custom_form, form = get_dolfinx_forms(a, unf_mesh)

    # Warm up (first call triggers JIT compile + shim build + cache fill).
    custom_form.pack_coefficients()
    coeffs = custom_form.pack_coefficients()
    dolfinx.fem.assemble_matrix(custom_form, coeffs=coeffs)
    dolfinx.fem.assemble_matrix(form)

    times_pack, times_custom, times_std = [], [], []
    for _ in range(repeats):
        t = time.perf_counter()
        coeffs = custom_form.pack_coefficients()
        times_pack.append(time.perf_counter() - t)

        t = time.perf_counter()
        dolfinx.fem.assemble_matrix(custom_form, coeffs=coeffs)
        times_custom.append(time.perf_counter() - t)

        t = time.perf_counter()
        dolfinx.fem.assemble_matrix(form)
        times_std.append(time.perf_counter() - t)

    n_cells = unf_mesh.topology.index_map(dim).size_local
    return n_cells, min(times_pack), min(times_custom), min(times_std)


def header() -> None:
    print(f"{'dim':>3} {'p':>2} {'simplex':>7} {'N':>4} {'cells':>7} "
          f"{'pack(ms)':>9} {'custom(ms)':>11} {'std(ms)':>9} "
          f"{'cust/std':>8}")


def main() -> int:
    if MPI.COMM_WORLD.rank != 0:
        return 0
    header()
    # 2D scalar Lagrange p=1,2 across mesh sizes.
    for p in (1, 2):
        for simplex in (True, False):
            for N in (16, 32, 64):
                nc, pack, cu, st = bench(2, N, p, simplex, np.float64)
                print(f"  2 {p:>2} {str(simplex):>7} {N:>4} {nc:>7} "
                      f"{pack*1e3:>9.2f} {cu*1e3:>11.2f} {st*1e3:>9.2f} "
                      f"{cu/st:>8.2f}")
    # 3D scalar Lagrange p=1 (3D blows up fast).
    for simplex in (True, False):
        for N in (8, 16):
            nc, pack, cu, st = bench(3, N, 1, simplex, np.float64)
            print(f"  3  1 {str(simplex):>7} {N:>4} {nc:>7} "
                  f"{pack*1e3:>9.2f} {cu*1e3:>11.2f} {st*1e3:>9.2f} "
                  f"{cu/st:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
