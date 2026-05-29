"""Isolated M0 prototype driver.

Compiles the C++ shim (mimicking the future JIT-compile-and-cache step),
loads it through a plain C ABI (ctypes), and checks that on-the-fly tabulation
matches python-basix across:
  - real types: float64 AND float32 (the scalar-type axis),
  - several element families (scalar Lagrange, native-vector Raviart-Thomas),
  - cells of dim 2 and 3,
  - derivative orders 0 and 1.

Also asserts the element is created exactly once (registry does not grow when
tabulating repeatedly) -- the hard performance requirement.

Run with the qugar-0.10.0 env python, e.g.:
    /Users/antolin/miniconda3/envs/qugar-0.10.0/bin/python proto/onthefly_shim/test_shim.py
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import basix
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shim import build, load  # noqa: E402


CELL_GDIM = {
    basix.CellType.triangle: 2,
    basix.CellType.quadrilateral: 2,
    basix.CellType.tetrahedron: 3,
    basix.CellType.hexahedron: 3,
}


def check(lib, np_dtype, family, cell, degree, lvariant, dvariant, nd):
    """Compare shim tabulation against python-basix for one element/type."""
    is64 = np_dtype == np.float64
    suffix = "f64" if is64 else "f32"
    ct = ctypes.c_double if is64 else ctypes.c_float

    el = basix.create_element(family, cell, degree, lvariant, dvariant, False,
                              dtype=np_dtype)
    # Pass the element's *resolved* parameters so the shim builds the identical
    # element (variants may be resolved from 'unset').
    params = (int(el.family), int(el.cell_type), el.degree,
              int(el.lagrange_variant), int(el.dpc_variant),
              int(el.discontinuous))

    handle = getattr(lib, f"qugar_register_element_{suffix}")(*params)
    assert handle >= 0, f"register failed ({handle}) for {family} deg {degree}"

    gdim = CELL_GDIM[cell]
    rng = np.random.default_rng(0)
    npts = 5
    pts = (rng.random((npts, gdim)) * 0.25).astype(np_dtype)

    # Shape from the shim.
    out4 = (ctypes.c_long * 4)()
    rc = getattr(lib, f"qugar_tabulate_shape_{suffix}")(handle, nd, npts, out4)
    assert rc == 0, f"shape rc={rc}"
    shape = tuple(int(x) for x in out4)  # (nderiv, npts, ndofs, vs)
    need = int(np.prod(shape))
    basis = np.zeros(need, dtype=np_dtype)

    # Create-once proof: tabulate many times, registry must not grow.
    size_before = getattr(lib, f"qugar_registry_size_{suffix}")()
    tab = getattr(lib, f"qugar_tabulate_{suffix}")
    for _ in range(50):
        rc = tab(handle, nd, pts.ctypes.data_as(ctypes.POINTER(ct)), npts, gdim,
                 basis.ctypes.data_as(ctypes.POINTER(ct)), need)
        assert rc == 0, f"tabulate rc={rc}"
    size_after = getattr(lib, f"qugar_registry_size_{suffix}")()
    assert size_after == size_before, "registry grew -> element recreated!"

    # Reference from python-basix.
    ref = np.asarray(el.tabulate(nd, pts))
    got = basis.reshape(shape)
    assert ref.size == got.size, f"size mismatch {ref.shape} vs {shape}"
    tol = 1e-12 if is64 else 1e-5
    ok = np.allclose(ref.ravel(), got.ravel(), atol=tol, rtol=tol)
    status = "OK " if ok else "FAIL"
    maxerr = float(np.max(np.abs(ref.ravel() - got.ravel())))
    print(f"  [{status}] {str(family):28s} deg={degree} cell={str(cell):24s} "
          f"nd={nd} dtype={np_dtype.__name__:8s} shape={shape} maxerr={maxerr:.2e}")
    return ok


def main() -> int:
    build()
    lib = load()
    P = basix.ElementFamily.P
    RT = basix.ElementFamily.RT
    LV = basix.LagrangeVariant.gll_isaac
    DV = basix.DPCVariant.unset
    UNSET = basix.LagrangeVariant.unset

    cases = [
        # scalar Lagrange, both real types, 2D + 3D, nd 0 and 1
        (P, basix.CellType.triangle, 1, UNSET, DV, 1),
        (P, basix.CellType.triangle, 2, UNSET, DV, 1),
        (P, basix.CellType.triangle, 3, LV, DV, 1),
        (P, basix.CellType.quadrilateral, 2, UNSET, DV, 1),
        (P, basix.CellType.tetrahedron, 2, UNSET, DV, 0),
        (P, basix.CellType.hexahedron, 1, UNSET, DV, 1),
        # native-vector element (value_size > 1) -> exercises the 4th axis
        (RT, basix.CellType.triangle, 1, basix.LagrangeVariant.legendre, DV, 1),
        (RT, basix.CellType.tetrahedron, 1, basix.LagrangeVariant.legendre, DV, 0),
    ]

    all_ok = True
    for dtype in (np.float64, np.float32):
        print(f"--- dtype={dtype.__name__} ---")
        for fam, cell, deg, lv, dv, nd in cases:
            all_ok &= check(lib, dtype, fam, cell, deg, lv, dv, nd)
        lib.qugar_shim_reset()  # exercise teardown between type sweeps

    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
