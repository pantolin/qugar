# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Direct tests of the basix tabulation shim used by the on-the-fly kernel.

Loads the production shim through ctypes and checks that on-the-fly
tabulation matches python-basix across:

  - real types: float64 AND float32 (the scalar-type axis),
  - several element families (scalar Lagrange, native-vector Raviart-Thomas),
  - cells of dim 2 and 3,
  - derivative orders 0 and 1.

Also asserts the element is created exactly once (registry does not grow
when tabulating repeatedly).
"""

from __future__ import annotations

import ctypes

import basix
import numpy as np
import pytest
from onthefly_helpers import load_shim  # type: ignore

_CELL_GDIM = {
    basix.CellType.triangle: 2,
    basix.CellType.quadrilateral: 2,
    basix.CellType.tetrahedron: 3,
    basix.CellType.hexahedron: 3,
}

_CT = {np.float64: ctypes.c_double, np.float32: ctypes.c_float}
_SUFFIX = {np.float64: "f64", np.float32: "f32"}


@pytest.fixture(scope="module")
def shim_lib():
    return load_shim()


_P = basix.ElementFamily.P
_RT = basix.ElementFamily.RT
_LV_ISAAC = basix.LagrangeVariant.gll_isaac
_LV_LEG = basix.LagrangeVariant.legendre
_DV_UNSET = basix.DPCVariant.unset
_LV_UNSET = basix.LagrangeVariant.unset

_CASES = [
    (_P, basix.CellType.triangle, 1, _LV_UNSET, _DV_UNSET, 1),
    (_P, basix.CellType.triangle, 2, _LV_UNSET, _DV_UNSET, 1),
    (_P, basix.CellType.triangle, 3, _LV_ISAAC, _DV_UNSET, 1),
    (_P, basix.CellType.quadrilateral, 2, _LV_UNSET, _DV_UNSET, 1),
    (_P, basix.CellType.tetrahedron, 2, _LV_UNSET, _DV_UNSET, 0),
    (_P, basix.CellType.hexahedron, 1, _LV_UNSET, _DV_UNSET, 1),
    (_RT, basix.CellType.triangle, 1, _LV_LEG, _DV_UNSET, 1),
    (_RT, basix.CellType.tetrahedron, 1, _LV_LEG, _DV_UNSET, 0),
]


def _case_id(case):
    fam, cell, deg, _lv, _dv, nd = case
    return f"{str(fam).split('.')[-1]}-{str(cell).split('.')[-1]}-d{deg}-nd{nd}"


@pytest.mark.parametrize("np_dtype", [np.float64, np.float32],
                         ids=["f64", "f32"])
@pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
def test_shim_matches_basix(shim_lib, np_dtype, case):
    """Per-element check: shim tabulation == python-basix, with the
    registry not growing when the same element is tabulated repeatedly."""
    family, cell, degree, lvariant, dvariant, nd = case
    is64 = np_dtype == np.float64
    suffix = _SUFFIX[np_dtype]
    ct = _CT[np_dtype]

    el = basix.create_element(family, cell, degree, lvariant, dvariant, False,
                              dtype=np_dtype)
    params = (int(el.family), int(el.cell_type), el.degree,
              int(el.lagrange_variant), int(el.dpc_variant),
              int(el.discontinuous))

    handle = getattr(shim_lib, f"qugar_register_element_{suffix}")(*params)
    assert handle >= 0, f"register failed ({handle}) for {family} deg {degree}"

    gdim = _CELL_GDIM[cell]
    rng = np.random.default_rng(0)
    npts = 5
    pts = (rng.random((npts, gdim)) * 0.25).astype(np_dtype)

    out4 = (ctypes.c_long * 4)()
    rc = getattr(shim_lib, f"qugar_tabulate_shape_{suffix}")(
        handle, nd, npts, out4)
    assert rc == 0, f"shape rc={rc}"
    shape = tuple(int(x) for x in out4)
    need = int(np.prod(shape))
    basis = np.zeros(need, dtype=np_dtype)

    # Create-once proof: tabulate many times, registry must not grow.
    size_before = getattr(shim_lib, f"qugar_registry_size_{suffix}")()
    tab = getattr(shim_lib, f"qugar_tabulate_{suffix}")
    for _ in range(50):
        rc = tab(handle, nd, pts.ctypes.data_as(ctypes.POINTER(ct)),
                 npts, gdim,
                 basis.ctypes.data_as(ctypes.POINTER(ct)), need)
        assert rc == 0, f"tabulate rc={rc}"
    size_after = getattr(shim_lib, f"qugar_registry_size_{suffix}")()
    assert size_after == size_before, "registry grew -> element recreated!"

    ref = np.asarray(el.tabulate(nd, pts))
    got = basis.reshape(shape)
    assert ref.size == got.size, f"size mismatch {ref.shape} vs {shape}"
    tol = 1e-12 if is64 else 1e-5
    np.testing.assert_allclose(got.ravel(), ref.ravel(), atol=tol, rtol=tol)
