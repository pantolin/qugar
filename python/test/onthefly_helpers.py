# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Test helpers for the on-the-fly basix tabulation shim.

Loads the *production* shim (the one ``qugar.dolfinx.jit`` links against)
through ctypes so tests can drive it directly without going via a JIT'd
kernel.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from qugar.dolfinx._shim import _LIB_FILENAME, ensure_built

_CT = {np.float64: ctypes.c_double, np.float32: ctypes.c_float}
_SUFFIX = {np.float64: "f64", np.float32: "f32"}


def _libfile() -> Path:
    cache, _ = ensure_built()
    return cache / _LIB_FILENAME


def load_shim() -> ctypes.CDLL:
    """Build (if needed) and load the production shim through ctypes."""
    lib = ctypes.CDLL(str(_libfile()))
    for dtype, suffix in _SUFFIX.items():
        ct = _CT[dtype]
        reg = getattr(lib, f"qugar_register_element_{suffix}")
        reg.argtypes = [ctypes.c_int] * 6
        reg.restype = ctypes.c_int
        shp = getattr(lib, f"qugar_tabulate_shape_{suffix}")
        shp.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.POINTER(ctypes.c_long)]
        shp.restype = ctypes.c_int
        tab = getattr(lib, f"qugar_tabulate_{suffix}")
        tab.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ct),
                        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ct),
                        ctypes.c_long]
        tab.restype = ctypes.c_int
        getattr(lib, f"qugar_registry_size_{suffix}").restype = ctypes.c_int
    lib.qugar_shim_reset.restype = None
    return lib


def core_params(ufl_element):
    """Map a basix.ufl element to core-basix create_element parameters.

    Returns (family, cell, degree, lvariant, dvariant, discontinuous, block_size).
    For blocked (vector/tensor) elements, returns the scalar sub-element's
    parameters plus its block size -- the shim tabulates the scalar element and
    the block expansion is a codegen concern.
    """
    block_size = getattr(ufl_element, "block_size", 1)
    scalar = ufl_element
    if block_size > 1:
        scalar = ufl_element.sub_elements[0]
    be = scalar.basix_element
    return (int(be.family), int(be.cell_type), be.degree,
            int(be.lagrange_variant), int(be.dpc_variant),
            int(be.discontinuous), block_size)


def tabulate(lib, ufl_element, dtype, nd, points):
    """Tabulate the (scalar core of the) element at ``points`` via the shim.

    Returns a numpy array of shape (n_derivs, n_points, n_dofs, value_size).
    """
    np_dtype = np.dtype(dtype).type
    suffix = _SUFFIX[np_dtype]
    ct = _CT[np_dtype]
    family, cell, degree, lv, dv, disc, _block = core_params(ufl_element)

    handle = getattr(lib, f"qugar_register_element_{suffix}")(
        family, cell, degree, lv, dv, disc)
    if handle < 0:
        raise RuntimeError(f"shim register failed: {handle}")

    pts = np.ascontiguousarray(points, dtype=np_dtype)
    npts, gdim = pts.shape

    out4 = (ctypes.c_long * 4)()
    rc = getattr(lib, f"qugar_tabulate_shape_{suffix}")(handle, nd, npts, out4)
    if rc != 0:
        raise RuntimeError(f"shim shape failed: {rc}")
    shape = tuple(int(x) for x in out4)
    need = int(np.prod(shape))

    basis = np.zeros(need, dtype=np_dtype)
    rc = getattr(lib, f"qugar_tabulate_{suffix}")(
        handle, nd, pts.ctypes.data_as(ctypes.POINTER(ct)), npts, gdim,
        basis.ctypes.data_as(ctypes.POINTER(ct)), need)
    if rc != 0:
        raise RuntimeError(f"shim tabulate failed: {rc}")
    return basis.reshape(shape)
