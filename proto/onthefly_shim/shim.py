"""Reusable loader + pythonic wrapper for the isolated basix tabulation shim.

Builds the C++ shim (mimicking the future JIT-compile-and-cache step), loads it
through ctypes, and exposes a numpy-friendly ``tabulate`` that mirrors what the
generated C kernel will call at runtime.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import basix
import basix.ufl
import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = Path(sys.prefix)
# Canonical shim source now lives in the qugar package; proto compiles from
# the same file so the M0/M1 validation tests track any change to the shipped
# shim.
SRC = HERE.parent.parent / "python" / "qugar" / "dolfinx" / "_shim" / "qugar_basix_shim.cpp"
LIB = HERE / "libqugar_basix_shim.dylib"

_CT = {np.float64: ctypes.c_double, np.float32: ctypes.c_float}
_SUFFIX = {np.float64: "f64", np.float32: "f32"}


def build() -> None:
    """Compile the shim against the env's basix (cache: skip if up to date)."""
    if LIB.exists() and LIB.stat().st_mtime >= SRC.stat().st_mtime:
        return
    bindir = PREFIX / "bin"
    candidates = [bindir / "clang++", *sorted(bindir.glob("clang++-*"))]
    cxx = next((c for c in candidates if c.exists()), None)
    if cxx is None:
        raise FileNotFoundError(f"no clang++ driver found in {bindir}")
    cmd = [
        str(cxx), "-std=c++20", "-O2", "-fPIC", "-shared",
        f"-I{PREFIX / 'include'}", f"-L{PREFIX / 'lib'}", "-lbasix",
        f"-Wl,-rpath,{PREFIX / 'lib'}", "-o", str(LIB), str(SRC),
    ]
    subprocess.run(cmd, check=True)


def load() -> ctypes.CDLL:
    """Load the shim and configure ctypes signatures."""
    build()
    lib = ctypes.CDLL(str(LIB))
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

    Returns a numpy array of shape (n_derivs, n_points, n_dofs, value_size),
    matching basix's C++ tabulate layout.
    """
    suffix = _SUFFIX[dtype]
    ct = _CT[dtype]
    family, cell, degree, lv, dv, disc, _block = core_params(ufl_element)

    handle = getattr(lib, f"qugar_register_element_{suffix}")(
        family, cell, degree, lv, dv, disc)
    if handle < 0:
        raise RuntimeError(f"shim register failed: {handle}")

    pts = np.ascontiguousarray(points, dtype=dtype)
    npts, gdim = pts.shape

    out4 = (ctypes.c_long * 4)()
    rc = getattr(lib, f"qugar_tabulate_shape_{suffix}")(handle, nd, npts, out4)
    if rc != 0:
        raise RuntimeError(f"shim shape failed: {rc}")
    shape = tuple(int(x) for x in out4)
    need = int(np.prod(shape))

    basis = np.zeros(need, dtype=dtype)
    rc = getattr(lib, f"qugar_tabulate_{suffix}")(
        handle, nd, pts.ctypes.data_as(ctypes.POINTER(ct)), npts, gdim,
        basis.ctypes.data_as(ctypes.POINTER(ct)), need)
    if rc != 0:
        raise RuntimeError(f"shim tabulate failed: {rc}")
    return basis.reshape(shape)
