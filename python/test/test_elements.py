# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests exercising custom-form assembly with element families other
than continuous Lagrange.

The standard ``test_scalar`` / ``test_vector`` / ``test_matrix`` suites
only use ``Lagrange`` (and its vector variant). The qugar custom
form / quadrature pipeline must also handle:

* Discontinuous Galerkin (``Lagrange`` with ``discontinuous=True``) --
  same basis as Lagrange but with a different DOF layout.
* H(curl) Nedelec elements -- covariant Piola transform applied by the
  ffcx kernel; qugar's job is to provide reference-space basis values
  in the right component ordering.
* H(div) Raviart-Thomas / BDM elements -- contravariant Piola transform
  applied by the ffcx kernel; same constraint on qugar.
* Mixed elements built via ``basix.ufl.mixed_element`` -- multiple
  sub-elements sharing a single function space. qugar drills into the
  correct sub-element per flat component (see
  ``fe_table_eval._resolve_mixed_component``) instead of calling
  ``tabulate`` on the mixed element directly.
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import basix.ufl
import dolfinx
import dolfinx.fem
import numpy as np
import pytest
import ufl
from utils import create_mock_unfitted_mesh, dtypes, run_matrix_check  # type: ignore

# Small, fixed N: these tests exercise element-handling code paths, not
# mesh refinement.
_N = 4
_NNZ = 0.3
_MAX_QUAD = 3


# ---------------------------------------------------------------------------
# Continuous Lagrange baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_lagrange_scalar_mass(dim, simplex_cell, dtype):
    """Sanity baseline: scalar Lagrange mass matrix."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    el = basix.ufl.element("Lagrange", unf.basix_cell(), 2, dtype=dtype)
    V = dolfinx.fem.functionspace(unf, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)


# ---------------------------------------------------------------------------
# Discontinuous Galerkin (DG)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_dg_scalar_mass(dim, simplex_cell, dtype):
    """Discontinuous scalar Lagrange (DG) mass matrix."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    el = basix.ufl.element(
        "Lagrange", unf.basix_cell(), 2, discontinuous=True, dtype=dtype
    )
    V = dolfinx.fem.functionspace(unf, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_dg_vector_grad_grad(dim, simplex_cell, dtype):
    """Vector-valued DG, ``grad-grad`` style integral (still continuous
    on each cell so ``grad`` is well-defined cell-wise)."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    el = basix.ufl.element(
        "Lagrange", unf.basix_cell(), 2, discontinuous=True, shape=(dim,), dtype=dtype
    )
    V = dolfinx.fem.functionspace(unf, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)


# ---------------------------------------------------------------------------
# Piola-mapped elements -- currently broken in qugar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", [2, 3])
def test_nedelec_1st_mass(dim, dtype):
    """Nedelec 1st kind H(curl) mass matrix on simplex cells."""
    unf = create_mock_unfitted_mesh(
        dim, _N, simplex_cell=True, nnz=_NNZ, max_quad_sets=_MAX_QUAD, dtype=dtype
    )
    el = basix.ufl.element(
        "Nedelec 1st kind H(curl)", unf.basix_cell(), 1, dtype=dtype
    )
    V = dolfinx.fem.functionspace(unf, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", [2, 3])
def test_nedelec_2nd_mass(dim, dtype):
    """Nedelec 2nd kind H(curl) mass matrix on simplex cells."""
    unf = create_mock_unfitted_mesh(
        dim, _N, simplex_cell=True, nnz=_NNZ, max_quad_sets=_MAX_QUAD, dtype=dtype
    )
    el = basix.ufl.element(
        "Nedelec 2nd kind H(curl)", unf.basix_cell(), 1, dtype=dtype
    )
    V = dolfinx.fem.functionspace(unf, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", [2, 3])
def test_raviart_thomas_mass(dim, dtype):
    """Raviart-Thomas H(div) mass matrix on simplex cells."""
    unf = create_mock_unfitted_mesh(
        dim, _N, simplex_cell=True, nnz=_NNZ, max_quad_sets=_MAX_QUAD, dtype=dtype
    )
    el = basix.ufl.element("Raviart-Thomas", unf.basix_cell(), 1, dtype=dtype)
    V = dolfinx.fem.functionspace(unf, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", [2, 3])
def test_bdm_mass(dim, dtype):
    """Brezzi-Douglas-Marini H(div) mass matrix on simplex cells."""
    unf = create_mock_unfitted_mesh(
        dim, _N, simplex_cell=True, nnz=_NNZ, max_quad_sets=_MAX_QUAD, dtype=dtype
    )
    el = basix.ufl.element(
        "Brezzi-Douglas-Marini", unf.basix_cell(), 1, dtype=dtype
    )
    V = dolfinx.fem.functionspace(unf, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)


# ---------------------------------------------------------------------------
# Mixed elements (basix.ufl.mixed_element)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_mixed_taylor_hood(dim, simplex_cell, dtype):
    """Taylor-Hood style mixed space (P2 vector velocity + P1 scalar
    pressure)."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    el_v = basix.ufl.element("Lagrange", unf.basix_cell(), 2, shape=(dim,), dtype=dtype)
    el_q = basix.ufl.element("Lagrange", unf.basix_cell(), 1, dtype=dtype)
    el_mixed = basix.ufl.mixed_element([el_v, el_q])
    V = dolfinx.fem.functionspace(unf, el_mixed)
    up, vq = ufl.TrialFunction(V), ufl.TestFunction(V)
    u, p = ufl.split(up)
    v, q = ufl.split(vq)
    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v))
        + p * ufl.div(v)
        + q * ufl.div(u)
    ) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_mixed_lagrange_dg(dim, simplex_cell, dtype):
    """Mixed continuous Lagrange (vector) + discontinuous Lagrange
    (scalar) -- the latter is the typical pressure space in DG-CR
    formulations."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    el_v = basix.ufl.element("Lagrange", unf.basix_cell(), 2, shape=(dim,), dtype=dtype)
    el_q = basix.ufl.element(
        "Lagrange", unf.basix_cell(), 1, discontinuous=True, dtype=dtype
    )
    el_mixed = basix.ufl.mixed_element([el_v, el_q])
    V = dolfinx.fem.functionspace(unf, el_mixed)
    up, vq = ufl.TrialFunction(V), ufl.TestFunction(V)
    u, p = ufl.split(up)
    v, q = ufl.split(vq)
    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v))
        + p * ufl.div(v)
        + q * ufl.div(u)
    ) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)


@pytest.mark.parametrize("dtype", dtypes)
def test_mixed_nedelec_lagrange(dtype):
    """Mixed H(curl) formulation: Nedelec + Lagrange (3D simplex)."""
    unf = create_mock_unfitted_mesh(
        dim=3, N=_N, simplex_cell=True, nnz=_NNZ, max_quad_sets=_MAX_QUAD, dtype=dtype
    )
    el_e = basix.ufl.element(
        "Nedelec 1st kind H(curl)", unf.basix_cell(), 1, dtype=dtype
    )
    el_q = basix.ufl.element("Lagrange", unf.basix_cell(), 1, dtype=dtype)
    el_mixed = basix.ufl.mixed_element([el_e, el_q])
    V = dolfinx.fem.functionspace(unf, el_mixed)
    up, vq = ufl.TrialFunction(V), ufl.TestFunction(V)
    u, p = ufl.split(up)
    v, q = ufl.split(vq)
    a = (
        ufl.inner(ufl.curl(u), ufl.curl(v))
        + ufl.inner(u, ufl.grad(q))
        + ufl.inner(ufl.grad(p), v)
    ) * ufl.dx(domain=unf)
    run_matrix_check(a, unf)
