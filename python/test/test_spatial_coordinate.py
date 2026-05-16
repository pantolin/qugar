# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for ``ufl.SpatialCoordinate`` inside qugar custom forms.

When a UFL form references ``x = ufl.SpatialCoordinate(mesh)``, ffcx
emits FE tables for the *coordinate element* (typically the linear
Lagrange element used to define the mesh geometry). qugar's custom
quadrature path must therefore correctly tabulate the coordinate
element at the runtime quadrature points -- a code path that is
exercised indirectly in ``test_div_thm`` but never with an explicit
assertion on the SpatialCoordinate-driven values.

These tests build forms whose integrands depend explicitly on the
spatial coordinates and verify the assembled value matches what
standard DOLFINx assembly produces on the same mock unfitted mesh
(where the two are by construction equivalent).
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import dolfinx
import dolfinx.fem
import pytest
import ufl
from utils import (  # type: ignore
    create_mock_unfitted_mesh,
    dtypes,
    run_matrix_check,
    run_scalar_check,
    run_vector_check,
)

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_spatial_coordinate_scalar(dim, simplex_cell, dtype):
    """Integrate a polynomial ``x[0]**2 + x[1]**2`` over the cut
    domain. ffcx will tabulate the coordinate element at the custom
    quadrature points to evaluate the integrand."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    x = ufl.SpatialCoordinate(unf)
    expr = x[0] ** 2 + x[1] ** 2
    if dim == 3:
        expr = expr + x[2] ** 2
    ufl_form = expr * ufl.dx(domain=unf)
    run_scalar_check(ufl_form, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_spatial_coordinate_in_rhs_vector(dim, simplex_cell, dtype):
    """Vector RHS where the load depends on ``SpatialCoordinate``.
    Combines coordinate-element tabulation with the
    ``TestFunction`` basis evaluation."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(unf)
    load = ufl.sin(x[0]) * ufl.cos(x[1])
    ufl_form = load * v * ufl.dx(domain=unf)
    run_vector_check(ufl_form, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_spatial_coordinate_in_matrix_coefficient(dim, simplex_cell, dtype):
    """Bilinear form whose coefficient is a closed-form function of
    ``SpatialCoordinate`` (no ``Function`` involved at all). The
    custom quadrature must evaluate the coefficient correctly per
    point."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(unf)
    coeff = 1.0 + x[0] + x[1]
    ufl_form = coeff * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)
    run_matrix_check(ufl_form, unf)
