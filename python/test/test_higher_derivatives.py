# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for forms involving second (and mixed) derivatives.

All existing assembly tests use only first-order ``grad(u)``. Forms
with higher-order derivatives like ``grad(grad(u))`` exercise a
separate code path in qugar's FE-table evaluation: ffcx emits FE
tables with non-zero derivative indices (the ``_D###`` suffix in the
table names), and qugar must tabulate the *derivatives* of the basis
functions at the custom quadrature points -- not just the basis
values.

The forms below assemble against the mock unfitted mesh and compare
to standard DOLFINx assembly, which by construction must match.
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
)

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_hessian_scalar_integral(dim, simplex_cell, dtype):
    """Scalar integral of the Frobenius norm of the Hessian of a
    Function: ``inner(grad(grad(u)), grad(grad(u))) * dx``.

    Requires second derivatives of the basis. For a constant
    Function this is trivially zero, so the assertion is that
    custom assembly agrees with standard assembly on a non-trivial
    Function.
    """
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 3))
    u = dolfinx.fem.Function(V, dtype=dtype)
    # Non-degenerate (non-quadratic) Function so grad(grad(u)) is
    # non-trivial on the support of each basis function.
    u.interpolate(lambda x: 1.0 + x[0] ** 3 + x[1] ** 3)  # type: ignore

    ufl_form = ufl.inner(ufl.grad(ufl.grad(u)), ufl.grad(ufl.grad(u))) * ufl.dx(
        domain=unf
    )
    run_scalar_check(ufl_form, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_biharmonic_matrix(dim, simplex_cell, dtype):
    """Biharmonic-like bilinear form
    ``inner(grad(grad(u)), grad(grad(v))) * dx`` with C^1 basis
    enforced weakly (we only assemble the matrix here -- the
    assembled result with C^0 Lagrange is non-physical, but the
    assembly itself must match standard ffcx output)."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 3))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ufl_form = ufl.inner(ufl.grad(ufl.grad(u)), ufl.grad(ufl.grad(v))) * ufl.dx(
        domain=unf
    )
    run_matrix_check(ufl_form, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_laplacian_term_in_matrix(dim, simplex_cell, dtype):
    """Bilinear form with ``laplace(u)`` (trace of ``grad(grad(u))``)
    -- another typical higher-derivative usage that goes through the
    same FE-table-with-D### path."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 3))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ufl_form = ufl.div(ufl.grad(u)) * v * ufl.dx(domain=unf)
    run_matrix_check(ufl_form, unf)
