# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for the coefficient-packing pipeline of qugar custom forms.

Targets blind spots that the standard ``test_scalar`` / ``test_vector``
/ ``test_matrix`` suites only cover lightly:

* the ``update_coefficients`` flow (used by LinearProblem /
  NonlinearProblem to avoid re-packing the custom part when a Function
  changes between solves);
* forms with many ``Function`` coefficients;
* ``fem.Constant`` participating in custom forms (alongside
  ``Function`` coefficients).
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import dolfinx
import dolfinx.fem
import numpy as np
import pytest
import ufl
from utils import (  # type: ignore
    check_vals,
    create_mock_unfitted_mesh,
    dtypes,
    get_dolfinx_forms,
    run_matrix_check,
    run_scalar_check,
    run_vector_check,
)


_N = 4
_NNZ = 0.3
_MAX_QUAD = 3


# ---------------------------------------------------------------------------
# update_coefficients flow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_update_coefficients_scalar(dim, simplex_cell, dtype):
    """``CustomForm.update_coefficients`` should produce the same
    assembled value as a fresh ``pack_coefficients`` after a coefficient
    Function is mutated.

    This is the path Newton iterations / time stepping rely on to avoid
    re-packing the custom-quadrature portion of the coefficient blob on
    every solve.
    """
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    f = dolfinx.fem.Function(V, dtype=dtype)
    f.interpolate(lambda x: 1 + x[0] ** 2)  # type: ignore
    v = ufl.TestFunction(V)
    ufl_form = ufl.inner(f, v) * ufl.dx(domain=unf)

    custom_form, _ = get_dolfinx_forms(ufl_form, unf)

    # Initial pack + assemble.
    coeffs_v1 = custom_form.pack_coefficients()
    b1 = dolfinx.fem.assemble_vector(custom_form, coeffs=coeffs_v1)

    # Mutate the coefficient.
    f.interpolate(lambda x: 1 + 3 * x[0] ** 2 + x[1] ** 2)  # type: ignore

    # Update the coefficients in place via update_coefficients...
    coeffs_v2 = custom_form.update_coefficients(coeffs_v1)
    b2_update = dolfinx.fem.assemble_vector(custom_form, coeffs=coeffs_v2)

    # ...and via a fresh pack from scratch.
    coeffs_v2_fresh = custom_form.pack_coefficients()
    b2_fresh = dolfinx.fem.assemble_vector(custom_form, coeffs=coeffs_v2_fresh)

    # update_coefficients must reflect the new f values.
    assert not np.allclose(b1.array, b2_update.array), (
        "update_coefficients silently ignored the coefficient mutation"
    )
    # ...and must match the fresh-pack result.
    check_vals(b2_update.array, b2_fresh.array, dtype=dtype)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_update_coefficients_matrix(dim, simplex_cell, dtype):
    """Same as ``test_update_coefficients_scalar`` but for a matrix form
    with a coefficient inside ``grad-grad``."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    e = dolfinx.fem.Function(V, dtype=dtype)
    e.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)  # type: ignore
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ufl_form = e * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)

    custom_form, _ = get_dolfinx_forms(ufl_form, unf)

    coeffs_v1 = custom_form.pack_coefficients()
    A1 = dolfinx.fem.assemble_matrix(custom_form, coeffs=coeffs_v1)

    e.interpolate(lambda x: 5 + 2 * x[0] - 3 * x[1])  # type: ignore

    coeffs_v2 = custom_form.update_coefficients(coeffs_v1)
    A2_update = dolfinx.fem.assemble_matrix(custom_form, coeffs=coeffs_v2)

    coeffs_v2_fresh = custom_form.pack_coefficients()
    A2_fresh = dolfinx.fem.assemble_matrix(custom_form, coeffs=coeffs_v2_fresh)

    assert not np.allclose(A1.data, A2_update.data), (
        "update_coefficients silently ignored the coefficient mutation"
    )
    check_vals(A2_update.data, A2_fresh.data, dtype=dtype)


# ---------------------------------------------------------------------------
# Multi-coefficient forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_many_function_coefficients(dim, simplex_cell, dtype):
    """Scalar form with four distinct ``Function`` coefficients on
    different function spaces. Exercises the coefficient-offset and
    layout bookkeeping in ``CustomCoeffsPacker``."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V_s1 = dolfinx.fem.functionspace(unf, ("Lagrange", 1))
    V_s2 = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    V_v1 = dolfinx.fem.functionspace(unf, ("Lagrange", 1, (dim,)))
    V_v2 = dolfinx.fem.functionspace(unf, ("Lagrange", 2, (dim,)))

    f1 = dolfinx.fem.Function(V_s1, dtype=dtype)
    f1.interpolate(lambda x: 1 + x[0])  # type: ignore
    f2 = dolfinx.fem.Function(V_s2, dtype=dtype)
    f2.interpolate(lambda x: 2 + x[0] ** 2)  # type: ignore

    g1 = dolfinx.fem.Function(V_v1, dtype=dtype)
    if dim == 2:
        g1.interpolate(lambda x: np.vstack((x[0], x[1])))  # type: ignore
    else:
        g1.interpolate(lambda x: np.vstack((x[0], x[1], x[2])))  # type: ignore

    g2 = dolfinx.fem.Function(V_v2, dtype=dtype)
    if dim == 2:
        g2.interpolate(lambda x: np.vstack((x[1] ** 2, x[0] * x[1])))  # type: ignore
    else:
        g2.interpolate(
            lambda x: np.vstack((x[1] ** 2, x[0] * x[1], x[2] - x[0]))  # type: ignore
        )

    ufl_form = (
        f1 * f2 + ufl.inner(g1, g2)
    ) * ufl.dx(domain=unf)
    run_scalar_check(ufl_form, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_many_coefficients_matrix(dim, simplex_cell, dtype):
    """Bilinear form with three Function coefficients (scalar field +
    vector field + tensor-like contraction). Verifies the matrix
    assembly path handles multiple coefficients sharing a single
    integrand."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    V_v = dolfinx.fem.functionspace(unf, ("Lagrange", 1, (dim,)))

    e = dolfinx.fem.Function(V, dtype=dtype)
    e.interpolate(lambda x: 1 + x[0])  # type: ignore
    rho = dolfinx.fem.Function(V, dtype=dtype)
    rho.interpolate(lambda x: 2 + x[1] ** 2)  # type: ignore

    beta = dolfinx.fem.Function(V_v, dtype=dtype)
    if dim == 2:
        beta.interpolate(lambda x: np.vstack((1 + x[0], 1 - x[1])))  # type: ignore
    else:
        beta.interpolate(
            lambda x: np.vstack((1 + x[0], 1 - x[1], x[2]))  # type: ignore
        )

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ufl_form = (
        e * ufl.inner(ufl.grad(u), ufl.grad(v))
        + rho * u * v
        + u * ufl.inner(beta, ufl.grad(v))
    ) * ufl.dx(domain=unf)
    run_matrix_check(ufl_form, unf)


# ---------------------------------------------------------------------------
# Constants in custom forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_constant_scalar(dim, simplex_cell, dtype):
    """``fem.Constant`` (scalar) participates in a custom scalar
    integral."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    c = dolfinx.fem.Constant(unf, dtype(2.5))
    ufl_form = c * ufl.dx(domain=unf)
    run_scalar_check(ufl_form, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_constant_vector(dim, simplex_cell, dtype):
    """``fem.Constant`` (vector) used in a custom vector integral
    alongside a ``Function``."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2, (dim,)))
    c = dolfinx.fem.Constant(unf, np.array([2.5] * dim, dtype=dtype))
    f = dolfinx.fem.Function(V, dtype=dtype)
    if dim == 2:
        f.interpolate(lambda x: np.vstack((x[0], x[1])))  # type: ignore
    else:
        f.interpolate(lambda x: np.vstack((x[0], x[1], x[2])))  # type: ignore
    v = ufl.TestFunction(V)
    ufl_form = ufl.inner(c + f, v) * ufl.dx(domain=unf)
    run_vector_check(ufl_form, unf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_constant_and_function_in_matrix(dim, simplex_cell, dtype):
    """A bilinear form mixing a ``Constant`` multiplier with a
    ``Function`` coefficient inside a ``grad-grad``."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    mu = dolfinx.fem.Constant(unf, dtype(0.75))
    rho = dolfinx.fem.Function(V, dtype=dtype)
    rho.interpolate(lambda x: 1 + x[0] ** 2)  # type: ignore
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ufl_form = mu * rho * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)
    run_matrix_check(ufl_form, unf)
