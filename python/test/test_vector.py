# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for custom Python vector assemblers."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import dolfinx
import dolfinx.fem
import numpy as np
import pytest
import ufl
from utils import clean_cache, create_mock_unfitted_mesh, dtypes, run_vector_check  # type: ignore


@pytest.mark.parametrize("max_quad_sets", [3])
@pytest.mark.parametrize("nnz", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("dx", [ufl.dx, ufl.ds])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("N", [1, 4])
def test_v_f(
    N: int,
    dim: int,
    dx: ufl.Measure,
    p: int,
    simplex_cell: bool,
    dtype: type[np.float32 | np.float64],
    nnz: float,
    max_quad_sets: int,
) -> None:
    """Tests the integral of a vector quantity (a function times a
    test function).

    It checks that the integral computed using a mock custom quadrature
    is the same as the one using standard integrals.

    Args:
        N (int): Number of cells per direction in the mes.
        dim (int): Domain's dimension (either 2D or 3D).
        dx (ufl.Measure): Measure to be used (dx for cell integrals or
            ds for boundary integrals).
        p (int): Base degree to used in finite element spaces.
        simplex_cell (bool): If ``True`` simplex cells (triangles or
            tetrahedra) are used. Otherwise, quadrilaterals or
            hexahedra.
        dtype (type[np.float32 | np.float64]): `numpy` type to be used for
            scalars
        nnz (float): Ratio of entities with custom quadratures respect
            to the total number of entities. It is a value in the range
            [0.0, 1.0].
        max_quad_sets (int): Maximum number of repetitions of the
            standard quadrature to be used for each custom entity
            quadrature. For each custom entity a random number is
            generated between 1 and `max_quad_sets`.
    """

    unf_mesh = create_mock_unfitted_mesh(dim, N, simplex_cell, nnz, max_quad_sets, dtype)

    V0 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p))
    V1 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p + 1))

    v = ufl.TestFunction(V0)

    e = dolfinx.fem.Function(V1, dtype=dtype)
    e.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)  # type: ignore

    ufl_form_0 = e * v * dx(domain=unf_mesh)  # type: ignore
    ufl_form = ufl_form_0

    run_vector_check(ufl_form, unf_mesh)


@pytest.mark.parametrize("max_quad_sets", [3])
@pytest.mark.parametrize("nnz", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("N", [1, 4])
def test_interior_facet(
    N: int,
    dim: int,
    p: int,
    simplex_cell: bool,
    dtype: type[np.float32 | np.float64],
    nnz: float,
    max_quad_sets: int,
) -> None:
    """Tests the integral of a vector quantity in interior facets
    (the jump of a test function times another function)

    It checks that the integral computed using a mock custom quadrature
    is the same as the one using standard integrals.

    Args:
        N (int): Number of cells per direction in the mes.
        dim (int): Domain's dimension (either 2D or 3D).
        p (int): Base degree to used in finite element spaces.
        simplex_cell (bool): If ``True`` simplex cells (triangles or
            tetrahedra) are used. Otherwise, quadrilaterals or
            hexahedra.
        dtype (type[np.float32 | np.float64]): `numpy` type to be used for
            scalars
        nnz (float): Ratio of entities with custom quadratures respect
            to the total number of entities. It is a value in the range
            [0.0, 1.0].
        max_quad_sets (int): Maximum number of repetitions of the
            standard quadrature to be used for each custom entity
            quadrature. For each custom entity a random number is
            generated between 1 and `max_quad_sets`.
    """

    unf_mesh = create_mock_unfitted_mesh(dim, N, simplex_cell, nnz, max_quad_sets, dtype)

    V0 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p))
    V1 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p))

    v = ufl.TestFunction(V0)

    f = dolfinx.fem.Function(V1, dtype=dtype)
    ufl_form = f * ufl.jump(v) * ufl.dS  # type: ignore

    run_vector_check(ufl_form, unf_mesh)


if __name__ == "__main__":
    clean_cache()
    # test_v_f(
    #     N=1, dim=2, p=1, simplex_cell=False, dtype=np.float64, nnz=0.2, max_quad_sets=1
    # )
    test_interior_facet(
        N=6, dim=2, p=1, simplex_cell=True, dtype=np.float32, nnz=0.3, max_quad_sets=3
    )
