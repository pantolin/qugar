# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for custom Python scalar assemblers."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from typing import cast

import dolfinx
import dolfinx.fem
import numpy as np
import pytest
import ufl
from utils import clean_cache, create_mock_unfitted_mesh, dtypes, run_scalar_check  # type: ignore


@pytest.mark.parametrize("max_quad_sets", [3])
@pytest.mark.parametrize("nnz", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dx", [ufl.dx, ufl.ds])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("N", [1, 4])
def test_measure(
    N: int,
    dim: int,
    dx: ufl.Measure,
    simplex_cell: bool,
    dtype: type[np.float32 | np.float64],
    nnz: float,
    max_quad_sets: int,
) -> None:
    """Test the measture of a domain (volume/area or boundary
    area/perimeter) for a custom quadrature generated with the
    mock custom quadrature generator.

    It checks that the integral computed using such mock custom
    quadrature is the same as the one using standard integrals.

    Args:
        N (int): Number of cells per direction in the mes.
        dim (int): Domain's dimension (either 2D or 3D).
        dx (ufl.Measure): Measure to be used (dx for cell integrals or
            ds for boundary integrals).
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

    one = dolfinx.fem.Constant(unf_mesh, dtype(1.0))

    ufl_form = one * dx(domain=unf_mesh)

    run_scalar_check(ufl_form, unf_mesh)


@pytest.mark.parametrize("max_quad_sets", [3])
@pytest.mark.parametrize("nnz", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("dx", [ufl.dx, ufl.ds])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("N", [1, 4])
def test_scalar(
    N: int,
    dim: int,
    dx: ufl.Measure,
    p: int,
    simplex_cell: bool,
    dtype: type[np.float32 | np.float64],
    nnz: float,
    max_quad_sets: int,
) -> None:
    """Tests the integral of a scalar quantity.

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

    V0 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p, (2,)))
    V1 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p + 1, (2,)))
    V2 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p))

    e = dolfinx.fem.Function(V0, dtype=dtype)
    e.interpolate(  # type: ignore
        lambda x: np.vstack((1 + x[0] ** 2 + 2 * x[1] ** 2, 1 + x[0] ** 2 + 3 * x[1] ** 2))
    )

    f = dolfinx.fem.Function(V1, dtype=dtype)
    f.interpolate(  # type: ignore
        lambda x: np.vstack((2 + x[0] ** 2 + 4 * x[1] ** 2, 1 + x[0] ** 2 + 5 * x[1] ** 2))
    )
    constant_0 = dolfinx.fem.Constant(unf_mesh, np.array([5.0, 6.0], dtype=dtype))

    ff = dolfinx.fem.Function(V2, dtype=dtype)
    ff.interpolate(lambda x: np.sin(x[0] * x[1]))  # type: ignore

    ufl_form_0 = cast(ufl.Form, ufl.inner(e, constant_0) * dx(domain=unf_mesh))
    ufl_form_1 = cast(ufl.Form, ufl.inner(f, constant_0) * dx(domain=unf_mesh))
    ufl_form = ufl_form_0 + ufl_form_1

    run_scalar_check(ufl_form, unf_mesh)


if __name__ == "__main__":
    clean_cache()
    test_measure(
        N=1,
        dim=2,
        dx=ufl.ds,
        simplex_cell=True,
        dtype=np.float32,
        nnz=0.0,
        max_quad_sets=3,
    )
    # test_scalar(
    #     N=1,
    #     dim=2,
    #     dx=ufl.dx,
    #     p=1,
    #     simplex_cell=True,
    #     dtype=np.float32,
    #     nnz=0.0,
    #     max_quad_sets=3,
    # )
