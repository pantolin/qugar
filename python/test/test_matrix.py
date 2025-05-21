# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for custom Python matrix assemblers."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from utils import clean_cache, create_mock_unfitted_mesh, dtypes, run_matrix_check  # type: ignore


@pytest.mark.parametrize("max_quad_sets", [3])
@pytest.mark.parametrize("nnz", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("dx", [ufl.dx, ufl.ds])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("N", [1, 4])
def test_grad_grad(
    N: int,
    dim: int,
    dx: ufl.Measure,
    p: int,
    simplex_cell: bool,
    dtype: type[np.float32 | np.float64],
    nnz: float,
    max_quad_sets: int,
):
    """Tests the integral of the grad grad matrix with a non-homogeneous
    coefficient.

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
            scalars.
        nnz (float): Ratio of entities with custom quadratures respect
            to the total number of entities. It is a value in the range
            [0.0, 1.0].
        max_quad_sets (int): Maximum number of repetitions of the
            standard quadrature to be used for each custom entity
            quadrature. For each custom entity a random number is
            generated between 1 and `max_quad_sets`.
    """

    unf_mesh = create_mock_unfitted_mesh(dim, N, simplex_cell, nnz, max_quad_sets, dtype)

    V0 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p, (dim,)))
    V1 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p + 1, (dim,)))
    V2 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p))

    u, v = ufl.TrialFunction(V0), ufl.TestFunction(V1)

    e = dolfinx.fem.Function(V2, dtype=dtype)
    e.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)  # type: ignore

    ufl_form_0 = e * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(domain=unf_mesh)  # type: ignore
    ufl_form = ufl_form_0

    run_matrix_check(ufl_form, unf_mesh)


@pytest.mark.parametrize("max_quad_sets", [3])
@pytest.mark.parametrize("nnz", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("dx", [ufl.dx, ufl.ds])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("N", [1, 4])
def test_elasticity(
    N: int,
    dim: int,
    dx: ufl.Measure,
    p: int,
    simplex_cell: bool,
    dtype: type[np.float32 | np.float64],
    nnz: float,
    max_quad_sets: int,
) -> None:
    """Tests the integral of the elasticity matrix with a
    non-homogeneous coefficient.

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
            scalars.
        nnz (float): Ratio of entities with custom quadratures respect
            to the total number of entities. It is a value in the range
            [0.0, 1.0].
        max_quad_sets (int): Maximum number of repetitions of the
            standard quadrature to be used for each custom entity
            quadrature. For each custom entity a random number is
            generated between 1 and `max_quad_sets`.
    """

    unf_mesh = create_mock_unfitted_mesh(dim, N, simplex_cell, nnz, max_quad_sets, dtype)

    lambda_ = 1.0
    mu_ = 1.0

    def epsilon(u):
        return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu_ * epsilon(u)  # type: ignore

    V0 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p + 1, (dim,)))
    V1 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p, (dim,)))
    V2 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p))

    u, v = ufl.TrialFunction(V0), ufl.TestFunction(V1)

    e = dolfinx.fem.Function(V2, dtype=dtype)
    e.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)  # type: ignore

    ufl_form_0 = e * ufl.inner(sigma(u), epsilon(v)) * dx(domain=unf_mesh)  # type: ignore

    ufl_form = ufl_form_0

    run_matrix_check(ufl_form, unf_mesh)


@pytest.mark.parametrize("max_quad_sets", [3])
@pytest.mark.parametrize("nnz", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("N", [1, 4])
def test_grad_grad_surface(
    N: int,
    dim: int,
    p: int,
    simplex_cell: bool,
    dtype: type[np.float32 | np.float64],
    nnz: float,
    max_quad_sets: int,
) -> None:
    """Tests the integral of a matrix quantity on the domain's boundary.

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
            scalars.
        nnz (float): Ratio of entities with custom quadratures respect
            to the total number of entities. It is a value in the range
            [0.0, 1.0].
        max_quad_sets (int): Maximum number of repetitions of the
            standard quadrature to be used for each custom entity
            quadrature. For each custom entity a random number is
            generated between 1 and `max_quad_sets`.
    """

    unf_mesh = create_mock_unfitted_mesh(dim, N, simplex_cell, nnz, max_quad_sets, dtype)

    facet_indices, facet_markers = [], []
    fdim = unf_mesh.topology.dim - 1

    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[0], 1)),
        (3, lambda x: np.isclose(x[1], 0)),
        (4, lambda x: np.isclose(x[1], 1)),
    ]

    for marker, locator in boundaries:
        facets = dolfinx.mesh.locate_entities(unf_mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dolfinx.mesh.meshtags(
        unf_mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

    V0 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p))
    V1 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p + 1, (2,)))

    u, v = ufl.TrialFunction(V0), ufl.TestFunction(V0)

    f = dolfinx.fem.Function(V1, dtype=dtype)
    f.interpolate(  # type: ignore
        lambda x: np.vstack((2 + x[0] ** 2 + 4 * x[1] ** 2, 1 + x[0] ** 2 + 5 * x[1] ** 2))
    )
    constants = []
    constants.append(dolfinx.fem.Constant(unf_mesh, np.array([5.0, 6.0], dtype=dtype)))
    constants.append(dolfinx.fem.Constant(unf_mesh, np.array([6.0, 7.0], dtype=dtype)))
    constants.append(dolfinx.fem.Constant(unf_mesh, np.array([6.0, 7.0], dtype=dtype)))
    constants.append(dolfinx.fem.Constant(unf_mesh, np.array([6.0, 7.0], dtype=dtype)))
    ds = ufl.ds(domain=unf_mesh, subdomain_data=facet_tag)

    ufl_form = None
    for i in range(4):
        new_ufl_form = (
            ufl.inner(f, constants[i]) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ds(i + 1)  # type: ignore
        )
        if ufl_form is None:
            ufl_form = new_ufl_form
        else:
            ufl_form += new_ufl_form

    run_matrix_check(ufl_form, unf_mesh)


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
):
    """Tests the integral of a matrix quantity on the interior facets,
    testing jumps of quantities.

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

    n = ufl.FacetNormal(unf_mesh)

    u = ufl.TrialFunction(V0)
    v = ufl.TestFunction(V0)

    f = dolfinx.fem.Function(V1, dtype=dtype)

    ufl_form = (
        f * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)) * ufl.dS  # type: ignore
    )

    run_matrix_check(ufl_form, unf_mesh)


if __name__ == "__main__":
    clean_cache()
    test_grad_grad(
        N=1,
        dim=2,
        dx=ufl.ds,
        p=1,
        simplex_cell=True,
        dtype=np.float32,
        nnz=0.3,
        max_quad_sets=3,
    )

    # test_interior_facet(
    #     N=4, dim=2, p=1, simplex_cell=True, dtype=np.float32, nnz=0.3, max_quad_sets=3
    # )

    # test_grad_grad_surface(
    #     N=4,
    #     dim=2,
    #     p=3,
    #     simplex_cell=False,
    #     dtype=np.float32,
    #     nnz=0.3,
    #     max_quad_sets=3,
    # )

    # test_elasticity(
    #     N=1, dim=2, p=1, simplex_cell=False, dtype=np.float64, nnz=0.2, max_quad_sets=1
    # )
