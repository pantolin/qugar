# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for mixed dimension integrals."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from utils import (  # type: ignore
    check_vals,
    clean_cache,
    create_mock_unfitted_mesh,
    dtypes,
    get_dtype,
)

from qugar.dolfinx import CustomForm, form_custom


@pytest.mark.parametrize("max_quad_sets", [3])
@pytest.mark.parametrize("nnz", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("N", [1, 4])
def test_mixed_dimension(
    N: int,
    dim: int,
    p: int,
    simplex_cell: bool,
    dtype: type[np.float32 | np.float64],
    nnz: float,
    max_quad_sets: int,
) -> None:
    """Test integral with mixed dimensions (combining quantities
    evaluated over facets of cells with quantities computed of cells
    with one dimension less). It evaluates them for a case of a mass
    matrix.

    It checks that the integral computed using such mock custom
    quadrature is the same as the one using standard integrals.

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

    mesh = create_mock_unfitted_mesh(dim, N, simplex_cell, nnz, max_quad_sets, dtype)

    # Create a submesh of the boundary
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    smesh, smesh_to_mesh = dolfinx.mesh.create_submesh(mesh, fdim, boundary_facets)[:2]

    # Define function spaces over the mesh and submesh
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", p))
    # W = dolfinx.fem.functionspace(mesh, ("Lagrange", p))
    Vbar = dolfinx.fem.functionspace(smesh, ("Lagrange", p))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    # v = ufl.TestFunction(W)
    vbar = ufl.TestFunction(Vbar)

    # Coefficients
    def coeff_expr(x):
        return np.sin(np.pi * x[0])

    # Coefficient defined over the mesh
    f = dolfinx.fem.Function(V, dtype=dtype)
    f.interpolate(coeff_expr)  # type: ignore

    # Coefficient defined over the submesh
    g = dolfinx.fem.Function(Vbar, dtype=dtype)
    g.interpolate(coeff_expr)  # type: ignore

    # Create the integration measure. Mixed-dimensional forms use the
    # higher-dimensional domain as the integration domain
    ds = ufl.Measure("ds", domain=mesh)

    facet_imap = mesh.topology.index_map(tdim - 1)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    mesh_to_smesh = np.full(num_facets, -1)
    mesh_to_smesh[smesh_to_mesh] = np.arange(len(smesh_to_mesh))
    entity_maps = {smesh: mesh_to_smesh}

    ufl_form = ufl.inner(f * g * u, vbar) * ds  # type: ignore

    dtype = get_dtype(ufl_form)

    form = dolfinx.fem.form(ufl_form, dtype=dtype, entity_maps=entity_maps)  # type: ignore
    matrix = dolfinx.fem.assemble_matrix(form)

    custom_form = form_custom(ufl_form, dtype=dtype, entity_maps=entity_maps)
    assert isinstance(custom_form, CustomForm)
    custom_matrix = dolfinx.fem.assemble_matrix(form)

    custom_coeffs = custom_form.pack_coefficients()
    custom_matrix = dolfinx.fem.assemble_matrix(custom_form, coeffs=custom_coeffs)

    dtype = get_dtype(ufl_form)
    check_vals(custom_matrix.data, matrix.data, dtype=dtype)


if __name__ == "__main__":
    clean_cache()
    test_mixed_dimension(
        N=1, dim=3, p=1, simplex_cell=True, dtype=np.float64, nnz=1.0, max_quad_sets=1
    )
