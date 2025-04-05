# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for quadrature point generation and assemblers."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import ufl
from utils import (
    get_Gauss_quad_degree,  # type: ignore
)

import qugar.cpp
import qugar.impl
from qugar.dolfinx import form_custom
from qugar.mesh import UnfittedCartMesh, create_unfitted_impl_Cartesian_mesh


def create_vector_func(dlf_mesh: dolfinx.mesh.Mesh) -> ufl.Coefficient:
    """
    Create a vector function for testing the divergence theorem.

    Args:
        dlf_mesh (dolfinx.mesh.Mesh): The input mesh for which the UFL function is created.

    Returns:
        ufl.Coefficient: The generated UFL vector field as a coefficient.
    """
    x = ufl.SpatialCoordinate(dlf_mesh)
    if dlf_mesh.geometry.dim == 2:
        return ufl.as_vector([ufl.sin(x[0]) * ufl.cos(x[1]), ufl.cos(x[0]) * ufl.sin(x[1])])  # type: ignore
    else:
        return ufl.as_vector(
            [
                ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.exp(x[2]),  # type: ignore
                ufl.cos(x[0]) * ufl.sin(x[1]) * ufl.exp(x[2]),  # type: ignore
                ufl.cos(x[0]) * ufl.cos(x[1]) * ufl.exp(x[2]),  # type: ignore
            ]
        )


def create_div_thm_volume_ufl_form(domain: UnfittedCartMesh, n_quad_pts: int):
    """
    Creates a UFL form representing the volume integral of the divergence theorem
    for a given unfitted domain.

    Args:
        domain (UnfittedImplDomain): The unfitted domain for which the UFL form is created.
        n_quad_pts (int): The number of quadrature points to be used for integration of
            cut cells.

    Returns:
        ufl.Form: The UFL form representing the volume integral of the divergence theorem.
    """

    full_tag = 1
    cut_tag = 0
    cell_tags = domain.create_cell_subdomain_data(cut_tag=cut_tag, full_tag=full_tag)

    quad_degree = get_Gauss_quad_degree(n_quad_pts)
    dx_ = ufl.dx(
        domain=domain,
        subdomain_data=cell_tags,
    )
    dx = dx_(subdomain_id=cut_tag, degree=quad_degree) + dx_(full_tag)

    # Note: if no specific number of quadrature points is set for the cut cells,
    # it would be enough to use a single tag for both cut and full cells.
    # and invoke dx_ only once for that tag.

    func = create_vector_func(domain)
    div_func = ufl.div(func)

    ufl_form_vol = div_func * dx
    return ufl_form_vol


n_cells_dir = 2
dtype = np.float64


radius = 0.35
center = np.array([0.51, 0.45], dtype=dtype)
func_0 = qugar.impl.create_disk(radius=radius, center=center, use_bzr=True)

radius = 0.25
center = np.array([0.51, 0.45], dtype=dtype)
func_1 = qugar.impl.create_disk(radius=radius, center=center, use_bzr=True)


dim = func_0.dim
comm = MPI.COMM_WORLD
n_cells = [n_cells_dir] * dim
xmin = np.zeros(dim, dtype)
xmax = np.ones(dim, dtype)

unf_mesh_0 = create_unfitted_impl_Cartesian_mesh(comm, func_0, n_cells, xmin, xmax)
unf_mesh_1 = create_unfitted_impl_Cartesian_mesh(comm, func_1, n_cells, xmin, xmax)

ufl_form_vol_0 = create_div_thm_volume_ufl_form(unf_mesh_0, 5)
ufl_form_vol_1 = create_div_thm_volume_ufl_form(unf_mesh_1, 5)
ufl_form_vol = ufl_form_vol_0 + ufl_form_vol_1

form_vol = form_custom(ufl_form_vol_0, dtype=dtype)
