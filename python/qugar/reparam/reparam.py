# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------


from typing import TypeAlias

from mpi4py import MPI

import basix.ufl
import dolfinx.cpp as dlf_cpp
import dolfinx.mesh as dlf_mesh
import numpy as np
import numpy.typing as npt
import ufl
from basix import CellType
from dolfinx.fem import create_interpolation_data as dlf_create_interpolation_data
from dolfinx.fem.function import FunctionSpace
from dolfinx.geometry import PointOwnershipData

import qugar.cpp as cpp
from qugar.mesh import (
    DOLFINx_to_lexicg_nodes,
    UnfittedDomain,
)


def identity_partitioner(
    mpi_comm: MPI.Comm,
    n_parts: int,
    cell_types: list[dlf_mesh.CellType],
    cells: list[npt.NDArray[np.int32]],
) -> dlf_cpp.graph.AdjacencyList_int32:
    """(Dummy) mesh partitioner for leaving cells on the current rank.

    Args:
        comm (MPI.Comm): Mesh's MPI communicator.
        n_parts (int): Number of part in which the mesh will be partitioned.
        cell_types (list[dlf_mesh.CellType]): Cell types.
        cells (list[npt.NDArray[np.int32]]): List of cells to distribute.

    Returns:
        dlf_cpp.graph.AdjacencyList_int32: Adjaceny list assigning to
        every cell in the list cells, the MPI rank of destination.
        In this case, the destination rank will be the current one.
    """

    assert len(cell_types) == 1, "Not implemented for multiple cell types"

    valid_cells = [
        dlf_mesh.CellType.interval,
        dlf_mesh.CellType.quadrilateral,
        dlf_mesh.CellType.hexahedron,
    ]
    cell_type = cell_types[0]
    assert cell_type in valid_cells, (
        f"Cell type {cell_type} not supported. Only quadrilateral and hexahedron are supported."
    )

    tdim = valid_cells.index(cell_type) + 1

    n_pts_per_cell = 2**tdim

    n_cells = cells[0].size // n_pts_per_cell
    rank_dest = np.full(n_cells, mpi_comm.rank, dtype=np.int32)
    rank_dest = dlf_cpp.graph.AdjacencyList_int32(rank_dest)
    return rank_dest


class UnfDomainReparamMesh:
    """A class to represent the unfitted domain raparameterization mesh."""

    ReparamMesh: TypeAlias = (
        cpp.ReparamMesh_1_2 | cpp.ReparamMesh_2_2 | cpp.ReparamMesh_2_3 | cpp.ReparamMesh_3_3
    )

    def __init__(self, unf_domain: UnfittedDomain, cpp_object: ReparamMesh) -> None:
        """
        Initializes the Reparam object with a given C++ ReparamMesh object.

        Args:
            unf_domain (UnfittedDomain): An instance of the UnfittedDomain class.
            cpp_object (ReparamMesh): An instance of a C++ ReparamMesh object, either 2D or 3D.
        """
        self._unf_domain = unf_domain
        self._cpp_object = cpp_object

    @property
    def cpp_object(self) -> ReparamMesh:
        """
        Returns the C++ object associated with this instance.

        Returns:
            ReparamMesh: The C++ object underlying this instance.
        """
        return self._cpp_object

    @property
    def unf_domain(self) -> UnfittedDomain:
        """
        Returns the unfitted domain associated with this instance.

        Returns:
            UnfittedDomain: The unfitted domain associated with this instance.
        """
        return self._unf_domain

    def create_mesh(self, wirebasket: bool = False) -> dlf_mesh.Mesh:
        """Creates a DOLFINx mesh from the reparameterization data.

        This method generates a `dlf_mesh.Mesh` object representing either
        the reparameterized cells or the wirebasket of the reparameterization,
        based on the `wirebasket` flag.

        Args:
            wirebasket (bool, optional): If True, creates the mesh for the
                wirebasket connectivity. If False, creates the mesh for the
                cell connectivity. Defaults to False.

        Returns:
            dlf_mesh.Mesh: The generated DOLFINx mesh object.
        """
        mesh_tdim = self._unf_domain.mesh.tdim

        degree = self._cpp_object.order - 1
        points = self._cpp_object.points
        conn = self._cpp_object.wirebasket_conn if wirebasket else self._cpp_object.cells_conn

        reparam_tdim = 1 if wirebasket else mesh_tdim
        conn_mask = DOLFINx_to_lexicg_nodes(reparam_tdim, degree)
        conn = conn[:, conn_mask]

        cell_type = [CellType.interval, CellType.quadrilateral, CellType.hexahedron][
            reparam_tdim - 1
        ]

        gdim = points.shape[1]

        element = basix.ufl.element(
            "Lagrange",
            cell_type,
            degree,
            # lagrange_variant=basix.LagrangeVariant.equispaced,
            lagrange_variant=basix.LagrangeVariant.gll_warped,
            discontinuous=False,
            shape=(gdim,),
            dtype=points.dtype,
        )
        domain = ufl.Mesh(element)

        comm = self._unf_domain.mesh.comm

        # For parallel meshes, we need to use an identity partitioner
        # as the (default) ParMetis partitioner fails for non-connected
        # meshes as this one.

        # partitioner = identity_partitioner if comm.size > 1 else None
        partitioner = identity_partitioner

        return dlf_mesh.create_mesh(comm, conn.astype(np.int64), points, domain, partitioner)


def create_reparam_mesh(
    unf_domain: UnfittedDomain, degree: int = 3, levelset: bool = False
) -> UnfDomainReparamMesh:
    """
    Creates a reparameterized mesh for a given unfitted domain.

    Args:
        unf_domain (UnfittedDomain): The unfitted domain to be reparameterized.
        degree (int, optional): The reparameterization degree along each direction.
            It must be a positive value. Defaults to 3.
        levelset (bool, optional): Whether to create a levelset reparameterization (True)
            or a solid one (False). Defaults to False.

    Returns:
        UnfDomainReparam: The reparameterized domain.
    """
    assert degree > 0, "The reparameterization degree must be positive."

    algo = cpp.create_reparameterization_levelset if levelset else cpp.create_reparameterization
    n_pts_dir = degree + 1
    merge_points = False
    reparam_cpp = algo(unf_domain.cpp_unf_domain_object, n_pts_dir, merge_points)
    return UnfDomainReparamMesh(unf_domain, reparam_cpp)


def create_interpolation_data(
    V_to: FunctionSpace,
    V_from: FunctionSpace,
    padding: float = 1.0e-14,
) -> tuple[npt.NDArray[np.int32], PointOwnershipData]:
    """
    Generate data needed to interpolate discrete functions across different meshes.

    Note:
        Note that the function spaces `V_to` and `V_from` must be defined
        on meshes with different dimensions. E.g., `V_from` (2D or 3D)
        can be the function space of a computed solution and `V_to` the
        function space for interpolating the solution in the
       reparameterization wirebasket (1D).

    Args:
        V_to (FunctionSpace): Function space to interpolate into.
        V_from (FunctionSpace): Function space to interpolate from.
        padding (float): Absolute padding of bounding boxes of all
            entities on the mesh associated to `V_to`.

    Returns:
        npt.NDArray[np.int32]: Cell indices of the mesh associated
        to `V_to` on which to interpolate into.
        PointOwnershipData: Data needed to interpolation functions
        defined on function spaces on the meshes.
    """

    mesh = V_to.mesh
    cmap = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cmap.size_local + cmap.num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)

    return cells, dlf_create_interpolation_data(V_to, V_from, cells, padding)
