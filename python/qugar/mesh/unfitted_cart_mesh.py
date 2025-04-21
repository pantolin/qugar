# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from typing import Optional

from qugar import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from mpi4py import MPI

import numpy as np
import numpy.typing as npt
from dolfinx.cpp.mesh import GhostMode

import qugar.cpp
from qugar.cpp import CartGridTP_2D, CartGridTP_3D, UnfittedDomain_2D, UnfittedDomain_3D
from qugar.impl import ImplicitFunc
from qugar.mesh.tp_mesh import CartesianMesh
from qugar.mesh.unfitted_domain import UnfittedDomain


class UnfittedCartMesh(CartesianMesh, UnfittedDomain):
    """Class for storing an unfitted Cartesian mesh domain
    and access its cut, full, and empty cells and facets.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        cpp_unf_domain: UnfittedDomain_2D | UnfittedDomain_3D,
        cpp_cart_grid_tp: CartGridTP_2D | CartGridTP_3D,
        degree: int = 1,
        exclude_empty_cells: bool = True,
        ghost_mode: GhostMode = GhostMode.none,
        dtype: type[np.float32 | np.float64] = np.float64,
    ):
        """Constructor.

        Note:
            This constructor is not intended to be called directly,
            but rather use the function `create_Cartesian_mesh`.

        Args:
            comm (MPI.Comm): MPI communicator to be used for
                distributing the mesh.
            cpp_unf_domain (UnfittedDomain_2D | UnfittedDomain_3D):
                C++ unfitted domain object.
            cpp_cart_grid_tp (CartGridTP_2D | CartGridTP_3D):
                C++ Cartesian grid object.
            degree (int, optional): Degree of the mesh. Defaults to 1.
                It must be greater than zero.
            exclude_empty_cells (bool, optional): If True, empty cells
                are excluded from the mesh. Defaults to True.
            ghost_mode (GhostMode, optional): Ghost mode used for mesh
                partitioning. Defaults to `none`.
            dtype (type[np.float32 | np.float64], optional): Type to
                be used in the grid. Defaults to `np.float64`.
        """

        if exclude_empty_cells:
            active_cells = cpp_unf_domain.get_full_cells()
            active_cells = np.sort(np.append(active_cells, cpp_unf_domain.get_cut_cells()))
        else:
            active_cells = None

        CartesianMesh.__init__(
            self, comm, cpp_cart_grid_tp, degree, active_cells, ghost_mode, dtype
        )
        UnfittedDomain.__init__(self, self, cpp_unf_domain)


def create_unfitted_impl_Cartesian_mesh(
    comm: MPI.Comm,
    impl_func: ImplicitFunc,
    n_cells: npt.NDArray[np.int32] | list[np.int32] | list[int],
    xmin: Optional[npt.NDArray[np.float32 | np.float64]] = None,
    xmax: Optional[npt.NDArray[np.float32 | np.float64]] = None,
    degree: int = 1,
    exclude_empty_cells: bool = True,
    ghost_mode: GhostMode = GhostMode.none,
) -> UnfittedCartMesh:
    """Creates an uniftted Cartesian mesh generated with a bounding box and the number of
    cells per direction, and an implicit function describing the domain

    Args:
        comm: MPI communicator to be used for distributing the mesh.
        impl_func (ImplicitFunc_2D | ImplicitFunc_3D): Implicit function
            that describes the domain.
        n_cells (npt.NDArray[np.int32] | list[np.int32] | list[int]):
            Number of cells per direction in the mesh.
        xmin (Optional[npt.NDArray[np.float32 | np.float64]]): Minimum
            coordinates of the mesh's bounding box. Defaults to a vector
            of zeros with double floating precision.
        xmax (Optional[npt.NDArray[np.float32 | np.float64]]): Maximum
            coordinates of the mesh's bounding box. Defaults to a vector
            of ones with double floating precision.
        degree (int, optional): Degree of the mesh. Defaults to 1.
        exclude_empty_cells (bool, optional): If True, empty cells
            are excluded from the mesh. Defaults to True.
        ghost_mode (GhostMode, optional): Ghost mode used for mesh
            partitioning. Defaults to `none`.

    Returns:
        UnfittedCartMesh: Generated unfitted Cartesian mesh.
    """

    n_cells = np.array(n_cells)
    dim = len(n_cells)
    assert 2 <= dim <= 3, "Invalid dimension."

    dtype = (np.dtype(np.float64) if xmin is None else xmin.dtype) if xmax is None else xmax.dtype
    xmin_ = np.zeros(dim, dtype=dtype) if xmin is None else xmin
    xmax_ = np.ones(dim, dtype=dtype) if xmax is None else xmax

    assert dim == xmin_.size and dim == xmax_.size
    assert xmin_.dtype == xmax_.dtype and xmin_.dtype in [np.float32, np.float64]
    assert np.all(xmax_ > xmin_)

    cell_breaks = []
    for dir in range(dim):
        cell_breaks.append(np.linspace(xmin_[dir], xmax_[dir], n_cells[dir] + 1, dtype=np.float64))

    cpp_grid = qugar.cpp.create_cart_grid(cell_breaks)

    cpp_unf_domain = qugar.cpp.create_unfitted_impl_domain(
        impl_func.cpp_object,
        cpp_grid,
    )

    return UnfittedCartMesh(
        comm, cpp_unf_domain, cpp_grid, degree, exclude_empty_cells, ghost_mode, xmin_.dtype.type
    )
