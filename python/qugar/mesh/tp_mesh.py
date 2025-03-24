# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tensor-product and Cartesian meshes."""

from typing import Optional, cast

import qugar.utils

if not qugar.utils.has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import basix.ufl
import dolfinx.cpp as dlf_cpp
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.cpp.mesh import GhostMode

import qugar.cpp
from qugar.mesh.tp_index import TensorProdIndex
from qugar.mesh.utils import (
    DOLFINx_to_lexicg_faces,
    DOLFINx_to_lexicg_nodes,
    create_cells_to_facets_map,
)


def _create_first_cell_conn_Cartesian(
    dim: int, degree: int, n_cells: npt.NDArray[np.int32]
) -> npt.NDArray[np.int64]:
    """Creates the connectivity of the first cell of a tensor-product
    mesh.

    Args:
        dim (int): Parametric dimension of the grid (1D, 2D, or 3D).
        degree (int): Cells's degree. It must be greater than 0.
        n_cells (npt.NDArray[np.int32]): Number of cells per direction.

    Returns:
        npt.NDArray[np.int64]: Connectivity of the first cell following
        the basix ordering convention. See
        https://github.com/FEniCS/basix/#supported-elements
    """

    assert 1 <= dim <= 3, "Invalid dimension."
    assert 1 <= degree, "Invalid degree."
    assert np.all(n_cells > 0), "Invalid number of cells per direction."

    n_pts = n_cells * degree + 1
    first_cell_1d_lex = np.arange(degree + 1, dtype=np.int64)

    if dim == 1:
        first_cell_lex = first_cell_1d_lex
    else:
        first_cell_2d_lex = first_cell_1d_lex + np.arange(
            0, (degree + 1) * n_pts[0], n_pts[0], dtype=np.int64
        ).reshape(degree + 1, 1)
        first_cell_2d_lex = first_cell_2d_lex.ravel()

        if dim == 2:
            first_cell_lex = first_cell_2d_lex
        else:  # if dim == 3:
            first_cell_lex = first_cell_2d_lex + np.arange(
                0,
                (degree + 1) * n_pts[0] * n_pts[1],
                n_pts[0] * n_pts[1],
                dtype=np.int64,
            ).reshape(degree + 1, 1)
            first_cell_lex = first_cell_lex.ravel()

    dlf_to_lex = DOLFINx_to_lexicg_nodes(dim, degree)
    return np.array(
        [first_cell_lex[dlf_to_lex[dlf_i]] for dlf_i in range(len(dlf_to_lex))],
        dtype=np.int64,
    )


def _create_tensor_prod_mesh_conn(
    dim: int, degree: int, n_cells: npt.NDArray[np.int32]
) -> npt.NDArray[np.int64]:
    """Creates the cell's connectivity of a Cartesian mesh.

    Args:
        dim (int): Parametric dimension of the grid (1D, 2D, or 3D).
        degree (int): Cells's degree. It must be greater than 0.
        n_cells (npt.NDArray[np.int32]): Number of cells per direction.

    Returns:
        npt.NDArray[np.int64]: Generated connectivity. It is a 2D
        array where the rows correspond to the cells and the columns
        to the nodes of every cell. The connectivity of
        https://github.com/FEniCS/basix/#supported-elements
    """

    assert 1 <= dim <= 3, "Invalid dimension."
    assert 1 <= degree, "Invalid degree."
    assert np.all(n_cells > 0), "Invalid number of cells per direction."

    n_pts = n_cells * degree + 1

    conn_first_cell = _create_first_cell_conn_Cartesian(dim, degree, n_cells)
    n_nodes_per_cell = conn_first_cell.size

    conn = conn_first_cell + np.arange(0, n_cells[0] * degree, degree, dtype=np.int64).reshape(
        n_cells[0], 1
    )
    if 1 < dim:
        conn = conn.ravel() + np.arange(
            0, n_pts[0] * degree * n_cells[1], n_pts[0] * degree, dtype=np.int64
        ).reshape(n_cells[1], 1)
        if 2 < dim:
            conn = conn.ravel() + np.arange(
                0,
                n_pts[0] * n_pts[1] * degree * n_cells[2],
                n_pts[0] * n_pts[1] * degree,
                dtype=np.int64,
            ).reshape(n_cells[2], 1)

    return conn.reshape(-1, n_nodes_per_cell)


def _merge_coincident_points_in_mesh(
    nodes: npt.NDArray[np.float32 | np.float64],
    conn: npt.NDArray[np.int64],
    tolerance: Optional[type[np.float32 | np.float64]] = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Finds conincident points from a mesh and merge them updating the
    cells' connectivity.

    This functionality requires `scipy` package to be installed.

    Args:
        nodes (npt.NDArray[np.float32 | np.float64]): Coordinates of the
            nodes stored in a 2D array. Rows correspond to the different
            points and columns to their coordinates.
        conn (npt.NDArray[np.int64]): Connectivity to update. It is a 2D
            array, where every row stores a list of nodes ids. The
            connectivity of each cells follows the DOLFINx convention.
            See https://github.com/FEniCS/basix/#supported-elements
        tolerance (Optional[type[np.float32 | np.float64]]): Absolute
            tolerance to be used in the comparison of coincident points.
            If not set, it is computed as 1.0e-5 times the maximum
            length along the Cartesian directions of the the bounding
            box enclosing the nodes.

    Raises:
        ValueError: If `scipy` is not found, an exception is thrown.

    Returns:
        npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int64],
        npt.NDArray[np.int64]:
        The first entry is the new array of node coordinates (with same
        format as `nodes`) containing unique (non-coincident) nodes.
        The second entry is the update connectivity (with same format as
        `conn`). The last entry is the map from old nodes to new ones.
    """

    try:
        import scipy as sp
    except ImportError as e:
        raise ValueError(e)

    kdtree = sp.spatial.KDTree(nodes)

    if tolerance is None:
        tolerance = cast(nodes.dtype.type, 1.0e-5 * (np.max(nodes) - np.min(nodes)))
    coincidences = kdtree.query_pairs(tolerance, output_type="ndarray")

    master = coincidences[:, 0]
    slave = coincidences[:, 1]

    n_pts = nodes.shape[0]
    slave_to_master = np.arange(n_pts, dtype=np.int64)
    slave_to_master[slave] = slave_to_master[master]

    convergence = False

    while not convergence:
        slave_to_master_prev = slave_to_master.copy()
        slave_to_master = slave_to_master[slave_to_master]
        convergence = np.all(slave_to_master == slave_to_master_prev)

    master_unique = np.unique(np.sort(slave_to_master))
    master_to_new_ordering = np.empty(n_pts, dtype=np.int64)
    master_to_new_ordering[master_unique] = np.arange(master_unique.size, dtype=np.int64)
    old_to_new = master_to_new_ordering[slave_to_master]

    new_nodes = nodes[master_unique]
    new_conn = np.empty_like(conn)
    for i, cell_conn in enumerate(conn):
        for j, node in enumerate(cell_conn):
            new_conn[i, j] = old_to_new[node]

    return new_nodes, new_conn, old_to_new


def _create_mesh(
    comm: MPI.Intracomm,
    dim: int,
    coords: npt.NDArray[np.float32 | np.float64],
    conn: npt.NDArray[np.int64],
    degree: int,
    cell_type: Optional[str] = None,
    ghost_mode: GhostMode = GhostMode.none,
) -> dolfinx.mesh.Mesh:
    """Creates a DOLFINx mesh of Lagrange elements.

    of intervals, quadrilaterals, or
    hexahedra.

    Args:
        comm (MPI.Intracomm): MPI communicator to be used for
            distributing the mesh. Right now, only the serial case is
            implemented.
        dim (int): Parametric dimension of the mesh.
        coords (npt.NDArray[np.float32 | np.float64]): Coordinates of
            the nodes stored in a 2D np.ndarray. The rows correspond to
            the different points and columns to the coordinates.
        conn (npt.NDArray[np.int64]): Generated connectivity. It is a
            list, where every entry is a list of nodes ids. The
            connectivity of each cells follows the DOLFINx convention.
            See https://github.com/FEniCS/basix/#supported-elements
        degree (int): Cells's degree. It must be greater than 0.
        cell_type (Optional[str]): Cell type. It can be, `interval`,
            `triangle`, `quadrilateral`, `tetrahedron`, or `hexahedron`.
            It defaults to None, therefore, the type will be set to
            `interval`, `quadrilateral`, or `hexahedron`, depending on
            the dimension `dim`.
        ghost_mode (GhostMode, optional): Ghost mode used for mesh
            partitioning. Defaults to `none`.

    Returns:
        dolfinx.mesh.Mesh: Generated mesh.
    """

    assert 1 <= dim <= 3, "Invalid dimension."
    assert degree >= 1, "Invalid degree."

    if comm.rank > 1:
        assert coords.shape[0] == 0 and conn.shape[0] == 0
        partitioner = dlf_cpp.mesh.create_cell_partitioner(ghost_mode)
    else:
        partitioner = None

    if cell_type is None:
        cell_type = ["interval", "quadrilateral", "hexahedron"][dim - 1]

    dtype = coords.dtype
    gdim = coords.shape[1]
    element = basix.ufl.element("Lagrange", cell_type, degree, shape=(gdim,), dtype=dtype)

    return dolfinx.mesh.create_mesh(comm, conn, coords, ufl.Mesh(element), partitioner=partitioner)


def _create_Cartesian_mesh_nodes_from_points(
    pts_1D: list[npt.NDArray[np.float32 | np.float64]],
) -> npt.NDArray[np.float32 | np.float64]:
    """Creates the coordinates matrix of the nodes of a tensor-product
    Cartesian mesh.

    The ordering of the generated nodes follows the lexicographical
    ordering convetion.

    Args:
        pts_1D (list[npt.NDArray[np.float32 | np.float64]]): Points
            coordinates along the Cartesian directions.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Coordinates of the nodes
        stored in a 2D array. Rows correspond to the different points
        and columns to their coordinates.
    """

    dim = len(pts_1D)
    assert 1 <= dim <= 3, "Invalid dimension"

    if dim == 1:
        x = pts_1D[0].copy()
    elif dim == 2:
        x = np.meshgrid(pts_1D[0], pts_1D[1], indexing="xy")
    else:  # dim == 3
        x = np.meshgrid(pts_1D[2], pts_1D[1], pts_1D[0], indexing="ij")[::-1]

    nodes = np.zeros((x[0].size, dim), dtype=x[0].dtype)
    for dir in range(dim):
        nodes[:, dir] = x[dir].ravel()

    return nodes


def _create_Cartesian_mesh_equidistant_nodes(
    n_pts_dir: npt.NDArray[np.int32],
    xmin: npt.NDArray[np.float32 | np.float64],
    xmax: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Creates the coordinates matrix of equidistant nodes of a
    tensor-product Cartesian mesh.

    The mesh domain is the bounding box defined by `xmin` and `xmax`.

    The ordering of the generated nodes follow the lexicographical
    ordering convetion.

    Args:
        n_pts_dir (npt.NDArray[np.int32]): Number of points per
            direction.
        xmin (npt.NDArray[np.float32 | np.float64]): Minimum coordinates
            of the bounding box enclosing the mesh.
        xmax (npt.NDArray[np.float32 | np.float64]): Maximum coordinates
            of the bounding box enclosing the mesh.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Coordinates of the nodes
        stored in a 2D array. Rows correspond to the different points
        and columns to their coordinates.
    """

    dim = len(n_pts_dir)
    assert 1 <= dim <= 3, "Invalid dimension."
    assert xmin.size == dim and xmax.size == dim

    dtype = xmin.dtype
    assert dtype == xmax.dtype, "Non matching dtypes."

    pts_1D = [np.linspace(xmin[dir], xmax[dir], n_pts_dir[dir], dtype=dtype) for dir in range(dim)]

    return _create_Cartesian_mesh_nodes_from_points(pts_1D)


def _create_Cartesian_mesh_nodes_from_cells(
    cell_breaks_1D: list[npt.NDArray[np.float32 | np.float64]], degree: int = 1
) -> npt.NDArray[np.float32 | np.float64]:
    """Creates the coordinates matrix of the nodes of a tensor-product
    Cartesian mesh.

    The ordering of the generated nodes follows the lexicographical
    ordering convetion.

    Args:
        cell_breaks_1D (list[npt.NDArray[np.float32 | np.float64]]):
            Points coordinates associated to the cells breaks along the
            Cartesian directions.
        degree (int, optional): Degree of the mesh. Defaults to 1.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Coordinates of the nodes
        stored in a 2D array. Rows correspond to the different points
        and columns to their coordinates.
    """

    assert 1 <= len(cell_breaks_1D) <= 3, "Invalid dimension"
    assert degree >= 1, "Invalid degree."

    if degree == 1:
        pts = cell_breaks_1D
    else:
        pts = []
        for cell_breaks in cell_breaks_1D:
            dtype = cell_breaks.dtype
            n_cells = cell_breaks.size - 1
            pts_1D = np.empty(n_cells * degree + 1, dtype=dtype)
            pts_1D[::degree] = cell_breaks

            for i in range(1, degree):
                coeff_1 = dtype.type(i) / dtype.type(degree)
                coeff_0 = dtype.type(1.0) - coeff_1
                pts_1D[i::degree] = coeff_0 * cell_breaks[:-1] + coeff_1 * cell_breaks[1:]

            pts.append(pts_1D)

    return _create_Cartesian_mesh_nodes_from_points(pts)


def _find_in_array(values_to_search: npt.NDArray, all_values: npt.NDArray) -> npt.NDArray[np.int64]:
    """Finds the index positions of the `values_to_search` referred to
    the array `all_values`.

    Args:
        values_to_search (npt.NDArray): Values to search.
        all_values (npt.NDArray): Array in which the values are sought.
            It must have the same type as `values_to_search`.

    Returns:
        npt.NDArray[np.int64]: Array indicating the position in
        `all_values` of each item in `values_to_search`. It has the same
        length as `values_to_search`. If a value is not present,
        it returns -1.
    """

    index = np.argsort(all_values)
    sorted_all_values = all_values[index]

    sorted_index = np.searchsorted(sorted_all_values, values_to_search)
    values_to_search_index = np.take(index, sorted_index, mode="clip").astype(np.int64)
    mask = all_values[values_to_search_index] != values_to_search

    return np.ma.array(values_to_search_index, mask=mask).filled(-1)


class TensorProductMesh:
    """Tensor-product mesh data structure.

    This class is a helper class to ease the management of structured
    tensor-product meshes of intervals (1D), quadrangles (2D), or
    hexahedra (3D). The geometric dimension can be arbitrarity according
    to the DOLFINx capabilities.

    The class provides methods for easily quering the id of a cell in
    the reference mesh or its corresponding counterpart referred to a
    mesh created with DOLFINx.

    It also provides functionalities for extracting the facets of a
    cell, and the facets that belong to the mesh's boundary, among
    others.

    The mesh can be partitioned among different processes using a MPI
    communicator passed to the contructor. So, when queried about
    indices of vertices, facets, cells, etc., the returns are referred
    to the indices present in the current subdomain of the mesh.

    The created mesh follows the denoted as lexicographical convention
    for numbering cells and nodes. This ordering is such that the
    parametric direction 0 iterates the fastest, i.e., is the
    inner-most, and the direction dim-1 iterates the slowest, i.e., the
    outer-most. The nodes and cells numbering for a  2D case would be:

       8 --- 9 ---10 ---11              . --- . --- . --- .
       |     |     |     |              |  3  |  4  |  5  |
       4 --- 5 --- 6 --- 7              . --- . --- . --- .
       |     |     |     |              |  0  |  1  |  2  |
       0 --- 1 --- 2 --- 3              . --- . --- . --- .

    Note that after the mesh creation, DOLFINx renumbers and partitions
    the mesh, so, this numeration will be different. It is always
    possible to retrieve the original numbering using the maps
    `self.dolfinx_to_lexicg_nodes` or `self.dolfinx_to_lexicg_cells`.
    Or, from the lexicographical to the DOLFINx numbering using
    `self.lexicg_to_dolfinx_nodes` or `self.lexicg_to_dolfinx_cells`.
    """

    def __init__(
        self,
        comm: MPI.Intracomm,
        nodes_coords: npt.NDArray[np.float32 | np.float64],
        n_cells: npt.NDArray[np.int32] | list[np.int32] | list[int],
        degree: int = 1,
        ghost_mode: GhostMode = GhostMode.none,
        merge_nodes: bool = False,
        merge_tol: Optional[type[np.float32 | np.float64]] = None,
    ) -> None:
        """Constructor.

        Args:
            comm (MPI.Intracomm): MPI communicator to be used for
                distributing the mesh.
            n_cells (npt.NDArray[np.int32] | list[np.int32] | list[int]):
                Number of cells per direction in the mesh.
            nodes_coords (npt.NDArray[np.float32 | np.float64]): Nodes
                coordinates. The rows correspond to the different nodes
                (following lexicographical ordering), and the columns to
                the coordinates of each point.
            degree (int): Degree of the mesh.
            ghost_mode (GhostMode, optional): Ghost mode used for mesh
                partitioning. Defaults to `none`.
            merge_nodes (bool, optional): If `True`, coincident nodes
                will be merged together into a single one. Otherwise,
                duplicated nodes will not be merged. Defaults to
                `False`.
            merge_tol (Optional[type[np.float32 | np.float64]]): Absolute
                tolerance to be used for seeking coincident nodes.
                If not set, and if `merge_nodes` is set to `True`,
                this tolerance will be automatically computed in the
                function `merge_coincident_points_in_mesh`.
        """

        self._n_cells = np.array(n_cells).astype(np.int32)
        assert np.all(self._n_cells > 0)
        assert 1 <= self.tdim <= 3, "Invalid dimension."

        self._degree = degree
        assert 1 <= degree, "Invalid degree."

        self._create_mesh(comm, nodes_coords, merge_nodes, merge_tol, ghost_mode)

    @property
    def dolfinx_mesh(self) -> dolfinx.mesh.Mesh:
        """Gets the underlying DOLFINx mesh.

        Returns:
            dolfinx.mesh.Mesh: Underlying DOLFINx mesh.
        """
        return self._mesh

    @property
    def degree(self) -> int:
        """Gets the mesh' degree.

        Returns:
            int: Mesh's degree.
        """
        return self._degree

    @property
    def num_pts_dir(self) -> npt.NDArray[np.int32]:
        """Gets the number of points per direction in the mesh
        taking into account the number of cells per direction and
        the mesh's degree.

        Returns:
            npt.NDArray[np.int32]: Number of points per direction.
        """
        return self._n_cells * self._degree + 1

    @property
    def num_cells_dir(self) -> npt.NDArray[np.int32]:
        """Gets the number of cells per direction in the mesh.

        Returns:
            npt.NDArray[np.int32]: Number of cells per direction.
        """
        return self._n_cells

    @property
    def tdim(self) -> int:
        """Gets the topological dimension of the mesh.

        Returns:
            int: Mesh's topological dimension (1, 2, or 3).
        """
        return self._n_cells.size

    @property
    def gdim(self) -> int:
        """Gets the geometrical dimension of the mesh.

        Returns:
            int: Mesh's geometrical dimension.
        """
        return self._mesh.geometry.dim

    @property
    def has_merged_nodes(self) -> bool:
        """Checks if the mesh has merged nodes.

        Returns:
            bool: Whether the mesh has merged nodes.
        """
        return self._merged_nodes_map.size > 0

    @property
    def merged_nodes_map(self) -> npt.NDArray[np.int64]:
        """Gets the map of lexicographical merged nodes: from original
        ids to unique ones.

        Returns:
            npt.NDArray[np.int64]: Map from old to new lexicographical
            nodes.
        """
        return self._merged_nodes_map

    @property
    def num_global_cells(self) -> np.int64:
        """Gets the total (global) number of cells.

        Returns:
            np.int64: Number of global cells.
        """
        return np.prod(self._n_cells, dtype=np.int64)

    @property
    def num_local_cells(self) -> np.int64:
        """Gets the local number of cells (associated to the current
        subdomain, i.e., MPI process).

        Returns:
            np.int64: Number of local cells.
        """
        return np.int64(self._mesh.topology.original_cell_index.size)

    def _create_mesh(
        self,
        comm: MPI.Intracomm,
        nodes_coords: npt.NDArray[np.float32 | np.float64],
        merge_nodes: bool,
        merge_tol: Optional[type[np.float32 | np.float64]] = None,
        ghost_mode=GhostMode.none,
    ) -> None:
        """Creates the underlying DOLFINx mesh and stores it in
        `self._mesh`.

        Args:
            comm (MPI.Intracomm): MPI communicator to be used for
                distributing the mesh.
            nodes_coords (npt.NDArray[np.float32 | np.float64]): Nodes
                coordinates. The rows correspond to the different nodes
                (following lexicographical ordering), and the columns to
                the coordinates of each point.
            ghost_mode (GhostMode, optional): Ghost mode used for mesh
                partitioning. Defaults to `none`.
            merge_nodes (bool): If `True`, coincident nodes will be
                merged together into a single one. Otherwise, duplicated
                nodes will not be merged.
            merge_tol (Optional[type[np.float32 | np.float64]]): Absolute
                tolerance to be used for seeking coincident nodes.
                If not set, and if `merge_nodes` is set to `True`,
                this tolerance will be automatically computed in the
                function `merge_coincident_points_in_mesh`. Defaults to
                None.
        """

        if comm.rank == 0:
            conn = _create_tensor_prod_mesh_conn(self.tdim, self.degree, self.num_cells_dir)
            assert nodes_coords.shape[0] == np.prod(self.num_pts_dir)
        else:
            # TODO: this is a temporary hack. All the nodes and cells
            # are created in rank 0, and then distributed. Better
            # performance could be achieved by distributing nodes and
            # cells since their creation.
            n_nodes_per_cell = 2**self.tdim
            conn = np.empty((0, n_nodes_per_cell), dtype=np.int64)
            assert nodes_coords.shape[0] == 0

        self._merged_nodes_map = np.empty(0, dtype=np.int64)
        if comm.rank == 0 and merge_nodes:
            nodes_coords, conn, self._merged_nodes_map = _merge_coincident_points_in_mesh(
                nodes_coords, conn, merge_tol
            )

        self._mesh = _create_mesh(
            comm, self.tdim, nodes_coords, conn, self._degree, ghost_mode=ghost_mode
        )

    def get_DOLFINx_local_cell_ids(
        self,
        cell_ids: npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64,
        lexicg: bool = True,
    ) -> npt.NDArray[np.int32] | np.int32:
        """Transforms the given `cell_ids` from the tensor-product mesh
        into the corresponding local ids of the underlying DOLFINx mesh.

        Args:
            cell_ids (npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64):
                Cell indices to be transformed. They may be local
                indices referred to the underlying DOLFINx mesh (if
                `lexicg` is set to `False`) or lexicographical indices
                (if `lexicg` is `True`). If `lexicg` is set to `False`,
                all the local DOLFINx indices must belong to the current
                subdomain (process).
            lexicg (bool, optional): Whether `cell_ids` follow the
                tensor-product lexicographical numbering or the DOLFINx
                one. Defaults to `True`.

        Returns:
            npt.NDArray[np.int32] | np.int32: Indices of the cells in
            the underlying DOLFINx mesh. These indices correspond to the
            local numbering associated to the current subdomain
            (process). Note that some indices may be set to -1 in the
            case the input indices are lexicographical and some of them
            are not contained in the subdomain. The length of the output
            array is the same as the input.
        """

        if not isinstance(cell_ids, np.ndarray):
            cell_ids = np.array(cell_ids)
            return cast(
                npt.NDArray[np.int32],
                self.get_DOLFINx_local_cell_ids(cell_ids, lexicg),
            )

        dlf_to_lex = self.dolfinx_mesh.topology.original_cell_index

        if lexicg:
            lex_ids = np.array(cell_ids, dtype=np.int64)
            return _find_in_array(lex_ids, dlf_to_lex).astype(np.int32)
        else:
            n_local_cells = dlf_to_lex.size
            assert np.all(cell_ids < n_local_cells) and np.all(cell_ids >= 0), (
                "Cells not contained in subdomain"
            )

            return cell_ids.astype(np.int32)

    def get_DOLFINx_global_cell_ids(
        self,
        cell_ids: npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64,
        lexicg: bool = True,
    ) -> npt.NDArray[np.int32]:
        """Transforms the given local `cell_ids` into the corresponding global ids
        of the underlying DOLFINx mesh.

        Args:
            cell_ids (npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64):
                Cell indices to be transformed. They may be local
                indices referred to the underlying DOLFINx mesh (if
                `lexicg` is set to `False`) or lexicographical indices
                (if `lexicg` is `True`). If `lexicg` is set to `False`,
                all the local DOLFINx indices must belong to the current
                subdomain (process).
            lexicg (bool, optional): Whether `cell_ids` follow the
                tensor-product lexicographical numbering or the DOLFINx
                one. Defaults to `True`.

        Note:
            All the indices in `cell_ids` must be contained in the
            current subdomain.

        Returns:
            npt.NDArray[np.int32]: Indices of the cells in the underlying DOLFINx mesh.
            These indices correspond to the global numbering associated to the
            current subdomain (process).
        """

        dlf_cell_ids = self.get_DOLFINx_local_cell_ids(cell_ids, lexicg)
        index_map = self.dolfinx_mesh.topology.index_map(self.tdim)
        return index_map.local_to_global(dlf_cell_ids)

    def get_lexicg_cell_ids(
        self,
        cell_ids: npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64,
        lexicg: bool = False,
    ) -> npt.NDArray[np.int64] | np.int64:
        """Transforms given `cell_ids` from the tensor-product mesh into
        the tensor-product mesh numbering, that follows a lexicographical
        indexing.

        Args:
            cell_ids (npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64):
                Cell indices to be transformed. They may be local
                indices referred to the underlying DOLFINx mesh (if
                `lexicg` is set to `False`) or lexicographical indices
                (if `lexicg` is `True`). If `lexicg` is set to `False`,
                all the local DOLFINx indices must belong to the current
                subdomain (process).
            lexicg (bool, optional): Whether `cell_ids` follow the
                tensor-product lexicographical numbering or the DOLFINx
                one. Defaults to `True`.

        Returns:
            npt.NDArray[np.int64] | np.int64: Ids of the cells in the
            tensor-product lexicographical indexing. Note that some
            indices may be set to -1 if their associated cells do not
            belong to current subdomain. The length of the output array
            is the same as the input.
        """

        if not isinstance(cell_ids, np.ndarray):
            cell_ids = np.array(cell_ids)
            return cast(
                npt.NDArray[np.int64],
                self.get_lexicg_cell_ids(cell_ids, lexicg),
            )

        dlf_to_lex = self.dolfinx_mesh.topology.original_cell_index

        if lexicg:
            lex_ids = cell_ids.astype(np.int64, copy=True)
            # Getting cells that don't belong to the subdomain.
            not_in_subdomain = np.logical_not(np.isin(lex_ids, dlf_to_lex))

            return np.ma.array(lex_ids, mask=not_in_subdomain).filled(-1)

        else:
            n_local_cells = dlf_to_lex.size
            assert np.all(cell_ids < n_local_cells) and np.all(cell_ids >= 0), (
                "Cells not contained in subdomain"
            )

            return dlf_to_lex[cell_ids]

    def get_DOLFINx_node_ids(
        self,
        node_ids: npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64,
        lexicg: bool = True,
    ) -> npt.NDArray[np.int32] | np.int32:
        """Transforms the given `node_ids` from the tensor-product mesh
        into the corresponding local ids of the underlying DOLFINx mesh.

        Args:
            node_ids (npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64):
                Node indices to be transformed. They may be local
                indices referred to the underlying DOLFINx mesh (if
                `lexicg` is set to `False`) or lexicographical indices
                (if `lexicg` is `True`). If `lexicg` is set to `False`,
                all the local DOLFINx indices must belong to the current
                subdomain (process).
            lexicg (bool, optional): Whether `node_ids` follow the
                tensor-product lexicographical numbering or the DOLFINx
                one. Defaults to `True`.

        Returns:
            npt.NDArray[np.int32] | np.int32: Indices of the nodes in
            the underlying DOLFINx mesh. These indices correspond to the
            local numbering associated to the current subdomain
            (process). Note that some indices may be set to -1 in the
            case the input indices are lexicographical and some of them
            are not contained in the subdomain. The length of the output
            array is the same as the input.
        """

        if not isinstance(node_ids, np.ndarray):
            node_ids = np.array(node_ids)
            return cast(
                npt.NDArray[np.int32],
                self.get_DOLFINx_node_ids(node_ids, lexicg),
            )

        dlf_to_lex = self.dolfinx_mesh.geometry.input_global_indices

        if lexicg:
            lex_ids = np.array(node_ids, dtype=np.int64)
            if self.has_merged_nodes:
                # Mapping "old" to "new" ids.
                lex_ids = self.merged_nodes_map[lex_ids]

            return _find_in_array(lex_ids, dlf_to_lex).astype(np.int32)

        else:
            n_local_nodes = dlf_to_lex.size
            assert np.all(node_ids < n_local_nodes) and np.all(node_ids >= 0), (
                "Cells not contained in subdomain"
            )

            return node_ids.astype(np.int32)

    def get_lexicg_node_ids(
        self,
        node_ids: npt.NDArray[np.int32] | np.int32,
        lexicg: bool = False,
    ) -> npt.NDArray[np.int64] | np.int64:
        """Transforms given `node_ids` from the tensor-product mesh into
        the tensor-product mesh numbering, that follows a lexicographical
        indexing.

        Args:
            node_ids (npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64):
                Node indices to be transformed. They may be local
                indices referred to the underlying DOLFINx mesh (if
                `lexicg` is set to `False`) or lexicographical indices
                (if `lexicg` is `True`). If `lexicg` is set to `False`,
                all the local DOLFINx indices must belong to the current
                subdomain (process).
            lexicg (bool, optional): Whether `node_ids` follow the
                tensor-product lexicographical numbering or the DOLFINx
                one. Defaults to `True`.

        Returns:
            npt.NDArray[np.int64] | np.int64: Ids of the nodes in the
            tensor-product lexicographical indexing. Note that some
            indices may be set to -1 if their associated cells do not
            belong to current subdomain. The length of the output array
            is the same as the input.
        """

        if not isinstance(node_ids, np.ndarray):
            node_ids = np.array(node_ids)
            return cast(
                npt.NDArray[np.int64],
                self.get_DOLFINx_node_ids(node_ids, lexicg),
            )

        dlf_to_lex = self.dolfinx_mesh.geometry.input_global_indices

        if lexicg:
            lex_ids = node_ids.astype(np.int64, copy=True)
            # Getting nodes that don't belong to the subdomain.
            not_in_subdomain = np.logical_not(np.isin(lex_ids, dlf_to_lex))

            return np.ma.array(lex_ids, mask=not_in_subdomain).filled(-1)
        else:
            n_local_nodes = dlf_to_lex.size
            assert np.all(node_ids < n_local_nodes) and np.all(node_ids >= 0), (
                "Nodes not contained in subdomain."
            )

            lex_nodes = dlf_to_lex[node_ids]
            if self.has_merged_nodes:
                return _find_in_array(lex_nodes, self.merged_nodes_map)
            else:
                return lex_nodes

    def create_facet_bdry_tags(self, lexicg: bool = False) -> dolfinx.mesh.MeshTags:
        """Creates face facet tags for the tensor product mesh.

        It creates the tags for the facets of the underlying DOLFINx mesh,
        setting a different tag for each boundary face of the domain.

        Args:
            lexicg (bool, optional): Defines how the tag values for the
                different boundary facets are set. It set to `True`, the
                lexicographical ordering is used (i.e., umin:0, umax:1,
                vmin:2, vmax:3, ...). Otherwise, the faces correspond to the
                FEniCSx cell faces ordering, but applied to the full
                hypercube. See
                https://github.com/FEniCS/basix/#supported-elements
                It defaults to `False`.

        Returns:
            dolfinx.mesh.MeshTags: Generated tags.
        """

        faces_facets, faces_markers = [], []
        n_faces = 2 * self.tdim
        for face_id in range(n_faces):
            facets = self.get_face_facets(face_id, lexicg)
            faces_facets.append(facets)
            faces_markers.append(np.full_like(facets, face_id))

        faces_facets = np.hstack(faces_facets)
        faces_markers = np.hstack(faces_markers)
        sorted_facets = np.argsort(faces_facets)
        faces_facets = faces_facets[sorted_facets]
        faces_markers = faces_markers[sorted_facets]

        facet_tdim = self.tdim - 1
        return dolfinx.mesh.meshtags(
            self.dolfinx_mesh,
            facet_tdim,
            faces_facets,
            faces_markers,
        )

    def get_cells_exterior_facets(
        self,
        cell_ids: npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64,
        lexicg: bool = False,
    ) -> npt.NDArray[np.int32]:
        """Extracts the exterior facets of associated to a list of cells for
        the tensor-product mesh.

        The exterior facets are those that belong to one single cell
        (they are not at the interface between two cells, but on the
        mesh boundary).

        Args:
            cell_ids (npt.NDArray[np.int32 | np.int64] | np.int32 | np.int64):
                Array of cells whose exterior facets are extracted.
                They must be all contained in the subdomain (process).
            lexicg (bool, optional): Whether `cell_ids` follow
                the tensor-product lexicographical numbering or the one
                associated to the DOLINFx mesh. Defaults to `False`.

        Returns:
            npt.NDArray[np.int32]: Sorted unique array of exterior
            facets present in the current subdomain.
        """

        if cell_ids.size == 0:
            return np.empty(0, dtype=np.int32)

        if not isinstance(cell_ids, np.ndarray):
            cell_ids = np.array(cell_ids)
            return self.get_cells_exterior_facets(cell_ids, lexicg)

        dolfinx_cell_ids = self.get_DOLFINx_local_cell_ids(cell_ids, lexicg)
        assert np.all(dolfinx_cell_ids >= 0), "Cells out of the subdomain."

        topology = self.dolfinx_mesh.topology

        cell_dim = self.tdim
        facet_dim = cell_dim - 1

        cells_to_facets = create_cells_to_facets_map(self.dolfinx_mesh)

        facets_in_cells = np.sort(np.unique(cells_to_facets[dolfinx_cell_ids].ravel()))

        topology.create_connectivity(facet_dim, cell_dim)
        exterior_facets = dolfinx.mesh.exterior_facet_indices(topology)
        return np.setdiff1d(exterior_facets, facets_in_cells)

    def get_face_DOLFINx_nodes(
        self,
        face_id: int,
        face_lexicg: bool = False,
    ) -> npt.NDArray[np.int32]:
        """Extracts the nodes of a given boundary face that belong to
        the subdomain of the tensor-product mesh.

        Args:
            face_id (int): Id of the face whose nodes are extracted.
            face_lexicg (bool, optional): Whether `face_id` follows the
                tensor-product lexicographical numbering or the
                FEniCSx one. See
                https://github.com/FEniCS/basix/#supported-elements
                Defaults to `False`.
            nodes_lexicg (bool, optional): Whether the returned nodes
                indices must follow the tensor-product lexicographical
                numbering or the one associated to the underlying
                DOLINFx mesh. Defaults to `False`.

        Returns:
            npt.NDArray[np.int32]: Sorted unique array of local DOLFINx
            nodes of the boundary face that belong to the current subdomain
            (process).
        """

        all_lex_bndry_nodes = TensorProdIndex.get_face(self.num_pts_dir, face_id, face_lexicg)

        bndry_nodes = cast(
            npt.NDArray[np.int32],
            self.get_DOLFINx_node_ids(all_lex_bndry_nodes, lexicg=True),
        )
        return np.sort(bndry_nodes[bndry_nodes >= 0])  # Removing nodes out of the mesh domain.

    def get_face_DOLFINx_cells(self, face_id: int, lexicg: bool = False) -> npt.NDArray[np.int32]:
        """Extracts the DOLFINx indices of the cells touching the given face
        of the tensor-product mesh.

        Args:
            face_id (int): Id of the face whose cells are extracted.
            lexicg (bool, optional): Whether `face_id` follows the
                tensor-product lexicographical ordering or the FEniCSx
                one. See
                https://github.com/FEniCS/basix/#supported-elements
                Defaults to `False`.

        Returns:
            npt.NDArray[np.int32]: Array storing the local cells indices
            referred to the underlying DOLFINx mesh numbering for the
            current subdomain (process).
            This array is unique and sorted. Note that it may be empty
            if the cells associated to the face do not belong to the
            current subdomain.
        """

        lex_cells = TensorProdIndex.get_face(self.num_cells_dir, face_id, lexicg=lexicg)
        cells = cast(
            npt.NDArray[np.int32],
            self.get_DOLFINx_local_cell_ids(lex_cells, lexicg=True),
        )
        return np.sort(cells[cells >= 0])  # Removing cells out of the mesh domain.

    def get_face_facets(self, face_id: int, lexicg: bool = False) -> npt.NDArray[np.int32]:
        """Extracts the facets indices of the given face of the tensor-product
        mesh.

        Args:
            face_id (int): Id of the face whose facets are extracted.
            lexicg (bool, optional): Whether `face_id` follows the
                tensor-product lexicographical ordering or the FEniCSx
                one. See
                https://github.com/FEniCS/basix/#supported-elements
                Defaults to `False`.

        Returns:
            npt.NDArray[np.int32]: Array storing the facets indices
            referred to the underlying DOLFINx mesh numbering.
            This array is unique and sorted. Note that it may be empty
            if the facets associated to the face do not belong to the
            current subdomain.
        """

        topology = self.dolfinx_mesh.topology

        tdim = self.tdim
        topology.create_connectivity(tdim, tdim - 1)
        cell_to_facets = topology.connectivity(tdim, tdim - 1)
        cell_to_facets = cell_to_facets.array.reshape(cell_to_facets.num_nodes, -1)

        cells = self.get_face_DOLFINx_cells(face_id, lexicg)
        return np.sort(np.unique(cell_to_facets[cells, face_id]))

    def get_cell_facets(
        self,
        cell_id: np.int32,
        out_lexicg: bool = False,
    ) -> npt.NDArray[np.int32]:
        """Extracts the facet ids of the given cell.

        Args:
            cell_id (np.int32): Id of the queried cell. It is a DOLFINx
                (local) cell id. It must be contained in the subdomain.
            out_lexicg (bool, optional): Whether the returned facet ids
                are reordered according to the lexicographical ordering
                of the cell's bounding box. I.e., according to the
                directions umin, umax, vmin, vmax, ... Otherwise, they
                follow the FEniCSx convention. See
                https://github.com/FEniCS/basix/#supported-elements
                Defaults to `False`.

        Returns:
            npt.NDArray[np.int32]: Array of cell facets.
        """

        tdim = self.tdim

        topology = self.dolfinx_mesh.topology
        topology.create_connectivity(tdim, tdim - 1)
        cell_to_facets = self.dolfinx_mesh.topology.connectivity(tdim, tdim - 1)

        facets = cell_to_facets.links(cell_id)

        if out_lexicg:
            mask = DOLFINx_to_lexicg_faces(tdim)
            facets = facets[mask]

        return facets

    def get_cell_tensor_index(
        self, cell_id: np.int64 | np.int32, lexicg: bool = False
    ) -> npt.NDArray[np.int32]:
        """Computes the tensor index referred to the tensor-product mesh
        of a given cell index.

        Args:
            cell_id (np.int64 | np.int32): Cell index to be transformed.
                It must be contained in the subdomain (process).
            lexicg (bool, optional): Whether `cell_id` follows
                the tensor-product lexicographical numbering or the one
                associated to the DOLINFx mesh. Defaults to `False`.

        Returns:
            npt.NDArray[np.int32]: Cell tensor index
        """

        lex_cell_id = cast(np.int64, self.get_lexicg_cell_ids(cell_id, lexicg))
        assert lex_cell_id >= 0, "Cell out of the subdomain."
        return TensorProdIndex.get_tensor_index(self.num_cells_dir, lex_cell_id)


class CartesianMesh(TensorProductMesh):
    """Cartesian mesh data structure. This class is a helper class to
    ease the construction and management of tensor-product (hypercube)
    meshes aligned with the Cartesian axes.

    It is only implemented for 2D and 3D.

    I.e., tensor-product meshes (see `TensorProductMesh`) whose points
    are distributed according to the Cartesian axes.
    """

    def __init__(
        self,
        comm: MPI.Intracomm,
        grid_cpp: qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D,
        degree: int = 1,
        ghost_mode: GhostMode = GhostMode.none,
        dtype: type[np.float32 | np.float64] = np.float64,
    ):
        """Constructor.

        Note:
            This constructor is not intended to be called directly,
            but rather use the function `create_Cartesian_mesh`.

        Args:
            comm (MPI.Intracomm): MPI communicator to be used for
                distributing the mesh.
            grid_cpp (qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D):
                C++ Cartesian grid object.
            degree (int, optional): Degree of the mesh. Defaults to 1.
                It must be greater than zero.
            ghost_mode (GhostMode, optional): Ghost mode used for mesh
                partitioning. Defaults to `none`.
            dtype (type[np.float32 | np.float64], optional): Type to
                be used in the grid. Defaults to `np.float64`.
        """

        dim = grid_cpp.dim
        assert 2 <= dim <= 3, "Only implemented for 2D and 3D."
        self._cell_breaks = [breaks.astype(dtype) for breaks in grid_cpp.cell_breaks]

        if comm.rank == 0:
            nodes_coords = _create_Cartesian_mesh_nodes_from_cells(self._cell_breaks, degree)
        else:
            # TODO: this is a temporary hack. All the nodes and cells
            # are created in rank 0, and then distributed. Better
            # performance could by achieving by distributing nodes and
            # cells since their creation.
            nodes_coords = np.empty((0, dim), dtype=dtype)

        self._cpp_object = grid_cpp
        n_cells = grid_cpp.num_cells_dir

        super().__init__(comm, nodes_coords, n_cells, degree, ghost_mode)

    @property
    def grid_cpp_object(self) -> qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_2D:
        """Returns the stored (C++) Cartesian grid object.

        Returns:
            qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_2D: Stored Cartesian
            grid object.
        """
        return self._cpp_object

    @property
    def cell_breaks(self) -> list[npt.NDArray[np.float32 | np.float64]]:
        """Gets the coordinates of the Cartesian grid cells along each direction

        Returns:
            list[npt.NDArray[np.float32 | np.float64]]: Cell coordinates
            along the parametric directions.
        """
        return self._cell_breaks

    @property
    def dtype(self) -> type[np.float32 | np.float64]:
        """Gets the type associated to the breaks.

        Returns:
            type[np.float32 | np.float64]: Type associated to the
            breaks.
        """
        return self._cell_breaks[0].dtype

    def get_cell_bbox(
        self, cell_id: np.int32, lexicg: bool = False
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Computes the bounding box of a single cell.

        Arguments:
            cell_id (np.int32): Cell whose bounding box is created.
            lexicg (bool, optional): If `True`, the given `cell_id` is
                assumed to be referred to the lexicographical mesh
                numbering convention. Otherwise, if `False`, `cell_id`
                follows the numbering of the underlying DOLFINx mesh.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Computed bounding box
                of the cell. It is a 2D with two rows and as many columns
                as coordinates. Both rows represent the minimum and maximum
                coordintes of the bounding box, respectively.
        """

        bbox = np.empty((2, self.gdim), dtype=self.dtype)

        cart_cell_tid = self.get_cell_tensor_index(cell_id, lexicg)
        for dir in range(self.tdim):
            bbox[0, dir] = self._cell_breaks[dir][cart_cell_tid[dir]]
            bbox[1, dir] = self._cell_breaks[dir][cart_cell_tid[dir] + 1]

        return bbox


def create_Cartesian_mesh(
    comm: MPI.Intracomm,
    n_cells: npt.NDArray[np.int32] | list[np.int32] | list[int],
    xmin: Optional[npt.NDArray[np.float32 | np.float64]] = None,
    xmax: Optional[npt.NDArray[np.float32 | np.float64]] = None,
    degree: int = 1,
    ghost_mode: GhostMode = GhostMode.none,
) -> CartesianMesh:
    """Creates a Cartesian mesh from a bounding box and the number of
    cells per direction.

    Args:
        comm: MPI communicator to be used for distributing the mesh.
        n_cells (npt.NDArray[np.int32] | list[np.int32] | list[int]):
            Number of cells per direction in the mesh.
        xmin (Optional[npt.NDArray[np.float32 | np.float64]]): Minimum
            coordinates of the mesh's bounding box. Defaults to a vector
            of zeros with double floating precision.
        xmax (Optional[npt.NDArray[np.float32 | np.float64]]): Maximum
            coordinates of the mesh's bounding box. Defaults to a vector
            of ones with double floating precision.
        degree (int, optional): Degree of the mesh. Defaults to 1.
        ghost_mode (GhostMode, optional): Ghost mode used for mesh
            partitioning. Defaults to `none`.

    Returns:
        CartesianMesh: Generated Cartesian mesh.
    """

    n_cells = np.array(n_cells)
    dim = len(n_cells)
    assert 2 <= dim <= 3, "Invalid dimension."

    if xmin is None:
        dtype = np.dtype(np.float64) if xmax is None else xmax.dtype
        xmin = np.zeros(dim, dtype=dtype)

    if xmax is None:
        xmax = np.ones(dim, dtype=xmin.dtype)

    assert dim == xmin.size and dim == xmax.size
    assert xmin.dtype == xmax.dtype and xmin.dtype in [np.float32, np.float64]
    assert np.all(xmax > xmin)

    cell_breaks = []
    for dir in range(dim):
        cell_breaks.append(np.linspace(xmin[dir], xmax[dir], n_cells[dir] + 1, dtype=np.float64))

    cpp_grid = qugar.cpp.create_cart_grid(cell_breaks)

    return CartesianMesh(comm, cpp_grid, degree, ghost_mode, xmin.dtype.type)
