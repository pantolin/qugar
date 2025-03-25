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
from dolfinx.fem import coordinate_element as _coordinate_element

import qugar.cpp
from qugar.mesh.tp_index import TensorProdIndex
from qugar.mesh.utils import (
    DOLFINx_to_lexicg_faces,
    DOLFINx_to_lexicg_nodes,
    create_cells_to_facets_map,
    lexicg_to_DOLFINx_faces,
    map_facets_to_cells_and_local_facets,
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


class TensorProductMesh(dolfinx.mesh.Mesh):
    """Tensor-product mesh data structure.

    This class derives from `dolfinx.mesh.Mesh`, easing the management of
    structured tensor-product meshes of intervals (1D), quadrangles (2D), or
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
    `self.get_lexicg_node_ids` or `self.get_lexicg_cell_ids`.
    Or, from the lexicographical to the local DOLFINx numbering using
    `self.get_DOLFINx_local_node_ids` or `self.get_DOLFINx_local_cell_ids`,
    or the global with `self.get_DOLFINx_global_node_ids` or
    `self.get_DOLFINx_global_cell_ids`,
    """

    def __init__(
        self,
        comm: MPI.Comm,
        nodes_coords: npt.NDArray[np.float32 | np.float64],
        n_cells: npt.NDArray[np.int32] | list[np.int32] | list[int],
        degree: int = 1,
        ghost_mode: GhostMode = GhostMode.none,
        merge_nodes: bool = False,
        merge_tol: Optional[type[np.float32 | np.float64]] = None,
    ) -> None:
        """Constructor.

        Args:
            comm (MPI.Comm): MPI communicator to be used for
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
        assert 1 <= self.degree, "Invalid degree."

        self._create_mesh(comm, nodes_coords, merge_nodes, merge_tol, ghost_mode)

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
        return self.geometry.dim

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
        """Gets the total (global) number of cells of the DOLFINx mesh.

        Returns:
            np.int64: Number of global cells of the DOLFINx mesh.
        """
        return np.int64(self.topology.index_map(self.tdim).size_global)

    @property
    def num_local_cells(self) -> np.int64:
        """Gets the local number of cells (associated to the current
        subdomain, i.e., MPI process).

        Returns:
            np.int64: Number of local cells.
        """
        return np.int64(self.topology.index_map(self.tdim).size_local)

    @property
    def num_cells_tp(self) -> np.int64:
        """Gets the total number of cells of the underlying tensor-product mesh.

        Returns:
            np.int64: Number of cells of the tensor-product mesh.
        """
        return np.prod(self._n_cells, dtype=np.int64)

    def _create_mesh(
        self,
        comm: MPI.Comm,
        nodes_coords: npt.NDArray[np.float32 | np.float64],
        merge_nodes: bool,
        merge_tol: Optional[type[np.float32 | np.float64]] = None,
        ghost_mode=GhostMode.none,
    ) -> None:
        """Creates the base DOLFINx mesh.

        Args:
            comm (MPI.Comm): MPI communicator to be used for
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

        self._merged_nodes_map = np.empty(0, dtype=np.int64)

        nodes_coords = np.asarray(nodes_coords, order="C")
        if nodes_coords.ndim == 1:
            gdim = 1
        else:
            gdim = nodes_coords.shape[1]

        if comm.rank == 0:
            conn = _create_tensor_prod_mesh_conn(self.tdim, self.degree, self.num_cells_dir)
            assert nodes_coords.shape[0] == np.prod(self.num_pts_dir)

            if merge_nodes:
                nodes_coords, conn, self._merged_nodes_map = _merge_coincident_points_in_mesh(
                    nodes_coords, conn, merge_tol
                )
            conn = np.asarray(conn, dtype=np.int64, order="C")

        else:
            # TODO: this is a temporary hack. All the nodes and cells
            # are created in rank 0, and then distributed. Better
            # performance could be achieved by distributing nodes and
            # cells since their creation.
            # See https://jsdokken.com/dolfinx_docs/meshes.html#mpi-communication
            # for further details.
            assert nodes_coords.shape[0] == 0

            n_nodes_per_cell = 2**self.tdim
            conn = np.empty((0, n_nodes_per_cell), dtype=np.int64, order="C")

        if comm.size > 1:
            partitioner = dlf_cpp.mesh.create_cell_partitioner(ghost_mode)
        else:
            partitioner = None

        cell_type = ["interval", "quadrilateral", "hexahedron"][self.tdim - 1]

        element = basix.ufl.element(
            "Lagrange", cell_type, self.degree, shape=(gdim,), dtype=nodes_coords.dtype
        )
        domain = ufl.Mesh(element)

        e_ufl = domain.ufl_coordinate_element()
        cmap = _coordinate_element(e_ufl.basix_element)
        # TODO: Resolve UFL vs Basix geometric dimension issue
        # assert domain.geometric_dimension() == gdim

        msh_cpp = dlf_cpp.mesh.create_mesh(comm, conn, cmap._cpp_object, nodes_coords, partitioner)

        super().__init__(msh_cpp, domain)

    def get_DOLFINx_local_cell_ids(
        self,
        lex_cell_ids: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int32]:
        """Transforms the given `lex_cell_ids` from the lexicographical tensor-product mesh
        into the corresponding local ids of the underlying DOLFINx mesh.

        Args:
            lex_cell_ids (npt.NDArray[np.int64]):
                Cell indices to be transformed. They are lexicographical indices
                referred to the tensor-product mesh.

        Returns:
            npt.NDArray[np.int32]: Indices of the cells in
            the underlying DOLFINx mesh. These indices correspond to the
            local numbering associated to the current subdomain
            (process). Note that some indices may be set to -1 in the
            case the associated input lexicographical indices are not contained
            in the subdomain. The length of the output
            array is the same as the input.
        """

        dlf_to_lex = self.topology.original_cell_index
        return _find_in_array(lex_cell_ids, dlf_to_lex).astype(np.int32)

    def get_DOLFINx_global_cell_ids(
        self,
        lex_cell_ids: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """Transforms the given local `lex_cell_ids` into the corresponding global ids
        of the underlying DOLFINx mesh.

        Args:
            lex_cell_ids (npt.NDArray[np.int64]):
                Cell indices to be transformed. They are lexicographical indices

        Note:
            All the indices in `lex_cell_ids` must be contained in the
            current subdomain.

        Returns:
            npt.NDArray[np.int64]: Indices of the cells in the underlying DOLFINx mesh.
            These indices correspond to the global indices belonging to the
            current subdomain (process).
        """

        dlf_local_cell_ids = self.get_DOLFINx_local_cell_ids(lex_cell_ids)
        return self.get_DOLFINx_local_to_global_cell_ids(dlf_local_cell_ids)

    def get_DOLFINx_local_to_global_cell_ids(
        self,
        dlf_local_cell_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int64]:
        """Transforms the given local `dlf_local_cell_ids` (DOLFINx) local
        cell ids into the corresponding global ones associated to the current process.

        Args:
            dlf_local_cell_ids (npt.NDArray[np.int32]): DOLFINx local cell indices
                to be transformed.

        Returns:
            npt.NDArray[np.int64]: Global indices of the DOLFINx cells.
        """

        index_map = self.topology.index_map(self.tdim)
        return index_map.local_to_global(dlf_local_cell_ids)

    def get_lexicg_cell_ids(
        self,
        dlf_local_cell_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int64]:
        """Transforms given `dlf_local_cell_ids` into
        the tensor-product mesh numbering, that follows a lexicographical
        indexing.

        Args:
            dlf_local_cell_ids (npt.NDArray[np.int32]):
                Cell indices to be transformed. They arelocal
                indices referred to the underlying DOLFINx mesh.

        Returns:
            npt.NDArray[np.int64]: Ids of the cells in the
            tensor-product lexicographical indexing. The length of the output array
            is the same as the input.
        """

        dlf_to_lex = self.topology.original_cell_index
        n_local_cells = dlf_to_lex.size
        assert np.all(dlf_local_cell_ids < n_local_cells) and np.all(dlf_local_cell_ids >= 0), (
            "Cells not contained in subdomain"
        )

        return dlf_to_lex[dlf_local_cell_ids]

    def get_DOLFINx_local_node_ids(
        self,
        lex_node_ids: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int32]:
        """Transforms the given `lx_node_ids` from the tensor-product mesh
        into the corresponding local ids of the underlying DOLFINx mesh.

        Args:
            lex_node_ids (npt.NDArray[np.int64]):
                Node indices to be transformed. They are lexicographical
                indices referred to the tensor-product mesh.

        Returns:
            npt.NDArray[np.int32]: Indices of the nodes in
            the underlying DOLFINx mesh. These indices correspond to the
            local numbering associated to the current subdomain
            (process). Note that some indices may be set to -1 in the
            case the input indices are lexicographical and some of them
            are not contained in the subdomain. The length of the output
            array is the same as the input.
        """

        dlf_to_lex = self.geometry.input_global_indices

        if self.has_merged_nodes:
            # Mapping "old" to "new" ids.
            lex_node_ids = self.merged_nodes_map[lex_node_ids]

        return _find_in_array(lex_node_ids, dlf_to_lex).astype(np.int32)

    def get_lexicg_node_ids(
        self,
        dlf_local_node_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int64]:
        """Transforms given `dlf_local_node_ids` from the (local) DOLFINx
        mesh numbering to the lexicographical (tensor-product mesh) one.

        Args:
            dlf_local_node_ids (npt.NDArray[np.int32]):
                Node indices to be transformed. They are referred to the
                underlying (local) DOLFINx mesh.

        Returns:
            npt.NDArray[np.int64]: Ids of the nodes in the
            tensor-product lexicographical indexing.
        """

        dlf_to_lex = self.geometry.input_global_indices
        n_local_nodes = dlf_to_lex.size
        assert np.all(dlf_local_node_ids < n_local_nodes) and np.all(dlf_local_node_ids >= 0), (
            "Nodes not contained in subdomain."
        )

        lex_nodes = dlf_to_lex[dlf_local_node_ids]
        if self.has_merged_nodes:
            return _find_in_array(lex_nodes, self.merged_nodes_map)
        else:
            return lex_nodes

    def create_facet_bdry_tags(self) -> dolfinx.mesh.MeshTags:
        """Creates face facet tags for the tensor product mesh.

        It creates the tags for the facets of the underlying DOLFINx mesh,
        setting a different tag for each boundary face of the domain.

        The created tags follow the lexicographical ordering of the
        tensor-product mesh. I.e., umin:0, umax:1, vmin:2, vmax:3, ...

        Returns:
            dolfinx.mesh.MeshTags: Generated tags.
        """

        faces_facets, faces_markers = [], []
        n_faces = 2 * self.tdim
        for face_id in range(n_faces):
            facets = self.get_face_facets(face_id)
            faces_facets.append(facets)
            faces_markers.append(np.full_like(facets, face_id))

        faces_facets = np.hstack(faces_facets)
        faces_markers = np.hstack(faces_markers)
        sorted_facets = np.argsort(faces_facets)
        faces_facets = faces_facets[sorted_facets]
        faces_markers = faces_markers[sorted_facets]

        facet_tdim = self.tdim - 1
        return dolfinx.mesh.meshtags(
            self,
            facet_tdim,
            faces_facets,
            faces_markers,
        )

    def get_cells_exterior_facets(
        self,
        dlf_local_cell_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """Extracts the exterior facets of associated to a list of
        (local) DOLFINx cell ids.

        The exterior facets are those that belong to one single cell
        (they are not at the interface between two cells, but on the
        mesh boundary).

        Args:
            dlf_local_cell_ids (npt.NDArray[np.int32] | np.int32):
                Array of cells whose exterior facets are extracted.
                They are (local to the current subdomain) DOLFINx cell
                indices.

        Returns:
            npt.NDArray[np.int32]: Sorted unique array of exterior
            facets present in the current subdomain.
        """

        self.topology.create_connectivity(self.tdim - 1, self.tdim)

        cells_to_facets = create_cells_to_facets_map(self)

        facets_in_cells = np.sort(np.unique(cells_to_facets[dlf_local_cell_ids].ravel()))

        exterior_facets = dolfinx.mesh.exterior_facet_indices(self.topology)
        return np.setdiff1d(exterior_facets, facets_in_cells)

    def get_face_DOLFINx_local_node_ids(
        self,
        face_id: int,
    ) -> npt.NDArray[np.int32]:
        """Extracts the nodes of a given boundary face that belong to
        the subdomain of the tensor-product mesh.

        Args:
            face_id (int): Id of the face whose nodes are extracted.
                This id follows the lexicographical ordering.

        Returns:
            npt.NDArray[np.int32]: Sorted unique array of local DOLFINx
            nodes of the boundary face that belong to the current subdomain
            (process).
        """

        all_lex_bndry_nodes = TensorProdIndex.get_face(self.num_pts_dir, face_id, lexicg=True)

        bndry_nodes = self.get_DOLFINx_local_node_ids(all_lex_bndry_nodes)
        return np.sort(bndry_nodes[bndry_nodes >= 0])  # Removing nodes out of the mesh domain.

    def get_face_DOLFINx_cells(self, tp_face_id: int) -> npt.NDArray[np.int32]:
        """Extracts the (local) DOLFINx indices of the cells touching the given face
        of the tensor-product mesh.

        Args:
            tp_face_id (int): Id of the face whose cells are extracted.
                This id follows the lexicographical ordering.

        Returns:
            npt.NDArray[np.int32]: Array storing the local cells indices
            referred to the underlying DOLFINx mesh numbering for the
            current subdomain (process).
            This array is unique and sorted. Note that it may be empty
            if the cells associated to the face do not belong to the
            current subdomain.
        """

        lex_cells = TensorProdIndex.get_face(self.num_cells_dir, tp_face_id, lexicg=True)
        cells = self.get_DOLFINx_local_cell_ids(lex_cells)
        return np.sort(cells[cells >= 0])  # Removing cells out of the mesh domain.

    def get_face_facets(self, face_id: int) -> npt.NDArray[np.int32]:
        """Extracts the facets indices of the given face of the tensor-product
        mesh.

        Args:
            face_id (int): Id of the face whose facets are extracted.
                This id follows the lexicographical ordering.

        Returns:
            npt.NDArray[np.int32]: Array storing the facets indices
            referred to the underlying DOLFINx mesh numbering.
            This array is unique and sorted. Note that it may be empty
            if the facets associated to the face do not belong to the
            current subdomain.
        """

        tdim = self.tdim
        self.topology.create_connectivity(tdim, tdim - 1)
        cell_to_facets = self.topology.connectivity(tdim, tdim - 1)
        cell_to_facets = cell_to_facets.array.reshape(cell_to_facets.num_nodes, -1)

        dlf_local_cells = self.get_face_DOLFINx_cells(face_id)
        return np.sort(np.unique(cell_to_facets[dlf_local_cells, face_id]))

    def get_cell_facets(
        self,
        dlf_local_cell_id: np.int32,
    ) -> npt.NDArray[np.int32]:
        """Extracts the facet ids of the given cell.

        Args:
            dlf_local_cell_id (np.int32): Id of the queried cell. It is a DOLFINx
                (local) cell id.

        Returns:
            npt.NDArray[np.int32]: Array of DOLFINx cell facets.
        """

        tdim = self.tdim

        self.topology.create_connectivity(tdim, tdim - 1)
        cell_to_facets = self.topology.connectivity(tdim, tdim - 1)

        return cell_to_facets.links(dlf_local_cell_id)

    def get_cell_tensor_index(self, dlf_local_cell_id: np.int32) -> npt.NDArray[np.int32]:
        """Computes the tensor index referred to the tensor-product mesh
        of a given (local) DOLFINx cell index.

        Args:
            dlf_local_cell_id (np.int32): Local DOLFINx cell index to be transformed.

        Returns:
            npt.NDArray[np.int32]: Cell tensor index
        """

        lex_cell_id = self.get_lexicg_cell_ids(np.array(dlf_local_cell_id, dtype=np.int32))[0]
        return TensorProdIndex.get_tensor_index(self.num_cells_dir, lex_cell_id)

    def transform_lexicg_facet_ids_to_DOLFINx_local(
        self,
        lex_cell_ids: npt.NDArray[np.int64],
        lex_facet_ids: npt.NDArray[np.int32],
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Transforms the given lexicographical facets indices to
        DOLFINx local numbering (otherwise)

        Args:
            lex_cell_ids (npt.NDArray[np.int64]): Indices of the facets
                following the lexicographical ordering. All the cell
                indices must be associated to the current process.

            lex_facet_ids (npt.NDArray[np.int32]): Local indices of the
                facets referred to `lex_cell_ids` (both arrays should have
                the same length). The face ids follow the
                lexicographical ordering.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Transformed facets returned as one array of cells and another
            one of local facets referred to those cells. The indices of the cells
            and local facets follow the DOLFINx local numbering.
        """

        dlf_cells = self.get_DOLFINx_local_cell_ids(lex_cell_ids)

        dlf_to_lex_facets = DOLFINx_to_lexicg_faces(self.tdim)
        dlf_facets = dlf_to_lex_facets[lex_facet_ids]
        return dlf_cells, dlf_facets

    def transform_DOLFINx_local_facet_ids_to_lexicg(
        self,
        dlf_local_cell_ids: npt.NDArray[np.int32],
        dlf_local_facet_ids: npt.NDArray[np.int32],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
        """Transforms the given local DOLFINx facets indices to
        either lexicographical numbering (both for cells and local facet ids).

        Args:
            dlf_local_cell_ids (npt.NDArray[np.int32]): Indices of the facets
                following the DOLFINx local ordering.

            dlf_local_facet_ids (npt.NDArray[np.int32]): Local indices of the
                facets referred to `dlf_local_cell_ids` (both arrays should have
                the same length). The face ids follow the
                DOLFINx ordering.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Transformed facets returned as one array of cells and another
            one of local facets referred to those cells. The indices of the cells
            and local facets follow the DOLFINx (local) numbering.
        """

        lex_cell_ids = self.get_lexicg_cell_ids(dlf_local_cell_ids)
        lex_to_dlf_facets = lexicg_to_DOLFINx_faces(self.tdim)
        lex_facet_is = lex_to_dlf_facets[dlf_local_facet_ids]
        return lex_cell_ids, lex_facet_is

    def transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(
        self,
        dlf_local_facet_ids: npt.NDArray[np.int32],
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Given the ids of (local) DOLFINx facets, returns the DOLFINx
        cells and local facet ids corresponding to those facets.

        Note:
            Interior facets belong to more whan one cell, in those cases
            only one cell (and local facet) is returned for that
            particular facet. The one chosen depends on the way in which
            that information is stored in the mesh connectivity.

        Args:
            dlf_local_facet_ids (npt.NDArray[np.int32]): Array of DOLFINx facets
                (local to the current process) to be transformed.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: DOLFINx cells
                and local facet ids the associated to the facets. The first
                entry of the tuple corresponds to the cells and the second
                to the the local facets.

                The local facets indices follow the FEniCSx convention. See
                https://github.com/FEniCS/basix/#supported-elements
        """
        return map_facets_to_cells_and_local_facets(self, dlf_local_facet_ids)


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
        comm: MPI.Comm,
        cart_grid_tp_cpp: qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D,
        degree: int = 1,
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
            cart_grid_tp_cpp (qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D):
                C++ Cartesian grid object.
            degree (int, optional): Degree of the mesh. Defaults to 1.
                It must be greater than zero.
            ghost_mode (GhostMode, optional): Ghost mode used for mesh
                partitioning. Defaults to `none`.
            dtype (type[np.float32 | np.float64], optional): Type to
                be used in the grid. Defaults to `np.float64`.
        """

        dim = cart_grid_tp_cpp.dim
        assert 2 <= dim <= 3, "Only implemented for 2D and 3D."
        self._cell_breaks = [breaks.astype(dtype) for breaks in cart_grid_tp_cpp.cell_breaks]

        if comm.rank == 0:
            nodes_coords = _create_Cartesian_mesh_nodes_from_cells(self._cell_breaks, degree)
        else:
            # TODO: this is a temporary hack. All the nodes and cells
            # are created in rank 0, and then distributed. Better
            # performance could by achieving by distributing nodes and
            # cells since their creation.
            # See https://jsdokken.com/dolfinx_docs/meshes.html#mpi-communication
            # for further details.
            nodes_coords = np.empty((0, dim), dtype=dtype)

        self._cart_grid_tp_cpp_object = cart_grid_tp_cpp
        n_cells = cart_grid_tp_cpp.num_cells_dir

        super().__init__(comm, nodes_coords, n_cells, degree, ghost_mode)

    @property
    def cart_grid_tp_cpp_object(self) -> qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_2D:
        """Returns the stored (C++) Cartesian grid object.

        Returns:
            qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_2D: Stored Cartesian
            grid object.
        """
        return self._cart_grid_tp_cpp_object

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

    def get_cell_bbox(self, dlf_local_cell_id: np.int32) -> npt.NDArray[np.float32 | np.float64]:
        """Computes the bounding box of a single cell.

        Arguments:
            dlf_local_cell_id (np.int32): Local DOLFINx cell whose bounding box is created.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Computed bounding box
                of the cell. It is a 2D with two rows and as many columns
                as coordinates. Both rows represent the minimum and maximum
                coordintes of the bounding box, respectively.
        """

        bbox = np.empty((2, self.gdim), dtype=self.dtype)

        lex_cell_tid = self.get_cell_tensor_index(dlf_local_cell_id)
        for dir in range(self.tdim):
            bbox[0, dir] = self._cell_breaks[dir][lex_cell_tid[dir]]
            bbox[1, dir] = self._cell_breaks[dir][lex_cell_tid[dir] + 1]

        return bbox


def create_Cartesian_mesh(
    comm: MPI.Comm,
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
