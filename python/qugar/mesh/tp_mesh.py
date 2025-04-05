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

from typing import Optional

import qugar.utils

if not qugar.utils.has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import numpy.typing as npt
from basix import CellType
from dolfinx.cpp.mesh import GhostMode

import qugar.cpp
from qugar.mesh.mesh import Mesh
from qugar.mesh.tp_index import TensorProdIndex
from qugar.mesh.utils import (
    DOLFINx_to_lexicg_nodes,
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


class TensorProductMesh(Mesh):
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
    `self.get_original_node_ids` or `self.get_original_cell_ids`.
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

        tdim = len(n_cells)
        assert 1 <= tdim <= 3, "Invalid dimension."

        cell_type = [CellType.interval, CellType.quadrilateral, CellType.hexahedron][
            len(n_cells) - 1
        ]

        self._n_cells = np.array(n_cells).astype(np.int32)
        assert np.all(self._n_cells > 0)

        if comm.rank == 0:
            conn = _create_tensor_prod_mesh_conn(tdim, degree, self.num_cells_dir)
            conn = np.asarray(conn, dtype=np.int64, order="C")

        else:
            # TODO: this is a temporary hack. All the nodes and cells
            # are created in rank 0, and then distributed. Better
            # performance could be achieved by distributing nodes and
            # cells since their creation.
            # See https://jsdokken.com/dolfinx_docs/meshes.html#mpi-communication
            # for further details.
            assert nodes_coords.shape[0] == 0

            n_nodes_per_cell = 2**tdim
            conn = np.empty((0, n_nodes_per_cell), dtype=np.int64, order="C")

        Mesh.__init__(
            self, comm, nodes_coords, conn, cell_type, degree, ghost_mode, merge_nodes, merge_tol
        )

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
    def num_cells_tp(self) -> np.int64:
        """Gets the total number of cells of the underlying tensor-product mesh.

        Returns:
            np.int64: Number of cells of the tensor-product mesh.
        """
        return np.prod(self._n_cells, dtype=np.int64)

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

    def get_cell_tensor_index(self, dlf_local_cell_id: np.int32) -> npt.NDArray[np.int32]:
        """Computes the tensor index referred to the tensor-product mesh
        of a given (local) DOLFINx cell index.

        Args:
            dlf_local_cell_id (np.int32): Local DOLFINx cell index to be transformed.

        Returns:
            npt.NDArray[np.int32]: Cell tensor index
        """

        lex_cell_id = self.get_original_cell_ids(np.array(dlf_local_cell_id, dtype=np.int32))[0]
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

        TensorProductMesh.__init__(self, comm, nodes_coords, n_cells, degree, ghost_mode)

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

    dtype = (np.dtype(np.float64) if xmin is None else xmin.dtype) if xmax is None else xmax.dtype
    xmin_ = np.zeros(dim, dtype=dtype) if xmin is None else xmin
    xmax_ = np.ones(dim, dtype=dtype) if xmax is None else xmax

    assert dim == xmin_.size and dim == xmax_.size
    assert xmin_.dtype == xmax_.dtype and xmin_.dtype in [np.float32, np.float64]
    assert np.all(xmax > xmin_)

    cell_breaks = []
    for dir in range(dim):
        cell_breaks.append(np.linspace(xmin_[dir], xmax_[dir], n_cells[dir] + 1, dtype=np.float64))

    cpp_grid = qugar.cpp.create_cart_grid(cell_breaks)

    return CartesianMesh(comm, cpp_grid, degree, ghost_mode, xmin_.dtype.type)
