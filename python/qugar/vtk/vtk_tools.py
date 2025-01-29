# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Functionalities for generating VTK data structure of quadratures and reparameterizations"""

# import importlib.util
from typing import TypeAlias, cast

from qugar import has_FEniCSx, has_VTK

if not has_VTK:
    raise ValueError("VTK installation not found is required.")

import numpy as np
import numpy.typing as npt
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonDataModel import (
    VTK_LAGRANGE_CURVE,
    VTK_LAGRANGE_HEXAHEDRON,
    VTK_LAGRANGE_QUADRILATERAL,
    VTK_LINE,
    VTK_PIXEL,
    VTK_VERTEX,
    VTK_VOXEL,
    vtkCellArray,
    vtkCompositeDataSet,
    vtkMultiBlockDataSet,
    vtkUnstructuredGrid,
)
from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter

import qugar.cpp
from qugar.cpp import (
    ReparamMesh_1_2,
    ReparamMesh_2_2,
    ReparamMesh_2_3,
    ReparamMesh_3_3,
)
from qugar.mesh import (
    DOLFINx_to_lexicg_faces,
    TensorProdIndex,
    VTK_to_lexicg_nodes,
    map_cells_and_local_facets_to_facets,
)

if has_FEniCSx:
    from qugar.mesh import map_cells_and_local_facets_to_facets

ReparamMesh: TypeAlias = ReparamMesh_1_2 | ReparamMesh_2_2 | ReparamMesh_2_3 | ReparamMesh_3_3


def _get_dimension(reparam: ReparamMesh) -> int:
    """
    Determines the dimension of the given reparametrization mesh.

    Args:
        reparam (ReparamMesh): The reparametrization mesh object.

    Returns:
        int: The dimension of the mesh. Returns 1 for ReparamMesh_1_2,
             2 for ReparamMesh_2_2 or ReparamMesh_2_3, and 3 for ReparamMesh_3_3.
    """
    if isinstance(reparam, ReparamMesh_1_2):
        return 1
    elif isinstance(reparam, ReparamMesh_2_2) or isinstance(reparam, ReparamMesh_2_3):
        return 2
    else:
        return 3


def _scale_points(
    bbox: npt.NDArray[np.float32 | np.float64], points_0_1: npt.NDArray[np.float32 | np.float64]
) -> npt.NDArray[np.float32 | np.float64]:
    """
    Scales points from a normalized range [0, 1] to a specified bounding box.

    Args:
        bbox (npt.NDArray[np.float32 | np.float64]): A 2-element array containing
            the minimum and maximum values of the bounding box.
        points_0_1 (npt.NDArray[np.float32 | np.float64]): An array of points in
            the normalized range [0, 1].

    Returns:
        npt.NDArray[np.float32 | np.float64]: An array of points scaled to the
        specified bounding box.
    """
    xmin = bbox[0]
    xmax = bbox[1]
    return points_0_1 * (xmax - xmin) + xmin


def _facets_const_dirs(dim: int) -> npt.NDArray[np.int32]:
    """
    Generate an array of constant directions of the facets of a reference cell.

    Args:
        dim (int): The dimension of the reference cell. Must be either 2 or 3.

    Returns:
        npt.NDArray[np.int32]: An array of integers representing the constant directions
            of every facet.

    Raises:
        AssertionError: If the dimension is not 2 or 3.
    """
    assert 2 <= dim <= 3, "Invalid dimension."
    if dim == 2:
        return np.array([1, 0, 0, 1], dtype=np.int32)
    else:
        return np.array([2, 1, 0, 0, 1, 2], dtype=np.int32)


def _facets_const_sides(dim: int) -> npt.NDArray[np.int32]:
    """
    Generate an array representing the constant sides of facets for a reference cell.

    Args:
        dim (int): The dimension of the reference cell. Must be either 2 or 3.

    Returns:
        npt.NDArray[np.int32]: An array of integers representing the constant sides of the
            facets of a cell.

    Raises:
        AssertionError: If the dimension is not 2 or 3.
    """
    assert 2 <= dim <= 3, "Invalid dimension."
    if dim == 2:
        return np.array([0, 0, 1, 1], dtype=np.int32)
    else:
        return np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)


def _facets_active_dirs(dim: int) -> npt.NDArray[np.int32]:
    """
    Returns the active directions for the facets of a 2D or 3D reference cell.

    Args:
        dim (int): The dimension of the reference cell. Must be either 2 or 3.

    Returns:
        npt.NDArray[np.int32]: A numpy array containing the active directions for each
            facet of the reference cell.

    Raises:
        AssertionError: If the dimension is not 2 or 3.
    """
    assert 2 <= dim <= 3, "Invalid dimension."
    if dim == 2:
        return np.array([[0], [1], [1], [0]], dtype=np.int32)
    else:
        return np.array([[0, 1], [0, 2], [1, 2], [1, 2], [0, 2], [0, 1]], dtype=np.int32)


def _create_Cartesian_voxel_points(
    bbox: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """
    Generates the (voxel) points of a given bounding box.

    Args:
        bbox (npt.NDArray[np.float32 | np.float64]): A 2x2 or 2x3 array
            representing the bounding box.
            The first row represents the minimum coordinates, and the second
            row represents the maximum coordinates.

    Returns:
        npt.NDArray[np.float32 | np.float64]: An array of points representing the
            corners of the voxel.
            For a 2D bounding box, the array will have shape (4, 2).
            For a 3D bounding box, the array will have shape (8, 3).

    Raises:
        AssertionError: If the shape of bbox is not (2, 2) or (2, 3).
    """
    assert bbox.shape[0] == 2
    dim = bbox.shape[1]
    assert 2 <= dim <= 3, "Invalid dimension."

    dtype = bbox.dtype

    if dim == 2:
        return np.array(
            [
                [bbox[0, 0], bbox[0, 1]],
                [bbox[1, 0], bbox[0, 1]],
                [bbox[0, 0], bbox[1, 1]],
                [bbox[1, 0], bbox[1, 1]],
            ],
            dtype=dtype,
        )
    else:
        return np.array(
            [
                [bbox[0, 0], bbox[0, 1], bbox[0, 2]],
                [bbox[1, 0], bbox[0, 1], bbox[0, 2]],
                [bbox[0, 0], bbox[1, 1], bbox[0, 2]],
                [bbox[1, 0], bbox[1, 1], bbox[0, 2]],
                [bbox[0, 0], bbox[0, 1], bbox[1, 2]],
                [bbox[1, 0], bbox[0, 1], bbox[1, 2]],
                [bbox[0, 0], bbox[1, 1], bbox[1, 2]],
                [bbox[1, 0], bbox[1, 1], bbox[1, 2]],
            ],
            dtype=dtype,
        )


def _create_local_facets_point_ids(dim: int) -> npt.NDArray[np.int32]:
    """
    Create the local point ids for the different facets of a cell.

    Args:
        dim (int): The topological dimension of the cell.

    Returns:
        npt.NDArray[np.int32]: An array of local point IDs for every facet.
    """
    n_facets = dim * 2
    n_inds = np.full(dim, 2, dtype=np.int32)

    return np.array(
        [TensorProdIndex.get_face(n_inds, face_id, lexicg=False) for face_id in range(n_facets)],
        np.int32,
    )


def _get_VTK_cell_type(dim: int) -> int:
    """
    Returns the VTK cell type based on the given dimension.

    Args:
        dim (int): The dimension of the cell (1 for curve, 2 for quadrilateral, 3 for hexahedron).

    Returns:
        int: The VTK cell type corresponding to the given dimension.

    Raises:
        IndexError: If the dimension is not in the range [1, 3].
    """

    cell_types = [
        VTK_LAGRANGE_CURVE,
        VTK_LAGRANGE_QUADRILATERAL,
        VTK_LAGRANGE_HEXAHEDRON,
    ]
    return cell_types[dim - 1]


def _create_VTK_grid_generic(
    points: npt.NDArray[np.floating], conn: npt.NDArray[np.int64 | np.int32], cell_type: int
) -> vtkUnstructuredGrid:
    """
    Creates a VTK unstructured grid from given points and connectivity information.

    Args:
        points (npt.NDArray[np.floating]): Array of point coordinates. If the points are 2D,
            a zero z-coordinate will be added to each point.
        conn (npt.NDArray[np.int64 | np.int32]): Connectivity array specifying the indices of
            points that form each cell.
        cell_type (int): VTK cell type identifier.

    Returns:
        vtkUnstructuredGrid: The constructed VTK unstructured grid.
    """
    if points.shape[1] == 2:
        zeros = np.zeros([points.shape[0], 1], dtype=points.dtype)
        points = np.hstack((points, zeros))

    vtk_points = vtk.vtkPoints()  # type: ignore
    vtk_points.SetData(numpy_to_vtk(points))

    assert len(conn.shape) == 2

    n_cells = conn.shape[0]

    # Add left column specifying number of nodes per cell and flatten array
    n_nodes_per_cell = np.full([n_cells, 1], conn.shape[1], dtype=np.int64)
    conn = np.hstack((n_nodes_per_cell, conn)).ravel()

    cells = vtkCellArray()
    cells.SetCells(
        conn.shape[0],
        numpy_to_vtkIdTypeArray(conn, deep=True),
    )

    grid = vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    grid.SetCells(cell_type, cells)

    return grid


def _create_VTK_grid(
    points: npt.NDArray[np.floating],
    cell_type: int,
    n_pts_cell: int,
) -> vtkUnstructuredGrid:
    """
    Creates a VTK unstructured grid from given points and cell information.

    Args:
        points (npt.NDArray[np.floating]): A numpy array of points coordinates.
        cell_type (int): The VTK cell type identifier.
        n_pts_cell (int): Number of points per cell.

    Returns:
        vtkUnstructuredGrid: The generated VTK unstructured grid.
    """
    vtk_points = vtk.vtkPoints()  # type: ignore
    vtk_points.SetData(numpy_to_vtk(points))
    conn = np.arange(points.shape[0]).reshape(-1, n_pts_cell)
    return _create_VTK_grid_generic(points, conn, cell_type)


def _create_Cartesian_cells_grid(
    grid: qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D,
    lex_cell_ids: npt.NDArray[np.int64 | np.int32],
) -> vtkUnstructuredGrid:
    """
    Create a VTK unstructured grid from a Cartesian grid and a list of
    lexicographical cell IDs.

    Args:
        grid (qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D): The input
            Cartesian grid, either 2D or 3D.
        lex_cell_ids (npt.NDArray[np.int64 | np.int32]): Array of lexicographical cell IDs
            to be included in the VTK grid.

    Returns:
        vtkUnstructuredGrid: The resulting VTK unstructured grid.
    """
    assert np.all(lex_cell_ids >= 0)

    tdim = grid.dim
    n_pts_cell = 2**tdim
    n_cells = lex_cell_ids.size
    n_pts = n_cells * n_pts_cell
    dtype = np.float64
    points = np.zeros((n_pts, 3), dtype=dtype)

    ofs = 0
    for cell_id in lex_cell_ids:
        domain = grid.get_cell_domain(cell_id)
        bbox = np.vstack([domain.min_corner, domain.max_corner])
        points[ofs : ofs + n_pts_cell, :tdim] = _create_Cartesian_voxel_points(bbox)
        ofs += n_pts_cell

    cell_type = VTK_PIXEL if tdim == 2 else VTK_VOXEL
    grid = _create_VTK_grid(points, cell_type, n_pts_cell)

    cell_data = grid.GetCellData()
    cell_data.AddArray(dsa.numpyTovtkDataArray(lex_cell_ids, "Lexicographical cell ids"))

    return grid


def _create_Cartesian_facets_grid(
    grid: qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D,
    lex_cells: npt.NDArray[np.int32],
    local_facets: npt.NDArray[np.int32],
) -> vtkUnstructuredGrid:
    """
    Create a VTK grid including some facets a Cartesian grid.

    Args:
        grid (qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D): The input grid,
            either 2D or 3D, from which facets are extracted.
        lex_cells (npt.NDArray[np.int32]): Array of lexicographical cell indices
            corresponding to the facets to be included in the VTK grid.
        local_facets (npt.NDArray[np.int32]): Array of local facet indices, referred to
            `lex_cells`, to be included in the VTK grid.

    Returns:
        vtkUnstructuredGrid: The resulting VTK unstructured grid containing the facets.
    """
    tdim = grid.dim
    n_local_facets = 2 * tdim

    assert np.all(lex_cells >= 0)
    assert np.all(local_facets >= 0) and np.all(local_facets < n_local_facets)
    n_facets = lex_cells.size
    assert n_facets == local_facets.size

    n_pts_cell = 2 ** (tdim - 1)
    n_pts = n_facets * n_pts_cell
    dtype = np.float64

    points = np.zeros((n_pts, 3), dtype=dtype)

    facets_point_ids = _create_local_facets_point_ids(tdim)

    ofs = 0
    for cell_id, local_facet_id in zip(lex_cells, local_facets):
        domain = grid.get_cell_domain(cell_id)
        bbox = np.vstack([domain.min_corner, domain.max_corner])
        bbox_points = _create_Cartesian_voxel_points(bbox)
        facet_point_ids = facets_point_ids[local_facet_id]

        points[ofs : ofs + n_pts_cell, :tdim] = bbox_points[facet_point_ids]
        ofs += n_pts_cell

    cell_type = VTK_LINE if tdim == 2 else VTK_PIXEL
    grid = _create_VTK_grid(points, cell_type, n_pts_cell)

    cell_data = grid.GetCellData()
    cell_data.AddArray(dsa.numpyTovtkDataArray(lex_cells, "Lexicographical cell ids"))
    cell_data.AddArray(dsa.numpyTovtkDataArray(local_facets, "Lexicographical local facet ids"))

    return grid


def _create_quad_points_grid(
    grid: qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D,
    quad: qugar.cpp.CutCellsQuad_2D | qugar.cpp.CutCellsQuad_3D,
    add_normals: bool = False,
) -> vtkUnstructuredGrid:
    """
    Create a VTK unstructured grid containing quadrature points.

    Args:
        grid (qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D): The Cartesian grid
            associated to teh quadrature `quad`.
        quad (qugar.cpp.CutCellsQuad_2D | qugar.cpp.CutCellsQuad_3D): The quadrature
            containing the points to be included in the VTK grid.
        add_normals (bool, optional): Whether to add normals to the VTK grid. Defaults to False.

    Returns:
        vtkUnstructuredGrid: The resulting VTK unstructured grid.
    """
    n_pts, dim = quad.points.shape

    points = np.zeros((n_pts, 3), quad.points.dtype)
    ofs = 0
    for cell_id, n_pts_cell in zip(quad.cells, quad.n_pts_per_entity):
        domain = grid.get_cell_domain(cell_id)
        bbox = np.vstack([domain.min_corner, domain.max_corner])
        points_0_1 = quad.points[ofs : ofs + n_pts_cell]
        points[ofs : ofs + n_pts_cell, :dim] = _scale_points(bbox, points_0_1)
        ofs += n_pts_cell

    vtk_grid = _create_VTK_grid(points, VTK_VERTEX, 1)

    cell_data = vtk_grid.GetCellData()

    lex_cell_ids = np.repeat(quad.cells.reshape(-1), quad.n_pts_per_entity)
    cell_data.AddArray(dsa.numpyTovtkDataArray(lex_cell_ids, "Lexicographical cell ids"))

    cell_data.AddArray(dsa.numpyTovtkDataArray(quad.weights, "Weights"))

    if add_normals:
        # assert isinstance(quad, CustomQuadIntBoundary)
        normals = np.zeros((n_pts, 3), quad.points.dtype)
        normals[:, :dim] = quad.normals
        cell_data.AddArray(dsa.numpyTovtkDataArray(normals, "Normals"))

    return vtk_grid


def _create_quad_facet_points_grid(
    grid: qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D,
    quad: qugar.cpp.CutIsoBoundsQuad_1D | qugar.cpp.CutIsoBoundsQuad_2D,
) -> vtkUnstructuredGrid:
    """
    Creates a VTK unstructured grid containing quadrature points contained
    in `quad` and associated to the facets of `grid`.

    Args:
        grid (qugar.cpp.CartGridTP_2D | qugar.cpp.CartGridTP_3D): The Cartesian grid for
            whose facets quadrature points are associated to.
        quad (qugar.cpp.CutIsoBoundsQuad_2D | qugar.cpp.CutIsoBoundsQuad_3D): The
            quadrature containing the facet quadrature points.

    Returns:
        vtkUnstructuredGrid: The generated VTK unstructured grid.
    """
    n_pts, dim = quad.points.shape
    dim += 1

    faces_const_dirs = _facets_const_dirs(dim)
    faces_const_sides = _facets_const_sides(dim)
    faces_active_dirs = _facets_active_dirs(dim)

    dtype = quad.points.dtype
    points = np.zeros((n_pts, 3), dtype)
    ofs = 0
    for cell_id, facet_id, n_pts_cell in zip(quad.cells, quad.facets, quad.n_pts_per_entity):
        domain = grid.get_cell_domain(cell_id)
        bbox = np.vstack([domain.min_corner, domain.max_corner])

        const_dir = faces_const_dirs[facet_id]
        const_side = dtype.type(faces_const_sides[facet_id])
        active_dirs = faces_active_dirs[facet_id]

        points_facet = points[ofs : ofs + n_pts_cell, :dim]
        points_facet[:, const_dir] = const_side
        for i, dir in enumerate(active_dirs):
            points_facet[:, dir] = quad.points[ofs : ofs + n_pts_cell, i]

        points[ofs : ofs + n_pts_cell, :dim] = _scale_points(bbox, points_facet)
        ofs += n_pts_cell

    vtk_grid = _create_VTK_grid(points, VTK_VERTEX, 1)

    cell_data = vtk_grid.GetCellData()

    lex_cell_ids = np.repeat(quad.cells.reshape(-1), quad.n_pts_per_entity)
    cell_data.AddArray(dsa.numpyTovtkDataArray(lex_cell_ids, "Lexicographical cell ids"))

    lex_local_facets = np.repeat(quad.facets.reshape(-1), quad.n_pts_per_entity)
    cell_data.AddArray(dsa.numpyTovtkDataArray(lex_local_facets, "Lexicographical local facet ids"))

    cell_data.AddArray(dsa.numpyTovtkDataArray(quad.weights, "Weights"))

    return vtk_grid


def _quadrature_to_VTK(
    unf_domain: qugar.cpp.UnfittedDomain_2D | qugar.cpp.UnfittedDomain_3D,
    n_pts_dir: int,
) -> vtkMultiBlockDataSet:
    """
    Generates quadrature data for an unfitted domain and exports it to a
    VTK data structure. It generates, and export, quadrature points
    for the interior of the cells, the interior boundaries, and the facets.

    Args:
        unf_domain (qugar.cpp.UnfittedDomain_2D | qugar.cpp.UnfittedDomain_3D):
            The unfitted domain for which the quadratures are generated.
        n_pts_dir (int): Number of points per direction for quadrature.

    Returns:
        vtkMultiBlockDataSet:
            A VTK multiblock dataset containing the quadrature data.
    """
    multiblock = vtkMultiBlockDataSet()

    def set_name(block_id, block_name):
        multiblock.GetMetaData(block_id).Set(vtkCompositeDataSet.NAME(), block_name)

    def add_cells(cells, block_id, block_name):
        vtk_grid = _create_Cartesian_cells_grid(unf_domain.grid, cells)
        multiblock.SetBlock(block_id, vtk_grid)
        set_name(block_id, block_name)

    def add_facets(lex_cells_facets, block_id, block_name):
        grid = _create_Cartesian_facets_grid(
            unf_domain.grid, lex_cells_facets[0], lex_cells_facets[1]
        )
        multiblock.SetBlock(block_id, grid)
        set_name(block_id, block_name)

    add_cells(unf_domain.cut_cells, 0, "Cut cells")
    add_cells(unf_domain.full_cells, 1, "Full cells")
    add_cells(unf_domain.empty_cells, 2, "Empty cells")

    add_facets(unf_domain.cut_facets, 3, "Cut facets")
    add_facets(unf_domain.full_facets, 4, "Full facets")
    add_facets(unf_domain.empty_facets, 5, "Empty facets")

    cells_quad = qugar.cpp.create_quadrature(unf_domain, unf_domain.cut_cells, n_pts_dir)
    multiblock.SetBlock(6, _create_quad_points_grid(unf_domain.grid, cells_quad))
    set_name(6, "Points cells")

    int_bdry_quad = qugar.cpp.create_interior_bound_quadrature(
        unf_domain, unf_domain.cut_cells, n_pts_dir
    )
    multiblock.SetBlock(
        7, _create_quad_points_grid(unf_domain.grid, int_bdry_quad, add_normals=True)
    )
    set_name(7, "Points interior boundaries")

    cut_facets_cells, cut_facets_local_facets = unf_domain.cut_facets
    facets_quad = qugar.cpp.create_facets_quadrature(
        unf_domain, cut_facets_cells, cut_facets_local_facets, n_pts_dir
    )
    multiblock.SetBlock(8, _create_quad_facet_points_grid(unf_domain.grid, facets_quad))
    set_name(8, "Points facets")

    return multiblock


if has_FEniCSx:
    from qugar.unfitted_domain import UnfittedDomain

    def _quadrature_to_VTK_dolfinx(
        unf_domain: UnfittedDomain,
        n_pts_dir: int = 4,
    ) -> vtkMultiBlockDataSet:
        """
        Generates quadrature data for an unfitted domain and exports it to a
        VTK data structure. It generates, and export, quadrature points
        for the interior of the cells, the interior boundaries, and the facets.

        In addition to the lexicographical indices of cells and facets,
        this function also appends the DOLFINx associated indices.

        @note: This function is only available when FEniCSx is installed.

        Args:
            unf_domain (UnfittedDomain):
                The unfitted domain for which the quadratures are generated.
            n_pts_dir (int): Number of points per direction for quadrature.

        Returns:
            vtkMultiBlockDataSet:
                A VTK multiblock dataset containing the quadrature data.
        """
        vtk_mb = _quadrature_to_VTK(unf_domain.cpp_object, n_pts_dir)

        cart_mesh = unf_domain.cart_mesh
        dim = cart_mesh.tdim
        index_map = cart_mesh.mesh.topology.index_map(cart_mesh.tdim)

        def add_dolfinx_ids(block_id: int, add_facets: bool):
            cell_data = vtk_mb.GetBlock(block_id).GetCellData()
            lex_cell_ids = dsa.vtkDataArrayToVTKArray(cell_data.GetArray(0))
            dlf_cell_ids = cast(
                npt.NDArray[np.int32],
                cart_mesh.get_DOLFINx_local_cell_ids(lex_cell_ids, lexicg=True),
            )
            global_dlf_cell_ids = index_map.local_to_global(dlf_cell_ids)

            cell_data.AddArray(dsa.numpyTovtkDataArray(dlf_cell_ids, "DOLFINx local cell ids"))
            cell_data.AddArray(
                dsa.numpyTovtkDataArray(global_dlf_cell_ids, "DOLFINx global cell ids")
            )

            if add_facets:
                lex_local_facets = dsa.vtkDataArrayToVTKArray(cell_data.GetArray(1))
                dlf_to_lex_facets = DOLFINx_to_lexicg_faces(dim)
                dlf_local_facets = dlf_to_lex_facets[lex_local_facets]
                dlf_facets = map_cells_and_local_facets_to_facets(
                    cart_mesh.mesh, dlf_cell_ids, dlf_local_facets
                )

                cell_data.AddArray(
                    dsa.numpyTovtkDataArray(
                        dlf_local_facets, "DOLFINx local facet ids (local to cell)"
                    )
                )
                cell_data.AddArray(
                    dsa.numpyTovtkDataArray(
                        dlf_facets, "DOLFINx local facet ids (local to the process)"
                    )
                )

        for block_id in range(9):
            add_dolfinx_ids(block_id, False)

        for block_id in [3, 4, 5, 8]:
            add_dolfinx_ids(block_id, True)

        return vtk_mb


def quadrature_to_VTK(
    unf_domain,
    n_pts_dir: int = 4,
) -> vtkMultiBlockDataSet:
    """
    Generates quadrature data for an unfitted domain and exports it to a
    VTK data structure. It generates, and export, quadrature points
    for the interior of the cells, the interior boundaries, and the facets.

    Args:
        unf_domain (
            qugar.cpp.UnfittedDomain_2D | qugar.cpp.UnfittedDomain_3D |
            qugar.unfitted_domain.UnfittedDomain
        ):
            The unfitted domain for which the quadratures are generated.
        n_pts_dir (int, optional): Number of points per direction for quadrature. Defaults to 4.

    Returns:
        vtkMultiBlockDataSet:
            A VTK multiblock dataset containing the quadrature data.
    """
    if isinstance(unf_domain, (qugar.cpp.UnfittedDomain_2D, qugar.cpp.UnfittedDomain_3D)):
        return _quadrature_to_VTK(unf_domain, n_pts_dir)
    else:
        if not has_FEniCSx:
            raise ValueError("FEniCSx installation not found is required.")

        from qugar.unfitted_domain import UnfittedDomain

        assert isinstance(unf_domain, UnfittedDomain), "Invalid type."
        return _quadrature_to_VTK_dolfinx(unf_domain, n_pts_dir)


def reparam_to_VTK(reparam: ReparamMesh) -> vtkMultiBlockDataSet:
    """
    Converts a ReparamMesh object to a vtkMultiBlockDataSet.

    Args:
        reparam (ReparamMesh): The reparametrized mesh object to be converted.

    Returns:
        vtkMultiBlockDataSet: A VTK multiblock dataset containing the interior and boundary grids.

    """
    dim = _get_dimension(reparam)

    degree = reparam.order - 1
    vtk_mask = VTK_to_lexicg_nodes(dim, degree)

    conn = reparam.cells_conn.astype(np.int64)
    conn = conn[:, vtk_mask]

    cell_type = _get_VTK_cell_type(dim)
    bulk = _create_VTK_grid_generic(reparam.points, conn, cell_type)

    multiblock = vtkMultiBlockDataSet()
    multiblock.SetBlock(0, bulk)
    multiblock.GetMetaData(0).Set(vtkCompositeDataSet.NAME(), "interior")

    vtk_mask_wire = VTK_to_lexicg_nodes(1, degree)
    wire_conn = reparam.wirebasket_conn.astype(np.int64)
    wire_conn = wire_conn[:, vtk_mask_wire]

    cell_type = _get_VTK_cell_type(1)
    wirebasket = _create_VTK_grid_generic(reparam.points, wire_conn, cell_type)
    multiblock.SetBlock(1, wirebasket)
    multiblock.GetMetaData(1).Set(vtkCompositeDataSet.NAME(), "boundary")

    return multiblock


def write_VTK_to_file(vtk_mb: vtkMultiBlockDataSet, fname: str, use_ascii: bool = False):
    """
    Writes a vtkMultiBlockDataSet to a VTK file.

    Args:
        vtk_mb (vtkMultiBlockDataSet): The multi-block dataset to write.
        fname (str): The base name of the output file (without extension).
        use_ascii (bool, optional): If True, writes the file in ASCII format. Defaults to False.
    """

    writer = vtkXMLMultiBlockDataWriter()
    writer.SetFileName(f"{fname}.vtm")
    writer.SetInputData(vtk_mb)
    if use_ascii:
        writer.SetDataModeToAscii()
    writer.Update()
    writer.Write()
