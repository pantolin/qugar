#' --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------
"""
Functionalities for generating PyVista data structures of grids, unfitted domains,
quadratures and reparameterizations.
"""

from typing import Any, Optional, TypeAlias, cast

from qugar.utils import has_FEniCSx, has_PyVista

if not has_PyVista:
    raise ValueError("PyVista installation not found is required.")

import numpy as np
import numpy.typing as npt
import pyvista as pv

import qugar.cpp
from qugar.cpp import (
    CartGridTP_2D,
    CartGridTP_3D,
    CutCellsQuad_2D,
    CutCellsQuad_3D,
    CutIsoBoundsQuad_1D,
    CutIsoBoundsQuad_2D,
    CutUnfBoundsQuad_2D,
    CutUnfBoundsQuad_3D,
    ReparamMesh_1_2,
    ReparamMesh_2_2,
    ReparamMesh_2_3,
    ReparamMesh_3_3,
    UnfittedDomain_2D,
    UnfittedDomain_3D,
)
from qugar.mesh import (
    DOLFINx_to_lexicg_faces,
    VTK_to_lexicg_faces,
    VTK_to_lexicg_nodes,
    lexicg_to_VTK_faces,
)


def _scale_points(
    bbox: npt.NDArray[np.float32 | np.float64], points_0_1: npt.NDArray[np.float32 | np.float64]
) -> npt.NDArray[np.float32 | np.float64]:
    """
    Scales points from a normalized range [0, 1] to a specified bounding box.

    Args:
        bbox (npt.NDArray[np.float32 | np.float64]): A 2-element array containing
            the minimum (first row) and maximum (second row) values of the bounding box.
        points_0_1 (npt.NDArray[np.float32 | np.float64]): An array of points in
            the normalized range [0, 1].

    Returns:
        npt.NDArray[np.float32 | np.float64]: An array of points scaled to the
        specified bounding box.
    """
    assert bbox.dtype == points_0_1.dtype, "Incompatible data types."
    assert bbox.shape == (2, points_0_1.shape[1]), "Incompatible shapes."

    xmin = bbox[0]
    xmax = bbox[1]
    return points_0_1 * (xmax - xmin) + xmin


def _facets_const_dirs(dim: int) -> npt.NDArray[np.int32]:
    """
    Generates an array of the constant directions of the facets of a reference cell.

    Args:
        dim (int): The dimension of the reference cell. Must be either 2 or 3.

    Returns:
        npt.NDArray[np.int32]: An array of integers representing the constant directions
            of every facet.

    Note:
        The indices of the facets correspond to the lexicographical ordering.

    Raises:
        AssertionError: If the dimension is not 2 or 3.
    """
    assert 2 <= dim <= 3, "Invalid dimension."
    if dim == 2:
        return np.array([0, 0, 1, 1], dtype=np.int32)
    else:
        return np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)


def _facets_const_sides(dim: int) -> npt.NDArray[np.int32]:
    """
    Generates an array representing the constant sides of the facets for a reference cell.

    Args:
        dim (int): The dimension of the reference cell. Must be either 2 or 3.

    Returns:
        npt.NDArray[np.int32]: An array of integers representing the constant sides of the
            facets of a cell.

    Note:
        The indices of the facets correspond to the lexicographical ordering.

    Raises:
        AssertionError: If the dimension is not 2 or 3.
    """
    assert 2 <= dim <= 3, "Invalid dimension."
    if dim == 2:
        return np.array([0, 1, 0, 1], dtype=np.int32)
    else:
        return np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)


def _facets_active_dirs(dim: int) -> npt.NDArray[np.int32]:
    """
    Returns the active directions for the facets of a 2D or 3D reference cell.

    Args:
        dim (int): The dimension of the reference cell. Must be either 2 or 3.

    Returns:
        npt.NDArray[np.int32]: A numpy array containing the active directions for each
            facet of the reference cell.

    Note:
        The indices of the facets correspond to the lexicographical ordering.

    Raises:
        AssertionError: If the dimension is not 2 or 3.
    """
    assert 2 <= dim <= 3, "Invalid dimension."
    if dim == 2:
        return np.array([[1], [1], [0], [0]], dtype=np.int32)
    else:
        return np.array([[1, 2], [1, 2], [0, 2], [0, 2], [0, 1], [0, 1]], dtype=np.int32)


def _get_PyVista_cell_type(dim: int, degree: int) -> pv.CellType:
    """
    Returns the PyVista cell type based on the given dimension.

    For degree 1, the `LINE`, `QUAD`, and `HEXAHEDRON` cell types are returned.
    For higher-degrees, the lagrange versions are returned, namely
    `LAGRANGE_CURVE`, `LAGRANGE_QUADRILATERAL`, and `LAGRANGE_HEXAHEDRON`.

    Args:
        dim (int): The dimension of the cell (1 for curve, 2 for quadrilateral, 3 for hexahedron).
        degree (int): The degree of the cell.

    Returns:
        pv.CellType: The PyVista cell type corresponding to the given dimension.

    Raises:
        IndexError: If the dimension is not in the range [1, 3].
    """

    assert 1 <= dim <= 3, "Invalid dimension."

    if degree == 1:
        cell_types = [
            pv.CellType.LINE,
            pv.CellType.QUAD,
            pv.CellType.HEXAHEDRON,
        ]
    else:  # High-order
        cell_types = [
            pv.CellType.LAGRANGE_CURVE,
            pv.CellType.LAGRANGE_QUADRILATERAL,
            pv.CellType.LAGRANGE_HEXAHEDRON,
        ]
    return cast(pv.CellType, cell_types[dim - 1])


def _create_unstructured_grid_from_points(points: npt.NDArray[np.float64]) -> pv.UnstructuredGrid:
    """
    Create a PyVista unstructured grid from a set of points.

    This function generates an unstructured grid where each point is treated as a vertex.

    Args:
        points (npt.NDArray[np.float64]): A 2D numpy array of shape (n_points, 3) containing
            the coordinates of the points.

    Returns:
        pv.UnstructuredGrid: An unstructured grid object with the given points as vertices.
    """
    assert points.shape[1] == 3, "Points must be 3D."

    n_points = points.shape[0]
    conn = np.ones((n_points, 2), dtype=np.int64)
    conn[:, 1] = np.arange(n_points)

    cell_type = pv.CellType.VERTEX
    cell_types = np.full(n_points, cell_type, dtype=np.uint8)
    return pv.UnstructuredGrid(conn, cell_types, points)


def _create_quad_points_grid(
    grid: CartGridTP_2D | CartGridTP_3D,
    points: npt.NDArray[np.float64],
    n_pts_per_entity: npt.NDArray[np.int32],
    cells_ids: npt.NDArray[np.int32],
    weights: npt.NDArray[np.float64],
    normals: Optional[npt.NDArray[np.float64]] = None,
) -> pv.UnstructuredGrid:
    """
    Creates a PyVista unstructured grid containing the given set of points.

    This function scales the points from the unit domain to the bounding box of the cell,
    creates an unstructured grid from the points, and appends cell data such as
    lexicographical cell ids, weights, and optionally normals.

    Args:
        grid (CartGridTP_2D | CartGridTP_3D): The grid object containing cell domain information.
        points (npt.NDArray[np.float64]): Array of points in the unit domain that will be added
            to the grid.
        n_pts_per_entity (npt.NDArray[np.int32]): Number of points per entity (cell or facet).
        cells_ids (npt.NDArray[np.int32]): Array of cell ids associated to the points.
        weights (npt.NDArray[np.float64]): Array of weights for each point.
        normals (Optional[npt.NDArray[np.float64]], optional): Array of normals for each point.
            Defaults to None.

    Returns:
        pv.UnstructuredGrid: The resulting unstructured grid with appended cell data.
    """
    assert cells_ids.size == n_pts_per_entity.size

    n_pts = weights.size
    dim = grid.dim

    assert points.shape == (n_pts, dim)

    # First we scale the points from the unit domain to the bounding box of the cell.
    # If 2D, we add a zero component to the points.
    new_points = np.zeros((n_pts, 3), points.dtype)
    ofs = 0
    for cell_id, n_pts_cell in zip(cells_ids, n_pts_per_entity):
        domain = grid.get_cell_domain(cell_id)
        bbox = np.vstack([domain.min_corner, domain.max_corner])
        points_0_1 = points[ofs : ofs + n_pts_cell]
        new_points[ofs : ofs + n_pts_cell, :dim] = _scale_points(bbox, points_0_1)
        ofs += n_pts_cell

    points_grid = _create_unstructured_grid_from_points(new_points)

    # Append cell data.

    lex_cell_ids = np.repeat(cells_ids.reshape(-1), n_pts_per_entity)
    points_grid.cell_data["Lexicographical cell ids"] = lex_cell_ids

    points_grid.cell_data["Weights"] = weights

    if normals is not None:
        assert normals.shape == (n_pts, dim)
        new_normals = np.zeros((n_pts, 3), points.dtype)
        if n_pts == 0:
            new_normals = np.empty(0, points.dtype)
        else:
            new_normals[:, :dim] = normals
        points_grid.cell_data["Normals"] = new_normals

    return points_grid


def _cut_quad_to_PyVista(
    grid: CartGridTP_2D | CartGridTP_3D,
    quad: CutCellsQuad_2D | CutCellsQuad_3D | CutUnfBoundsQuad_2D | CutUnfBoundsQuad_3D,
) -> pv.UnstructuredGrid:
    """
    Creates a PyVista unstructured grid containing quadrature points and weights.

    For the case of quadrature of unfitted boundaries, it also includes normals.

    Args:
        grid (CartGridTP_2D | CartGridTP_3D): The Cartesian grid associated
            to the quadrature `quad`.
        quad (CutCellsQuad_2D | CutCellsQuad_3D | CutUnfBoundsQuad_2D | CutUnfBoundsQuad_3D): The
            quadrature containing the points to be included in the PyVista grid.

    Returns:
        pv.UnstructuredGrid: The resulting PyVista unstructured grid.
    """

    return _create_quad_points_grid(
        grid,
        quad.points,
        quad.n_pts_per_entity,
        quad.cells,
        quad.weights,
        quad.normals if hasattr(quad, "normals") else None,
    )


def _create_quad_facet_points_grid(
    grid: CartGridTP_2D | CartGridTP_3D,
    quads: list[CutIsoBoundsQuad_1D | CutIsoBoundsQuad_2D],
) -> pv.UnstructuredGrid:
    """
    Creates a PyVista unstructured grid containing quadrature points contained
    in `quad` and associated to the facets of `grid`.

    Args:
        grid (CartGridTP_2D | CartGridTP_3D): The Cartesian grid for
            whose facets quadrature points are associated to.
        quads (list[CutIsoBoundsQuad_2D | CutIsoBoundsQuad_3D]): The
            list of quadratures containing the facet quadrature points.

    Returns:
        pv.UnstructuredGrid: The generated PyVista unstructured grid.
    """

    assert len(quads) > 0, "Empty list of quadratures."

    # First we transform point from the reference facet (with one dimension less)
    # to facets on the reference cell (with the same dimension as the cell).
    dim = quads[0].points.shape[1] + 1
    dtype = quads[0].points.dtype

    n_pts = sum(quad.points.shape[0] for quad in quads)
    n_facets = sum(quad.cells.size for quad in quads)

    facets_const_dirs = _facets_const_dirs(dim)
    facets_const_sides = _facets_const_sides(dim)
    facets_active_dirs = _facets_active_dirs(dim)

    points = np.zeros((n_pts, dim), dtype)
    weights = np.zeros(n_pts, dtype)
    cells = np.zeros(n_facets, dtype=np.int32)
    n_pts_per_entity = np.zeros(n_facets, dtype=np.int32)
    facets = np.zeros(n_facets, dtype=np.int32)

    ofs_pts = 0
    ofs_cell = 0
    for quad in quads:
        ofs_pts_quad = 0
        for cell_id, facet_id, n_pts_facet in zip(quad.cells, quad.facets, quad.n_pts_per_entity):
            const_dir = facets_const_dirs[facet_id]
            const_side = dtype.type(facets_const_sides[facet_id])
            active_dirs = facets_active_dirs[facet_id]

            points_facet = points[ofs_pts : ofs_pts + n_pts_facet, :dim]
            points_facet[:, const_dir] = const_side
            for i, dir in enumerate(active_dirs):
                points_facet[:, dir] = quad.points[ofs_pts_quad : ofs_pts_quad + n_pts_facet, i]
            weights[ofs_pts : ofs_pts + n_pts_facet] = quad.weights[
                ofs_pts_quad : ofs_pts_quad + n_pts_facet
            ]
            cells[ofs_cell] = cell_id
            facets[ofs_cell] = facet_id
            n_pts_per_entity[ofs_cell] = n_pts_facet

            ofs_pts += n_pts_facet
            ofs_pts_quad += n_pts_facet
            ofs_cell += 1

    # Then, we create the grid.
    points_grid = _create_quad_points_grid(grid, points, n_pts_per_entity, cells, weights, None)

    # Finally we append the lexicographical local facet ids as cell data.
    lex_local_facets = np.repeat(facets.reshape(-1), n_pts_per_entity)
    points_grid.cell_data["Lexicographical local facets ids"] = lex_local_facets

    return points_grid


if has_FEniCSx:
    from qugar.mesh import Mesh

    def _append_DOLFINx_cell_ids(grid: pv.UnstructuredGrid, mesh: Mesh) -> None:
        """
        Appends DOLFINx local and global cell IDs to the given PyVista UnstructuredGrid.

        This function adds two new cell data arrays to the provided `grid`:
        "DOLFINx local cell ids" and "DOLFINx global cell ids". These arrays are
        derived from the lexicographical cell IDs present in the grid and the
        corresponding DOLFINx cell IDs from the provided `mesh`.

        Args:
            grid (pv.UnstructuredGrid): The PyVista UnstructuredGrid to which the
                DOLFINx cell IDs will be appended.
            mesh (Mesh): The QUGaR Mesh object containing the DOLFINx
                cell ID information.

        Raises:
            AssertionError: If the number of global cells in the mesh does not match
            the number of local cells. This is not implemented yet.
        """

        lex_cell_ids = grid.cell_data["Lexicographical cell ids"]

        grid.cell_data["DOLFINx local cell ids"] = mesh.get_DOLFINx_local_cell_ids(lex_cell_ids)
        grid.cell_data["DOLFINx global cell ids"] = mesh.get_DOLFINx_global_cell_ids(lex_cell_ids)

        return None

    def _append_DOLFINx_facets_ids(grid: pv.UnstructuredGrid, dim: int) -> None:
        """
        Appends DOLFINx local facet IDs to the given PyVista UnstructuredGrid.

        This function maps the lexicographical local facet IDs to DOLFINx local facet IDs
        and appends them to the cell data of the provided grid.

        Args:
            grid (pv.UnstructuredGrid): The PyVista UnstructuredGrid to which the DOLFINx
                local facet IDs will be appended.
            dim (int): The dimension of the grid.
        """

        dlf_lex_mask = DOLFINx_to_lexicg_faces(dim)

        lex_facets = grid.cell_data["Lexicographical local facets ids"]
        dlf_facets = dlf_lex_mask[lex_facets]
        grid.cell_data["DOLFINx local facets ids"] = dlf_facets

        return None


def quadrature_to_PyVista(
    domain: Any,
    n_pts_dir: int = 4,
) -> pv.MultiBlock:
    """
    Generates quadrature data for an unfitted domain and exports it to a
    PyVista data structure. It generates, and export, quadrature points
    for the interior of the cells, the unfitted boundaries, and the facets.

    In addition to the lexicographical indices of cells and facets,
    this function also appends the DOLFINx associated indices.

    @note: This function is only available when FEniCSx is installed.

    Args:
        domain (UnfittedDomain): The unfitted domain for which the quadratures
            are generated.
        n_pts_dir (int): Number of points per direction for quadrature.

    Returns:
        pv.MultiBlock: A PyVista multiblock dataset containing the quadratures
            of the cut cells, cut facets, and unfitted boundaries.
    """

    is_cpp = isinstance(domain, (UnfittedDomain_2D, UnfittedDomain_3D))
    if is_cpp:
        domain_ = domain
    else:
        assert has_FEniCSx, "FEniCSx is required to convert a TensorProductMesh to PyVista."
        from qugar.mesh import UnfittedDomain

        assert isinstance(domain, UnfittedDomain), "Invalid type."
        domain_ = domain.cpp_unf_domain_object

    cells_quad = qugar.cpp.create_quadrature(domain_, domain_.get_cut_cells(), n_pts_dir)

    cells_points_set = _cut_quad_to_PyVista(domain_.grid, cells_quad)

    include_facet_unf_bry = True
    exclude_ext_bdry = True
    unf_bdry_quad = qugar.cpp.create_unfitted_bound_quadrature(
        domain_, domain_.get_cut_cells(), n_pts_dir, include_facet_unf_bry, exclude_ext_bdry
    )
    unf_bdry_points_set = _cut_quad_to_PyVista(domain_.grid, unf_bdry_quad)

    cut_facets_cells, cut_facets_local_facets = domain_.get_cut_facets()
    cut_facets_quad = qugar.cpp.create_facets_quadrature_exterior_integral(
        domain_,
        cut_facets_cells,
        cut_facets_local_facets,
        n_pts_dir,
    )
    unf_bdry_facets_cells, unf_bdry_facets_local_facets = domain_.get_unf_bdry_facets()
    unf_bdry_facets_quad = qugar.cpp.create_facets_quadrature_exterior_integral(
        domain_,
        unf_bdry_facets_cells,
        unf_bdry_facets_local_facets,
        n_pts_dir,
    )
    facets_points_set = _create_quad_facet_points_grid(
        domain_.grid, [cut_facets_quad, unf_bdry_facets_quad]
    )

    if not is_cpp:
        _append_DOLFINx_cell_ids(cells_points_set, domain)  # type: ignore
        _append_DOLFINx_cell_ids(unf_bdry_points_set, domain)  # type: ignore
        _append_DOLFINx_cell_ids(facets_points_set, domain)  # type: ignore
        _append_DOLFINx_facets_ids(facets_points_set, domain.tdim)  # type: ignore

    return pv.MultiBlock(
        {
            "Cut cells quadrature": cells_points_set,
            "Unfitted boundary quadrature": unf_bdry_points_set,
            "Cut/unfitted facets quadrature": facets_points_set,
        }
    )


def unfitted_domain_facets_to_PyVista(
    domain: Any, cut: bool = True, full: bool = True, empty: bool = False
) -> pv.UnstructuredGrid:
    """
    Converts the facets of the unfitted domain to a PyVista UnstructuredGrid.
    Extra information regarding the status of the facets and parent cells is added as cell data.

    Args:
        domain (Any): The unfitted domain object whose facets are transformed.
            It can be an instance of UnfittedDomain_2D, UnfittedDomain_3D, or UnfittedDomain
            (if FEniCSx is available).
        cut (bool, optional): Whether to include cut facets. Defaults to True.
        full (bool, optional): Whether to include full facets. Defaults to True.
        empty (bool, optional): Whether to include empty facets. Defaults to False.

    Returns:
        pv.UnstructuredGrid: A PyVista UnstructuredGrid containing the facets of the domain.

    Raises:
        AssertionError: If the domain is an instance of UnfittedDomain and
            FEniCSx installation is not found.
    """

    grid = unfitted_domain_to_PyVista(domain, cut=True, full=True, empty=True)
    n_cells = grid.n_cells

    dim = domain.tdim
    assert 2 <= dim <= 3, "Invalid dimension."

    grid = cast(pv.DataSet, grid.explode(factor=0.0))
    if dim == 2:
        grid = cast(pv.DataSet, grid.extract_all_edges())
    else:
        grid = cast(pv.DataSet, grid.extract_surface())

    n_facets_per_cell = 2 * dim
    grid.cell_data["VTK local facets ids"] = np.tile(np.arange(n_facets_per_cell), n_cells)
    grid.cell_data["Facet status (0: full, 1: cut, 2: empty)"] = np.full(
        grid.n_cells, 3, dtype=np.uint8
    )

    vtk_lex_mask = VTK_to_lexicg_faces(dim)

    active_facets = np.empty(0, dtype=np.int32)

    facets = {}
    if full:
        facets[0] = domain.get_full_facets()
    if cut:
        facets[1] = domain.get_cut_facets()
    if empty:
        facets[2] = domain.get_empty_facets()

    for status, lex_cells_facets in facets.items():
        vtk_facets = vtk_lex_mask[lex_cells_facets.local_facet_ids]
        vtk_cells = lex_cells_facets.cell_ids * n_facets_per_cell + vtk_facets
        grid.cell_data["Facet status (0: full, 1: cut, 2: empty)"][vtk_cells] = status
        active_facets = np.append(active_facets, vtk_cells)

    facets_grid = cast(pv.UnstructuredGrid, grid.extract_cells(active_facets))

    lex_vtk_mask = lexicg_to_VTK_faces(dim)

    vtk_facets = facets_grid.cell_data.pop("VTK local facets ids")
    lex_facets = lex_vtk_mask[vtk_facets]
    facets_grid.cell_data["Lexicographical local facets ids"] = lex_facets

    is_cpp = isinstance(domain, (UnfittedDomain_2D, UnfittedDomain_3D))

    if not is_cpp:
        _append_DOLFINx_facets_ids(facets_grid, dim)

    return facets_grid


def unfitted_domain_to_PyVista(
    domain: Any, cut: bool = True, full: bool = True, empty: bool = False
) -> pv.UnstructuredGrid:
    """
    Converts the grid of an unfitted domain to a PyVista `UnstructuredGrid`.
    Extra information regarding the status of the cells is added as cell data.

    Args:
        domain (Any): The domain to be converted. It can be an instance of
            UnfittedDomain_2D, UnfittedDomain_3D, or UnfittedDomain (if FEniCSx is available).
        cut (bool, optional): Whether to include cut cells. Defaults to True.
        full (bool, optional): Whether to include full cells. Defaults to True.
        empty (bool, optional): Whether to include empty cells. Defaults to True.

    Returns:
        pv.UnstructuredGrid: The resulting PyVista UnstructuredGrid with cell data
            indicating the status of each cell (full, cut, or empty).

    Raises:
        AssertionError: If the domain is an instance of UnfittedDomain and
            FEniCSx installation is not found.
    """
    is_cpp = isinstance(domain, (UnfittedDomain_2D, UnfittedDomain_3D))

    if is_cpp:
        grid = cart_grid_tp_to_PyVista(domain.grid)
        n_cells = domain.num_total_cells
    else:
        assert has_FEniCSx, "FEniCSx is required to convert a TensorProductMesh to PyVista."
        from qugar.mesh import UnfittedDomain

        assert isinstance(domain, UnfittedDomain), "Invalid type."

        n_cells = domain.mesh.num_global_cells

        grid = grid_tp_to_PyVista(domain.mesh)

    value = np.full(n_cells, empty, dtype=bool)
    status = np.full(n_cells, 2, dtype=np.uint8)
    cut_cells = domain.get_cut_cells()
    full_cells = domain.get_full_cells()
    value[cut_cells] = cut
    status[cut_cells] = 1
    value[full_cells] = full
    status[full_cells] = 0

    grid.cell_data["value"] = value
    grid.cell_data["Cell status (0: full, 1: cut, 2: empty)"] = status

    grid = cast(pv.UnstructuredGrid, grid.threshold(True, scalars="value"))
    grid.cell_data.pop("value")

    return grid


def cart_grid_tp_to_PyVista(grid: Any) -> pv.RectilinearGrid:
    """
    Converts a Cartesian grid to a PyVista `RectilinearGrid`.

    Args:
        grid (CartGridTP_2D | CartGridTP_3D | CartesianMesh): The input grid, which can
            be either a CartGridTP_2D, CartGridTP_3D, or CartesianMesh (if FEniCSx is available).

    Returns:
        pv.RectilinearGrid: The converted PyVista `RectilinearGrid`.

    Raises:
        AssertionError: If a TensorProductMesh is passed as input and the number of global cells
            does not match the number of local cells in the grid. This is not implemented yet.
    """

    is_cpp = isinstance(grid, (CartGridTP_2D, CartGridTP_3D))

    cell_breaks = grid.cell_breaks
    dim = len(cell_breaks)
    assert 1 <= dim <= 3, "Invalid dimension."
    n_cells = np.prod([len(breaks) - 1 for breaks in cell_breaks])

    pv_grid = pv.RectilinearGrid(*grid.cell_breaks)
    lex_cell_ids = np.arange(n_cells, dtype=np.int32)
    pv_grid.cell_data["Lexicographical cell ids"] = lex_cell_ids

    if not is_cpp:
        assert has_FEniCSx, "FEniCSx is required to convert a TensorProductMesh to PyVista."

        if grid.has_inactive_cells:
            pv_grid = pv_grid.extract_cells(grid.original_active_cells)

        _append_DOLFINx_cell_ids(pv_grid, grid)

    return pv_grid


def grid_tp_to_PyVista(grid: Any) -> pv.Grid:
    """
    Converts a tensor-product grid to a PyVista `Grid`.

    Args:
        grid (CartGridTP_2D | CartGridTP_3D | TensorProductMesh): The input grid, which can
            be either a CartGridTP_2D, CartGridTP_3D, or TensorProductMesh
            (if FEniCSx is available).

    Returns:
        pv.Grid: The converted PyVista `Grid`.
    """

    if isinstance(grid, (CartGridTP_2D, CartGridTP_3D)):
        return cart_grid_tp_to_PyVista(grid)
    else:
        assert has_FEniCSx, "FEniCSx is required to convert a TensorProductMesh to PyVista."
        from qugar.mesh import CartesianMesh

        if isinstance(grid, CartesianMesh):
            return cart_grid_tp_to_PyVista(grid)
        else:
            assert False, "Not implemented yet."


ReparamMesh: TypeAlias = ReparamMesh_1_2 | ReparamMesh_2_2 | ReparamMesh_2_3 | ReparamMesh_3_3


def _reparam_mesh_to_PyVista(
    reparam: ReparamMesh,
) -> pv.MultiBlock:
    """
    Converts a reparameterization mesh object to a PyVista `MultiBlock`.

    Args:
        reparam (ReparamMesh): The reparametrized mesh object to be converted.

    Returns:
        pv.MultiBlock: Reparameterization translated to PyVista data structures.
            It is a multiblock containing two unstructured grids: one for the
            reparameterization itselt and one for the reparmeterization's wirebasket.
            They can be accessed using ``.get("reparam")`` and ``.get("wirebasket")``,
            respectively.

    """

    assert isinstance(
        reparam, (ReparamMesh_1_2, ReparamMesh_2_2, ReparamMesh_2_3, ReparamMesh_3_3)
    ), "Invalid type."

    dim = reparam.dim
    degree = reparam.order - 1

    points = reparam.points
    # Enforcing points to be 3D.
    if points.shape[1] == 2:
        zeros = np.zeros([points.shape[0], 1], dtype=points.dtype)
        points = np.hstack((points, zeros))

    def _create_unstructured_grid(conn: npt.NDArray[np.int64], dim: int) -> pv.UnstructuredGrid:
        vtk_mask = VTK_to_lexicg_nodes(dim, degree)

        n_cells = conn.shape[0]
        new_conn = np.zeros((n_cells, conn.shape[1] + 1), dtype=np.int64)
        new_conn[:, 0] = np.full(n_cells, conn.shape[1], dtype=np.int64)
        new_conn[:, 1:] = conn.astype(np.int64)[:, vtk_mask]

        cell_type = _get_PyVista_cell_type(dim, degree)
        cell_types = np.full(n_cells, cell_type, dtype=np.uint8)

        return pv.UnstructuredGrid(new_conn, cell_types, points)

    int_grid = _create_unstructured_grid(reparam.cells_conn, dim)
    wb_grid = _create_unstructured_grid(reparam.wirebasket_conn, dim=1)
    return pv.MultiBlock({"reparam": int_grid, "wirebasket": wb_grid})


def reparam_mesh_to_PyVista(reparam: Any) -> pv.MultiBlock:
    """
    Converts a reparameterization mesh object to a PyVista `MultiBlock`.

    Args:
        reparam (ReparamMesh | UnfDomainReparamMesh): The reparametrized mesh object
            to be converted.

    Returns:
        pv.MultiBlock: Reparameterization translated to PyVista data structures.
            It is a multiblock containing two unstructured grids: one for the
            reparameterization itselt and one for the reparmeterization's wirebasket.
            They can be accessed using ``.get("reparam")`` and ``.get("wirebasket")``,
            respectively.

    """

    if isinstance(reparam, (ReparamMesh_1_2 | ReparamMesh_2_2 | ReparamMesh_2_3 | ReparamMesh_3_3)):
        return _reparam_mesh_to_PyVista(reparam)
    else:
        assert has_FEniCSx, "FEniCSx is required to convert a TensorProductMesh to PyVista."
        import dolfinx.plot as plot_dlf

        from qugar.reparam import UnfDomainReparamMesh

        assert isinstance(reparam, UnfDomainReparamMesh), "Invalid type."

        mesh = reparam.create_mesh()
        pv_mesh = pv.UnstructuredGrid(*plot_dlf.vtk_mesh(mesh))

        mesh_wb = reparam.create_mesh(wirebasket=True)
        pv_mesh_wb = pv.UnstructuredGrid(*plot_dlf.vtk_mesh(mesh_wb))

        return pv.MultiBlock({"reparam": pv_mesh, "wirebasket": pv_mesh_wb})
