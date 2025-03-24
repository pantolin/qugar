# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from typing import Callable, Optional, cast

from qugar import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import dolfinx.mesh
import numpy as np
import numpy.typing as npt

from qugar.cpp import UnfittedDomain_2D, UnfittedDomain_3D
from qugar.mesh import CartesianMesh, DOLFINx_to_lexicg_faces, lexicg_to_DOLFINx_faces
from qugar.mesh.utils import create_cells_to_facets_map, map_facets_to_cells_and_local_facets


class UnfittedDomain:
    """Class for storing an unfitted domain
    and access its cut, full, and empty cells and facets.
    """

    def __init__(
        self,
        cart_mesh: CartesianMesh,
        cpp_object: UnfittedDomain_2D | UnfittedDomain_3D,
    ) -> None:
        """Constructor.

        Args:
            cart_mesh (CartesianMesh): Cartesian tensor-product mesh
                for which the cell decomposition is generated.
            cpp_object (UnfittedDomain_2D | UnfittedDomain_3D):
                Already generated unfitted domain binary object.
        """

        assert cpp_object.dim == cart_mesh.tdim, "Non-matching dimensions."
        assert cart_mesh.tdim == cart_mesh.gdim, "Mesh must have co-dimension 0."
        assert cart_mesh.grid_cpp_object == cpp_object.grid, "Non-matching grids."

        self._cart_mesh = cart_mesh
        self._cpp_object = cpp_object

    @property
    def dim(self) -> int:
        """Gets the dimension of the mesh.

        Returns:
            int: Mesh's topological dimension (2 or 3).
        """
        return self.cart_mesh.tdim

    @property
    def cart_mesh(self) -> CartesianMesh:
        """Gets the stored Cartesian mesh.

        Returns:
            CartesianMesh: Stored Cartesian mesh.
        """
        return self._cart_mesh

    @property
    def cpp_object(self) -> UnfittedDomain_2D | UnfittedDomain_3D:
        """Gets the internal C++ unfitted domain.

        Returns:
            UnfittedDomain_2D | UnfittedDomain_3D:
            Internal C++ unfitted domain.
        """
        return self._cpp_object

    def _transform_lexicg_cell_ids_to_DOLFINx_local(
        self, lex_cell_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int32]:
        """Transforms the given `lex_cell_ids` to  DOLFINx local numbering.

        Args:
            lex_cell_ids (npt.NDArray[np.int32]): Lexicographical cell
                ids to transform.

        Returns:
            npt.NDArray[np.int32]: Transformed indices following the
            DOLFINx local numbering.
        """

        return cast(
            npt.NDArray[np.int32],
            self.cart_mesh.get_DOLFINx_local_cell_ids(lex_cell_ids),
        )

    def _transform_dlf_cell_ids_to_lexicg(
        self, dlf_cell_ids: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.int32]:
        """Transforms the given `dlf_cell_ids` to lexicographical numbering.

        Args:
            dlf_cell_ids (npt.NDArray[np.int32]): Local DOLFINx cell
                ids to transform.

        Returns:
            npt.NDArray[np.int32]: Transformed indices following the lexicographical
            numbering.
        """

        return cast(
            npt.NDArray[np.int32],
            self.cart_mesh.get_lexicg_cell_ids(dlf_cell_ids),
        )

    def _transform_lexicg_facet_ids_to_DOLFINx_local(
        self,
        lex_cells: npt.NDArray[np.int64],
        lex_facets: npt.NDArray[np.int32],
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Transforms the given lexicographical facets indices to
        DOLFINx local numbering (otherwise)

        Args:
            lex_cells (npt.NDArray[np.int64]): Indices of the facets
                following the lexicographical ordering. All the cell
                indices must be associated to the current process.

            lex_facets (npt.NDArray[np.int32]): Local indices of the
                facets referred to `lex_cells` (both arrays should have
                the same length). The face ids follow the
                lexicographical ordering.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Transformed facets returned as one array of cells and another
            one of local facets referred to those cells. The indices of the cells
            and local facets follow the DOLFINx local numbering.
        """

        tdim = self._cart_mesh.tdim
        dlf_to_lex_facets = DOLFINx_to_lexicg_faces(tdim)

        dlf_cells = self._transform_lexicg_cell_ids_to_DOLFINx_local(lex_cells)
        dlf_facets = dlf_to_lex_facets[lex_facets]
        return dlf_cells, dlf_facets

    def _transform_dlf_facet_ids(
        self,
        dlf_cells: npt.NDArray[np.int32],
        dlf_facets: npt.NDArray[np.int32],
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Transforms the given DOLFINx facets indices to
        either lexicographical numbering (both for cells and local facet ids).

        Args:
            dlf_cells (npt.NDArray[np.int32]): Indices of the facets
                following the DOLFINx local ordering.

            dlf_facets (npt.NDArray[np.int32]): Local indices of the
                facets referred to `dlf_cells` (both arrays should have
                the same length). The face ids follow the
                DOLFINx ordering.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Transformed facets returned as one array of cells and another
            one of local facets referred to those cells. The indices of the cells
            and local facets follow the DOLFINx (local) numbering.
        """

        tdim = self._cart_mesh.tdim
        lex_to_dlf_facets = lexicg_to_DOLFINx_faces(tdim)

        lex_cells = self._transform_dlf_cell_ids_to_lexicg(dlf_cells)
        lex_facets = lex_to_dlf_facets[dlf_facets]
        return lex_cells, lex_facets

    def _transform_cell_facet_pairs_to_facet_ids(
        self,
        dlf_cells: npt.NDArray[np.int32],
        dlf_facets: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """Transforms the given pairs of cells and local facets to
        DOLFINx local facets indices.

        Args:
            dlf_cells (npt.NDArray[np.int32]): Indices of the cells
                following the DOLFINx local ordering.

            dlf_facets (npt.NDArray[np.int32]): Local indices of the
                facets referred to `dlf_cells` (both arrays should have
                the same length). The face ids follow the DOLFINx
                ordering.

        Returns:
            npt.NDArray[np.int32]: DOLFINx (local) facet indices associated to
            the current proces.
        """

        cells_to_facets = create_cells_to_facets_map(self._cart_mesh.dolfinx_mesh)
        return cells_to_facets[dlf_cells, dlf_facets]

    def get_cut_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the cut cells.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process following the DOLFINx local numbering.
        """

        return self._transform_lexicg_cell_ids_to_DOLFINx_local(self._cpp_object.cut_cells)

    def get_full_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the full cells.

        Returns:
            npt.NDArray[np.int32]: Array of full cells associated to the
            current process following the DOLFINx local numbering.
        """
        return self._transform_lexicg_cell_ids_to_DOLFINx_local(self._cpp_object.full_cells)

    def get_empty_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the empty cells.

        Returns:
            npt.NDArray[np.int32]: Array of empty cells associated to the
            current process following the DOLFINx local numbering.
        """
        return self._transform_lexicg_cell_ids_to_DOLFINx_local(self._cpp_object.empty_cells)

    def _get_exterior_facets(self) -> npt.NDArray[np.int32]:
        """Gets the exterior facets of the mesh.

        Returns:
            npt.NDArray[np.int32]: Sorted list of owned facet indices that are
            exterior facets of the mesh.
        """
        msh = self._cart_mesh.dolfinx_mesh
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
        return dolfinx.mesh.exterior_facet_indices(msh.topology)

    def _get_facets(
        self,
        facets_getter: Callable[
            [Optional[npt.NDArray[np.int32]], Optional[npt.NDArray[np.int32]]],
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]],
        ],
        only_exterior: bool = False,
        only_interior: bool = False,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Accesses a list of facets either through the function `facets_getter`.

        The list of facets can be filtered to only exterior or interior facets.

        Args:
            facets_getter (Callable[[Optional[npt.NDArray[np.int32]], Optional[npt.NDArray[np.int32]]],
              tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]): Function to
                access the facets. It returns the facets as pairs of cells and local facets
                following lexicographical ordering. Optionally, it can receive as arguments
                the (lexicographical) cell ids and local facet ids of the subsets of
                facets to be considered.
            only_exterior (bool): If `True`, only the exterior facets are returned.
                Defaults to False.
            only_interior (bool): If `True`, only the interior facets are returned.
                Defaults to False.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Extracted facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local) DOLFINx ordering.
        """

        assert not (only_exterior and only_interior), "Cannot be both exterior and interior."

        mesh = self.cart_mesh.dolfinx_mesh

        if only_exterior or only_interior:
            ext_facets_ids = self._get_exterior_facets()
            if only_exterior:
                facets_ids = ext_facets_ids
            else:
                topology = mesh.topology
                imap = topology.index_map(topology.dim - 1)
                all_facets = np.arange(imap.local_range[0], imap.local_range[1], dtype=np.int32)
                facets_ids = np.setdiff1d(all_facets, ext_facets_ids, assume_unique=True)

            dlf_cell_ids, dlf_local_facet_ids = map_facets_to_cells_and_local_facets(
                mesh, facets_ids
            )
            lex_cell_ids, lex_local_facet_ids = self._transform_dlf_facet_ids(
                dlf_cell_ids, dlf_local_facet_ids
            )

            lex_cell_ids, lex_local_facet_ids = facets_getter(lex_cell_ids, lex_local_facet_ids)

        else:
            lex_cell_ids, lex_local_facet_ids = facets_getter(None, None)

        return self._transform_lexicg_facet_ids_to_DOLFINx_local(lex_cell_ids, lex_local_facet_ids)

    def get_cut_facets(
        self,
        only_exterior: bool = False,
        only_interior: bool = False,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the cut facets as pairs of cells and local facets.

        The list of facets can be filtered to only exterior or interior facets.

        Note:
            Cut facet may also contain unfitted boundaries parts.

        Args:
            only_exterior (bool): If `True`, only the exterior facets are considered.
                Defaults to False.
            only_interior (bool): If `True`, only the interior facets are considered.
                Defaults to False.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Cut facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local) DOLFINx ordering.
        """
        return self._get_facets(self._cpp_object.get_cut_facets, only_exterior, only_interior)

    def get_full_facets(
        self,
        only_exterior: bool = False,
        only_interior: bool = False,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the full facets as pairs of cells and local facets.

        The list of facets can be filtered to only exterior or interior facets.

        Note:
            Facets contained full unfitted boundaries are not considered as full.

        Args:
            only_exterior (bool): If `True`, only the exterior facets are considered.
                Defaults to False.
            only_interior (bool): If `True`, only the interior facets are considered.
                Defaults to False.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Cut facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local) DOLFINx ordering.
        """

        return self._get_facets(self._cpp_object.get_full_facets, only_exterior, only_interior)

    def get_empty_facets(
        self,
        only_exterior: bool = False,
        only_interior: bool = False,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the empty facets as pairs of cells and local facets.

        Args:
            cell_ids (Optional[npt.NDArray[np.int32]]): Indices of the
                candidate cells to get the empty facets. If follows the DOLFINx local numbering.
                If not provided, all the empty facets are returned. Defaults to None.
            local_facet_ids (Optional[npt.NDArray[np.int32]]): Local
                indices of the candidate facets referred to `cell_ids` (both arrays
                should have the same length). The face ids follow the
                DOLFINx ordering. If not provided, all the empty facets are returned.
                Defaults to None.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Empty facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following DOLFINx ordering.
        """

        return self._get_facets(self._cpp_object.get_empty_facets, only_exterior, only_interior)

    def get_unf_bdry_facets(
        self,
        only_exterior: bool = False,
        only_interior: bool = False,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the facets that contain unfitted boundaries as pairs of cells and local facets.

        The list of facets can be filtered to only exterior or interior facets.

        Note:
            Facets that unfitted boundaries may also be cut.

        Args:
            only_exterior (bool): If `True`, only the exterior facets are considered.
                Defaults to False.
            only_interior (bool): If `True`, only the interior facets are considered.
                Defaults to False.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Facets that contain unfitted boundaries. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local) DOLFINx ordering.
        """

        return self._get_facets(self._cpp_object.get_unf_bdry_facets, only_exterior, only_interior)

    def create_cell_subdomain_data(
        self,
        cut_tag: Optional[int] = None,
        full_tag: Optional[int] = None,
        empty_tag: Optional[int] = None,
    ) -> list[tuple[int, npt.NDArray[np.int32]]]:
        """Creates subdomain data that may contain the cut, full, and/or
        empty cells.

        If the tag for cut, full, or empty tags are not provided, those
        cells will not be included.

        Args:
            cut_tag (Optional[int]): Tag to assign to cut cells. Defaults to None.
            full_tag (Optional[int]): Tag to assign to full cells. Defaults to None.
            empty_tag (Optional[int]): Tag to assign to empty cells. Defaults to None.

        Returns:
            list[tuple[int, npt.NDArray[np.int32]]]: Generated cells subdomain data.
            It is a list where is entry is a tuple with a tag (identifier) and an array
            of cell ids.
        """

        subdomain_data = {}

        def add_cells(cells, tag):
            if tag in subdomain_data:
                subdomain_data[tag] = np.concatenate([subdomain_data[tag], cells])
            else:
                subdomain_data[tag] = cells

        if cut_tag is not None:
            add_cells(self.get_cut_cells(), cut_tag)

        if full_tag is not None:
            add_cells(self.get_full_cells(), full_tag)

        if empty_tag is not None:
            add_cells(self.get_empty_cells(), empty_tag)

        return list((tag, np.sort(cells)) for tag, cells in subdomain_data.items())

    def create_exterior_facet_subdomain_data(
        self,
        cut_tag: Optional[int] = None,
        full_tag: Optional[int] = None,
        unf_bdry_tag: Optional[int] = None,
        empty_tag: Optional[int] = None,
    ) -> list[tuple[int, npt.NDArray[np.int32]]]:
        """Creates subdomain data that may contain the cut, full, unfitted and/or
        empty exterior facets

        If the tag for cut, full, unfitted, or empty tags are not provided, those
        facets will not be included.

        Args:
            cut_tag (Optional[int]): Tag to assign to cut exterior facets. Defaults to None.
            full_tag (Optional[int]): Tag to assign to full exterior facets. Defaults to None.
            unf_bdry_tag (Optional[int]): Tag to assign to facets that contain unfitted boundaries.
                They are not necessarily exterior facets in the sense of exterior facets
                of the underlying (non-cut) mesh. Defaults to None.
            empty_tag (Optional[int]): Tag to assign to empty exterior facets. Defaults to None.

        Returns:
            list[tuple[int, npt.NDArray[np.int32]]]: Generated tags subdomain data.
            It is a list where is entry is a tuple with a tag (identifier) and an array
            of exterior facets. The array of facets is made of consecutive pairs of cells and
            local facet ids.

        """

        subdomain_data = {}

        def add_facets(cells, local_facets, tag):
            facets = np.empty((len(cells), 2), dtype=np.int32)
            facets[:, 0] = cells
            facets[:, 1] = local_facets
            facets = facets.ravel()

            if tag in subdomain_data:
                subdomain_data[tag] = np.concatenate([subdomain_data[tag], facets])
            else:
                subdomain_data[tag] = facets

        if cut_tag is not None:
            cells, local_facets = self.get_cut_facets(only_exterior=True, only_interior=False)
            add_facets(cells, local_facets, cut_tag)

        if full_tag is not None:
            cells, local_facets = self.get_full_facets(only_exterior=True, only_interior=False)
            add_facets(cells, local_facets, full_tag)

        if unf_bdry_tag is not None:
            cells, local_facets = self.get_unf_bdry_facets(only_exterior=False, only_interior=False)
            add_facets(cells, local_facets, unf_bdry_tag)

        if empty_tag is not None:
            cells, local_facets = self.get_empty_facets(only_exterior=True, only_interior=False)
            add_facets(cells, local_facets, cut_tag)

        return list((tag, entities) for tag, entities in subdomain_data.items())
