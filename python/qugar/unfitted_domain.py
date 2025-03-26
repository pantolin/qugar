# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from typing import Callable, Optional

from qugar import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import dolfinx.mesh
import numpy as np
import numpy.typing as npt

from qugar.cpp import UnfittedDomain_2D, UnfittedDomain_3D
from qugar.mesh import TensorProductMesh


class UnfittedDomain:
    """Class for storing an unfitted domain
    and access its cut, full, and empty cells and facets.
    """

    def __init__(
        self,
        tp_mesh: TensorProductMesh,
        cpp_object: UnfittedDomain_2D | UnfittedDomain_3D,
    ) -> None:
        """Constructor.

        Args:
            tp_mesh (TensorProductMesh): Tensor-product mesh
                for which the cell decomposition is generated.
            cpp_object (UnfittedDomain_2D | UnfittedDomain_3D):
                Already generated unfitted domain binary object.
        """

        assert cpp_object.dim == tp_mesh.tdim, "Non-matching dimensions."
        assert tp_mesh.tdim == tp_mesh.gdim, "Mesh must have co-dimension 0."
        # assert cart_mesh.cart_grid_tp_cpp_object == cpp_object.grid, "Non-matching grids."

        self._tp_mesh = tp_mesh
        self._cpp_object = cpp_object

        self._full_cells = None
        self._empty_cells = None
        self._cut_cells = None

    @property
    def dim(self) -> int:
        """Gets the dimension of the mesh.

        Returns:
            int: Mesh's topological dimension (2 or 3).
        """
        return self.tp_mesh.tdim

    @property
    def tp_mesh(self) -> TensorProductMesh:
        """Gets the stored tensor-product mesh.

        Returns:
            TensorProductMesh: Stored tensor-product mesh.
        """
        return self._tp_mesh

    @property
    def cpp_object(self) -> UnfittedDomain_2D | UnfittedDomain_3D:
        """Gets the internal C++ unfitted domain.

        Returns:
            UnfittedDomain_2D | UnfittedDomain_3D:
            Internal C++ unfitted domain.
        """
        return self._cpp_object

    def get_cut_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the cut cells.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process following the DOLFINx local numbering.
        """

        if self._cut_cells is None:
            if self._tp_mesh.num_local_cells != self._tp_mesh.num_cells_tp:
                dlf_cell_ids = np.arange(self._tp_mesh.num_local_cells, dtype=np.int32)
                lex_cell_ids = self._tp_mesh.get_lexicg_cell_ids(dlf_cell_ids)
                lex_cut_cell_ids = self._cpp_object.get_cut_cells(lex_cell_ids)
            else:
                lex_cut_cell_ids = self._cpp_object.get_cut_cells()
            self._cut_cells = self.tp_mesh.get_DOLFINx_local_cell_ids(lex_cut_cell_ids)

        return self._cut_cells

    def get_full_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the full cells.

        Returns:
            npt.NDArray[np.int32]: Array of full cells associated to the
            current process following the DOLFINx local numbering.
        """

        if self._full_cells is None:
            if self._tp_mesh.num_local_cells != self._tp_mesh.num_cells_tp:
                dlf_cell_ids = np.arange(self._tp_mesh.num_local_cells, dtype=np.int32)
                lex_cell_ids = self._tp_mesh.get_lexicg_cell_ids(dlf_cell_ids)
                lex_full_cell_ids = self._cpp_object.get_full_cells(lex_cell_ids)
            else:
                lex_full_cell_ids = self._cpp_object.get_full_cells()
            self._full_cells = self.tp_mesh.get_DOLFINx_local_cell_ids(lex_full_cell_ids)

        return self._full_cells

    def get_empty_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the empty cells.

        Returns:
            npt.NDArray[np.int32]: Array of empty cells associated to the
            current process following the DOLFINx local numbering.
        """

        if self._empty_cells is None:
            if self._tp_mesh.num_local_cells != self._tp_mesh.num_cells_tp:
                dlf_cell_ids = np.arange(self._tp_mesh.num_local_cells, dtype=np.int32)
                lex_cell_ids = self._tp_mesh.get_lexicg_cell_ids(dlf_cell_ids)
                lex_empty_cell_ids = self._cpp_object.get_empty_cells(lex_cell_ids)
            else:
                lex_empty_cell_ids = self._cpp_object.get_empty_cells()
            self._empty_cells = self.tp_mesh.get_DOLFINx_local_cell_ids(lex_empty_cell_ids)

        return self._empty_cells

    def _get_exterior_facets(self) -> npt.NDArray[np.int32]:
        """Gets the exterior facets of the mesh.

        Returns:
            npt.NDArray[np.int32]: Sorted list of owned facet indices that are
            exterior facets of the mesh.
        """
        msh = self._tp_mesh
        msh.topology.create_connectivity(msh.tdim - 1, msh.tdim)
        return dolfinx.mesh.exterior_facet_indices(msh.topology)

    def _get_facets(
        self,
        facets_getter: Callable[
            [Optional[npt.NDArray[np.int64]], Optional[npt.NDArray[np.int32]]],
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]],
        ],
        only_exterior: bool = False,
        only_interior: bool = False,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Accesses a list of facets either through the function `facets_getter`.

        The list of facets can be filtered to only exterior or interior facets.

        Args:
            facets_getter (Callable[[Optional[npt.NDArray[np.int64]], Optional[npt.NDArray[np.int32]]],
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
        """  # noqa: E501

        assert not (only_exterior and only_interior), "Cannot be both exterior and interior."

        if only_exterior or only_interior:
            ext_facets_ids = self._get_exterior_facets()
            if only_exterior:
                facets_ids = ext_facets_ids
            else:
                topology = self.tp_mesh.topology
                imap = topology.index_map(topology.dim - 1)
                all_facets = np.arange(imap.local_range[0], imap.local_range[1], dtype=np.int32)
                facets_ids = np.setdiff1d(all_facets, ext_facets_ids, assume_unique=True)

            dlf_cell_ids, dlf_local_facet_ids = (
                self.tp_mesh.transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(facets_ids)
            )
            lex_cell_ids, lex_local_facet_ids = (
                self.tp_mesh.transform_DOLFINx_local_facet_ids_to_lexicg(
                    dlf_cell_ids, dlf_local_facet_ids
                )
            )

            lex_cell_ids, lex_local_facet_ids = facets_getter(lex_cell_ids, lex_local_facet_ids)

        else:
            lex_cell_ids, lex_local_facet_ids = facets_getter(None, None)

        return self.tp_mesh.transform_lexicg_facet_ids_to_DOLFINx_local(
            lex_cell_ids, lex_local_facet_ids
        )

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
