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

import numpy as np
import numpy.typing as npt

import qugar.cpp
from qugar.cpp import UnfittedDomain_2D, UnfittedDomain_3D
from qugar.mesh.mesh import Mesh
from qugar.mesh.unfitted_domain_abc import UnfittedDomainABC
from qugar.mesh.utils import lexicg_to_DOLFINx_faces
from qugar.quad.custom_quad import (
    CustomQuad,
    CustomQuadFacet,
    CustomQuadUnfBoundary,
)


class UnfittedDomain(UnfittedDomainABC):
    """Class for storing an unfitted domains
    and access its cut, full, and empty cells and facets,
    and create custom quadratures associated to those cells and facets.

    Note:
        This class is not intended to be used directly, but rather
        as a class to be inherited by other classes, such as
        `UnfittedCartMesh`.
    """

    def __init__(
        self,
        mesh: Mesh,
        cpp_unf_domain_object: UnfittedDomain_2D | UnfittedDomain_3D,
    ) -> None:
        """Constructor.

        Args:
            mesh (Mesh): Mesh object associated to the unfitted domain.
            cpp_unf_domain_object (UnfittedDomain_2D | UnfittedDomain_3D):
                Already generated unfitted domain binary object.
        """

        self._mesh = mesh
        self._cpp_unf_domain_object = cpp_unf_domain_object

        self._full_cells = None
        self._empty_cells = None
        self._cut_cells = None

    @property
    def cpp_unf_domain_object(self) -> UnfittedDomain_2D | UnfittedDomain_3D:
        """Gets the internal C++ unfitted domain.

        Returns:
            UnfittedDomain_2D | UnfittedDomain_3D:
            Internal C++ unfitted domain.
        """
        return self._cpp_unf_domain_object

    def _get_cells(
        self,
        cell_getter: Callable[
            [Optional[npt.NDArray[np.int64]]],
            npt.NDArray[np.int64],
        ],
    ) -> npt.NDArray[np.int64]:
        """Accesses a list of cells either through the function `cells_getter`.

        Args:
            cells_getter (Callable[[Optional[npt.NDArray[np.int64]]], npt.NDArray[np.int32]]):
                Function to access the cells. It returns the cells following the original numbering.

        Returns:
            npt.NDArray[np.int32]: Extracted cells. They are sorted.
        """
        if self._mesh.num_local_cells != self._cpp_unf_domain_object.num_total_cells:
            orig_cell_ids = self._mesh.get_all_original_cell_ids()
        else:
            orig_cell_ids = None
        return self._mesh.get_DOLFINx_local_cell_ids(cell_getter(orig_cell_ids))

    def _get_facets(
        self,
        facets_getter: Callable[
            [Optional[npt.NDArray[np.int64]], Optional[npt.NDArray[np.int32]]],
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]],
        ],
        only_exterior: bool = False,
        only_interior: bool = False,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Accesses a list of facets through the function `facets_getter`.

        The list of facets can be filtered to only exterior or interior facets.

        Args:
            facets_getter (Callable[[Optional[npt.NDArray[np.int64]], Optional[npt.NDArray[np.int32]]],
              tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]): Function to
                access the facets. It returns the facets as pairs of cells (referred to the
                local ordering) and local facets following lexicographical ordering.
                Optionally, it can receive as arguments (origina) cell ids and local
                (lexicographical) facet ids of the subsets of facets to be considered.
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
            ext_facets_ids = self._mesh.get_exterior_facets()
            if only_exterior:
                facets_ids = ext_facets_ids
            else:
                topology = self._mesh.topology
                imap = topology.index_map(topology.dim - 1)
                all_facets = np.arange(imap.local_range[0], imap.local_range[1], dtype=np.int32)
                facets_ids = np.setdiff1d(all_facets, ext_facets_ids, assume_unique=True)

            dlf_cell_ids, dlf_local_facet_ids = (
                self._mesh.transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(facets_ids)
            )
            orig_cell_ids, orig_local_facet_ids = (
                self._mesh.transform_DOLFINx_local_facet_ids_to_original(
                    dlf_cell_ids, dlf_local_facet_ids
                )
            )

            orig_cell_ids, orig_local_facet_ids = facets_getter(orig_cell_ids, orig_local_facet_ids)

        else:
            orig_cell_ids, orig_local_facet_ids = facets_getter(None, None)

        return self._mesh.transform_original_facet_ids_to_DOLFINx_local(
            orig_cell_ids, orig_local_facet_ids
        )

    def get_cut_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the cut cells.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process following the DOLFINx local numbering.
            The cell id are sorted.
        """

        if self._cut_cells is None:
            self._cut_cells = self._get_cells(self._cpp_unf_domain_object.get_cut_cells)

        return self._cut_cells

    def get_full_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the full cells.

        Returns:
            npt.NDArray[np.int32]: Array of full cells associated to the
            current process following the DOLFINx local numbering.
            The cell id are sorted.
        """

        if self._full_cells is None:
            self._full_cells = self._get_cells(self._cpp_unf_domain_object.get_full_cells)

        return self._full_cells

    def get_empty_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the empty cells.

        Returns:
            npt.NDArray[np.int32]: Array of empty cells associated to the
            current process following the DOLFINx local numbering.
            The cell id are sorted.
        """

        if self._empty_cells is None:
            self._empty_cells = self._get_cells(self._cpp_unf_domain_object.get_empty_cells)

        return self._empty_cells

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
        return self._get_facets(
            self._cpp_unf_domain_object.get_cut_facets, only_exterior, only_interior
        )

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

        return self._get_facets(
            self._cpp_unf_domain_object.get_full_facets, only_exterior, only_interior
        )

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

        return self._get_facets(
            self._cpp_unf_domain_object.get_empty_facets, only_exterior, only_interior
        )

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

        return self._get_facets(
            self._cpp_unf_domain_object.get_unf_bdry_facets, only_exterior, only_interior
        )

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

    @staticmethod
    def get_num_Gauss_points(degree: int) -> int:
        """Returns the number of points per direction to be used
        for a given degree of exactness.

        It computes the number of points assuming the Gauss-Legendre quadrature.
        The degree of exactness is 2 times the number of points plues one.

        It does not consider the map of of the quadrature tile.
        A slightly higher number of points/degree may be needed.
        See the Appendix of the following paper for more details:

        Antolin, P., Wei, X. and Buffa, A., 2022. Robust numerical integration on curved
          polyhedra based on folded decompositions.
          Computer Methods in Applied Mechanics and Engineering, 395, p.114948.


        Args:
            degree (int): Expected degree of exactness of the quadrature.

        Returns:
            int: Number of quadrature points per direction for every quadrature tile.
        """

        assert degree >= 0, "Invalid degree."

        return max(int((degree - 1) / 2), 1)

    def create_quad_custom_cells(
        self,
        degree: int,
        dlf_cells: npt.NDArray[np.int32],
        tag: Optional[int] = None,
    ) -> CustomQuad:
        """Returns the custom quadratures for the given `cells`.

        Among the given `cells`, it only creates quadratures for a
        subset of those (the custom cells), while no points or
        weights are generated for the others. Thus, the cells without
        custom quadratures will be listed in the returned
        `CustomQuadInterface`, but will have 0 points associated to
        them.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            dlf_cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated.

            tag (Optional[int]): Mesh tag of the subdomain associated to
                the given cells. Defaults to None.

        Returns:
            CustomQuadInterface: Generated custom quadrature.
        """

        n_pts_dir = UnfittedDomain.get_num_Gauss_points(degree)

        orig_cells = self._mesh.get_original_cell_ids(dlf_cells)

        quad = qugar.cpp.create_quadrature(self.cpp_unf_domain_object, orig_cells, n_pts_dir)

        return CustomQuad(dlf_cells, quad.n_pts_per_entity, quad.points, quad.weights)

    def create_quad_unf_boundaries(
        self,
        degree: int,
        dlf_cells: npt.NDArray[np.int32],
        tag: Optional[int] = None,
    ) -> CustomQuadUnfBoundary:
        """Returns the custom quadrature for unfitted boundaries for the
        given `cells`.

        Warning:
            All the given cells associated should contain unfitted
            boundaries. I.e., they must be cut cells. If not, the
            custom coefficients generator will raise an
            exception.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            dlf_cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated. It must only contain cells with
                unfitted boundaries.

            tag (int): Mesh tag of the subdomain associated to the given
                cells.

        Returns:
            CustomQuadUnfBoundaryInterface: Generated custom quadrature
            for unfitted boundaries.
        """
        n_pts_dir = UnfittedDomain.get_num_Gauss_points(degree)

        orig_cells = self._mesh.get_original_cell_ids(dlf_cells)

        quad = qugar.cpp.create_unfitted_bound_quadrature(
            self._cpp_unf_domain_object, orig_cells, n_pts_dir
        )

        return CustomQuadUnfBoundary(
            dlf_cells, quad.n_pts_per_entity, quad.points, quad.weights, quad.normals
        )

    def create_quad_custom_facets(
        self,
        degree: int,
        dlf_cells: npt.NDArray[np.int32],
        dlf_local_facets: npt.NDArray[np.int32],
        integral_type: str,
        tag: Optional[int] = None,
    ) -> CustomQuadFacet:
        """Returns the custom quadratures for the given facets.

        Among the facets associated to `tag`, it only creates
        quadratures for a subset of those (the custom facets), while no
        points or weights are generated for the others. Thus, the facets
        without custom quadratures will be listed in the returned
        `CustomQuadFacetInterface`, but will have 0 points associated to

        Among the given facets, it only creates quadratures for a
        subset of those (the custom facets), while no points or
        weights are generated for the others. Thus, the facets without
        custom quadratures will be listed in the returned
        `CustomQuadFacetInterface`, but will have 0 points associated to
        them.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            dlf_cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated. It must only contain cells with
                unfitted boundaries. Beyond a cell id, for indentifying
                a facet, a local facet id (from the array
                `local_facets`) is also needed.

            dlf_local_facets (npt.NDArray[np.int32]): Array of local facet
                ids for which the custom quadratures are generated. Each
                facet is identified through a value in `cells` and a
                value in `local_facets`, having both arrays the same
                length. The numbering of these facets follows the
                FEniCSx convention. See
                https://github.com/FEniCS/basix/#supported-elements

            integral_type (str): Type of integral to be computed. It can
                be either 'interior_facet' or 'exterior_facet'.

            tag (int): Mesh tag of the subdomain associated to the given
                cells. Right now, it is not used. However, in the future
                it may be used to filter the different boundaries
                on the same cell.

        Returns:
            CustomQuadFacetInterface: Generated custom facet
            quadratures.
        """

        assert integral_type in ["interior_facet", "exterior_facet"], "Invalid integral type."

        n_pts_dir = UnfittedDomain.get_num_Gauss_points(degree)

        orig_cells = self._mesh.get_original_cell_ids(dlf_cells)

        tdim = self._cpp_unf_domain_object.dim
        lex_to_dlf_faces = lexicg_to_DOLFINx_faces(tdim)
        lex_local_facets = lex_to_dlf_faces[dlf_local_facets]

        if integral_type == "interior_facet":
            quad_func = qugar.cpp.create_interior_facets_quadrature
        else:
            quad_func = qugar.cpp.create_exterior_facets_quadrature

        quad = quad_func(
            self._cpp_unf_domain_object,
            orig_cells,
            lex_local_facets,
            n_pts_dir,
        )

        return CustomQuadFacet(
            dlf_cells,
            dlf_local_facets,
            quad.n_pts_per_entity,
            quad.points,
            quad.weights,
        )
