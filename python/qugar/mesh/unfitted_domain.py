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
        self._unf_bdry_cells = None

        self._exterior_cut_facets = None
        self._exterior_full_facets = None
        self._exterior_empty_facets = None

        self._interior_cut_facets = None
        self._interior_full_facets = None
        self._interior_empty_facets = None

    @property
    def mesh(self) -> Mesh:
        """Gets the mesh object associated to the unfitted domain.

        Returns:
            Mesh: Mesh object associated to the unfitted domain.
        """
        return self._mesh

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
    ) -> npt.NDArray[np.int32]:
        """Accesses a list of cells through the function `cells_getter`.

        Args:
            cells_getter (Callable[[Optional[npt.NDArray[np.int64]]], npt.NDArray[np.int64]]):
                Function to access the cells. It returns the cells following the original numbering.

        Returns:
            npt.NDArray[np.int32]: Extracted cells (following DOLFINx local
            numbering). They are sorted.
        """
        if self._mesh.num_local_cells == self._cpp_unf_domain_object.num_total_cells:
            orig_cell_ids = None  # All cells.
        else:
            orig_cell_ids = self._mesh.get_all_original_cell_ids()
        cell_ids = self._mesh.get_DOLFINx_local_cell_ids(cell_getter(orig_cell_ids))
        return np.sort(cell_ids)

    def _get_facets(
        self,
        facets_getter: Callable[
            [Optional[npt.NDArray[np.int64]], Optional[npt.NDArray[np.int32]]],
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]],
        ],
        exterior: bool,
    ) -> npt.NDArray[np.int32]:
        """Accesses a list of facets through the function `facets_getter`.

        The list of facets will be filtered to only exterior or interior facets.

        Args:
            facets_getter (Callable[npt.NDArray[np.int64], npt.NDArray[np.int32]],
              tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]): Function to
                access the facets. It returns the facets as pairs of cells (referred to the
                local ordering) and local facets following lexicographical ordering.
                Optionally, it can receive as arguments (origina) cell ids and local
                (lexicographical) facet ids of the subsets of facets to be considered.
            exterior (bool): If `True`, the exterior facets are considered.
                Otherwise, the interior ones. Defaults to True.

        Returns:
            npt.NDArray[np.int32]: Extracted facets. The facets are
            returned as one sorted array with the facets indices following
            the (local) DOLFINx ordering.
        """

        facets_ids = (
            self._mesh.get_exterior_facets() if exterior else self._mesh.get_interior_facets()
        )

        # From facet ids to cells and local facets
        dlf_cell_ids, dlf_local_facet_ids = (
            self._mesh.transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(facets_ids)
        )

        # From DOLFIN local ordering (of cells and local facets)
        # to original ordering (of cells and lexicographical local facets)
        orig_cell_ids, orig_local_facet_ids = (
            self._mesh.transform_DOLFINx_local_facet_ids_to_original(
                dlf_cell_ids, dlf_local_facet_ids
            )
        )

        # Filtering of the original facets using the facets_getter
        orig_cell_ids, orig_local_facet_ids = facets_getter(orig_cell_ids, orig_local_facet_ids)

        # From the original ordering (of cells and lexicographical local facets)
        # to DOLFINx local ordering (of cells and local facets)
        dlf_cell_ids, dlf_local_facet_ids = (
            self._mesh.transform_original_facet_ids_to_DOLFINx_local(
                orig_cell_ids, orig_local_facet_ids
            )
        )

        # From DOLFINx local ordering of cells and local facets
        # to DOLFINx local ordering of facets.
        facets_ids = self._mesh.transform_DOLFINx_cells_and_local_facets_to_local_facet_ids(
            dlf_cell_ids, dlf_local_facet_ids
        )

        return np.sort(facets_ids)

    def has_full_cells_with_unf_bdry(self) -> bool:
        """Checks if the unfitted domain has full cells with unfitted
        boundaries.

        Returns:
            bool: `True` if the unfitted domain has full cells with
            unfitted boundaries, `False` otherwise.
        """
        return self._cpp_unf_domain_object.has_full_cells_with_unf_bdry

    def get_cut_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the cut cells.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """

        if self._cut_cells is None:

            def caller(cells):
                return self._cpp_unf_domain_object.get_cut_cells(cells)

            self._cut_cells = self._get_cells(caller)

        return self._cut_cells

    def get_full_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the full cells.

        Note:
            We also consider as full the full cells that contain
            unfitted boundaries on their facets.

        Returns:
            npt.NDArray[np.int32]: Array of full cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """

        if self._full_cells is None:

            def caller(cells):
                return self._cpp_unf_domain_object.get_full_cells(cells)

            self._full_cells = self._get_cells(caller)

        return self._full_cells

    def get_empty_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the empty cells.

        Returns:
            npt.NDArray[np.int32]: Array of empty cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """

        if self._empty_cells is None:
            caller = self._cpp_unf_domain_object.get_empty_cells
            self._empty_cells = self._get_cells(caller)

        return self._empty_cells

    def get_cut_facets(
        self,
        exterior: bool = True,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the cut facets as DOLFINx cells and local facet ids

        The list of facets will be filtered to only exterior or interior
        facets according to the argument `exterior`.

        Note:
            The selection of cut facets is performed differently depending
            on whether exterior or interior facets are considered.
            For interior facets, simply facets that containg cut parts
            are considered. While for exterior facets, on top of those
            cut facets, facets that contain unfitted boundaries are also
            considered as cut facets, but only if the unfitted boundary
            does not correspond to a full facet.

        Args:
            exterior (bool): If `True`, the exterior facets are considered.
                Otherwise, the interior ones. Defaults to True.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Cut facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local) DOLFINx ordering.
        """

        if exterior:
            if self._exterior_cut_facets is None:
                # We include cut facets.
                exterior_cut_facets = self._get_facets(
                    self._cpp_unf_domain_object.get_cut_facets, exterior
                )

                # But also facets that contain unfitted boundaries, but
                # only when they are not full.
                unf_facets = self._get_facets(
                    self.cpp_unf_domain_object.get_unf_bdry_facets, exterior
                )
                if unf_facets.size > 0:
                    # They may contain full unfitted boundaries that need to be removed.
                    unf_full_facets = self._get_facets(
                        self.cpp_unf_domain_object.get_full_unf_bdry_facets, exterior
                    )
                    unf_facets = np.setdiff1d(unf_facets, unf_full_facets, assume_unique=exterior)

                    if unf_facets.size > 0:
                        exterior_cut_facets = np.concatenate((exterior_cut_facets, unf_facets))
                        exterior_cut_facets = np.unique(np.sort(exterior_cut_facets))

                self._exterior_cut_facets = (
                    self._mesh.transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(
                        exterior_cut_facets
                    )
                )

            return self._exterior_cut_facets

        else:  # Interior
            if self._interior_cut_facets is None:
                # Here we only include cut facets.
                interior_cut_facets = self._get_facets(
                    self._cpp_unf_domain_object.get_cut_facets, exterior
                )
                self._interior_cut_facets = (
                    self._mesh.transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(
                        interior_cut_facets
                    )
                )

            return self._interior_cut_facets

    def get_full_facets(
        self,
        exterior: bool = True,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the full facets as DOLFINx cells and local facet ids

        The list of facets will be filtered to only exterior or interior
        facets according to the argument `exterior`.

        Note:
            The selection of full facets is performed differently depending
            on whether exterior or interior facets are considered.
            For interior facets, simply facets that are fully inside
            the domain (not touching the boundary) are considered.
            While for exterior facets, we consider those that are fully
            contained in the domain's boundary, including the ones
            corresponding to unfitted boundaries, if the facet is full.

        Args:
            exterior (bool): If `True`, the exterior facets are considered.
                Otherwise, the interior ones. Defaults to True.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Full facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local) DOLFINx ordering.
        """

        if exterior:
            if self._exterior_full_facets is None:
                # Full facets
                exterior_full_facets = self._get_facets(
                    self._cpp_unf_domain_object.get_full_facets, exterior
                )

                # We also include facets that contain unfitted boudnaries
                # if the facet is full.
                unf_full_facets = self._get_facets(
                    self.cpp_unf_domain_object.get_full_unf_bdry_facets, exterior
                )
                if unf_full_facets.size > 0:
                    exterior_full_facets = np.unique(
                        np.sort(np.concatenate((exterior_full_facets, unf_full_facets)))
                    )

                self._exterior_full_facets = (
                    self._mesh.transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(
                        exterior_full_facets
                    )
                )

            return self._exterior_full_facets

        else:  # Interior
            if self._interior_full_facets is None:
                # Here we only the full facets, without considering the
                # unfitted boundaries fully contained in the facet.
                interior_full_facets = self._get_facets(
                    self._cpp_unf_domain_object.get_full_facets, exterior
                )

                self._interior_full_facets = (
                    self._mesh.transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(
                        interior_full_facets
                    )
                )

            return self._interior_full_facets

    def get_empty_facets(
        self,
        exterior: bool = True,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the empty facets as DOLFINx cells and local facet ids

        The list of facets will be filtered to only exterior or interior
        facets according to the argument `exterior`.

        Note:
            The selection of empty facets is performed differently depending
            on whether exterior or interior facets are considered.
            For exterior facets we only consider empty facets that do
            not touch the domain. While for interior facets, we consider
            empty facets that are completely outside of the domain, but
            also those that only touch its boundary.

        Args:
            exterior (bool): If `True`, the exterior facets are considered.
                Otherwise, the interior ones. Defaults to True.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Empty facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local) DOLFINx ordering.
        """

        if exterior:
            if self._exterior_empty_facets is None:
                # Here we only include empty facets.
                exterior_empty_facets = self._get_facets(
                    self._cpp_unf_domain_object.get_empty_facets, exterior
                )

                self._exterior_empty_facets = (
                    self._mesh.transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(
                        exterior_empty_facets
                    )
                )

            return self._exterior_empty_facets

        else:  # Interior
            if self._interior_empty_facets is None:
                interior_empty_facets = self._get_facets(
                    self._cpp_unf_domain_object.get_empty_facets, exterior
                )

                # Beyond the empty facets, we also need to consider
                # the facets that are not cut but contain unfitted
                # boundaries (full or not).

                unf_facets = self._get_facets(
                    self.cpp_unf_domain_object.get_unf_bdry_facets, exterior
                )
                if unf_facets.size > 0:
                    # They may contain cut facets need to be removed.
                    unf_facets = np.setdiff1d(
                        unf_facets, self.get_cut_facets(exterior), assume_unique=exterior
                    )

                    if unf_facets.size > 0:
                        interior_empty_facets = np.unique(
                            np.sort(np.concatenate((interior_empty_facets, unf_facets)))
                        )

                self._interior_empty_facets = (
                    self._mesh.transform_DOLFINx_local_facet_ids_to_cells_and_local_facets(
                        interior_empty_facets
                    )
                )

            return self._interior_empty_facets

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
    ) -> CustomQuad:
        """Returns the custom quadratures for the given `dlf_cells`.

        For empty cells, no quadrature is generated and will have 0 points
        associated to them. For full cells, the quadrature will be the
        standard one associated to the cell type. While for
        cut cells a custom quadrature for the cell's interior will be
        generated.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            dlf_cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated.

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
    ) -> CustomQuadUnfBoundary:
        """Returns the custom quadrature for unfitted boundaries for the
        given `cells`.

        For cells not containing unfitted boundaries,
        no quadrature is generated and will have 0 points associated to
        them. While for cells containing unfitted boundaries a custom
        quadrature for the unfitted boundary will be generated.

        Note:
            Some unfitted boundary parts may lay over facets.
            This function generates quadrature for those parts if they correspond
            to exterior facets. For the case of interior facets,
            the corresponding quadratures will be included in the
            quadrature generated with the method `create_quad_custom_facets`.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            dlf_cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated.

        Returns:
            CustomQuadUnfBoundaryInterface: Generated custom quadrature
            for unfitted boundaries.
        """
        n_pts_dir = UnfittedDomain.get_num_Gauss_points(degree)

        orig_cells = self._mesh.get_original_cell_ids(dlf_cells)

        include_facet_unf_bry = True
        exclude_ext_bdry = True
        quad = qugar.cpp.create_unfitted_bound_quadrature(
            self._cpp_unf_domain_object,
            orig_cells,
            n_pts_dir,
            include_facet_unf_bry,
            exclude_ext_bdry,
        )

        return CustomQuadUnfBoundary(
            dlf_cells, quad.n_pts_per_entity, quad.points, quad.weights, quad.normals
        )

    def create_quad_custom_facets(
        self,
        degree: int,
        dlf_cells: npt.NDArray[np.int32],
        dlf_local_faces: npt.NDArray[np.int32],
        exterior: bool,
    ) -> CustomQuadFacet:
        """Returns the custom quadratures for the given facets.

        For empty facets, no quadrature is generated and will have 0 points
        associated to them. For full facets, the quadrature will be the
        standard one associated to the facet type. While for
        cut facets a custom quadrature for the facet's interior will be
        generated.

        Note:
            Some unfitted boundary parts may lay over facets.
            This function generates quadrature for those parts if they correspond
            to interior facets. For the case of exterior facets,
            the corresponding quadratures will be included in the
            quadrature generated with the method `create_quad_unf_boundaries`.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            dlf_cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated. Beyond a cell id, for indentifying
                a facet, a local facet id (from the array
                `local_faces`) is also needed.

            dlf_local_faces (npt.NDArray[np.int32]): Array of local face
                ids for which the custom quadratures are generated. Each
                facet is identified through a value in `cells` and a
                value in `local_faces`, having both arrays the same
                length. The numbering of these facets follows the
                FEniCSx convention. See
                https://github.com/FEniCS/basix/#supported-elements

            exterior (bool): If `True` the quadratures will be generated
                considering the given facets as exterior facets.
                Otherwise, as interior.
                See the documentation of `get_cut_facets`,
                `get_full_facets`, and `get_empty_facets` methods
                for more information about the implications of considering
                interior or exterior facets.

        Returns:
            CustomQuadFacetInterface: Generated custom facet
            quadratures.
        """

        n_pts_dir = UnfittedDomain.get_num_Gauss_points(degree)

        orig_cells, lex_local_faces = self._mesh.transform_DOLFINx_local_facet_ids_to_original(
            dlf_cells, dlf_local_faces
        )

        quad_func = (
            qugar.cpp.create_exterior_facets_quadrature
            if exterior
            else qugar.cpp.create_interior_facets_quadrature
        )

        quad = quad_func(
            self._cpp_unf_domain_object,
            orig_cells,
            lex_local_faces,
            n_pts_dir,
        )

        return CustomQuadFacet(
            dlf_cells,
            dlf_local_faces,
            quad.n_pts_per_entity,
            quad.points,
            quad.weights,
        )

    def create_cell_tags(
        self,
        cut_tag: Optional[int] = None,
        full_tag: Optional[int] = None,
        empty_tag: Optional[int] = None,
    ) -> list[tuple[int, npt.NDArray[np.int32]]]:
        """Creates a cell tags container to identify the cut, full,
        and/or empty cells.

        These tags can be used to prescribe the `subdomain_data` to which
        integration measure apply. E.g., `ufl.dx(subdomain_data=cell_tags)`.

        TODO: transform this into a mesh tag object.

        If the tag for `cut`, `full`, `empty`, or `unf_bdry_tag` tags are not provided
        (they are set to `None`), those cells will not be included in the generated cell tags.

        Args:
            cut_tag (Optional[int]): Tag to assign to cut cells. Defaults to None.
            full_tag (Optional[int]): Tag to assign to full cells. Defaults to None.
            empty_tag (Optional[int]): Tag to assign to empty cells. Defaults to None.

        Returns:
            list[tuple[int, npt.NDArray[np.int32]]]: Generated cells tags.
            It is a list where each entry is a tuple with a tag (identifier) and an array
            of cell ids.
        """

        cell_tags = {}

        def add_cells(getter, tag):
            cells = getter()
            if tag in cell_tags:
                if cells.size > 0:
                    new_cells = np.concatenate([cell_tags[tag], cells])
                    cell_tags[tag] = np.unique(np.sort(new_cells))
            else:
                cell_tags[tag] = cells

        for getter, tag in {
            self.get_cut_cells: cut_tag,
            self.get_full_cells: full_tag,
            self.get_empty_cells: empty_tag,
        }.items():
            if tag is not None:
                add_cells(getter, tag)

        return list((tag, cells) for tag, cells in cell_tags.items())

    def create_facet_tags(
        self,
        cut_tag: Optional[int] = None,
        full_tag: Optional[int] = None,
        empty_tag: Optional[int] = None,
        exterior: bool = True,
    ) -> list[tuple[int, npt.NDArray[np.int32]]]:
        """Creates a facet tags container to identify the cut, full,
        and/or empty cells.

        These tags can be used to prescribe the `subdomain_data` to which
        integration measure apply. E.g., `ufl.ds(subdomain_data=cell_tags)`.

        Note:
            A DOFLINX mesh tag object could be generated in a straightforward manner
            from the information returned by this function. However, be aware
            that in the information returned by this function, the information
            for the interior facets is duplicated (provided information for the
            positive and negative sides of the facet).

        If the tag for cut, full, or empty tags are not provided, those
        facets will not be included in the list.

        Args:
            cut_tag (Optional[int]): Tag to assign to cut facets. Defaults to None.
            full_tag (Optional[int]): Tag to assign to full facets. Defaults to None.
            empty_tag (Optional[int]): Tag to assign to empty facets. Defaults to None.
            exterior (bool): If `True`, the exterior facets are considered.
                Otherwise, the interior ones. Check the documentation of
                `get_cut_facets`, `get_full_facets`, and `get_empty_facets`
                methods for more information about the implications of
                considering interior or exterior facets.
                Defaults to True.

        Returns:
            list[tuple[int, npt.NDArray[np.int32]]]: Generated facet tags.
            It is a list where is entry is a tuple with a tag (identifier) and an array
            of facets. The array of facets is made of consecutive pairs of cells and
            local face ids.
        """

        facet_tags = {}

        def add_facets(cells, local_facets, tag):
            facets = np.empty(len(cells) * 2, dtype=np.int32)
            facets[::2] = cells
            facets[1::2] = local_facets

            if tag in facet_tags:
                if facets.size > 0:
                    # No need for unique, as the sets of cut, empty, and
                    # full facets should be disjoint. Also, the facets
                    # are not required to be returned in any particular order.
                    facet_tags[tag] = np.concatenate([facet_tags[tag], facets])
            else:
                facet_tags[tag] = facets

        if cut_tag is not None:
            cells, local_facets = self.get_cut_facets(exterior)
            add_facets(cells, local_facets, cut_tag)

        if full_tag is not None:
            cells, local_facets = self.get_full_facets(exterior)
            add_facets(cells, local_facets, full_tag)

        if empty_tag is not None:
            cells, local_facets = self.get_empty_facets(exterior)
            add_facets(cells, local_facets, cut_tag)

        return list((tag, entities) for tag, entities in facet_tags.items())
