# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

from qugar.quad.custom_quad import (
    CustomQuad,
    CustomQuadFacet,
    CustomQuadUnfBoundary,
)


class UnfittedDomainABC(ABC):
    """Abstract base class for storing unfitted domains.

    This class provides (abstract) methods for accessing the cut, full, and empty
    cells and facets of the mesh. It also provides methods for creating
    quadrature rules for the cut cells and unfitted boundaries.

    Classes that inherit from this class should implement the
    `get_cut_cells`, `get_full_cells`, `get_empty_cells`, `get_cut_facets`,
    `get_full_facets`, `get_empty_facets`, and `get_unf_bdry_facets` methods
    to access the cut, full, empty, and unfitted boundary facets,
    respectively. The `create_quad_custom_cells`, `create_quad_unf_boundaries`,
    and `create_quad_custom_facets` methods should be implemented to create
    custom quadrature rules for the cut cells and unfitted boundaries.

    This class also provides methods for creating subdomain data for the
    different types of cells and faces.

    Note:
        This class is purely abstract and should not be instantiated
        directly. It is intended to be used as a base class for other
        classes that implement the abstract methods.
    """

    @abstractmethod
    def get_cut_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the cut cells.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process following the DOLFINx local numbering.
            The cell id are sorted.
        """
        pass

    @abstractmethod
    def get_full_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the full cells.

        Returns:
            npt.NDArray[np.int32]: Array of full cells associated to the
            current process following the DOLFINx local numbering.
            The cell id are sorted.
        """
        pass

    @abstractmethod
    def get_empty_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the empty cells.

        Returns:
            npt.NDArray[np.int32]: Array of empty cells associated to the
            current process following the DOLFINx local numbering.
            The cell id are sorted.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

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

    @abstractmethod
    def create_quad_custom_cells(
        self,
        degree: int,
        dlf_cells: npt.NDArray[np.int32],
        tag: Optional[int] = None,
    ) -> CustomQuad:
        """Returns the custom quadratures for the given `dlf_cells`.

        Among the given `dlf_cells`, it only creates quadratures for a
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
