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

import numpy as np
import numpy.typing as npt

from qugar.quad.custom_quad import (
    CustomQuad,
    CustomQuadFacet,
    CustomQuadUnfBoundary,
)


class UnfittedDomainABC(ABC):
    """Abstract base class for storing unfitted domains.

    This class provides (abstract) methods for accessing the cut, full,
    and empty cells and facets of the mesh. It also provides methods for
    creating quadrature rules for the cut cells and facets, as well as
    for unfitted boundaries.

    Classes that inherit from this class should implement the
    `get_cut_cells`, `get_full_cells`, `get_empty_cells`, `get_cut_facets`,
    `get_full_facets`, and `get_empty_facets` methods
    to access the cut, full, empty cells and facets, and the cells
    containing unfitted boundaries, respectively. The `create_quad_custom_cells`,
    `create_quad_unf_boundaries`, and `create_quad_custom_facets` methods
    should be implemented to create custom quadrature rules for the cut
    cells and unfitted boundaries.

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

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """
        pass

    @abstractmethod
    def get_full_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the full cells.

        Note:
            We also consider as full the full cells that contain
            unfitted boundaries on their facets.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Returns:
            npt.NDArray[np.int32]: Array of full cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """
        pass

    @abstractmethod
    def get_empty_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the empty cells.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Returns:
            npt.NDArray[np.int32]: Array of empty cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """
        pass

    @abstractmethod
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

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            exterior (bool): If `True`, the exterior facets are considered.
                Otherwise, the interior ones. Defaults to True.

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

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            exterior (bool): If `True`, the exterior facets are considered.
                Otherwise, the interior ones. Defaults to True.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Full facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local) DOLFINx ordering.
        """
        pass

    @abstractmethod
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

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            exterior (bool): If `True`, the exterior facets are considered.
                Otherwise, the interior ones. Defaults to True.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Empty facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local) DOLFINx ordering.
        """
        pass

    @abstractmethod
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

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            dlf_cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated.

        Returns:
            CustomQuadInterface: Generated custom quadrature.
        """
        pass

    @abstractmethod
    def create_quad_unf_boundaries(
        self,
        degree: int,
        dlf_cells: npt.NDArray[np.int32],
    ) -> CustomQuadUnfBoundary:
        """Returns the custom quadrature for unfitted boundaries for the
        given `cells`.

        Note:
            Some unfitted boundary parts may lay over facets.
            The quadrature corresponding to those facets will be generated
            with the method `create_quad_custom_facets`.

        Note:
            For cells not containing unfitted boundaries, no quadrature
            is generated and will have 0 points associated to them.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

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
        pass

    @abstractmethod
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

        Note:
            This is an abstract method and should be implemented in
            derived classes.

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
        pass
