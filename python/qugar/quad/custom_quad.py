# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Classes for describing custom quadratures and classes interfaces for
creating them."""

from enum import Enum
from typing import NamedTuple, Optional, Protocol

import numpy as np
import numpy.typing as npt

from qugar.mesh.mesh_facets import MeshFacets


class CellState(Enum):
    cut = 1
    full = 2
    empty = 3


class CustomQuad(NamedTuple):
    """Data class for storing custom quadratures for a collection of
    cells.

    The quadratures are defined by custom points and weights referred to
    the [0,1]^tdim unit domain of every cell, where tdim is the cell's
    topological dimension.

    All the points (and weights) are stored contiguously for all the
    cells involved. Thus, the way of differentiating which points
    belong to which cells is through `n_pts_per_entity`.

    Parameters:
        cells (npt.NDArray[np.int32]): Array of cell ids to which the
            quadrature points and weights are associated to.
            This is a unique array. The ids are local to the MPI rank.
        n_pts_per_entity (npt.NDArray[np.int32]): Array indicating the
            number of quadrature points and weights for every cell
            in `cells`.
        points (npt.NDArray[np.float64]): Array of custom quadrature
            points for all the cells. This is a 2D array that has as
            many rows as points and as many columns as coordinates. It
            stores all the points contiguously according the ordering of
            `cells`.
        weights (npt.NDArray[np.float64]): Array of custom quadrature
            weights for all the cells. This is a 1D array with one entry
            per point. It stores all the weights contiguously according
            the ordering of `cells`.
    """

    cells: npt.NDArray[np.int32]
    n_pts_per_entity: npt.NDArray[np.int32]
    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]


class CustomQuadFacet(NamedTuple):
    """Data class for storing custom quadratures for a collection of
    facets.

    It contains the same attributes as ``CustomQuad`` plus an additional
    array for identifying the local ids of the facets.

    The quadratures are defined by custom points and weights referred to
    the [0,1]^(tdim-1) unit domain of every facet, where tdim is the
    cell's topological dimension, and therefore (tdim-1) is the facet's
    topological dimension.

    Thus, the points have as many coordinates as the facets of a cell.
    E.g., for triangles or quadrilaterals, points will have 1
    coordinate, while for for tetrahedra or hexahedra they will have 2.

    Parameters:
        facest (MeshFacets): Mesh facets object that stores the
            information of the facets. The facets are local
            to the MPI rank.
        n_pts_per_entity (npt.NDArray[np.int32]): Array indicating the
            number of quadrature points and weights for every facet.
        points (npt.NDArray[np.float64]): Array of custom quadrature
            points for all the cells. This is a 2D array that has as
            many rows as points and as many columns as coordinates in
            the facet. It stores all the points contiguously according
            the ordering of `cells` and `facets`.
        weights (npt.NDArray[np.float64]): Array of custom quadrature
            weights for all the facets. This is a 1D array with one
            entry per point. It stores all the weights contiguously
            according the ordering of `cells` and `facets`.

    """

    facets: MeshFacets
    n_pts_per_entity: npt.NDArray[np.int32]
    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]


class CustomQuadUnfBoundary(NamedTuple):
    """Data class for storing custom quadratures for unfitted custom
    boundaries. I.e., boundaries that are not on the exterior boundary
    of a domain, but inside, as those derived from trimming operations.

    It contains the same attributes as ``CustomQuad`` plus an additional
    array for storing the unit outer normals on the boundary.

    The normal vectors (and points) have as many coordinates as the
    cell. E.g., for triangles or quadrilaterals, points and normals will
    have 2 coordinates, while for tetrahedra or hexahedra they will have
    3.

    Parameters:
        cells (npt.NDArray[np.int32]): Array of cell ids to which the
            quadrature points and weights are associated to.
            This is a unique array. The ids are local to the MPI rank.
        n_pts_per_entity (npt.NDArray[np.int32]): Array indicating the
            number of quadrature points and weights for every cell
            in `cells`.
        points (npt.NDArray[np.float64]): Array of custom quadrature
            points for all the cells. This is a 2D array that has as
            many rows as points and as many columns as coordinates. It
            stores all the points contiguously according the ordering of
            `cells`.
        weights (npt.NDArray[np.float64]): Array of custom quadrature
            weights for all the cells. This is a 1D array with one entry
            per point. It stores all the weights contiguously according
            the ordering of `cells`.
        normals (npt.NDArray[np.float64]): Array of custom quadrature
            unit outer normals for all the cells. The normals are
            referred to the [0,1]^tdim unit domain of every cell. This
            is a 2D array that has as many rows as quadrature points
            and as many columns as coordinates. Therefore, the shape of
            the array must be the same as `points`.
    """

    cells: npt.NDArray[np.int32]
    n_pts_per_entity: npt.NDArray[np.int32]
    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]


class QuadGenerator(Protocol):
    """Protocol class for generating quadratures for unfitted domains."""

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
        ...

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
        ...

    def create_quad_custom_facets(
        self,
        degree: int,
        dlf_facets: MeshFacets,
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

            dlf_facets (MeshFacets): Mesh facets object that stores the
                information of the facets. The facets are local
                to the MPI rank.

            integral_type (str): Type of integral to be computed. It can
                be either 'interior_facet' or 'exterior_facet'.

            tag (int): Mesh tag of the subdomain associated to the given
                cells.

        Returns:
            CustomQuadFacetInterface: Generated custom facet
            quadratures.
        """
        ...
