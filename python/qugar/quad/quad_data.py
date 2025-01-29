# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Classes for storing generated quadratures."""

from enum import Enum
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


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
        cells (npt.NDArray[np.int32]): Array of cell ids to which the
            quadrature points and weights are associated to.
            This is a unique array. The ids are local to the MPI rank.
        facets (npt.NDArray[np.int32]): Array of local facet ids for
            every cell index in the `cells` array.
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

    cells: npt.NDArray[np.int32]
    facets: npt.NDArray[np.int32]
    n_pts_per_entity: npt.NDArray[np.int32]
    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]


class CustomQuadIntBoundary(NamedTuple):
    """Data class for storing custom quadratures for interior custom
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
