# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from typing import Optional

import qugar.utils

if not qugar.has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import numpy as np
import numpy.typing as npt

import qugar.cpp
from qugar.mesh import (
    TensorProductMesh,
    lexicg_to_DOLFINx_faces,
)
from qugar.quad.quad_data import (
    CustomQuad,
    CustomQuadFacet,
    CustomQuadUnfBoundary,
)
from qugar.unfitted_domain import UnfittedDomain


class QuadGenerator:
    """Class for generating quadratures for unfitted domains."""

    def __init__(
        self,
        unf_domain: UnfittedDomain,
    ) -> None:
        """Constructor.

        Args:
            unf_domain (UnfittedDomain): Unfitted domain
                whose quadratures are generated.
        """

        self._unf_domain = unf_domain

    @property
    def tp_mesh(self) -> TensorProductMesh:
        """Gets the domain's tensor-product mesh.

        Returns:
            TensorProductMesh: Domain's tensor-product mesh.
        """
        return self._unf_domain.tp_mesh

    @property
    def unfitted_domain(self) -> UnfittedDomain:
        """Gets the unfitted domain.

        Returns:
            UnfittedDomain: Stored unfitted domain.
        """
        return self._unf_domain

    @staticmethod
    def get_num_points(degree: int) -> int:
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

        n_pts_dir = QuadGenerator.get_num_points(degree)

        orig_cells = self.tp_mesh.get_original_cell_ids(dlf_cells)

        quad = qugar.cpp.create_quadrature(
            self._unf_domain.cpp_object, orig_cells, n_pts_dir, full_cells=False
        )

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
        n_pts_dir = QuadGenerator.get_num_points(degree)

        if not np.all(np.isin(dlf_cells, self._unf_domain.get_cut_cells())):
            raise ValueError(
                "Unfitted boundary quadratures can only be generated for cells "
                "containing unfitted boundaries"
            )

        orig_cells = self.tp_mesh.get_original_cell_ids(dlf_cells)

        quad = qugar.cpp.create_unfitted_bound_quadrature(
            self._unf_domain._cpp_object, orig_cells, n_pts_dir
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

        n_pts_dir = QuadGenerator.get_num_points(degree)

        orig_cells = self.tp_mesh.get_original_cell_ids(dlf_cells)

        lex_to_dlf_faces = lexicg_to_DOLFINx_faces(self.tp_mesh.tdim)
        lex_local_facets = lex_to_dlf_faces[dlf_local_facets]

        if integral_type == "interior_facet":
            quad_func = qugar.cpp.create_interior_facets_quadrature
        else:
            quad_func = qugar.cpp.create_exterior_facets_quadrature

        quad = quad_func(
            self._unf_domain._cpp_object,
            orig_cells,
            lex_local_facets,
            n_pts_dir,
            full_facets=False,
        )

        return CustomQuadFacet(
            dlf_cells,
            dlf_local_facets,
            quad.n_pts_per_entity,
            quad.points,
            quad.weights,
        )


def create_quadrature_generator(
    unf_domain: UnfittedDomain,
) -> QuadGenerator:
    """Creates a quadrature generator for geometries.

    Args:
        unf_domain (UnfittedDomain): Unfitted domain
            whose quadratures are generated.

    Returns:
        QuadGenerator: Quadrature generator for unfitted geometries.
    """
    return QuadGenerator(unf_domain)
