# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Class for generating mock custom quadrature for testing purposes."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from typing import Optional

import dolfinx.mesh
import numpy as np
import numpy.typing as npt
from ffcx.ir.representationutils import create_quadrature_points_and_weights

from qugar.quad import CustomQuad, CustomQuadFacet, CustomQuadUnfBoundary


class MockQuadGenerator:
    """Class for generating mock custom quadrature for testing
    purposes.

    It generates custom quadratures for a few cells (number specified
    in the constructor) and standard quadratures for the rest. The
    custom quadratures are just the same as the standard quadrature but
    repeated several times in the cells, and scaling the weights
    inversely proportional to the number of times the quadrature is
    repeated in a cell.

    Using these custom quadratures the same results should be obtained
    as with the standard quadratures, what makes this mock quadrature
    interesting for testing purposes.

    Parameters:
        _mesh (dolfinx.mesh.Mesh): Mesh for which the quadratures are
            generated.
        _nnz (int, optional): Ratio of entities with custom quadratures
            respect to the total number of entities. It is a value in
            the range [0.0, 1.0].
        _max_quad_sets (int, optional): Maximum number of repetitions
            of the standard quadrature in each custom cell. For each
            custom entity a random number is generated between 1 and
            `max_quad_sets`.
        _custom_cells_ids (npt.NDArray[np.int32]): Ids of the cells
            for which custom quadratures are created.
        _n_quad_sets (npt.NDArray[np.int32]): Number of quadrature sets
            for the custom cells in `_custom_cells_ids`.
        _custom_facet_cells_ids (npt.NDArray[np.int32]): Ids of the
            cells whose one or more their facets have custom
            quadratures.
        _custom_facet_local_facets_ids (npt.NDArray[np.int32]): Local
            ids of the facets (referred to the reference cell) for which
            custom quadratures are generated.
        _n_quad_sets_facets (npt.NDArray[np.int32]): Number of
            quadrature sets for the custom facets defined by
            `_custom_facet_cells_ids` and
            `_custom_facet_local_facets_ids`.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, nnz: float = 0, max_quad_sets: int = 3) -> None:
        """Initializes.

        Args:
            mesh (dolfinx.mesh.Mesh): Mesh for which the quadratures are
                generated.
            nnz (int, optional): Ratio of entities with custom
                quadratures respect to the total number of entities.
                It must be a value in the range [0.0, 1.0].
            max_quad_sets (int, optional): Maximum number of repetitions
                of the standard quadrature in each custom cell. For each
                custom entity a random number is generated between 1 and
                `max_quad_sets`. Defaults to 3.
        """

        assert 0.0 <= nnz <= 1.0
        assert max_quad_sets > 0

        self._mesh = mesh
        self._nnz = nnz
        self._max_quad_sets = max_quad_sets

        self._compute_custom_cells()
        self._compute_custom_facets()

    def _get_cells_ids(self) -> npt.NDArray[np.int32]:
        """Gets the ids of all the cells in the mesh.

        Returns:
            npt.NDArray[np.int32]: Ids of all the cells in the mesh.
        """
        n_cells = self._mesh.geometry.dofmap.shape[0]
        return np.arange(n_cells, dtype=np.int32)

    def _get_interior_exterior_facets_ids(
        self,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the ids of all the interior and exterior facets in the
        mesh.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: Ids
            of all the interior (first) and exterior (second) facets
            in the mesh.
        """

        tdim = self._mesh.topology.dim

        topology = self._mesh.topology
        topology.create_connectivity(tdim - 1, tdim)
        f_to_c = topology.connectivity(tdim - 1, tdim)

        exterior_facets = dolfinx.mesh.exterior_facet_indices(topology)
        n_facets = f_to_c.num_nodes

        interior_facets = np.setdiff1d(
            np.arange(n_facets, dtype=np.int32), exterior_facets, assume_unique=True
        )

        return interior_facets, exterior_facets

    def _extract_cells_and_local_facets(
        self, facets: npt.NDArray[np.int32], is_interior: bool
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Given a collection of facets, extracts the cells that belong
        to and the local id of the facet referred to the cell.

        Args:
            facets (npt.NDArray[np.int32]): Facets whose cells and local
                indices are to be extracted.
            is_interior (bool): ``True`` if the facets are interior,
                ``False`` it they are exterior.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: Ids of
            the cells the facets belong to (first) and local ids of the
            facets referred to the cells (second).
        """

        topology = self._mesh.topology
        tdim = topology.dim
        topology.create_connectivity(tdim - 1, tdim)
        topology.create_connectivity(tdim, tdim - 1)

        c_to_f = topology.connectivity(tdim, tdim - 1)
        f_to_c = topology.connectivity(tdim - 1, tdim)

        n_cells_per_facet = 2 if is_interior else 1
        n_facets = len(facets) * n_cells_per_facet
        cells = np.empty(n_facets, dtype=np.int32)
        local_facets = np.empty(n_facets, dtype=np.int32)

        for i, facet in enumerate(facets):
            cells_in_facet = f_to_c.links(facet)
            assert len(cells_in_facet) == n_cells_per_facet

            for j in range(n_cells_per_facet):
                ind = n_cells_per_facet * i + j
                cell = cells_in_facet[j]
                cells[ind] = cell

                local_facets_in_cell = c_to_f.links(cell)
                local_pos = np.flatnonzero(local_facets_in_cell == facet)
                assert len(local_pos) == 1
                local_facets[ind] = local_pos[0]

        return cells, local_facets

    def _extract_custom_entities(
        self,
        entities_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """Given a list of `entities_ids`, extracts a random subset
        of the ones for which custom quadratures are generated. The
        number of custom cells is built following the ratio `self._nnz`.

        Despite the random generation, same result is produced among
        different Python sessions.

        The entities for which custom quadratures are generated is
        defined through the number of quadrature sets per custom cell
        (in a pseudo-random way) that is a value between 0 and
        `self._max_quad_sets`. If 0, no custom quadrature is generated.

        Args:
            entities_ids (npt.NDArray[np.int32]): Ids of the entities to
                be considered for custom quadratures.

        Returns:
            npt.NDArray[np.int32]: Nmber of custom quadrature sets for
            every entity.
        """

        n_entities = entities_ids.shape[0]
        n_quad_sets = np.arange(n_entities, dtype=np.int32) % self._max_quad_sets + 1

        n_custom_entities = int(np.ceil(n_entities * self._nnz))
        n_non_custom_entities = n_entities - n_custom_entities

        seed = n_entities
        rng = np.random.default_rng(seed)
        entities_ids_ = np.copy(np.arange(n_entities))
        rng.shuffle(entities_ids_)
        n_quad_sets[entities_ids_[:n_non_custom_entities]] = 0

        return n_quad_sets

    def _compute_custom_cells(self) -> None:
        """Sets the list of custom cells. The members
        `self._n_quad_sets` and `self._custom_cells_ids` are
        initialized.
        """

        cells_id = self._get_cells_ids()
        self._n_quad_sets = self._extract_custom_entities(cells_id)

    def _compute_custom_facets(self) -> None:
        """Sets the list of custom facets. The members
        `self._n_quad_sets_facets`, `self._custom_facet_cells_ids`, and
        `self._custom_facet_local_facets_ids` are initialized.
        """

        def extract_custom(
            facets: npt.NDArray[np.int32], is_interior: bool
        ) -> npt.NDArray[np.int32]:
            cells, local_facets = self._extract_cells_and_local_facets(facets, is_interior)
            facets = np.stack([cells, local_facets], axis=1)
            n_quad_sets = self._extract_custom_entities(facets)
            return np.hstack([facets, n_quad_sets.reshape(-1, 1)])

        int_facets, ext_facets = self._get_interior_exterior_facets_ids()
        int_facets = extract_custom(int_facets, True)
        ext_facets = extract_custom(ext_facets, False)

        all_facets = np.vstack([int_facets, ext_facets])
        all_facets = all_facets[np.lexsort((all_facets[:, 1], all_facets[:, 0]))]

        if all_facets.size > 0:
            self._facet_cells_ids = all_facets[:, 0]
            self._facet_local_facets_ids = all_facets[:, 1]
            self._n_quad_sets_facets = all_facets[:, 2]
        else:
            self._facet_cells_ids = np.empty(0, dtype=np.int32)
            self._facet_local_facets_ids = np.empty(0, dtype=np.int32)
            self._n_quad_sets_facets = np.empty(0, dtype=np.int32)

    def _create_ref_quadrature(
        self, degree: int, facet: bool
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Creates the reference quadrature for the given degree of
        exactness.

        Args:
            degree (int): Degree of exactness of the quadrature to be
                built.
            facet (bool): ``True`` if the quadrature to be generated
                corresponds to a facet, ``False`` otherwise.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
                Generated quadrature points (first) and weights
                (second).
        """
        # It doesn't matter to use exterior or interior facet here.
        itg_type = "exterior_facet" if facet else "cell"

        ufl_cell = self._mesh.ufl_domain().ufl_cell()

        points, weights, _ = create_quadrature_points_and_weights(
            itg_type, ufl_cell, degree, "default", [], False
        )
        return points, weights  # type: ignore

    def _create_quadrature(
        self, degree: int, n_sets_per_cell: npt.NDArray[np.int32], facet: bool
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generates custom quadratures for cells or facets.

        Args:
            degree (int): Degree of exactness of the quadrature to be
                generated.
            n_sets_per_cell (npt.NDArray[np.int32]): Number of
                quadrature sets (standard quadratures) per custom set.
            facet (bool): Flag indicating whether the quadrature is to
                be built for a facet (``True``) or a cell (``False``).

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.float64],
            npt.NDArray[np.float64]]: Generated quadrature. Namely,
            number of points per custom cell (first), quadrature
            weights (second) and points (third) for all the custom
            cells.
        """

        ref_points, ref_weights = self._create_ref_quadrature(degree, facet)

        n_pts_per_set = ref_weights.size
        n_pts_per_cell = n_sets_per_cell * n_pts_per_set
        n_tot_sets = np.sum(n_sets_per_cell)

        # TODO: to improve this
        scale = np.ones(n_tot_sets * n_pts_per_set)
        nz_n_sets_per_entity = n_sets_per_cell[n_sets_per_cell > 0]
        offset = 0
        for n in nz_n_sets_per_entity:
            scale[offset : offset + n * n_pts_per_set] = 1.0 / float(n)
            offset += n * n_pts_per_set

        weights = np.tile(ref_weights, (1, int(n_tot_sets))).reshape(-1)
        weights = weights * scale

        points = np.tile(ref_points, (int(n_tot_sets), 1))

        return n_pts_per_cell, weights.astype(np.float64), points

    def create_quad_custom_cells(
        self, degree: int, cells: npt.NDArray[np.int32], tag: Optional[int] = None
    ) -> CustomQuad:
        """Returns the custom quadratures for the given `cells`.

        Among the given `cells`, it only creates quadratures for a
        subset of those (the custom cells), while no points or
        weights are generated for the others. Thus, the cells without
        custom quadratures will be listed in the returned
        `CustomQuadProtocol`, but will have 0 points associated to
        them.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated.

            tag (Optional[int]): Mesh tag of the subdomain associated
                to the given cells. Defaults to None.

        Returns:
            CustomQuadProtocol: Generated custom quadrature.
        """

        subset_ids = np.searchsorted(self._get_cells_ids(), cells)
        n_quad_sets = self._n_quad_sets[subset_ids]

        n_pts_per_cell, weights, points = self._create_quadrature(degree, n_quad_sets, False)
        return CustomQuad(cells, n_pts_per_cell, points, weights)

    def create_quad_unf_boundaries(
        self, degree: int, cells: npt.NDArray[np.int32], tag: Optional[int] = None
    ) -> CustomQuadUnfBoundary:
        """Returns the custom quadrature for unfitted boundaries for the
        given `cells`.

        Warning:
            All the given cells associated should contain unfitted
            boundaries. If not, the custom coefficients generator will
            raise an exception.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated. It must only contain cells with
                unfitted boundaries.

            tag (Optional[int]): Mesh tag of the subdomain associated
                to the given cells. Defaults to None.

        Returns:
            CustomQuadUnfBoundaryProtocol: Generated custom quadrature
            for unfitted boundaries.
        """

        quad = self.create_quad_custom_cells(degree, cells, tag)

        s = quad.points.shape
        normals = np.random.rand(*s)
        normals_norm = np.linalg.norm(normals, axis=1)
        normals_norm = np.repeat(normals_norm, s[1]).reshape(s)
        normals /= normals_norm

        quad_bdry = CustomQuadUnfBoundary(
            quad.cells, quad.n_pts_per_entity, quad.points, quad.weights, normals
        )
        return quad_bdry

    def create_quad_custom_facets(
        self,
        degree: int,
        cells: npt.NDArray[np.int32],
        local_facets: npt.NDArray[np.int32],
        integral_type: str,
        tag: Optional[int] = None,
    ) -> CustomQuadFacet:
        """Returns the custom quadratures for the given facets.

        Among the facets associated to `tag`, it only creates
        quadratures for a subset of those (the custom facets), while no
        points or weights are generated for the others. Thus, the facets
        without custom quadratures will be listed in the returned
        `CustomQuadFacetProtocol`, but will have 0 points associated to

        Among the given facets, it only creates quadratures for a
        subset of those (the custom facets), while no points or
        weights are generated for the others. Thus, the facets without
        custom quadratures will be listed in the returned
        `CustomQuadFacetProtocol`, but will have 0 points associated to
        them.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated. It must only contain cells with
                unfitted boundaries. Beyond a cell id, for indentifying
                a facet, a local facet id (from the array
                `local_facets`) is also needed.

            local_facets (npt.NDArray[np.int32]): Array of local facet
                ids for which the custom quadratures are generated. Each
                facet is identified through a value in `cells` and a
                value in `local_facets`, having both arrays the same
                length. The numbering of these facets follows the
                FEniCSx convention. See
                https://github.com/FEniCS/basix/#supported-elements

            integral_type (str): Type of integral to be computed. It can
                be either 'interior_facet' or 'exterior_facet'.
                Right now it is not used.


            tag (Optional[int]): Mesh tag of the subdomain associated
                to the given facets. Right now it is not used.
                Defaults to None.

        Returns:
            CustomQuadFacetProtocol: Generated custom facet
            quadratures.
        """

        self_cells_facets = np.stack([self._facet_cells_ids, self._facet_local_facets_ids], axis=1)
        cells_facets = np.stack([cells, local_facets], axis=1)

        # TODO: this is potentially slow and should be done faster.
        n_quad_sets = np.zeros(cells_facets.shape[0], dtype=np.int32)
        for i, facet in enumerate(cells_facets):
            ind = (self_cells_facets == facet).all(axis=1).nonzero()[0][0]
            n_quad_sets[i] = self._n_quad_sets_facets[ind]

        n_pts_per_cell, weights, points = self._create_quadrature(degree, n_quad_sets, True)
        return CustomQuadFacet(cells, local_facets, n_pts_per_cell, points, weights)

    def get_unf_bdry_cells(self) -> npt.NDArray[np.int32]:
        """Gets the cells that contain unfitted boundaries.

        Returns:
            npt.NDArray[np.int32]: Array of DOLFINx cell ids (local to
            current MPI process) that contain unfitted boundaries.
        """
        return np.where(self._n_quad_sets != 0)[0].astype(np.int32)
