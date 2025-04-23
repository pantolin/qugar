# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from typing import Callable, Optional, Tuple, cast

from qugar import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import numpy as np
import numpy.typing as npt

import qugar.cpp
from qugar.cpp import UnfittedDomain_2D, UnfittedDomain_3D
from qugar.mesh.mesh import Mesh
from qugar.mesh.mesh_facets import (
    MeshFacets,
    create_facet_manager_all,
    create_facet_manager_exterior,
    create_facet_manager_interior,
)
from qugar.mesh.unfitted_domain_abc import UnfittedDomainABC
from qugar.quad.custom_quad import (
    CustomQuad,
    CustomQuadFacet,
    CustomQuadUnfBoundary,
)


def _sort_facets(
    cell_ids: npt.NDArray[np.int32], local_facet_ids: npt.NDArray[np.int32]
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Sorts facets based on cell ids and local facet ids.

    The sorting is performed first based on cell ids (increasing order)
    and then based on local facet ids (increasing order).

    Args:
        cell_ids (npt.NDArray[np.int32]): Array of cell ids.
        local_facet_ids (npt.NDArray[np.int32]): Array of local facet ids.

    Returns:
        Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        Sorted cell ids and local facet ids.
    """
    # Use lexsort to sort primarily by cell_ids and secondarily by local_facet_ids
    # Note: lexsort sorts by the last key first, so we provide local_facet_ids then cell_ids
    sorted_indices = np.lexsort((local_facet_ids, cell_ids))

    # Apply the sorted indices to both arrays
    sorted_cell_ids = cell_ids[sorted_indices]
    sorted_local_facet_ids = local_facet_ids[sorted_indices]

    return sorted_cell_ids, sorted_local_facet_ids


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

        super().__init__(mesh)

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
            orig_cell_ids = cell_getter(None)  # All cells.
        else:
            orig_cell_ids = cell_getter(self._mesh.get_all_original_cell_ids())

        return np.sort(self._mesh.get_DOLFINx_local_cell_ids(orig_cell_ids))

    def get_cut_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the cut cells.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """

        if self._cut_cells is None:
            caller = self._cpp_unf_domain_object.get_cut_cells
            self._cut_cells = self._get_cells(caller)

        return self._cut_cells

    def get_full_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the full cells.

        Returns:
            npt.NDArray[np.int32]: Array of full cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """

        if self._full_cells is None:
            caller = self._cpp_unf_domain_object.get_full_cells
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

    def _get_facets(
        self,
        facets_getter: Callable[
            [Optional[npt.NDArray[np.int64]], Optional[npt.NDArray[np.int32]]],
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]],
        ],
        facets_type: Optional[str] = None,
    ) -> MeshFacets:
        """Retrieves and filters mesh facets using a custom filtering function.

        This internal method orchestrates the process of selecting facets based on
        their type (all, interior, or exterior), transforming them into a
        representation based on original cell IDs and lexicographical local facet
        IDs, applying a user-provided filtering function (`facets_getter`) on this
        representation, and finally converting the filtered facets back to the
        standard DOLFINx local cell and facet indices.

        Args:
            facets_getter: A callable that accepts two optional NumPy arrays
                (original cell IDs `np.int64`, original lexicographical local
                facet IDs `np.int32`) and returns a tuple containing the filtered
                versions of these arrays. This function defines the custom
                filtering logic.
            facets_type: An optional string specifying the initial type of facets
                to retrieve. Valid options are:
                - `None`: Retrieve all facets. (Default)
                - `'interior'`: Retrieve only facets shared between two cells.
                - `'exterior'`: Retrieve only facets on the boundary (belonging
                  to a single cell).

        Returns:
            FacetManager: A `FacetManager` instance containing the filtered facets,
                represented by DOLFINx local cell indices and local facet indices,
                sorted lexicographically.
        """
        if facets_type is not None:
            assert facets_type in ["interior", "exterior"], "Invalid facets type."
            if facets_type == "interior":
                facets = create_facet_manager_interior(self._mesh, single_interior_facet=False)
            else:
                facets = create_facet_manager_exterior(self._mesh)
        else:
            facets = create_facet_manager_all(self._mesh, single_interior_facet=False)

        orig_facets = facets.to_original(self._mesh)

        # Filtering of the original facets using the facets_getter
        filtered_facets = facets_getter(
            cast(npt.NDArray[np.int64], orig_facets.cell_ids), orig_facets.local_facet_ids
        )

        dlf_facets = MeshFacets(*filtered_facets).to_DOLFINx(self._mesh)
        dlf_facets.sort()

        return dlf_facets

    def _get_cut_facets_ext_integral(self) -> MeshFacets:
        """Gets the cut facets as a FacetManager object for exterior integrals.

        Here we consider either interior or exterior (that belong to a
        single cell) facets that partially belong to the domain's
        boundary (either the mesh's domain or the unfitted boundary).
        If a facet is fully contained in an unfitted boundary, it is
        considered as full.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Cut facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local
            DOLFINx ordering.
        """

        if self._exterior_cut_facets is None:
            # We include cut facets.
            self._exterior_cut_facets = self._get_facets(
                self._cpp_unf_domain_object.get_cut_facets,
                facets_type="exterior",
            )

            # But also facets (exterior or interior) that contain
            # unfitted boundaries, but only when they are not full.
            unf_facets = self._get_facets(
                self.cpp_unf_domain_object.get_unf_bdry_facets, facets_type=None
            )
            if not unf_facets.empty:
                # They may contain full unfitted boundaries that need to be removed.
                unf_full_facets = self._get_facets(
                    self.cpp_unf_domain_object.get_full_unf_bdry_facets, facets_type=None
                )
                unf_partial_facets = unf_facets.difference(unf_full_facets)

                if not unf_partial_facets.empty:
                    self._exterior_cut_facets = self._exterior_cut_facets.concatenate(
                        unf_partial_facets
                    )
                    self._exterior_cut_facets.sort()
                    self._exterior_cut_facets.unique()

        return self._exterior_cut_facets

    def _get_cut_facets_int_integral(self) -> MeshFacets:
        """Gets the cut facets as a FacetManager object for interior integrals.

        Hwere we consider interior facets (shared by two cells) that are
        cut (i.e., they are partially inside the domain).

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Cut facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local
            DOLFINx ordering.
        """
        if self._interior_cut_facets is None:
            self._interior_cut_facets = self._get_facets(
                self._cpp_unf_domain_object.get_cut_facets,
                facets_type="interior",
            )

        return self._interior_cut_facets

    def get_cut_facets(
        self,
        exterior_integral: bool = True,
    ) -> MeshFacets:
        """Gets the cut facets as a FacetManager object.

        The list of facets will be filtered for exterior or interior
        integrals according to the argument `exterior_integral`.

        Note:
            The selection of facets is performed differently depending
            on whether exterior or interior integrals are to be
            computed.
            For interior integrals we consider interior facets (shared
            by two cells) that are cut (i.e., they are partially inside
            the domain).
            For exterior integrals we consider either interior or
            exterior (that belong to a single cell) facets that
            partially belong to the domain's boundary (either the mesh's
            domain or the unfitted boundary).
            In the case of exterior integrals, if a facet is fully
            contained in an unfitted boundary, it is considered as full.

        Args:
            exterior_integral (bool): Whether the list of facets is
                retrieved for computing exterior or interior integrals
                (see note above).

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Cut facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local
            DOLFINx ordering.
        """

        return (
            self._get_cut_facets_ext_integral()
            if exterior_integral
            else self._get_cut_facets_int_integral()
        )

    def _get_full_facets_ext_integral(self) -> MeshFacets:
        """Gets the full facets as a FacetManager object for
        exterior integrals.

        Here we consider facets (interior or exterior) that are fully
        contained in the domain's boundary (either the mesh's domain or
        the unfitted boundary).

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Full facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local
            DOLFINx ordering.
        """

        if self._exterior_full_facets is None:
            # Full facets for exterior facets.
            self._exterior_full_facets = self._get_facets(
                self._cpp_unf_domain_object.get_full_facets,
                facets_type="exterior",
            )

            # We also include facets that contain unfitted boudnaries
            # if the facet is full for both exterior and interior facets.
            unf_full_facets = self._get_facets(
                self.cpp_unf_domain_object.get_full_unf_bdry_facets,
                facets_type=None,
            )
            if not unf_full_facets.empty:
                self._exterior_full_facets = self._exterior_full_facets.concatenate(unf_full_facets)
                self._exterior_full_facets.sort()
                self._exterior_full_facets.unique()

        return self._exterior_full_facets

    def _get_full_facets_int_integral(self) -> MeshFacets:
        """Gets the full facets as a FacetManager object for
        interior integrals.

        Here we consider interior facets (shared by two cells) that are
        fully inside the domain (and not contained in any unfitted
        boundary).

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Full facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local
            DOLFINx ordering.
        """

        if self._interior_full_facets is None:
            # Here we only the full facets, without considering the
            # unfitted boundaries fully contained in the facet.
            self._interior_full_facets = self._get_facets(
                self._cpp_unf_domain_object.get_full_facets,
                facets_type="interior",
            )

        return self._interior_full_facets

    def get_full_facets(
        self,
        exterior_integral: bool = True,
    ) -> MeshFacets:
        """Gets the full facets as a FacetManager object.

        The list of facets will be filtered for exterior or interior
        integrals according to the argument `exterior_integral`.

        Note:
            The selection of facets is performed differently depending
            on whether exterior or interior integrals are to be
            computed.
            For interior integrals we consider interior facets (shared
            by two cells) that are fully inside the domain (and not
            contained in any unfitted boundary).
            For exterior integrals we consider facets (interior or
            exterior) that are fully contained in the domain's boundary
            (either the mesh's domain or the unfitted boundary).

        Args:
            exterior_integral (bool): Whether the list of facets is
                retrieved for computing exterior or interior integrals
                (see note above).

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Full facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local
            DOLFINx ordering.
        """

        return (
            self._get_full_facets_ext_integral()
            if exterior_integral
            else self._get_full_facets_int_integral()
        )

    def _get_empty_facets_ext_integral(
        self,
    ) -> MeshFacets:
        """Gets the empty facets as a FacetManager object
        for exterior integrals.

        Here we consider exterior facets (that belong to a single cell)
        that are not contained in the domain or its boundary.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Empty facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local)
            DOLFINx ordering.
        """

        if self._exterior_empty_facets is None:
            # Here we only include empty facets.
            self._exterior_empty_facets = self._get_facets(
                self._cpp_unf_domain_object.get_empty_facets,
                facets_type="exterior",
            )

        return self._exterior_empty_facets

    def _get_empty_facets_int_integral(
        self,
    ) -> MeshFacets:
        """Gets the empty facets as a FacetManager object
        for interior integrals.

        Here we consider interior facets (shared by two cells) that are
        not contained in the domain (they may be contained (fully or
        partially) in the unfitted boundary).

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Empty facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local)
            DOLFINx ordering.
        """

        if self._interior_empty_facets is None:
            self._interior_empty_facets = self._get_facets(
                self._cpp_unf_domain_object.get_empty_facets,
                facets_type="interior",
            )

            # Beyond the empty facets, we also need to consider
            # facets that are not cut but contain unfitted boundaries
            # (full or not).

            unf_facets = self._get_facets(
                self.cpp_unf_domain_object.get_unf_bdry_facets,
                facets_type="interior",
            )
            if not unf_facets.empty:
                # They may contain cut facets need to be removed.
                unf_non_cut_facets = unf_facets.difference(
                    self.get_cut_facets(exterior_integral=False), assume_unique=True
                )

                if not unf_non_cut_facets.empty:
                    self._interior_cut_facets = self._interior_empty_facets.concatenate(
                        unf_non_cut_facets
                    )
                    self._interior_cut_facets.sort()
                    self._interior_cut_facets.unique()

        return self._interior_empty_facets

    def get_empty_facets(
        self,
        exterior_integral: bool = True,
    ) -> MeshFacets:
        """Gets the empty facets as a FacetManager object

        The list of facets will be filtered to only exterior or interior
        facets according to the argument `exterior`.

        Note:
            The selection of facets is performed differently depending
            on whether exterior or interior integrals are to be
            computed.
            For interior integrals we consider interior facets (shared
            by two cells) that are not contained in the domain (they
            may be contained (fully or partially) in the unfitted
            boundary).
            For exterior integrals we consider exterior facets (that
            belong to a single cell) that are not contained in the
            domain or its boundary.

        Args:
            exterior_integral (bool): Whether the list of facets is
                retrieved for computing exterior or interior integrals
                (see note above).

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Empty facets. The facets are returned as one array of
            cells and another one of local facets referred to those
            cells both with the same length and both following (local)
            DOLFINx ordering.
        """

        return (
            self._get_empty_facets_ext_integral()
            if exterior_integral
            else self._get_empty_facets_int_integral()
        )

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
            This call requires the generation of the quadratures on
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

        quad = qugar.cpp.create_quadrature(
            self.cpp_unf_domain_object,
            cells=self._mesh.get_original_cell_ids(dlf_cells),
            n_pts_dir=UnfittedDomain.get_num_Gauss_points(degree),
        )

        return CustomQuad(dlf_cells, quad.n_pts_per_entity, quad.points, quad.weights)

    def create_quad_unf_boundaries(
        self,
        degree: int,
        dlf_cells: npt.NDArray[np.int32],
    ) -> CustomQuadUnfBoundary:
        """Returns the custom quadrature for unfitted boundaries for the
        given `cells`.

        Note:
            Some unfitted boundary parts may lay over facets.
            The quadrature corresponding to those parts will not be
            included in the quadrature generated by this function, but
            in the one generated with the method
            `create_quad_custom_facets`.

        Note:
            For cells not containing unfitted boundaries, no quadrature
            is generated and will have 0 points associated to them.

        Note:
            This call requires the generation of the quadratures on
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

        quad = qugar.cpp.create_unfitted_bound_quadrature(
            self._cpp_unf_domain_object,
            cells=self._mesh.get_original_cell_ids(dlf_cells),
            n_pts_dir=UnfittedDomain.get_num_Gauss_points(degree),
            include_facet_unf_bdry=False,
            exclude_ext_bdry=False,
        )

        return CustomQuadUnfBoundary(
            dlf_cells, quad.n_pts_per_entity, quad.points, quad.weights, quad.normals
        )

    def create_quad_custom_facets(
        self,
        degree: int,
        dlf_facets: MeshFacets,
        exterior_integral: bool,
    ) -> CustomQuadFacet:
        """Returns the custom quadratures for the given facets.

        For empty facets, no quadrature is generated and will have 0 points
        associated to them. For full facets, the quadrature will be the
        standard one associated to the facet type. While for
        cut facets a custom quadrature for the facet's interior will be
        generated.

        Note:
            This call requires the generation of the quadratures on
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
            exterior_integral (bool): Whether exterior integrals are
                to be computed using the generated quadratures.
                If `True`, the quadrature will be generated for the
                facets that belong to the domain's boundary (either
                the mesh's domain or the unfitted boundary).
                Otherwise, the quadrature will be generated only for
                the parts of the interior facets that are inside the
                domain (but not its boundary).

        Returns:
            CustomQuadFacetInterface: Generated custom facet
            quadratures.
        """

        n_pts_dir = UnfittedDomain.get_num_Gauss_points(degree)

        orig_facets = dlf_facets.to_original(self._mesh)

        quad_func = (
            qugar.cpp.create_facets_quadrature_exterior_integral
            if exterior_integral
            else qugar.cpp.create_facets_quadrature_interior_integral
        )

        quad = quad_func(
            self._cpp_unf_domain_object,
            orig_facets.cell_ids,
            orig_facets.local_facet_ids,
            n_pts_dir,
        )

        return CustomQuadFacet(
            dlf_facets,
            quad.n_pts_per_entity,
            quad.points,
            quad.weights,
        )
