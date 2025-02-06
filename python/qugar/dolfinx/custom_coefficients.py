# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import dolfinx.cpp.fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
from dolfinx.cpp.fem import IntegralType
from dolfinx.fem.forms import Form

from qugar.dolfinx.custom_quad_utils import map_facets_points, permute_facet_points
from qugar.dolfinx.fe_table import FETable
from qugar.dolfinx.fe_table_eval import evaluate_FE_tables
from qugar.dolfinx.integral_data import IntegralData
from qugar.dolfinx.quadrature_data import QuadratureData
from qugar.quad import CustomQuad, CustomQuadFacet, CustomQuadUnfBoundary, QuadGenerator

"""Type defining all the possible array types for the integrals
coefficients."""
FloatingArray = npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128]


class _CustomCoeffsPackerIntegral:
    """Class for computing coefficients for a single custom integral.

    Parameters:
        _custom_quads_types: All possible types for the custom
            quadratures. Namely ``CustomQuad``, ``CustomQuadFacet``, ``CustomQuadUnfBoundary``
        _itg_data (IntegralData): Data for the integral whose custom
            coefficients are computed.
        _subdomain_id (int): Id of the subdomain
        _domain (npt.NDArray[np.int32]): Domain in which the integrals
            are computed. If the integral is computed over a cell, this
            is just a 1D array containing the cells for which the
            integral is computed. If the integral is computed over an
            exterior facet, this is a 2D array, where each row
            corresponds to a particular facet, being the first column
            the cell id and the second the id of the local facet in the
            cell. Finally, if the integration domain is an interior
            facet `domain` is a 3D array where the first axis
            corresponds to a particular facet; the second axis (that has
            dimension 2) refers to the two sides of the interior facet;
            and the last axis is the id of the local facet in the cell
            (for each side, the local facet refers to cell the facet
            belongs to).
        _mesh (dolfinx.mesh.Mesh): DOLFINx associated to the integral.
        _quad_gen (QuadGenerator): Custom quadratures generator.
        _old_coeffs (FloatingArray): Original (standard) compute
            coefficients for non custom integrals.
        _new_coeffs (FloatingArray): New coefficients for custom
            integrals. See the documentation of the function
            `self._compute_new_coeffs` for a more detailed discussion of
            the structure of this array.
        _dtype (type[np.float32 | np.float64]): `numpy` scalar type of the
            integral quantities.
        _coeffs_dtype (FloatingArray): `numpy` scalar type of the
            integral coefficients.
        _offsets_dtype (np.uintp): `numpy` scalar type for the offset
            values.
        _custom_quads (dict[IntegralData, _custom_quads_types]):
            Dictionary mapping each integral data (integrand in the
            integral) to a custom quadrature (for cells, facets, or
            unfitted boundaries) for all the custom entities in the
            domain.
        _custom_quads_facets
            (Optional[dict[IntegralData, CustomQuadFacet]]):
            Dictionary mapping each integral data (integrand in the
            integral) to a custom facet quadrature for all the custom
            entities in the domain. This attribute is only initialized
            in the case of facet integrals.
        _n_vals_per_entity (npt.NDArray[np.int32]): Number of values to
            be computed / stored per entity, including the non-custom
            ones. The length of the array is the same as the number of
            entities, i.e., the same length along the first axis of
            `self._domain`.
        _offsets (npt.NDArray[np.uintp]): Array of absolute offsets.
            The offsets specify for every entity in the domain (cells,
            exterior facets, or first facet in the case of interior
            facets) in which position of the (flatten) array of extra
            custom coefficients needed for the custom integral start.
            It is an absolute position referred to the start of the
            array.
    """

    _custom_quads_types = CustomQuad | CustomQuadUnfBoundary | tuple[CustomQuad, CustomQuad]

    def __init__(
        self,
        itg_data: IntegralData,
        subdomain_id: int,
        domain: npt.NDArray[np.int32],
        mesh: dolfinx.mesh.Mesh,
        quad_gen: QuadGenerator,
        old_coeffs: npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128],
    ) -> None:
        """Initializes the class.

        Args:
            itg_data (IntegralData): Data for the integral whose custom
                coefficients are computed.
            subdomain_id (int): Id of the subdomain.
            domain (npt.NDArray[np.int32]): Domain in which the
                integrals are computed. If the integral is computed over
                a cell, this is just a 1D array containing the cells for
                which the integral is computed. If the integral is
                computed over an exterior facet, this is a 2D array,
                where each row corresponds to a particular facet, being
                the first column the cell id and the second the id of
                the local facet in the cell. Finally, if the integration
                domain is an interior facet `domain` is a 3D array where
                the first axis corresponds to a particular facet; the
                second axis (that has dimension 2) refers to the two
                sides of the interior facet; and the last axis is the id
                of the local facet in the cell (for each side, the local
                facet refers to cell the facet belongs to).
            mesh (dolfinx.mesh.Mesh): DOLFINx associated to the
                integral.
            quad_gen (QuadGenerator): Class for generating custom
                quadratures.
            old_coeffs (FloatingArray): `Standard` coefficients
                associated to the non-custom integral.
        """
        self._itg_data = itg_data
        self._subdomain_id = subdomain_id
        self._domain = domain
        self._mesh = mesh
        self._quad_gen = quad_gen
        self._old_coeffs = old_coeffs

        self._set_dtypes()
        self._create_quadratures()
        self._compute_new_coeffs()

        return

    @property
    def new_coeffs(
        self,
    ) -> npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128]:
        """Returns the generated new coefficients.

        Returns:
            New generated coefficients.
        """
        return self._new_coeffs

    def _set_dtypes(self) -> None:
        """Sets the `numpy` types associated to the integral, the
        coefficients, and the offsets. Namely, the members ``_dtype``,
        ``_coeffs_dtype``, and ``_offsets_dtype``.
        """

        self._dtype = np.dtype(self._itg_data.dtype)
        self._coeffs_dtype = np.dtype(self._old_coeffs.dtype)

        # We use np.uintp type  to store a size_t C type.
        # See https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uintp # noqa
        self._offsets_dtype = np.dtype(np.uintp)

    def _copy_old_coeffs_to_new(self) -> None:
        """Copies the old coefficients (the ones of the non-custom
        integral) in `self._old_coeffs` in the right position of the new
        custom coefficients array `self._new_coeffs`.
        """
        s = self._old_coeffs.shape
        self._new_coeffs[: s[0], : s[1]] = self._old_coeffs

    def _get_subdomain_tag(self) -> int:
        """Returns the integral subdomain id.

        Returns:
            int: Integral subdomain id.
        """
        return self._subdomain_id

    def _is_interior_facet(self) -> bool:
        """Checks if the current integral corresponds to an
        ``interior_facet``.

        Returns:
            bool: ``True`` if the current integral is on an interior
            facet, ``False`` otherwise.
        """
        return self._itg_data.integral_type == "interior_facet"

    def _is_mixed_dim(self) -> bool:
        """Checks if the current integral presents mixed dimensions.

        Returns:
            bool: ``True`` if the current integral has mixed dimensions,
            ``False`` otherwise.
        """
        return self._itg_data.is_mixed_dim

    def _is_exterior_facet(self) -> bool:
        """Checks if the current integral corresponds to an
        ``exterior_facet``.

        Returns:
            bool: ``True`` if the current integral is on an exterior
            facet, ``False`` otherwise.
        """
        return self._itg_data.integral_type == "exterior_facet"

    def _is_facet(self) -> bool:
        """Checks if the current integral corresponds to a facet, either
        exterior or interior.

        Returns:
            bool: ``True`` if the current integral is on a facet,
            ``False`` otherwise.
        """
        return self._is_exterior_facet() or self._is_interior_facet()

    def _get_n_extra_cols(self) -> int:
        """Gets the number of columns in the coefficients array required
        for storing the coefficients offset.

        As the offset may be represented with a type with more than
        32 bits (see `self._offsets_dtype`), in the case the
        coefficients array has 32 bits values one single column may not
        suffice for storing the offset. Thus, this function computes
        how many columns are needed depending on the offsets type used
        (`self._offsets_dtype`) and the coefficients type
        (`self._coeffs_dtype`).

        Returns:
            int: Number of columns required for storing custom
            coefficients offsets.
        """

        # A size_t value (unsigned 64 bits int used for offsets) may not
        # into a single coefficient value. In particular, in the case
        # they don't fit in the case coeffients are float 32 bits.
        # There should be no problem for float 64 bits, as well as for
        # complex 64 and 128 bits.
        # In the case of float 32 bits, we pack the size_t into
        # two values.
        n_extra_cols = np.ceil(self._offsets_dtype.itemsize / self._coeffs_dtype.itemsize)
        return int(n_extra_cols)

    def _allocate_new_coeffs(
        self,
    ) -> npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128]:
        """Allocates the `numpy` array for storing the custom
        coefficients.

        The size of the (original) non-custom array is enlarged for
        storing the original coefficients and the new ones. In
        particular, one or more columns are added to the original
        coefficients array for storing the custom coefficients offsets
        for every entity. In addition, as many rows as needed are
        added to store all the new coefficients to the custom cells.

        This method also initializes the member
        `self._n_vals_per_entity`.

        Returns:
            FloatingArray: Allocated `numpy` array for storing the
            custom coefficients. It is initialzed to zero.
        """

        self._n_vals_per_entity = self._compute_n_vals_per_entity()

        n_cols = self._old_coeffs.shape[1]
        n_cols += self._get_n_extra_cols()

        n_tot_vals = self._offsets_dtype.type(sum(self._n_vals_per_entity))

        n_rows = self._offsets_dtype.type(self._domain.shape[0])
        if self._is_interior_facet():
            n_rows *= 2
        n_rows += self._offsets_dtype.type(np.ceil(n_tot_vals / n_cols))

        return np.zeros((n_rows, n_cols), dtype=self._coeffs_dtype)

    def _compute_n_vals_per_entity(self) -> npt.NDArray[np.int32]:
        """Computes the number of values to be stored per custom entity.

        Warning:
            This method is yet not implemented for complex types.

        Returns:
            npt.NDArray[np.int32]: Number of values per custom entity.
            The length of the array is the same as the number of custom
            entities, i.e., the same length along the first axis of
            `self._custom_domain`.
        """

        n_entities = self._domain.shape[0]
        n_vals_per_entity = np.zeros(n_entities, dtype=np.int32)

        interior_facet = self._is_interior_facet()

        for quad_data, FE_tables in self._itg_data.quad_data_FE_tables.items():
            custom_quad = self._custom_quads[quad_data]
            if interior_facet:
                custom_quad = custom_quad[0]

            n_vals_per_pt = sum(
                [table.funcs for table in FE_tables if not table.is_constant_for_pts()]
            )

            if interior_facet:
                n_vals_per_pt *= 2

            n_vals_per_pt += 1  # One value per weight.

            if quad_data.unfitted_boundary:
                # Adding values for unfitted boundary normals
                n_vals_per_pt += custom_quad.points.shape[1]

            n_pts_per_entity = custom_quad.n_pts_per_entity

            n_vals_per_entity += n_pts_per_entity * n_vals_per_pt
            # 1 extra values is added for storing the quadrature's
            # number of points. This number is a 32bit int, therefore,
            # it can be casted from any coefficient type (all of them
            # should be >= 32bits).
            n_vals_per_entity[np.where(n_pts_per_entity != 0)] += 1

        if np.issubdtype(self._coeffs_dtype, np.complexfloating):
            # # If the coefficients are complex, we can pack two extra
            # # values into every complex coefficient.
            # assert (self._coeffs_dtype.itemsize
            #   / self._dtype.itemsize) == 2
            # dtype = n_vals_per_entity.dtype
            # # We roundup, so, at the end of the extra values for every
            # # cell there may be some extra (unused) space.
            # # This is done for simplying the offsets calculation and
            # # accesing.
            # n_vals_per_entity = np.ceil(n_vals_per_entity / 2).astype(
            #   dtype)
            assert False, "Not implemented yet for complex values."

        return n_vals_per_entity

    def _compute_offsets(self) -> None:
        """Computes the custom coefficients offset and copies them into
        the extra (right-most) columns of `self._new_coeffs`.

        The offsets specify for every entity in the domain (cells,
        exterior facets, or first facet in the case of interior facets)
        in which position of the (flatten) array the extra coefficients
        needed for the custom integral start. It is a relative position
        referred to the first entry of the current row (entity). If the
        offset is 0, then the entity does not require a custom integral
        and no extra coefficients are provided.

        In addition to copy the relative offsets to `self._new_coeffs`,
        the absolute (non-relative) offsets are stored in
        `self._offsets`.
        """

        interior_facet = self._is_interior_facet()

        n_tot_entities = self._domain.shape[0]
        n_entities = 2 * n_tot_entities if interior_facet else n_tot_entities

        n_cols = self._new_coeffs.shape[1]
        first_extra_val_pos = self._offsets_dtype.type(n_entities * n_cols)
        first_col_pos = np.arange(n_entities, dtype=self._offsets_dtype) * n_cols

        self._offsets = (
            np.concatenate(([0], np.cumsum(self._n_vals_per_entity[:-1]))) + first_extra_val_pos
        )
        self._offsets = self._offsets.astype(self._offsets_dtype)
        if interior_facet:
            first_col_pos = first_col_pos.reshape(-1, 2)[:, 0]
        rel_offsets = self._offsets - first_col_pos
        rel_offsets[np.where(self._n_vals_per_entity == 0)] = 0

        all_rel_offsets = np.empty(n_entities, dtype=rel_offsets.dtype)
        if interior_facet:
            all_rel_offsets_ = all_rel_offsets.reshape(-1, 2)
            all_rel_offsets_[:, 0] = rel_offsets
        else:
            all_rel_offsets[:] = rel_offsets

        # Copying np.uintp offsets to the coefficients array.
        # No cast performed, just a view. A cast will be required in C.

        n_extra_cols = self._get_n_extra_cols()
        all_rel_offsets_view = all_rel_offsets.view(self._coeffs_dtype).reshape([-1, n_extra_cols])
        self._new_coeffs[:n_entities, -n_extra_cols:] = all_rel_offsets_view

    def _copy_vals_complex(
        self,
        vals_quads: list[
            list[npt.NDArray[np.int32] | npt.NDArray[np.float32] | npt.NDArray[np.float64]]
        ],
    ) -> None:
        """Copies the generated extra custom coefficients to their
        position in the array `self._new_coeffs`.

        Note:
            This is the version for complex coefficients array.

        Warning:
            This method is yet not implemented for complex types.

        Args:
            vals_quads (list[list[FloatingArray]]): New custom
            coefficients to be copied. Check the documentation of
            `self._compute_new_vals` for a more detailed description.
        """

        assert self._dtype != self._coeffs_dtype
        raise ValueError("Not implemented for complex values.")

    def _copy_vals_real(
        self,
        vals_all_quads: list[
            list[npt.NDArray[np.int32] | npt.NDArray[np.float32] | npt.NDArray[np.float64]]
        ],
    ) -> None:
        """Copies the generated extra custom coefficients to their
        position in the array `self._new_coeffs`.

        Note:
            This is the version for real coefficients array.

        Args:
            vals_quads (list[list[FloatingArray]]): New custom
            coefficients to be copied. Check the documentation of
            `self._compute_new_vals` for a more detailed description.
        """

        assert self._dtype == self._coeffs_dtype

        new_coeffs = self._new_coeffs.ravel()

        odtype = self._offsets_dtype

        offsets = np.copy(self._offsets)
        offsets = offsets[np.where(self._n_vals_per_entity != 0)]
        n_custom = offsets.size

        for vals_for_quad in vals_all_quads:
            assert len(vals_for_quad) > 0

            vals_offsets = np.zeros(len(vals_for_quad), dtype=odtype)

            n_pts_per_cell = vals_for_quad.pop(0).astype(odtype)

            new_coeffs[offsets] = n_pts_per_cell
            offsets += 1

            n_vals_per_pt = np.array([vals.size for vals in vals_for_quad], dtype=odtype)

            n_tot_pts = np.sum(n_pts_per_cell)
            if n_tot_pts > 0:
                n_vals_per_pt //= n_tot_pts

            # TODO: This loop over the cells may be slow and could be
            # improved. Maybe to be done with numba.
            for cell_id in range(n_custom):
                n_pts = n_pts_per_cell[cell_id]
                c_off0 = offsets[cell_id]
                for i, vals in enumerate(vals_for_quad):
                    n_vals = n_pts * n_vals_per_pt[i]

                    c_off1 = c_off0 + n_vals

                    v_off0 = vals_offsets[i]
                    v_off1 = v_off0 + n_vals

                    new_coeffs[c_off0:c_off1] = vals[v_off0:v_off1]

                    c_off0 = c_off1
                    vals_offsets[i] = v_off1
                offsets[cell_id] = c_off0

    def _copy_vals(
        self,
        vals_all_quads: list[
            list[npt.NDArray[np.int32] | npt.NDArray[np.float32] | npt.NDArray[np.float64]]
        ],
    ) -> None:
        """Copies the generated extra custom coefficients to their
        position in the array `self._new_coeffs`.

        Args:
            vals_quads (list[list[FloatingArray]]): New custom
            coefficients to be copied. Check the documentation of
            `self._compute_new_vals` for a more detailed description.
        """

        if np.issubdtype(self._coeffs_dtype, np.complexfloating):
            self._copy_vals_complex(vals_all_quads)
        else:
            self._copy_vals_real(vals_all_quads)

    def _create_quadrature(self, quad_data: QuadratureData) -> CustomQuad | CustomQuadUnfBoundary:
        """Creates the custom quadratures for a certain integrand in the
        integral.

        This method manages the case in which the integral is performed
        either in a cell or in an unfitted boundary.

        Args:
            quad_data (QuadratureData): Quadrature data of the integrand
                for which the quadrature is generated.

        Returns:
            Generated quadrature.
        """

        assert not self._is_facet()

        degree = quad_data.degree
        quad_gen = self._quad_gen
        subdom_tag = self._get_subdomain_tag()
        cells = self._domain

        if quad_data.unfitted_boundary:
            return quad_gen.create_quad_unf_boundaries(degree, cells, subdom_tag)
        else:
            return quad_gen.create_quad_custom_cells(degree, cells, subdom_tag)

    def _create_quadrature_facet(
        self, quad_data: QuadratureData
    ) -> tuple[CustomQuadFacet, CustomQuad | tuple[CustomQuad, CustomQuad]]:
        """Creates the custom quadratures for a certain integrand in the
        the integral.

        This method manages the case in which the integral is performed
        over (internal or external) facets.

        Args:
            quad_data (QuadratureData): Quadrature data of the integrand
                for which the quadrature is generated.

        Returns:
            Generated quadrature. The first item of the tuple is the
            quadrature associated to the facet, while the second is
            that quadrature but mapped to the reference domain of the
            parente cell. In the case of internal facets, the second
            item is a tuple with two mapped quadratures (one for each
            side of the interface).
        """

        assert self._is_facet()

        degree = quad_data.degree
        subdom_tag = self._get_subdomain_tag()
        quad_gen = self._quad_gen

        interior_facet = self._is_interior_facet()
        if interior_facet:
            cells = self._domain[:, 0, 0].reshape(-1)
            facets = self._domain[:, 0, 1].reshape(-1)
        else:
            cells = self._domain[:, 0].reshape(-1)
            facets = self._domain[:, 1].reshape(-1)

        quad_facet = quad_gen.create_quad_custom_facets(degree, cells, facets, subdom_tag)

        quad = self._map_facet_quadrature(quad_facet, cells, facets)
        if not interior_facet:
            return quad_facet, quad

        cells_1 = self._domain[:, 1, 0].reshape(-1)
        facets_1 = self._domain[:, 1, 1].reshape(-1)
        quad_1 = self._map_facet_quadrature(quad_facet, cells_1, facets_1)

        return quad_facet, (quad, quad_1)

    def _map_facet_quadrature(
        self,
        quad_facet: CustomQuadFacet,
        cells: npt.NDArray[np.int32],
        facets: npt.NDArray[np.int32],
    ) -> CustomQuad:
        """Maps the given quadrature from a facet to the parent cell
        domain.

        Args:
            quad_facet (CustomQuadFacet): Facet quadrature to
                map.
            cells (npt.NDArray[np.int32]): Cells associated to the facet
                quadrature.
            facets (npt.NDArray[np.int32]): Facets to which the facet
                quadrature is associated to.

        Returns:
            CustomQuad: Generated cell quadrature.
        """

        assert self._is_facet()

        mesh = self._mesh
        needs_permutations = self._is_interior_facet() or self._is_mixed_dim()

        n_pts_per_entity = quad_facet.n_pts_per_entity
        points = quad_facet.points

        cells_rep = np.repeat(cells.reshape(-1), n_pts_per_entity)
        facets_rep = np.repeat(facets.reshape(-1), n_pts_per_entity)

        if needs_permutations:
            points = permute_facet_points(points, mesh, cells_rep, facets_rep)

        mapped_points = map_facets_points(points, facets_rep, mesh.topology.cell_type)

        return CustomQuad(cells, n_pts_per_entity, mapped_points, quad_facet.weights)

    def _create_quadratures(self) -> None:
        """Creates the custom quadratures for the integrands in the
        integral and stores them in `self._custom_quads` and
        `self._custom_quads_facets` (if mixed dimension integral).
        """

        self._custom_quads = {}
        self._custom_quads_facets = {}

        for quad_data in self._itg_data.quad_data_FE_tables.keys():
            if self._is_facet():
                quad_facet, quad = self._create_quadrature_facet(quad_data)
                if self._is_mixed_dim():
                    self._custom_quads_facets[quad_data] = quad_facet
            else:
                quad = self._create_quadrature(quad_data)

            self._custom_quads[quad_data] = quad

    def _compute_new_vals_tables(
        self, quad_data: QuadratureData, FE_tables: list[FETable]
    ) -> list[npt.NDArray[np.float32 | np.float64]]:
        """Computes the new values for the given FE tables.

        Args:
            quad_data (QuadratureData): Quadrature data of the integrand
                for which the values are computed.
            FE_tables (list[FETable]): List of FE tables which new
                values are computed.

        Returns:
            list[npt.NDArray[np.float32 | np.float64]]: Computed values.
        """

        if self._is_interior_facet():
            return self._compute_new_vals_tables_interior_facet(quad_data, FE_tables)
        elif self._is_mixed_dim():
            return self._compute_new_vals_tables_mixed_dim(quad_data, FE_tables)
        else:
            return self._compute_new_vals_tables_generic(quad_data, FE_tables)

    def _compute_new_vals_tables_mixed_dim(
        self, quad_data: QuadratureData, FE_tables: list[FETable]
    ) -> list[npt.NDArray[np.float32 | np.float64]]:
        """Computes the new values for the given FE tables for the case
        of integrals with mixed dimensions.

        Args:
            quad_data (QuadratureData): Quadrature data of the integrand
                for which the values are computed.
            FE_tables (list[FETable]): List of FE tables which new
                values are computed.

        Returns:
            list[npt.NDArray[np.float32 | np.float64]]: Computed values.
        """

        assert self._is_mixed_dim()

        custom_quad = self._custom_quads[quad_data]
        custom_quad_facet = self._custom_quads_facets[quad_data]

        tdim = custom_quad.points.shape[-1]
        FE_tables_cell = []
        FE_tables_facet = []
        for table in FE_tables:
            if tdim == table.element_dim:
                FE_tables_cell.append(table)
            else:
                FE_tables_facet.append(table)

        values = evaluate_FE_tables(FE_tables_cell, custom_quad.points)
        values.update(evaluate_FE_tables(FE_tables_facet, custom_quad_facet.points))

        vals_for_quad = []
        for table in FE_tables:
            if table in values:
                vals = values[table]
                assert len(vals.shape) == 2
                vals_for_quad.append(vals.ravel())

        return vals_for_quad

    def _compute_new_vals_tables_generic(
        self, quad_data: QuadratureData, FE_tables: list[FETable]
    ) -> list[npt.NDArray[np.float32 | np.float64]]:
        """Computes the new values for the given FE tables for the case
        in which the integral is not performed on interior facet and
        has no mixed dimensions.

        Args:
            quad_data (QuadratureData): Quadrature data of the integrand
                for which the values are computed.
            FE_tables (list[FETable]): List of FE tables which new
                values are computed.

        Returns:
            list[npt.NDArray[np.float32 | np.float64]]: Computed values.
        """

        assert not self._is_interior_facet() and not self._is_mixed_dim()

        custom_quad = self._custom_quads[quad_data]
        values = evaluate_FE_tables(FE_tables, custom_quad.points)
        vals_for_quad = []
        for vals in values.values():
            assert len(vals.shape) == 2
            vals_for_quad.append(vals.ravel())
        return vals_for_quad

    def _compute_new_vals_tables_interior_facet(
        self, quad_data: QuadratureData, FE_tables: list[FETable]
    ) -> list[npt.NDArray[np.float32 | np.float64]]:
        """Computes the new values for the given FE tables for the case
        of interior facet integrals.

        Args:
            quad_data (QuadratureData): Quadrature data of the integrand
                for which the values are computed.
            FE_tables (list[FETable]): List of FE tables which new
                values are computed.

        Returns:
            list[npt.NDArray[np.float32 | np.float64]]: Computed values.
        """

        assert self._is_interior_facet()

        custom_quad = self._custom_quads[quad_data]
        assert len(custom_quad) == 2
        values_0 = evaluate_FE_tables(FE_tables, custom_quad[0].points)
        values_1 = evaluate_FE_tables(FE_tables, custom_quad[1].points)

        vals_for_quad = []
        for table, vals_0 in values_0.items():
            vals_for_quad.append(vals_0.ravel())
            if table.permutations > 1:
                vals_for_quad.append(values_1[table].ravel())

        return vals_for_quad

    def _compute_new_vals(
        self,
    ) -> list[list[npt.NDArray[np.int32] | npt.NDArray[np.float32] | npt.NDArray[np.float64]]]:
        """Computes the extra coefficients associated to the custom
        integrals. Namely, number of points per entity, weights, normals
        (if needed), and finite element basis functions evaluations.

        Returns:
            list[list[npt.NDArray[np.int32] | npt.NDArray[np.float32] | npt.NDArray[np.float64]]]:
            Computed quantities. Every entry of the list corresponds to
            all the quantities for a particular integrand (quadrature).
            Each sublist corresponds to a particular quantity (e.g.,
            the weights or the basis functions of a certain finite
            element) for all the custom entities.
        """

        vals_all_quads = []
        for quad_data, FE_tables in self._itg_data.quad_data_FE_tables.items():
            custom_quad = self._custom_quads[quad_data]
            if self._is_interior_facet():
                custom_quad = custom_quad[0]

            nnz = np.where(custom_quad.n_pts_per_entity != 0)
            vals_for_quad = [custom_quad.n_pts_per_entity[nnz]]
            vals_for_quad.append(custom_quad.weights)  # type: ignore

            if quad_data.unfitted_boundary:
                vals_for_quad.append(custom_quad.normals.ravel())  # type: ignore

            vals_for_quad += self._compute_new_vals_tables(quad_data, FE_tables)

            vals_all_quads.append(vals_for_quad)

        return vals_all_quads

    def _compute_new_coeffs(self) -> None:
        """Creates the custom coefficients for the integral and stores
        them in `self._new_coeffs`. This new array also contains the
        coefficients of the (original) non-custom integral.

        After including the `self._old_coeffs` (the coefficients for the
        non-custom integral), this function appends all the extra
        coefficients for the custom integrals.

        In addition, it also appends one (or two, depending on the
        coefficients type) extra columns for including the coefficients
        relative offsets for every entity (cell or facet).

        If the offset is zero it means that the cell does not have a
        custom integration, but a standard one, and then the standard
        (old) coefficients should be used. Otherwise, that value
        indicates the relative position (respect to the first entry of
        the current row) where the extra custom coefficients for the
        integral of the current entity start.
        """

        # t0 = time.time()
        self._new_coeffs = self._allocate_new_coeffs()
        # print(f"Allocation time: {time.time() - t0} seconds.")

        # t0 = time.time()
        self._copy_old_coeffs_to_new()
        # print(f"Copying old coefficients: {time.time() - t0} seconds.")

        if self._old_coeffs.shape[0] == self._new_coeffs.shape[0]:
            return

        # t0 = time.time()
        self._compute_offsets()

        vals_all_quads = self._compute_new_vals()
        # print(f"Computing extra values: {time.time() - t0} seconds.")

        # t0 = time.time()
        self._copy_vals(vals_all_quads)
        # print(f"   Copying new values: {time.time() - t0} seconds.")


def _compute_custom_coeffs(
    form: Form,
    itg_data: IntegralData,
    itg_info: tuple[IntegralType, int],
    quad_gen: QuadGenerator,
    old_coeffs: FloatingArray,
) -> FloatingArray:
    """Generates the custom coefficients for an integral.

    Args:
        form (Form): DOLFINx form associated to the integral whose
            custom coefficients are computed.
        itg_data (IntegralData): Data of the integral whose custom
            coefficients are computed.
        quad_gen (QuadGenerator): Class for generating custom
            quadratures.
        old_coeffs (FloatingArray): `Standard` coefficients associated
            to the non-custom integral.

    Returns:
        FloatingArray: Computed coefficients for the custom integral.
    """

    domain = form._cpp_object.domains(*itg_info)
    mesh = form._cpp_object.mesh

    generator = _CustomCoeffsPackerIntegral(
        itg_data, itg_info[1], domain, mesh, quad_gen, old_coeffs
    )

    return generator.new_coeffs


class CustomCoeffsPacker:
    """Class for computing coefficients for custom integrals.

    Beyond its constructor, its public API is just the method
    `pack_coefficients`.

    Parameters:
        _form (Form): DOLFINx form (containing the custom integral
            module).
        _itg_data (list[IntegralData]): Data of all the integrals
            contained in the `form`.
    """

    def __init__(self, form: Form, itg_data: list[IntegralData]) -> None:
        """Initializes the class.

        Args:
            form (Form): DOLFINx form (containing the custom integral
                module).
            itg_data (list[IntegralData]): Data of all the integrals
                contained in the `form`.
        """

        self._form = form
        self._itg_data = itg_data
        return

    def _get_itg_data(self, itg_info: tuple[IntegralType, int]) -> IntegralData:
        """Gets the integral data whose info matches `itg_info`.

        Args:
            itg_info (tuple[IntegralType, int]): Integral info (type and
                subdomain id) sought.

        Returns:
            IntegralData: Integral data instance found.
        """

        valid_integral_types = (
            IntegralType.cell,
            IntegralType.exterior_facet,
            IntegralType.interior_facet,
        )

        assert itg_info[0] in valid_integral_types

        for data in self._itg_data:
            if itg_info in data.itg_infos:
                return data

        assert False, "Integral data not found."

    def pack_coefficients(
        self, quad_gen: QuadGenerator
    ) -> dict[
        tuple[IntegralType, int],
        npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128],
    ]:
        """Generates the custom coefficients consumed by custom
        integrals for all the integral types and subdomains.

        Args:
            quad_gen (QuadGenerator): Class for generating custom
                quadratures.

        Returns:
            dict[tuple[IntegralType, int], FloatingArray]: Generated
            custom coefficients (it follows the same data structure as
            ``dolfinx.cpp.fem.pack_coefficients``).
        """

        # t0 = time.time()
        coeffs = dolfinx.cpp.fem.pack_coefficients(self._form._cpp_object)
        # print(f"Computing original coefficients: {time.time() - t0} seconds.")

        new_pack_coeffs: dict[
            tuple[IntegralType, int],
            npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128],
        ] = {}

        for itg_info, old_coeffs in coeffs.items():
            itg_data = self._get_itg_data(itg_info)
            new_coeffs = _compute_custom_coeffs(
                self._form, itg_data, itg_info, quad_gen, old_coeffs
            )
            new_pack_coeffs[itg_info] = new_coeffs

        return new_pack_coeffs
