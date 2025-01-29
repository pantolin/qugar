# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Set of tools for evaluating FE tables."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import typing
from typing import Optional, Tuple, cast

import basix.ufl
import ffcx.ir.elementtables
import numpy as np
import numpy.typing as npt
import ufl.cell
from basix.ufl import _BasixElement as BasixElement
from basix.ufl import _ElementBase as ElementBase
from ffcx.element_interface import basix_index
from ffcx.ir.elementtables import (
    analyse_table_type,
    clamp_table_small_numbers,
    default_atol,
    default_rtol,
    is_permuted_table,
    permute_quadrature_interval,
    permute_quadrature_quadrilateral,
    permute_quadrature_triangle,
    piecewise_ttypes,
    uniform_ttypes,
)
from ffcx.ir.representationutils import create_quadrature_points_and_weights

from qugar.dolfinx.fe_table import FETable


def _get_points_weights_average_values(
    element: ElementBase, avg: str
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Computes the quadrature points and weights for computing the
    average of an element basis functions.

    Args:
        element (ElementBase): Element whose basis functions are
            computed.
        avg (str): Type of average to compute. It only can be
            ``cell`` or ``facet``.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        Computed quadrature points (first) and weights (second).
    """

    assert not isinstance(element, basix.ufl._QuadratureElement)
    assert avg in ("cell", "facet")

    # Doesn't matter if it's exterior or interior facet
    # integral, just need a valid integral type to create
    # quadrature rule
    integral_type = "cell" if avg == "cell" else "exterior_facet"

    # Make quadrature rule and get points and weights
    points, weights, _ = create_quadrature_points_and_weights(
        integral_type,
        element.cell,
        element.embedded_superdegree,
        "default",
        [element],
    )
    points = cast(npt.NDArray[np.float64], points)
    weights = cast(npt.NDArray[np.float64], weights)

    return points, weights


def _evaluate_scalar_element_derivatives(
    element: ElementBase,
    points: npt.NDArray[np.float64],
    avg: Optional[str],
    derivatives: list[tuple[int, ...]],
) -> dict[tuple[int, ...], npt.NDArray[np.float64]]:
    """Evaluates the derivatives of an element at the given
    points.

    Args:
        element (ElementBase): Element whose derivatives are computed.
        points (npt.NDArray[np.float64]): Points (in the reference
            domain of the element) where the derivatives are computed.
        avg (Optional[str]): String describing the type of average
            to compute for the values. If defined, it only can be
            ``cell`` or ``facet``.
        derivatives (list[tuple[int, ...]]): List of derivatives
            to compute.

    Returns:
        dict[tuple[int,...], npt.NDArray[np.float64]]: Computed
        Dictionary mapping derivatives indices to the computed values.
    """

    assert not isinstance(element, basix.ufl._QuadratureElement)
    assert len(derivatives)
    assert avg in (None, "cell", "facet")

    max_deriv_order = max([sum(der) for der in derivatives])
    if avg is not None:
        # Not expecting derivatives of averages
        assert len(derivatives) == 1 and not any(derivatives[0])
        assert max_deriv_order == 0
        points, weights = _get_points_weights_average_values(element, avg)

    tbl = element.tabulate(max_deriv_order, points)

    tables = {}
    for der_counts in derivatives:
        tables[der_counts] = tbl[basix_index(der_counts)]  # type: ignore

    if avg is not None:
        # Compute numeric integral of the each component table
        wsum = sum(weights)  # type: ignore

        for der, tbl in tables.items():
            avg_tbl = np.dot(weights, tbl) / wsum  # type: ignore
            tables[der] = np.reshape(avg_tbl, (1, 1, 1, tbl.shape[1]))
    else:
        for der, tbl in tables.items():
            tables[der] = tbl.reshape(1, 1, tbl.shape[0], tbl.shape[1])

    return tables


def _get_cell_permutations(cell: ufl.cell.Cell) -> tuple[list[list[int]], typing.Any]:
    """Creates all the permutations, and the function for performing
    them, for a given cell

    Args:
        cell (_type_): _description_

    Returns:
        tuple[list[list[int]], typing.Any]: First, list with all the
        permutations, that in 2D and 3D are defined by a reflection
        and rotation couple, while in 1D is just a reflection.
        Second, the function for permuting an array of points using
        the permutations.
    """
    perm_args = []

    tdim = cell.topological_dimension()
    if tdim == 1:
        perm_args.append([0])

        def identity(points, perm):
            return points

        permute_func = identity
    elif tdim == 2:
        perm_args.append([0])
        perm_args.append([1])
        permute_func = permute_quadrature_interval
    else:
        assert tdim == 3
        cell_type = cell.cellname()
        if cell_type == "tetrahedron":
            permute_func = permute_quadrature_triangle
            num_sides = 3
        else:
            assert cell_type == "hexahedron"
            permute_func = permute_quadrature_quadrilateral
            num_sides = 4

        for rot in range(num_sides):
            for ref in [0, 1]:
                perm_args.append([ref, rot])

    return perm_args, permute_func


def _evaluate_element(
    element: BasixElement,
    points: npt.NDArray[np.float64],
    integral_type: str,
    avg: Optional[str],
    entity_type: str,
    local_derivatives: tuple[int, ...],
    flat_component: int,
    is_mixed_dim: bool,
    codim: int,
) -> npt.NDArray[np.float64]:
    """Evaluates the given `element` for creating FE table values, given
    some additional inputs.

    This piece of code was initially copied from
    ``ffcx.ir.elementtables.build_optimized_tables``,
    and subsequently modified.

    Args:
        element (BasixElement): Element whose basis functions are
            evaluated. It may be a vector element (its corresponding
            component is specified by `flat_component`).
        points (npt.NDArray[np.float64]): Points (in the reference
            domain of the element) where the derivatives are computed.
            In the case of facet `integral_type`, these points will be
            referred to the facet reference domain.
        integral_type (str): Integral type. It may be ``cell``,
            ``interior_facet``, or ``exterior_facet``.
        avg (Optional[str]): String indicating indicating if basis
            functions average must be computed. If defined, it only
            can be ``cell`` or ``facet``.
        entity_type (str): Entity type. It may be ``facet`` or ``cell``.
            ``vertex`` is not allowed.
        local_derivatives (tuple[int, ...]): Order of the derivatives
            along the parametric directions to compute. If 0, no
            derivative is computed along that specific direction.
        flat_component (int): Component of the basis to compute.
        is_mixed_dim (bool): Flag indicating if the integral to which
            the element evaluation is associated to presents mixed
            dimensions or not.
        codim (int): Codimension of the evaluation domain respect
            to the element.

    Returns:
        npt.NDArray[np.float64]: Computed values.
    """

    def get_ffcx_table_values(
        points: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return ffcx.ir.elementtables.get_ffcx_table_values(
            points,
            cell,
            integral_type,
            element,
            avg,
            entity_type,
            local_derivatives,
            flat_component,
            codim,
        )["array"]

    cell = element.cell
    tdim = cell.topological_dimension()

    # Code adapted from ffcx.ir.elementtables.py build_optimized_tables

    # Only permute quadrature rules for interior facets integrals and
    # for the codim zero element in mixed-dimensional integrals. The
    # latter is needed because a cell may see its sub-entities as being
    # oriented differently to their global orientation
    if integral_type == "interior_facet" or (is_mixed_dim and codim == 0):
        # Do not add permutations if codim=1 as facets have already
        # gotten a global orientation in DOLFINx
        if 2 <= tdim and codim == 0:
            perm_args, permute_func = _get_cell_permutations(cell)

            new_table = []
            for perm_arg in perm_args:
                perm_points = permute_func(points, *perm_arg)
                new_table.append(get_ffcx_table_values(perm_points))

            return np.vstack(new_table)

    return get_ffcx_table_values(points)


def _format_table_values(
    tbl: npt.NDArray[np.float64],
    table_type: Optional[str] = None,
    is_permuted: Optional[bool] = None,
    rtol: float = default_rtol,
    atol: float = default_atol,
) -> npt.NDArray[np.float64]:
    """Formats the values for a FE table.

    This piece of code was copied from
    `ffcx.ir.elementtables.build_optimized_tables`, adapting it for the
    case in which sum factorization is not considered, and for accepting
    new arguments `tabletype` and `is_permuted`.

    Args:
        tbl (npt.NDArray[np.float64]): 4-dimensional array containing
            the table values.
        table_type (str, optional): Type of table, i.e., ``fixed``,
            ``piecewise``, ``uniform``, or ``varying``. If not provided,
            the table type is computed, what may be expensive (even
            more than computing the table itself). Defaults to None.
        is_permuted (bool, optional): Whether or not the table must
            contain permutations. If not provided, it is computed.
            Defaults to None.
        rtol (float, optional): Relative tolerance used for clamping
            near zero values. Defaults to default_rtol.
        atol (float, optional): Absolute tolerance used for clamping
            near zero values. Defaults to default_atol.

    Returns:
        npt.NDArray[np.float64]: Formated values.
    """

    assert len(tbl.shape) == 4

    # Clean up table
    if table_type is None or is_permuted is None:
        tbl = clamp_table_small_numbers(tbl, rtol=rtol, atol=atol)
        table_type = analyse_table_type(tbl)

    if table_type in piecewise_ttypes:
        # Reduce table to dimension 1 along num_points axis in
        # generated code
        tbl = tbl[:, :, :1, :]

    if table_type in uniform_ttypes:
        # Reduce table to dimension 1 along num_entities axis in
        # generated code
        tbl = tbl[:, :1, :, :]

    if is_permuted is None:
        is_permuted = is_permuted_table(tbl)

    if not is_permuted:
        # Reduce table along num_perms axis
        tbl = tbl[:1, :, :, :]

    return tbl


def _check_groupable_tables(table_0: FETable, table_1: FETable) -> bool:
    """Checks if the two given tables are groupable. We say that they
    are groupable if the share the same base element, they have same
    average, entity type, integral type, number of permuations and
    entities. Their component may differ.

    Args:
        table_0 (FETable): First table to compare.
        table_1 (FETable): Second table to compare.

    Returns:
        bool: ``True`` if both tables are groupable, ``False``
        otherwise.
    """

    sub_elems = []
    for table in [table_0, table_1]:
        elem = table.element
        if elem.block_size == 1:
            sub_elems.append(elem)
        else:
            sub_elem, _, _ = elem.get_component_element(table.component)
            sub_elems.append(sub_elem)

    return (
        sub_elems[0] == sub_elems[1]
        and table_0.avg == table_1.avg
        and table_0.entity == table_1.entity
        and table_0.integral_type == table_1.integral_type
        and table_0.permutations == table_1.permutations
        and table_0.entities == table_1.entities
    )


def _group_tables(FE_tables: list[FETable]) -> list[list[FETable]]:
    """Groups the given `FE_tables` such that all the tables in a group
    have the same base element.

    It disregards tables whose values are constant for all the points.

    Args:
        FE_tables (list[FETable]): List of FE tables to be grouped.

    Returns:
        list[list[FETable]]: Groups of FE tables. Each entry of the list
        correspond to one group.
    """

    groups = []
    for FE_table in FE_tables:
        if FE_table.is_constant_for_pts():
            continue

        found = False
        for group in groups:
            if _check_groupable_tables(FE_table, group[0]):
                group.append(FE_table)
                found = True
                break

        if not found:
            groups.append([FE_table])

    return groups


def _evaluate_FE_tables_same_element(
    fe_tables: list[FETable],
    points: npt.NDArray[np.float64],
) -> dict[FETable, npt.NDArray[np.float64]]:
    # If the table average (cell or facet) is computed, then, the
    # values should be constant for all the points in an entity -
    # permutation pair, and therefore this function should not be
    # called.

    values = {}

    assert len(fe_tables) > 0
    ref_fe_table = fe_tables[0]

    if ref_fe_table.avg is not None:
        # If the table average (cell or facet) is computed, then, the
        # values should be constant for all the points in an entity -
        # permutation pair, and therefore there is no need to recompute
        # them.
        for fe_table in fe_tables:
            assert fe_table.points == 1
            values[fe_table] = fe_table.values
        return values

    all_derivatives = [table.derivatives for table in fe_tables]

    raw_vals = _evaluate_scalar_element_derivatives(
        ref_fe_table.element,
        points,
        ref_fe_table.avg,
        all_derivatives,
    )

    for fe_table in fe_tables:
        vals = raw_vals[fe_table.derivatives]

        shape = vals.shape
        assert len(shape) == 4 and shape[0] == 1 and shape[1] == 1

        values[fe_table] = vals.reshape(shape[2], shape[3])

    return values


def evaluate_FE_table(
    fe_table: FETable,
    element: BasixElement,
    is_mixed_dim: bool,
    codim: int,
    tabletype: Optional[str] = None,
    is_permuted: Optional[bool] = None,
    rtol: float = default_rtol,
    atol: float = default_atol,
) -> npt.NDArray[np.float64]:
    """Creates the values associated to a FE table.

    This piece of code was copied from the function
    ``ffcx.ir.elementtables.build_optimized_tables``, adapting it for
    the case in which sum factorization is not considered, and for
    accepting new arguments `tabletype` and `is_permuted`.

    Args:
        fe_table (FETable): FE table whose associated values are going
            to be recomputed using the extra provided information in
            this function.
        element (BasixElement): Element to be used for the table
            evaluation.
        is_mixed_dim (bool): Flag indicating if the integral to which
            the element evaluation is associated to presents mixed
            dimensions or not.
        codim (int): Codimension of the evaluation domain respect
            to the element.
        tabletype (str, optional): Type of table, i.e., "fixed",
            "piecewise", "uniform", or "varying". If not provided,
            the table type is computed, what may be expensive (even
            more than computing the table itself). Defaults to None.
        is_permuted (bool, optional): Whether or not the table must
            contain permutations. If not provided, it is computed.
            Defaults to None.
        rtol (float, optional): Relative tolerance used for clamping
            near zero values. Defaults to default_rtol.
        atol (float, optional): Absolute tolerance used for clamping
            near zero values. Defaults to default_atol.

    Returns:
        npt.NDArray[np.float64]: Generated values in a 4-dimensional
        `numpy` array.
    """

    vals = _evaluate_element(
        element,
        fe_table._quad_data.rule.points,  # type: ignore
        fe_table.integral_type,
        fe_table.avg,
        fe_table.entity,
        fe_table.derivatives,
        fe_table.component,
        is_mixed_dim,
        codim,
    )

    return _format_table_values(vals, tabletype, is_permuted, rtol, atol)


def evaluate_FE_tables(
    fe_tables: list[FETable],
    points: npt.NDArray[np.float64],
    rtol: float = default_rtol,
    atol: float = default_atol,
) -> dict[FETable, npt.NDArray[np.float64]]:
    """Evaluates the given the elements of FE `tables` at the given
    `points`.

    Args:
        fe_tables (list[FETable]): List of tables whose elements' basis
            functions are evaluated. The tables provide extra
            information about which derivative to evaluate, which
            component and entity, if quantities, must be averaged, etc.
        points (npt.NDArray[np.float64]): Points at which the elements
            are evaluated. It is a 2D array that has as many rows a
            points and as many columns as coordinates. The number
            of coordinates is equal to the parametric dimension of the
            associated element.
        rtol (float, optional): Relative tolerance used for clamping
            values near zero in the generated table. Defaults to
            default_rtol.
        atol (float, optional): Absolute tolerance used for clamping
            values near zero in the generated table. Defaults to
            default_atol.

    Note:
        The implementation detects tables that share the same element,
        but associated to different derivatives, in order to minimize
        the number of evaluations to performed.

    Returns:
        dict[FETable, npt.NDArray[np.float64]]: Generated values for all
        the tables (the dictionary maps tables to values).
        Tables whose values are constants for all the points are
        excluded from the dictionary.
    """

    values_dict = {}
    if points.size == 0:
        for table in fe_tables:
            values_dict[table] = np.empty([0, table.funcs], dtype=table.dtype)
        return values_dict

    # Grouping tables with common elements for minimizing the number of
    # evaluations
    values = {}
    for group in _group_tables(fe_tables):
        values.update(_evaluate_FE_tables_same_element(group, points))

    for table in fe_tables:
        if not table.is_constant_for_pts():
            vals = clamp_table_small_numbers(values[table], rtol=rtol, atol=atol)
            values_dict[table] = vals

    return values_dict
