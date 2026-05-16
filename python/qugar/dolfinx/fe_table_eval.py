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

from typing import Optional, Tuple, cast

import basix.ufl
import numpy as np
import numpy.typing as npt
from basix.ufl import _ElementBase as ElementBase
from ffcx.element_interface import basix_index
from ffcx.ir.elementtables import (
    clamp_table_small_numbers,
    default_atol,
    default_rtol,
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
            tables[der] = np.reshape(avg_tbl, (1, 1, 1, *tbl.shape[1:]))
    else:
        for der, tbl in tables.items():
            tables[der] = tbl.reshape(1, 1, *tbl.shape)

    return tables


def _check_groupable_tables(table_0: FETable, table_1: FETable) -> bool:
    """Checks if the two given tables are groupable. We say that they
    are groupable if the share the same base element, they have same
    average, entity type, integral type, number of permutations and
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

    all_derivatives = list(set([table.derivatives for table in fe_tables]))

    element_ = ref_fe_table.element
    if element_.block_size > 1:
        # If the element is a vector element, we need to get the
        # sub-element for the component
        element_, _, _ = element_.get_component_element(ref_fe_table.component)
    raw_vals = _evaluate_scalar_element_derivatives(
        element_,
        points,
        ref_fe_table.avg,
        all_derivatives,
    )

    for fe_table in fe_tables:
        vals = raw_vals[fe_table.derivatives]

        shape = vals.shape
        assert (
            len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[-1] % fe_table.funcs == 0
        )

        if shape[-1] != fe_table.funcs:
            # Non-blocked vector elements (Raviart-Thomas, Nedelec, BDM,
            # ...): ``basix.ufl._BasixElement.tabulate`` flattens the
            # ``(n_basis_functions, value_size)`` tail into a single
            # axis in **component-first** order, i.e.
            # ``[phi0_c0, phi1_c0, ..., phi(n-1)_c0, phi0_c1, ...]``.
            # FFCx stores each component in its own FE table, so for a
            # ``fe_table`` referring to ``component = c`` we take the
            # ``c``-th slice along that axis.
            vals = vals.reshape(shape[2], -1, fe_table.funcs)
            vals = vals[:, fe_table.component, :]

        values[fe_table] = vals.reshape(shape[2], fe_table.funcs)

    return values


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
