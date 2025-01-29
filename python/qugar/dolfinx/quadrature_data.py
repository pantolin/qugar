# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Data structures and functionalities for quadratures."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from typing import NamedTuple

import ffcx.analysis
import numpy as np
import numpy.typing as npt
import ufl.algorithms.formdata
import ufl.integral
from ffcx.ir.representationutils import (
    QuadratureRule,
    create_quadrature_points_and_weights,
)


class QuadratureData(NamedTuple):
    """Class for storing inmutable infomation associated to a
    quadrature.

    Parameters:
        name (str): Name of the quadrature. Three alphanumeric
            characters derived from QuadratureRule.
        degree (int): Degree of exactness of the quadrature.
        interior_boundary (bool): Flag indicating if the quadrature
            corresponds to an interior custom boundary.
        rule (QuadratureRule): Quadrature rule of the class containing
            points coordinates and weights.
    """

    name: str
    degree: int
    interior_boundary: bool
    rule: QuadratureRule


def _extract_single_quadrature_data(
    ufl_integrand: ufl.integral.Integral,
    ufl_form: ufl.algorithms.formdata.FormData,
    ffcx_options: dict[str, int | float | npt.DTypeLike],
) -> QuadratureData:
    """Extract the quadrature data associated to the given
    `ufl_integrand`.

    Args:
        ufl_integrand (ufl.integral.Integral): UFL integrand from which
            the quadratures are extracted.
        ufl_form (ufl.algorithms.formdata.FormData): UFL form which
            contains the given `ufl_integrand`.
        ffcx_options (dict[str, int  |  float  |  npt.DTypeLike]): FFCx
            compilation options for the given UFL integrand (and form).

    Returns:
        QuadratureData: Extracted quadrature data.
    """

    use_sum_factorization = (
        bool(ffcx_options["sum_factorization"]) and ufl_integrand.integral_type == "cell"
    )

    metadata = ufl_integrand.metadata() or {}

    scheme = metadata["quadrature_rule"]
    assert scheme != "vertex"
    degree = int(metadata["quadrature_degree"])
    assert degree >= 0

    is_interior_bdry = (
        "custom_interior_boundary" in metadata and metadata["custom_interior_boundary"]
    )

    # Creating quadrature rule.
    if scheme == "custom":
        points = metadata["quadrature_points"]
        weights = metadata["quadrature_weights"]
        tensor_factors = None
    else:
        points, weights, tensor_factors = create_quadrature_points_and_weights(
            ufl_integrand.integral_type(),
            ufl_integrand.ufl_domain().ufl_cell(),
            degree,
            scheme,
            ufl_form.argument_elements,  # type: ignore
            use_sum_factorization,
        )

    points = np.asarray(points)
    weights = np.asarray(weights)
    rule = QuadratureRule(points, weights, tensor_factors)
    hash(rule)  # needed for calling rule.id()
    quad_name = rule.id()

    return QuadratureData(quad_name, degree, is_interior_bdry, rule)


def extract_quadrature_data(
    ufl_data: ffcx.analysis.UFLData,
    ffcx_options: dict[str, int | float | npt.DTypeLike],
) -> dict[str, QuadratureData]:
    """Extracts all the quadrature data associated to the given
    `ufl_data`.

    Args:
        ufl_data (ffcx.analysis.UFLData): UFL (analysis) data from which
            the quadrature data is extracted.
        ffcx_options (dict[str, int  |  float  |  npt.DTypeLike]): FFCx
            compilation options for the given UFL integrand (and form).

    Returns:
        dict[str, QuadratureData]: Map mapping quadrature names to
        quadrature data.
    """

    quads_data = {}

    for fd in ufl_data.form_data:
        for itg_data in fd.integral_data:
            for itg in itg_data.integrals:
                quad_data = _extract_single_quadrature_data(itg, fd, ffcx_options)
                quads_data[quad_data.name] = quad_data

    return quads_data
