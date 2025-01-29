# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Quadrature module for QUGaR"""

from qugar.quad.quad_data import CellState, CustomQuad, CustomQuadFacet, CustomQuadIntBoundary
from qugar.utils import has_FEniCSx

__all__ = [
    "CellState",
    "CustomQuad",
    "CustomQuadFacet",
    "CustomQuadIntBoundary",
]


if has_FEniCSx:
    from qugar.quad.quad_generator import QuadGenerator, create_quadrature_generator

    __all__ += ["QuadGenerator", "create_quadrature_generator"]
