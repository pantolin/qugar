# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Plot functionalities (using PyVista) for QUGaR"""

from qugar.utils import has_PyVista

if not has_PyVista:
    raise ValueError("PyVista installation not found is required.")

from qugar.plot.pyvista_tools import (
    cart_grid_tp_to_PyVista,
    quadrature_to_PyVista,
    reparam_mesh_to_PyVista,
    unfitted_domain_facets_to_PyVista,
    unfitted_domain_to_PyVista,
)

__all__ = [
    "quadrature_to_PyVista",
    "cart_grid_tp_to_PyVista",
    "unfitted_domain_to_PyVista",
    "unfitted_domain_facets_to_PyVista",
    "reparam_mesh_to_PyVista",
]
