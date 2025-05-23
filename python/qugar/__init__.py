# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Main module for QUGaR"""

from qugar.utils import has_FEniCSx, has_PyVista  # noqa: I001
from qugar.cpp import __doc__, __version__

from qugar import mesh, quad

__all__ = ["__version__", "__doc__", "has_FEniCSx", "has_PyVista", "quad", "mesh"]

if has_FEniCSx:
    from qugar import dolfinx, impl, reparam

    __all__ += [
        "dolfinx",
        "impl",
        "reparam",
    ]

if has_PyVista:
    from qugar import plot

    __all__ += [
        "plot",
    ]
