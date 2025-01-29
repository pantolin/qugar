# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""VTK functionalities for QUGaR"""

from qugar.utils import has_VTK

if not has_VTK:
    raise ValueError("VTK installation not found is required.")

from qugar.vtk.vtk_tools import quadrature_to_VTK, reparam_to_VTK, write_VTK_to_file

__all__ = ["reparam_to_VTK", "quadrature_to_VTK", "write_VTK_to_file"]
