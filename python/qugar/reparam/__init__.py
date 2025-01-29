# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Reparameterization module for QUGaR"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from qugar.reparam.reparam import UnfDomainReparamMesh, create_reparam_mesh

__all__ = [
    "UnfDomainReparamMesh",
    "create_reparam_mesh",
]
