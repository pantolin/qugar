# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------


"""Custom DOLFINx forms for runtime quadratures"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from qugar.dolfinx.boundary import ds_bdry_unf, mapped_normal
from qugar.dolfinx.forms import CustomForm, form_custom

__all__ = [
    "CustomForm",
    "ds_bdry_unf",
    "form_custom",
    "mapped_normal",
]
