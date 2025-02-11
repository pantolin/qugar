# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------


from typing import TypeAlias

import qugar.cpp as cpp
from qugar.unfitted_domain import UnfittedDomain


class UnfDomainReparamMesh:
    """A class to represent the unfitted domain raparameterization mesh wraps around a C++ object."""

    ReparamMesh: TypeAlias = (
        cpp.ReparamMesh_1_2 | cpp.ReparamMesh_2_2 | cpp.ReparamMesh_2_3 | cpp.ReparamMesh_3_3
    )

    def __init__(self, cpp_object: ReparamMesh) -> None:
        """
        Initializes the Reparam object with a given C++ ReparamMesh object.

        Args:
            cpp_object (ReparamMesh): An instance of a C++ ReparamMesh object, either 2D or 3D.
        """
        self._cpp_object = cpp_object

    @property
    def cpp_object(self) -> ReparamMesh:
        """
        Returns the C++ object associated with this instance.

        Returns:
            ReparamMesh: The C++ object underlying this instance.
        """
        return self._cpp_object


def create_reparam_mesh(
    unf_domain: UnfittedDomain, n_pts_dir: int = 4, levelset: bool = False
) -> UnfDomainReparamMesh:
    """
    Creates a reparameterized mesh for a given unfitted domain.

    Args:
        unf_domain (UnfittedDomain): The unfitted domain to be reparameterized.
        n_pts_dir (int, optional): The number of points in each direction.
            Must be greater than 1. Defaults to 4.
        levelset (bool, optional): Whether to create a levelset reparameterization (True)
            or a solid one (False). Defaults to False.

    Returns:
        UnfDomainReparam: The reparameterized domain.
    """
    assert n_pts_dir > 1, "The number of points in each direction must be greater than 1"

    algo = cpp.create_reparameterization_levelset if levelset else cpp.create_reparameterization
    return UnfDomainReparamMesh(algo(unf_domain.cpp_object, n_pts_dir))
