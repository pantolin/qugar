# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from typing import cast

import qugar.utils

if not qugar.has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import numpy as np
import numpy.typing as npt

import qugar.cpp
from qugar.cpp import UnfittedImplDomain_2D, UnfittedImplDomain_3D
from qugar.impl import ImplicitFunc
from qugar.mesh import CartesianMesh
from qugar.unfitted_domain import UnfittedDomain


class UnfittedImplDomain(UnfittedDomain):
    """Class for storing an unfitted implicit domain
    and access its cut, full, and empty cells and facets.
    """

    def __init__(
        self,
        cart_mesh: CartesianMesh,
        cpp_object: UnfittedImplDomain_2D | UnfittedImplDomain_3D,
    ) -> None:
        """Constructor.

        Args:
            cart_mesh (CartesianMesh): Cartesian tensor-product mesh
                for which the cell decomposition is generated.
            cpp_object (UnfittedImplDomain_2D | UnfittedImplDomain_3D):
                Already generated unfitted implicit domain binary object.
        """

        super().__init__(cart_mesh, cpp_object)


def create_unfitted_impl_domain(
    impl_func: ImplicitFunc,
    cart_mesh: CartesianMesh,
) -> UnfittedImplDomain:
    """Creates a new unfitted implicit domain using an implicit function.

    Args:
        impl_func (ImplicitFunc_2D | ImplicitFunc_3D): Implicit function
            that describes the domain.
        cart_mesh (CartesianMesh): Cartesian tensor-product mesh for
            which the unfitted domain is generated.

    Returns:
        UnfittedImplDomain: Created unfitted implicit domain.
    """

    n_local_cells = cart_mesh.num_local_cells

    if n_local_cells == cart_mesh.num_global_cells:
        unf_domain = qugar.cpp.create_unfitted_impl_domain(
            impl_func.cpp_object, cart_mesh.cpp_object
        )
    else:
        dlf_cells = np.arange(n_local_cells, dtype=np.int32)
        lex_cells = cast(
            npt.NDArray[np.int64],
            cart_mesh.get_lexicg_cell_ids(dlf_cells, lexicg=False),
        )
        unf_domain = qugar.cpp.create_unfitted_impl_domain(
            impl_func.cpp_object, cart_mesh.cpp_object, lex_cells
        )

    return UnfittedImplDomain(cart_mesh, unf_domain)
