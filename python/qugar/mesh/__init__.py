# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Mesh module for QUGaR"""

from qugar.mesh.utils import (
    DOLFINx_to_lexicg_edges,
    DOLFINx_to_lexicg_faces,
    DOLFINx_to_lexicg_nodes,
    DOLFINx_to_VTK_nodes,
    VTK_to_DOLFINx_nodes,
    VTK_to_lexicg_faces,
    VTK_to_lexicg_nodes,
    lexicg_to_DOLFINx_edges,
    lexicg_to_DOLFINx_faces,
    lexicg_to_DOLFINx_nodes,
    lexicg_to_VTK_faces,
    lexicg_to_VTK_nodes,
)

__all__ = [
    "DOLFINx_to_lexicg_edges",
    "DOLFINx_to_lexicg_faces",
    "DOLFINx_to_lexicg_nodes",
    "DOLFINx_to_VTK_nodes",
    "VTK_to_DOLFINx_nodes",
    "VTK_to_lexicg_nodes",
    "VTK_to_lexicg_faces",
    "lexicg_to_DOLFINx_edges",
    "lexicg_to_DOLFINx_faces",
    "lexicg_to_VTK_faces",
    "lexicg_to_DOLFINx_nodes",
    "lexicg_to_VTK_nodes",
]

from qugar.utils import has_FEniCSx

if has_FEniCSx:
    from qugar.mesh.tp_index import TensorProdIndex
    from qugar.mesh.tp_mesh import CartesianMesh, TensorProductMesh, create_Cartesian_mesh
    from qugar.mesh.utils import (
        create_cells_to_facets_map,
        map_cells_and_local_facets_to_facets,
        map_facets_to_cells_and_local_facets,
    )

    __all__ += [
        "CartesianMesh",
        "TensorProductMesh",
        "TensorProdIndex",
        "create_cells_to_facets_map",
        "map_cells_and_local_facets_to_facets",
        "map_facets_to_cells_and_local_facets",
        "create_Cartesian_mesh",
    ]
