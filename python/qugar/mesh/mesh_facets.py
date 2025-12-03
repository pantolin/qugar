# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""
MeshFacets: A helper class for managing facets of DOLFINx meshes.

This module provides the following functionalities:

- Concatenating facets from multiple instances.
- Computing intersections and differences of facets.
- Keeping unique facets.
- Transforming facets to the original mesh numbering.
- Creating MeshFacets instances from DOLFINx meshes for exterior,
  interior, or all facets.

Dependencies:
- numpy
- dolfinx (optional, for Mesh transformations)
- qugar.mesh.Mesh
"""

from typing import cast

import numpy as np
import numpy.typing as npt

import qugar.utils
from qugar.mesh.utils import (
    DOLFINx_to_lexicg_faces,
    lexicg_to_DOLFINx_faces,
)

if qugar.utils.has_FEniCSx:
    import dolfinx.mesh

    from qugar.mesh.mesh import Mesh


class MeshFacets:
    """Helper class for managing facets of meshes."""

    def __init__(
        self,
        cell_ids: npt.NDArray[np.int32 | np.int64],
        local_facet_ids: npt.NDArray[np.int32],
    ):
        """Initializes the MeshFacets with arrays of cell ids and local facet ids.

        In principle, the generated list of facets (cells and local facets) is not unique
        nor unique. The user can remove duplicates
        by calling the method `unique`.

        Args:
            cell_ids (npt.NDArray[np.int_]): Array of cell ids (can be np.int32 or np.int64).
            local_facet_ids (npt.NDArray[np.int32]): Array of local facet ids.
        """
        assert cell_ids.shape == local_facet_ids.shape, (
            "Cell ids and local facet ids must have the same shape."
        )

        self._cell_ids = cell_ids
        self._local_facet_ids = local_facet_ids

    @property
    def size(self) -> bool:
        """Returns number of facets."""
        return self._cell_ids.size

    @property
    def empty(self) -> bool:
        """Checks if the facets are empty."""
        return self.size == 0

    @property
    def cells_dtype(self) -> np.dtype:
        """Gets the data type of the cell ids."""
        return self._cell_ids.dtype

    @property
    def cell_ids(self) -> npt.NDArray[np.int32 | np.int64]:
        """Gets the cell ids."""
        return self._cell_ids

    @property
    def local_facet_ids(self) -> npt.NDArray[np.int32]:
        """Gets the local facet ids."""
        return self._local_facet_ids

    def __getitem__(self, index: int) -> tuple[int, int]:
        """Gets the cell id and local facet id at the specified index.

        Args:
            index (int): Index of the facet.

        Returns:
            tuple[int, int]: A tuple containing the cell id and local facet id.
        """
        return self._cell_ids[index], self._local_facet_ids[index]

    def unique(self):
        """Keeps only unique facets."""
        # Combine cell_ids and local_facet_ids into a 2D array
        # Ensure local_facet_ids has the same dtype as cell_ids for comparison
        combined = np.vstack((self._cell_ids, self._local_facet_ids.astype(self._cell_ids.dtype))).T

        # Find unique rows using np.unique along axis 0
        unique_combined = np.unique(combined, axis=0)

        # Update the internal arrays with the unique values, ensuring correct dtypes
        self._cell_ids = unique_combined[:, 0]  # This retains the original dtype of _cell_ids
        self._local_facet_ids = unique_combined[:, 1].astype(np.int32)  # Cast back to int32

    def as_array(self) -> npt.NDArray[np.int64 | np.int32]:
        """Transforms the stored cell ids and local facets into a 1D array.

        Returns:
            npt.NDArray[np.int64 | np.int32]: A 1D array of cell ids
            and local facet ids. The array is of the form
            [cell_id_1, local_facet_id_1, cell_id_2, local_facet_id_2, ...].
        """

        local_facet_ids = self._local_facet_ids.astype(self._cell_ids.dtype)

        return np.vstack((self._cell_ids, local_facet_ids)).T.ravel()

    def concatenate(self, other: "MeshFacets") -> "MeshFacets":
        """Concatenates the facets of another MeshFacets instance.

        Args:
            other (MeshFacets): Another MeshFacets instance.

        Returns:
            MeshFacets: A new MeshFacets instance with concatenated facets.
        """

        assert self.cells_dtype == other.cells_dtype, (
            "Cannot concatenate facets with different cell id types."
        )

        new_cell_ids = np.concatenate((self._cell_ids, other._cell_ids), dtype=self.cells_dtype)
        new_local_facet_ids = np.concatenate(
            (self._local_facet_ids, other._local_facet_ids), dtype=np.int32
        )
        return MeshFacets(new_cell_ids, new_local_facet_ids)

    def intersect(self, other: "MeshFacets", assume_unique: bool = False) -> "MeshFacets":
        """Computes the intersection of facets with another MeshFacets instance.

        Args:
            other (MeshFacets): Another MeshFacets instance.
            assume_unique (bool): If `True`, assumes that the facets are unique
                within each object (`self` and `other`). This can speed up
                the intersection calculation. Default is `False`.

        Returns:
            MeshFacets: A new MeshFacets instance with intersected facets.
        """

        assert self.cells_dtype == other.cells_dtype, (
            "Cannot compute intersection for facets with different cell id types."
        )

        # Combine cell_ids and local_facet_ids into structured arrays for set operations
        # Ensure local_facet_ids has the same dtype as cell_ids for comparison
        combined_dtype = [("cell", self._cell_ids.dtype), ("facet", self._cell_ids.dtype)]

        combined = np.empty(len(self._cell_ids), dtype=combined_dtype)
        combined["cell"] = self._cell_ids
        combined["facet"] = self._local_facet_ids.astype(self._cell_ids.dtype)

        other_combined = np.empty(len(other._cell_ids), dtype=combined_dtype)
        other_combined["cell"] = other._cell_ids
        other_combined["facet"] = other._local_facet_ids.astype(other._cell_ids.dtype)

        # Compute the intersection
        intersection_structured = np.intersect1d(
            combined, other_combined, assume_unique=assume_unique
        )

        # Extract cell_ids and local_facet_ids from the structured array
        intersect_cell_ids = intersection_structured["cell"]
        # Cast local_facet_ids back to int32
        intersect_local_facet_ids = intersection_structured["facet"].astype(np.int32)

        return MeshFacets(intersect_cell_ids, intersect_local_facet_ids)

    def difference(self, other: "MeshFacets", assume_unique: bool = False) -> "MeshFacets":
        """Computes the difference of facets with another MeshFacets instance.

        Args:
            other (MeshFacets): Another MeshFacets instance.
            assume_unique (bool): If `True`, assumes that the facets are
                both in the current object and in `other`.
                This can help speed up the difference calculation.
                Default is `False`.

        Returns:
            MeshFacets: A new MeshFacets instance with the difference of facets.
        """

        assert self.cells_dtype == other.cells_dtype, (
            "Cannot concatenate facets with different cell id types."
        )
        # Combine cell_ids and local_facet_ids into structured arrays for set operations
        # Ensure local_facet_ids has the same dtype as cell_ids for comparison
        combined_dtype = [("cell", self._cell_ids.dtype), ("facet", self._cell_ids.dtype)]
        combined = np.empty(len(self._cell_ids), dtype=combined_dtype)
        combined["cell"] = self._cell_ids
        combined["facet"] = self._local_facet_ids.astype(self._cell_ids.dtype)

        other_combined = np.empty(len(other._cell_ids), dtype=combined_dtype)
        other_combined["cell"] = other._cell_ids
        other_combined["facet"] = other._local_facet_ids.astype(other._cell_ids.dtype)

        # Compute the set difference
        difference_structured = np.setdiff1d(combined, other_combined, assume_unique=assume_unique)

        # Extract cell_ids and local_facet_ids from the structured array
        diff_cell_ids = difference_structured["cell"]
        # Cast local_facet_ids back to int32
        diff_local_facet_ids = difference_structured["facet"].astype(np.int32)

        return MeshFacets(diff_cell_ids, diff_local_facet_ids)

    def find(self, other: "MeshFacets") -> npt.NDArray[np.int32]:
        """Finds the positions of facets in `other` relative to `self`.

        Assumes that `self` contains unique facets. If not, the behavior for
        duplicate facets is undefined (it will return the index of the first
        occurrence found).

        Args:
            other (MeshFacets): Another MeshFacets instance whose facets are
                                expected to be a subset of `self`.

        Returns:
            npt.NDArray[np.int32]: An array of indices where each entry corresponds
                                   to the position of the facet (cell + local facet)
                                   of `other` in `self`.

        Raises:
            ValueError: If a facet in `other` is not found in `self`.
        """
        assert self.cells_dtype == other.cells_dtype, (
            "Cannot find facets with different cell id types."
        )

        # Create a mapping from (cell_id, local_facet_id) tuple to index in self
        # Using tuples as dict keys is efficient and handles different dtypes correctly.
        self_map = {
            (cell, facet): idx
            for idx, (cell, facet) in enumerate(zip(self._cell_ids, self._local_facet_ids))
        }

        # Find the indices for each facet in other
        indices = np.empty(other.size, dtype=np.int32)
        for i, (other_cell, other_facet) in enumerate(zip(other._cell_ids, other._local_facet_ids)):
            key = (other_cell, other_facet)
            idx = self_map.get(key)
            if idx is None:
                # If an element from other is not found, raise an error.
                # The previous implementation asserted this, here we make it explicit.
                raise ValueError(
                    f"Facet (cell={other_cell}, local_facet={other_facet}) "
                    f"from 'other' not found in 'self'."
                )
            indices[i] = idx

        return indices

    if qugar.utils.has_FEniCSx:

        def to_original(self, mesh: Mesh) -> "MeshFacets":
            """Transforms the cell ids and local facets ordering to the original
            numbering according to the given `mesh`.

            Args:
                mesh (Mesh): The Mesh object to use for the transformation.

            Returns:
                MeshFacets: A new MeshFacets instance with transformed cell ids and local facets.
            """
            assert self.cells_dtype == np.int32, (
                "Cannot transform to original numbering with cell ids of type "
                f"{self.cells_dtype}. Only int32 is supported."
            )

            tdim = mesh.topology.dim

            orig_cell_ids = mesh.get_original_cell_ids(cast(npt.NDArray[np.int32], self._cell_ids))
            lex_to_dlf_facets = lexicg_to_DOLFINx_faces(tdim)
            orig_local_facet_ids = lex_to_dlf_facets[self._local_facet_ids]

            return MeshFacets(orig_cell_ids, orig_local_facet_ids)

        def to_DOLFINx(self, mesh: Mesh) -> "MeshFacets":
            """Transforms the cell ids and local facets ordering to DOLFINx numbering
            according to the given `mesh`.

            Args:
                mesh (Mesh): The Mesh object to use for the transformation.

            Returns:
                MeshFacets: A new MeshFacets instance with transformed cell ids and local facets.
            """
            assert self.cells_dtype == np.int64, (
                "Cannot transform to DOLFINx numbering with cell ids of type "
                f"{self.cells_dtype}. Only int64 is supported."
            )

            dlf_cell_ids = mesh.get_DOLFINx_local_cell_ids(
                cast(npt.NDArray[np.int64], self._cell_ids)
            )
            dlf_to_orig_facets = DOLFINx_to_lexicg_faces(mesh.tdim)
            dlf_local_facet_ids = dlf_to_orig_facets[self._local_facet_ids]

            return MeshFacets(dlf_cell_ids, dlf_local_facet_ids)

        def get_facet_ids(self, mesh: Mesh) -> npt.NDArray[np.int32]:
            """Transforms the stored cell ids and local facets into DOLFINx facet ids,
            according to the given `mesh`.

            Args:
                mesh (Mesh): The Mesh object to use for the transformation.

            Returns:
                npt.NDArray[np.int32]: An array of unique DOLFINx facet ids corresponding
                                       to the stored cell/local facet pairs.
            """

            assert self.cells_dtype == np.int32, (
                "Cannot transform to original numbering with cell ids of type "
                f"{self.cells_dtype}. Only int32 is supported."
            )

            cells_to_facets = _create_cells_to_facets_map(mesh)
            return cells_to_facets[self._cell_ids, self._local_facet_ids]


if qugar.utils.has_FEniCSx:
    import dolfinx.mesh

    def _create_cells_to_facets_map(
        mesh: dolfinx.mesh.Mesh,
    ) -> npt.NDArray[np.int32]:
        """Creates a map that allows to find the facets ids in a mesh
        from their cell ids and the local facet ids referred to that
        cells.

        Args:
            mesh (dolfinx.mesh.Mesh): Mesh from which facet ids are
                extracted.

            Returns:
                npt.NDArray[np.int32]: Map from the cells and local facet
                ids to the facet ids. It is a 2D array where the first
                column corresponds to the cells and the second one to the
                local facet ids.

                Thus, given the cell and local facet ids of a particular
                facet, the facet id can be accessed as
                `facet_id = facets_map[cell_id, local_facet_id]`, where
                `facets_map` is the generated 2D array.
        """

        topology = mesh.topology
        tdim = topology.dim
        topology.create_connectivity(tdim, tdim - 1)
        conn = topology.connectivity(tdim, tdim - 1)

        return conn.array.reshape(conn.num_nodes, -1)

    def create_facets_from_ids(
        mesh: dolfinx.mesh.Mesh,
        facet_ids: npt.NDArray[np.int32],
        single_interior_facet: bool = False,
    ) -> MeshFacets:
        """Creates a MeshFacets instance from facet ids of a DOLFINx mesh.

        Args:
            mesh (dolfinx.mesh.Mesh): The mesh.
            facet_ids (npt.NDArray[np.int32]): Array of facet ids.
            single_interior_facet (bool): If `True`, only one cell and
                local facet is returned for each interior facet. If
                `False`, both cells and local facets are returned.
                This is useful for the case of interior facets, where
                the facet belongs to more than one cell. In that case,
                only one cell and local facet is returned for that
                particular facet. The one chosen depends on the way in
                which that information is stored in the mesh connectivity.

        Returns:
            MeshFacets: A MeshFacets instance with the specified facet ids.
            The entries in the manager are unique.
        """

        # TODO: this implementation can be likely improved.
        # Checking in DOLFINx code.

        topology = mesh.topology
        tdim = topology.dim
        topology.create_connectivity(tdim - 1, tdim)
        conn = topology.connectivity(tdim - 1, tdim)

        cells_to_facets = _create_cells_to_facets_map(mesh)

        n_cells = (
            facet_ids.size if single_interior_facet else facet_ids.size * 2
        )  # This is an overestimation

        cells = np.zeros(n_cells, dtype=np.int32)
        local_facets = np.zeros(n_cells, dtype=np.int32)

        i = 0
        for facet in facet_ids:
            facet_cells = conn.links(facet)
            if single_interior_facet:
                facet_cells = facet_cells[:1]

            for cell in facet_cells:
                cells[i] = cell
                local_facets[i] = np.where(cells_to_facets[cell] == facet)[0][0]
                i += 1

        if cells.size != i:
            cells = cells[:i]
            local_facets = local_facets[:i]

        return MeshFacets(
            cells,
            local_facets,
        )

    def create_exterior_mesh_facets(mesh: dolfinx.mesh.Mesh) -> MeshFacets:
        """Creates a MeshFacets instance including only the exterior facets of the mesh.

        Args:
            mesh (dolfinx.mesh.Mesh): The mesh.

        Returns:
            MeshFacets: A MeshFacets instance with exterior facets.
        """

        topology = mesh.topology
        topology.create_connectivity(topology.dim - 1, topology.dim)
        facet_ids = dolfinx.mesh.exterior_facet_indices(topology)
        return create_facets_from_ids(mesh, facet_ids, single_interior_facet=True)

    def create_interior_mesh_facets(
        mesh: dolfinx.mesh.Mesh, single_interior_facet: bool = False
    ) -> MeshFacets:
        """Creates a MeshFacets instance including only the interior facets of the mesh.

        Args:
            mesh (dolfinx.mesh.Mesh): The mesh.
            single_interior_facet (bool): If `True`, only one cell and
                local facet is returned for each interior facet. If
                `False`, both cells and local facets are returned.
                This is useful for the case of interior facets, where
                the facet belongs to more than one cell. In that case,
                only one cell and local facet is returned for that
                particular facet. The one chosen depends on the way in
                which that information is stored in the mesh connectivity.

        Returns:
            MeshFacets: A MeshFacets instance with interior facets.
        """

        topology = mesh.topology
        topology.create_connectivity(topology.dim - 1, topology.dim)
        ext_facets = dolfinx.mesh.exterior_facet_indices(topology)

        imap = topology.index_map(topology.dim - 1)
        all_facets = np.arange(imap.local_range[0], imap.local_range[1], dtype=np.int32)

        int_facets = np.setdiff1d(all_facets, ext_facets, assume_unique=True)

        return create_facets_from_ids(mesh, int_facets, single_interior_facet)

    def create_all_mesh_facets(
        mesh: dolfinx.mesh.Mesh, single_interior_facet: bool = False
    ) -> MeshFacets:
        """Creates a MeshFacets instance including all the facets of the mesh.

        Args:
            mesh (dolfinx.mesh.Mesh): The mesh.
            single_interior_facet (bool): If `True`, only one cell and
                local facet is returned for each interior facet. If
                `False`, both cells and local facets are returned.
                This is useful for the case of interior facets, where
                the facet belongs to more than one cell. In that case,
                only one cell and local facet is returned for that
                particular facet. The one chosen depends on the way in
                which that information is stored in the mesh connectivity.

        Returns:
            MeshFacets: A MeshFacets instance with all facets.
        """
        topology = mesh.topology
        imap = topology.index_map(topology.dim - 1)
        all_facets = np.arange(imap.local_range[0], imap.local_range[1], dtype=np.int32)

        return create_facets_from_ids(mesh, all_facets, single_interior_facet)
