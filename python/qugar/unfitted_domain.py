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
from qugar import has_FEniCSx, has_VTK

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import dolfinx
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
from dolfinx.mesh import MeshTags

import qugar.cpp
from qugar.cpp import UnfittedDomain_2D, UnfittedDomain_3D
from qugar.mesh import CartesianMesh, DOLFINx_to_lexicg_faces
from qugar.mesh.utils import create_cells_to_facets_map


class UnfittedDomain:
    """Class for storing an unfitted domain
    and access its cut, full, and empty cells and facets.
    """

    def __init__(
        self,
        cart_mesh: CartesianMesh,
        cpp_object: UnfittedDomain_2D | UnfittedDomain_3D,
    ) -> None:
        """Constructor.

        Args:
            cart_mesh (CartesianMesh): Cartesian tensor-product mesh
                for which the cell decomposition is generated.
            cpp_object (UnfittedDomain_2D | UnfittedDomain_3D):
                Already generated unfitted domain binary object.
        """

        assert cpp_object.dim == cart_mesh.tdim, "Non-matching dimensions."
        assert cart_mesh.tdim == cart_mesh.gdim, "Mesh must have co-dimension 0."
        assert cart_mesh.cpp_object == cpp_object.grid, "Non-matching grids."

        self._cart_mesh = cart_mesh
        self._cpp_object = cpp_object

    @property
    def cart_mesh(self) -> CartesianMesh:
        """Gets the stored Cartesian mesh.

        Returns:
            CartesianMesh: Stored Cartesian mesh.
        """
        return self._cart_mesh

    @property
    def cpp_object(self) -> UnfittedDomain_2D | UnfittedDomain_3D:
        """Gets the internal C++ unfitted domain.

        Returns:
            UnfittedDomain_2D | UnfittedDomain_3D:
            Internal C++ unfitted domain.
        """
        return self._cpp_object

    def _get_cells(
        self, lex_cell_ids: npt.NDArray[np.int32], lexicg: bool
    ) -> npt.NDArray[np.int32]:
        """Transforms the given `cell_ids`.

        Args:
            lex_cell_ids (npt.NDArray[np.int32]): Lexicographical cell
                ids to transform.
            lexicg (bool: If `True`, the returned indices follow the
                lexicographical ordering of the Cartesian mesh.
                Otherwise, they correspond to DOLFINx local numbering of
                the current submesh.

        Returns:
            npt.NDArray[np.int32]: Transformed indices.
        """

        if lexicg:
            return lex_cell_ids
        else:
            return cast(
                npt.NDArray[np.int32],
                self.cart_mesh.get_DOLFINx_local_cell_ids(lex_cell_ids, lexicg=True),
            )

    def get_cut_cells(self, lexicg: bool = False) -> npt.NDArray[np.int32]:
        """Gets the ids of the cut cells.

        Args:
            lexicg (bool, optional): If `True`, the returned indices
                follow the lexicographical ordering of the Cartesian
                mesh. Otherwise, they correspond to DOLFINx local
                numbering of the current submesh. Defaults to `False`.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process.
        """

        return self._get_cells(self._cpp_object.cut_cells, lexicg)

    def get_full_cells(self, lexicg: bool = False) -> npt.NDArray[np.int32]:
        """Gets the ids of the full cells.

        Args:
            lexicg (bool, optional): If `True`, the returned indices
                follow the lexicographical ordering of the Cartesian
                mesh. Otherwise, they correspond to DOLFINx local
                numbering of the current submesh. Defaults to `False`.

        Returns:
            npt.NDArray[np.int32]: Array of full cells associated to the
            current process.
        """
        return self._get_cells(self._cpp_object.full_cells, lexicg)

    def get_empty_cells(self, lexicg: bool = False) -> npt.NDArray[np.int32]:
        """Gets the ids of the empty cells.

        Args:
            lexicg (bool, optional): If `True`, the returned indices
                follow the lexicographical ordering of the Cartesian
                mesh. Otherwise, they correspond to DOLFINx local
                numbering of the current submesh. Defaults to `False`.

        Returns:
            npt.NDArray[np.int32]: Array of empty cells associated to
            the current process.
        """
        return self._get_cells(self._cpp_object.empty_cells, lexicg)

    def _get_facets(
        self,
        lex_cells: npt.NDArray[np.int32],
        lex_facets: npt.NDArray[np.int32],
        facet_ids: bool,
        lexicg: bool,
    ) -> npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Transforms the given facets indices.

        Args:
            lex_cells (npt.NDArray[np.int32]): Indices of the facets
                following the lexicographical ordering. All the cell
                indices must be associated to the current process.

            lex_facets (npt.NDArray[np.int32]): Local indices of the
                facets referred to `cell_ids` (both arrays should have
                the same length). The face ids follow the
                lexicographical ordering.

            facet_ids (bool, optional): If `True`, the DOLFINx (local)
                face indices will be returned. Otherwise, the facets are
                returned as one array of cells and another one of local
                facets referred to those cells. Defaults to `True`.

            lexicg (bool, optional): In the case in which `facet_ids` is
                set to `False`, `lexicg` describes the ordering for the
                returned cells and local facets. If `lexicg` is set to
                `True`, cell indices and local facets follow the
                lexicographical ordering of the Cartesian mesh.
                Otherwise, they correspond to DOLFINx local numbering
                of the current submesh. If `facet_ids` is set to `True`,
                this options does not play any role.
                Defaults to `False`.

        Returns:
            npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Transformed facet indices. If `facet_ids` is set to `True`,
            it returns the DOLFINx (local) facet indices associated to
            the current proces. Otherwise, the facets are returned as
            one array of cells and another one of local facets referred
            to those cells. In the latter case, the indices of the cells
            and local facets may follow the lexicographical ordering if
            `lexicg` is set to `True`. Otherwise, they follow the
            DOLFINx numbering.
        """
        if not facet_ids and lexicg:
            return lex_cells, lex_facets

        tdim = self._cart_mesh.tdim
        dlf_to_lex_facets = DOLFINx_to_lexicg_faces(tdim)

        dlf_cells = self._get_cells(lex_cells, lexicg=False)
        dlf_facets = dlf_to_lex_facets[lex_facets]
        if not facet_ids:
            return dlf_cells, dlf_facets

        cells_to_facets = create_cells_to_facets_map(self._cart_mesh.dolfinx_mesh)
        return cells_to_facets[dlf_cells, dlf_facets]

    def get_cut_facets(
        self, facet_ids: bool = True, lexicg: bool = False
    ) -> npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the the cut facets.

        Args:
            facet_ids (bool, optional): If `True`, the DOLFINx (local)
                face indices will be returned. Otherwise, the facets are
                returned as one array of cells and another one of local
                facets referred to those cells. Defaults to `True`.

            lexicg (bool, optional): In the case in which `facet_ids` is
                set to `False`, `lexicg` describes the ordering for the
                returned cells and local facets. If `lexicg` is set to
                `True`, cell indices and local facets follow the
                lexicographical ordering of the Cartesian mesh.
                Otherwise, they correspond to DOLFINx local numbering
                of the current submesh. If `facet_ids` is set to `True`,
                this options does not play any role.
                Defaults to `False`.

        Returns:
            npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Cut facets. If `facet_ids` is set to `True`, it returns
            the DOLFINx (local) facet indices associated to the current
            proces. Otherwise, the facets are returned as one array of
            cells and another one of local facets referred to those
            cells. In the latter case, the indices of the cells and
            local facets may follow the lexicographical ordering if
            `lexicg` is set to `True`. Otherwise, they follow the
            DOLFINx numbering.
        """

        lex_cells, lex_facets = self._cpp_object.cut_facets
        return self._get_facets(lex_cells, lex_facets, facet_ids, lexicg)

    def get_full_facets(
        self, facet_ids: bool = True, lexicg: bool = False
    ) -> npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the the full facets.

        Args:
            facet_ids (bool, optional): If `True`, the DOLFINx (local)
                face indices will be returned. Otherwise, the facets are
                returned as one array of cells and another one of local
                facets referred to those cells. Defaults to `True`.

            lexicg (bool, optional): In the case in which `facet_ids` is
                set to `False`, `lexicg` describes the ordering for the
                returned cells and local facets. If `lexicg` is set to
                `True`, cell indices and local facets follow the
                lexicographical ordering of the Cartesian mesh.
                Otherwise, they correspond to DOLFINx local numbering
                of the current submesh. If `facet_ids` is set to `True`,
                this options does not play any role.
                Defaults to `False`.

        Returns:
            npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Full facets. If `facet_ids` is set to `True`, it returns
            the DOLFINx (local) facet indices associated to the current
            proces. Otherwise, the facets are returned as one array of
            cells and another one of local facets referred to those
            cells. In the latter case, the indices of the cells and
            local facets may follow the lexicographical ordering if
            `lexicg` is set to `True`. Otherwise, they follow the
            DOLFINx numbering.
        """

        lex_cells, lex_facets = self._cpp_object.full_facets
        return self._get_facets(lex_cells, lex_facets, facet_ids, lexicg)

    def get_empty_facets(
        self, facet_ids: bool = True, lexicg: bool = False
    ) -> npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Gets the the empty facets.

        Args:
            facet_ids (bool, optional): If `True`, the DOLFINx (local)
                face indices will be returned. Otherwise, the facets are
                returned as one array of cells and another one of local
                facets referred to those cells. Defaults to `True`.

            lexicg (bool, optional): In the case in which `facet_ids` is
                set to `False`, `lexicg` describes the ordering for the
                returned cells and local facets. If `lexicg` is set to
                `True`, cell indices and local facets follow the
                lexicographical ordering of the Cartesian mesh.
                Otherwise, they correspond to DOLFINx local numbering
                of the current submesh. If `facet_ids` is set to `True`,
                this options does not play any role.
                Defaults to `False`.

        Returns:
            npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            Empty facets. If `facet_ids` is set to `True`, it returns
            the DOLFINx (local) facet indices associated to the current
            proces. Otherwise, the facets are returned as one array of
            cells and another one of local facets referred to those
            cells. In the latter case, the indices of the cells and
            local facets may follow the lexicographical ordering if
            `lexicg` is set to `True`. Otherwise, they follow the
            DOLFINx numbering.
        """

        lex_cells, lex_facets = self._cpp_object.empty_facets
        return self._get_facets(lex_cells, lex_facets, facet_ids, lexicg)

    def create_cell_tags(self, cut_tag: int = 0, full_tag: int = 1, empty_tag: int = 2) -> MeshTags:
        """Creates a `MeshTags` objects containing the cut, full, and
        empty cells.

        Args:
            cut_tag (int, optional): Tag to be used for the cut cells.
                Defaults to 0.
            full_tag (int, optional): Tag to be used for the full cells.
                Defaults to 1.
            empty_tag (int, optional): Tag to be used for the empty
                cells. Defaults to 2.

        Returns:
            MeshTags: Generated mesh tags.
        """

        cut_cells = self.get_cut_cells(lexicg=False)
        full_cells = self.get_full_cells(lexicg=False)

        cells = np.arange(self._cart_mesh.num_local_cells, dtype=np.int32)
        cells_values = np.full(cells.size, empty_tag, dtype=np.int32)
        cells_values[cut_cells] = cut_tag
        cells_values[full_cells] = full_tag

        tdim = self._cart_mesh.tdim
        return dolfinx.mesh.meshtags(self._cart_mesh.dolfinx_mesh, tdim, cells, cells_values)

    def create_facet_tags(
        self,
        cut_tag: int = 0,
        full_tag: int = 1,
        empty_tag: int = 2,
    ) -> MeshTags:
        """Creates a `MeshTags` objects containing the cut, full, and
        empty facets.

        Args:
            cut_tag (int, optional): Tag to be used for the cut facets.
                Defaults to 0.
            full_tag (int, optional): Tag to be used for the full
                facets. Defaults to 1.
            empty_tag (int, optional): Tag to be used for the empty
                facets. Defaults to 2.

        Returns:
            MeshTags: Generated mesh tags.
        """

        cut_facets = self.get_cut_facets(facet_ids=True)
        full_facets = self.get_full_facets(facet_ids=True)

        topology = self._cart_mesh.dolfinx_mesh.topology
        tdim = topology.dim
        topology.create_connectivity(tdim - 1, tdim)
        conn = topology.connectivity(tdim - 1, tdim)
        n_facets = conn.num_nodes

        facets = np.arange(n_facets, dtype=np.int32)
        facets_values = np.full_like(facets, empty_tag, dtype=np.int32)

        facets_values[full_facets] = full_tag
        facets_values[cut_facets] = cut_tag

        return dolfinx.mesh.meshtags(self._cart_mesh.dolfinx_mesh, tdim - 1, facets, facets_values)

    if has_VTK:
        from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet

        def quadrature_to_VTK(self, n_pts_dir: int = 4) -> vtkMultiBlockDataSet:
            """
            Generates quadrature data for an unfitted domain and exports it to a
            VTK data structure. It generates, and export, quadrature points
            for the interior of the cells, the interior boundaries, and the facets.

            Args:
                n_pts_dir (int, optional): Number of points per direction for quadrature.
                    Defaults to 4.

            Returns:
                vtkMultiBlockDataSet:
                    A VTK multiblock dataset containing the quadrature data.
            """
            return qugar.vtk.quadrature_to_VTK(self, n_pts_dir)

        def quadrature_to_VTK_file(self, name, n_pts_dir: int = 4):
            """
            Generates quadrature data for an unfitted domain, exports it to a
            VTK data structure and dumps it into a file.

            Args:
                name (str): The name of the file to write the VTK data to (without the extension).
                n_pts_dir (int, optional): Number of points per direction for quadrature.
                    Defaults to 4.
            """
            vtk_mb = self.quadrature_to_VTK(n_pts_dir)
            qugar.vtk.write_VTK_to_file(vtk_mb, name)
