# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tensor product indices."""

import numpy as np
import numpy.typing as npt

from qugar.mesh.utils import lexicg_to_DOLFINx_edges, lexicg_to_DOLFINx_faces


class TensorProdIndex:
    """Helper class for accessing indices in arbitrary dimensions
    tensor-product structures.

    The indices in the tensor product structure follow the
    lexicographical order, I.e., the direction 0 iterates the fastest,
    i.e., is the inner-most, and the direction dim-1 iterates the
    slowest, i.e., the outer-most. Where dim is the number of
    dimensions.

    I.e., in 2D this would be

        ^ y
        |
        8 --- 9 ---10 --- 11
        |     |     |     |
        4 --- 5 --- 6 --- 7
        |     |     |     |
        0 --- 1 --- 2 --- 3 --> x

    When requested for specific corner, face, or edge indices,
    the local index of the corresponding entity (corner, face, edge)
    can be specified in lexicographical ordering or following
    DOLFINx ordering.
    See https://github.com/FEniCS/basix/#supported-elements

    """

    @staticmethod
    def get_all_corners(n_inds: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
        """Extracts the indices of the 2^dim corners of the hypercube.

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.

        Returns:
            npt.NDArray[np.int64]: Array storing the sorted indices of
            the corners.
        """

        dim = len(n_inds)
        assert 1 <= dim <= 3

        if dim == 1:
            return np.array([0, n_inds[0] - 1], dtype=np.int64)
        elif dim == 2:
            corners_1D = TensorProdIndex.get_all_corners(n_inds[:-1])
            return np.hstack([corners_1D, corners_1D + n_inds[0] * (n_inds[1] - 1)]).astype(
                np.int64
            )
        else:  # if dim == 3:
            corners_2D = TensorProdIndex.get_all_corners(n_inds[:-1])
            return np.hstack(
                [corners_2D, corners_2D + n_inds[0] * n_inds[1] * (n_inds[2] - 1)]
            ).astype(np.int64)

    @staticmethod
    def get_all_internal(n_inds: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
        """Extracts the internal indices of the hypercube.

        The internal indices are those that are not on the boundaries.

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.

        Returns:
            npt.NDArray[np.int64]: Array storing the sorted internal
            indices.
        """

        dim = len(n_inds)
        assert 1 <= dim <= 3

        if dim == 1:
            return np.arange(1, n_inds[0] - 1, dtype=np.int64)
        elif dim == 2:
            internals_1D = TensorProdIndex.get_all_internal(n_inds[:-1])
            return internals_1D + np.arange(
                n_inds[0], n_inds[0] * (n_inds[1] - 1), n_inds[0], dtype=np.int64
            ).reshape(n_inds[1] - 2, 1)
        else:  # if dim == 3:
            internals_2D = TensorProdIndex.get_all_internal(n_inds[:-1])
            return internals_2D + np.arange(
                n_inds[0] * n_inds[1],
                n_inds[0] * n_inds[1] * (n_inds[2] - 1),
                n_inds[0] * n_inds[1],
                dtype=np.int64,
            ).reshape(n_inds[2] - 2, 1)

    @staticmethod
    def get_all_faces(n_inds: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
        """Extracts the face indices of the hypercube.

        The face indices are those that are on the boundaries.

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.

        Returns:
            npt.NDArray[np.int64]: Array storing the sorted boundary
            indices.
        """

        dim = len(n_inds)
        assert 1 <= dim <= 3

        if dim == 1:
            faces = TensorProdIndex.get_all_corners(n_inds)
        elif dim == 2:
            n_tot = 2 * n_inds[0] + 2 * n_inds[1] - 4
            faces = np.zeros(n_tot, dtype=np.int64)
            faces[0 : n_inds[0]] = TensorProdIndex.get_face(n_inds, face_id=2, lexicg=True)
            i = n_inds[0]
            for j in range(1, n_inds[1] - 1):
                faces[i] = j * n_inds[0]
                faces[i + 1] = (j + 1) * n_inds[0] - 1
                i += 2
            faces[-n_inds[0] :] = TensorProdIndex.get_face(n_inds, face_id=3, lexicg=True)
        else:  # if dim == 3:
            n_tot = 2 * n_inds[0] * n_inds[1] + 2 * (n_inds[0] + n_inds[1] - 2) * (n_inds[2] - 2)
            faces = np.zeros(n_tot, dtype=np.int64)
            n01 = np.prod(n_inds[0:2])

            faces[0:n01] = TensorProdIndex.get_face(n_inds, face_id=4, lexicg=True)

            i = n01
            bounds_2d = TensorProdIndex.get_all_faces(n_inds[0:2])
            m = len(bounds_2d)
            i = n01
            for k in range(1, n_inds[2] - 1):
                faces[i : i + m] = bounds_2d + k * n01
                i += m

            faces[-n01:] = TensorProdIndex.get_face(n_inds, face_id=5, lexicg=True)

        return faces

    @staticmethod
    def get_all_edges(n_inds: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
        """Extracts the edge indices of the hypercube.

        In 3D the edge indices are those that correspond to the
        intersection of two faces. In 1D and 2D, they are the same
        as the face indices.

        Warning:
            This method is not implemented yet for 3D.

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.

        Returns:
            npt.NDArray[np.int64]: Array storing the sorted edge
            indices.
        """
        dim = len(n_inds)
        assert 1 <= dim <= 3

        if dim < 3:
            return TensorProdIndex.get_all_faces(n_inds)
        else:  # if dim == 3:
            assert False

    @staticmethod
    def get_corner(n_inds: npt.NDArray[np.int32], local_corner_id: int | np.int32) -> np.int64:
        """Get the index associated to the given corner.

        Note:
            The local ordering of corners in DOLFINx coincides
            with the lexicographical ordering.
            See https://github.com/FEniCS/basix/#supported-elements

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.
            local_corner_id (int | np.int32): Corner whose index is
                extracted. It must be a value in the range `[0,2^dim[`
                following the lexicographical ordering.

        Returns:
            np.int64: Index of the corner.
        """

        dim = len(n_inds)
        assert 1 <= dim <= 3 and 0 <= local_corner_id < 2**dim

        return TensorProdIndex.get_all_corners(n_inds)[local_corner_id]

    @staticmethod
    def get_face(
        n_inds: npt.NDArray[np.int32], face_id: int | np.int32, lexicg=False
    ) -> npt.NDArray[np.int64]:
        """Get the indices associated to the given face.

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.
            face_id (int | np.int32): Face whose indices is extracted.
                It must be a value in the range [0, 2 * dim[, where
                `dim` is the dimension of the domain.
            lexicg (bool): If ``True``, the given `face_id` is referred
                to the lexicographical ordering, otherwise, to the
                DOLFINx one.
                See https://github.com/FEniCS/basix/#supported-elements

        Returns:
            npt.NDArray[np.int64]: Indices associated to the face.
        """

        dim = len(n_inds)
        assert 1 <= dim <= 3 and 0 <= face_id < 2 * dim

        if not lexicg:
            face_id = lexicg_to_DOLFINx_faces(dim)[face_id]

        if dim == 1:
            return np.array([TensorProdIndex.get_corner(n_inds, face_id)], dtype=np.int64)
        elif dim == 2:
            if face_id == 0:
                return np.arange(0, np.prod(n_inds), n_inds[0], dtype=np.int64)
            elif face_id == 1:
                return TensorProdIndex.get_face(n_inds, 0, lexicg=True) + n_inds[0] - 1
            elif face_id == 2:
                return np.arange(n_inds[0])
            else:  # if face_id == 3:
                return TensorProdIndex.get_face(n_inds, 2, lexicg=True) + n_inds[0] * (
                    n_inds[1] - 1
                )
        else:  # if dim == 3:
            if face_id == 0:
                return np.arange(0, np.prod(n_inds), n_inds[0], dtype=np.int64)
            elif face_id == 1:
                return TensorProdIndex.get_face(n_inds, 0, lexicg=True) + n_inds[0] - 1
            elif face_id == 2:
                return (
                    np.arange(n_inds[0], dtype=np.int64)
                    + np.arange(
                        0,
                        n_inds[0] * n_inds[1] * n_inds[2],
                        n_inds[0] * n_inds[1],
                        dtype=np.int64,
                    ).reshape(n_inds[2], 1)
                ).ravel()
            elif face_id == 3:
                return TensorProdIndex.get_face(n_inds, 2, lexicg=True) + n_inds[0] * (
                    n_inds[1] - 1
                )
            elif face_id == 4:
                return np.arange(n_inds[0] * n_inds[1], dtype=np.int64)
            else:  # if face_id == 5:
                face = TensorProdIndex.get_face(n_inds, 4, lexicg=True)
                face += n_inds[0] * n_inds[1] * (n_inds[2] - 1)
                return face

    @staticmethod
    def get_edge(
        n_inds: npt.NDArray[np.int32], edge_id: int | np.int32, lexicg=False
    ) -> npt.NDArray[np.int64]:
        """Get the indices associated to the given edge.

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.
            edge_id (int | np.int32): Local edge id whose indices are
                extracted.
            lexicg: If ``True``, the given `edge_id` is referred to
                the lexicographical ordering, otherwise, to the DOLFINx
                one.
                See https://github.com/FEniCS/basix/#supported-elements

        Returns:
            npt.NDArray[np.int64]: Indices associated to the edge.
        """

        dim = len(n_inds)
        assert 1 <= dim <= 3

        if not lexicg:
            edge_id = lexicg_to_DOLFINx_edges(dim)[edge_id]

        if dim < 3:
            return TensorProdIndex.get_face(n_inds, edge_id, lexicg=True)
        else:  # dim == 3:
            assert 0 <= edge_id < 12

            if edge_id < 4:
                i0 = TensorProdIndex.get_corner(n_inds, edge_id)
                return np.arange(i0, np.prod(n_inds), n_inds[0] * n_inds[1], dtype=np.int64)
            elif edge_id < 8:
                corner_id = [0, 1, 4, 5][edge_id - 4]
                i0 = TensorProdIndex.get_corner(n_inds, corner_id)
                return np.arange(i0, i0 + n_inds[0] * n_inds[1], n_inds[0], dtype=np.int64)
            else:  # if edge_id < 12:
                corner_id = [0, 2, 4, 6][edge_id - 8]
                i0 = TensorProdIndex.get_corner(n_inds, corner_id)
                return np.arange(i0, i0 + n_inds[0], 1, dtype=np.int64)

    @staticmethod
    def __create_dims_accumulation(
        n_inds: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int64]:
        """Creates the array of dimensions accumulation for transforming
        an id from tensor indices (according to the tensor-product
        directions) to the corresponding flat index (following the
        lexicographical convention), and viceversa.

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.

        Returns:
            npt.NDArray[np.int64]: Array of dimension accumulations. See
            the functions `get_tensor_index` and `get_flat_index` for an
            example of use.
        """

        dim = n_inds.size
        accum = np.ones(dim, dtype=np.int64)
        for dir in range(1, dim):
            accum[dir] = accum[dir - 1] * n_inds[dir - 1]
        return accum

    @staticmethod
    def get_flat_index(n_inds: npt.NDArray[np.int32], tid: npt.NDArray[np.int32]) -> np.int64:
        """Transform a given tensor index to a flat one considering
        `n_inds` indices per direction.

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.
            tid (npt.NDArray[np.int32]): Tensor indices to be
                transformed.

        Returns:
            np.int64: Computed flat index.
        """

        accum = TensorProdIndex.__create_dims_accumulation(n_inds)
        return np.dot(accum, tid)

    @staticmethod
    def get_tensor_index(
        n_inds: npt.NDArray[np.int32], id: int | np.int32 | np.int64
    ) -> npt.NDArray[np.int32]:
        """Transform a given index to a tensor one considering
        `n_inds` indices per direction.

        Args:
            n_inds (npt.NDArray[np.int32]): Number of indices per
                dimension.
            id (int | np.int32 | np.int64): Flat index to be
                transformed.

        Returns:
            npt.NDArray[np.int32]: Computed tensor index.
        """

        accum = TensorProdIndex.__create_dims_accumulation(n_inds)
        dim = accum.size

        tid = np.zeros(dim, dtype=np.int32)
        id = np.int64(id)
        for dir in range(dim - 1, -1, -1):
            tid[dir] = np.int32((id - np.dot(tid, accum)) // accum[dir])
        return tid
