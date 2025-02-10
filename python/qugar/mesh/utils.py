# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""A few utils for mesh connectivity."""

import numpy as np
import numpy.typing as npt

import qugar.utils


def _invert_map(map_ind: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Inverts a 1 to 1 map of indices.

    Args:
        map_ind (npt.NDArray[np.int32]): 1 to 1 map of indices to
            invert.

    Returns:
        npt.NDArray[np.int32]: Inverted map such that
            `inv_map_ind[map_ind[i]] = i` and `map_ind[inv_map_ind[i]] = i`.
    """
    return np.array([np.where(map_ind == i)[0][0] for i in range(map_ind.size)], dtype=np.int32)


def VTK_to_lexicg_nodes(dim: int, degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from VTK to lexicographical
    nodes ordering.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from VTK to
            lexicographical such that `a_vtk[i] = a_lex[perm_array[i]]`.

    Note:
        The VTK numbering is referred to the Arbitrary-order Lagrange
        Finite Elements as defined in
        https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit
        These are the cells with cell ids 68 (1D), 70 (2D), or 72 (3D),
        and not the linear ones (3, 9, 12, respec.) or quadratic (21,
        28, 29, respec.).
    """

    assert 1 <= dim <= 3, "Invalid dimension."
    assert 1 <= degree, "Invalid degree."

    if dim == 1:
        return _VTK_to_lexicg_1D_nodes(degree)
    elif dim == 2:
        return _VTK_to_lexicg_2D_nodes(degree)
    else:  # if dim == 3:
        return _VTK_to_lexicg_3D_nodes(degree)


def _VTK_to_lexicg_1D_nodes(degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from VTK to lexicographical
    nodes ordering for 1D cells.

    Args:
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from VTK to
            lexicographical such that `a_vtk[i] = a_lex[perm_array[i]]` or
            `a_lex[i] = perm_array[a_vtk[i]]`.

    Note:
        The VTK numbering is referred to the Arbitrary-order Lagrange
        https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit
        These are the cells with cell id 68 (1D) and not the linear
        (with id 3) or quadratic (21) ones.
    """

    assert 1 <= degree, "Invalid degree."

    # Vertices
    vtk_to_lex = [0, degree]

    # Internal points.
    for i in range(1, degree):
        vtk_to_lex.append(i)

    return np.array(vtk_to_lex, dtype=np.int32)


def _VTK_to_lexicg_2D_nodes(degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from VTK to lexicographical
    nodes ordering for 2D cells.

    Args:
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from VTK to
            lexicographical such that `a_vtk[i] = a_lex[perm_array[i]]` or
            `a_lex[i] = perm_array[a_vtk[i]]`.

    Note:
        The VTK numbering is referred to the Arbitrary-order Lagrange
        Finite Elements as defined in
        https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit
        These are the cells with cell id 70 (2D) and not the linear
        (with id 9) or quadratic (28) ones.
    """

    assert 1 <= degree, "Invalid degree."

    order = degree + 1

    def _tensor_to_flat(u: int, v: int) -> int:
        return u + v * order

    vtk_to_lex = []

    # Vertices
    vtk_to_lex.append(_tensor_to_flat(0, 0))
    vtk_to_lex.append(_tensor_to_flat(degree, 0))
    vtk_to_lex.append(_tensor_to_flat(degree, degree))
    vtk_to_lex.append(_tensor_to_flat(0, degree))

    # Edges
    for u in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(u, 0))
    for v in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(degree, v))
    for u in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(u, degree))
    for v in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(0, v))

    # Internal
    for v in range(1, degree):
        for u in range(1, degree):
            vtk_to_lex.append(_tensor_to_flat(u, v))

    return np.array(vtk_to_lex, dtype=np.int32)


def _VTK_to_lexicg_3D_nodes(degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from VTK to lexicographical
    nodes ordering for 3D cells.

    Args:
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from VTK to
            lexicographical such that `a_vtk[i] = a_lex[perm_array[i]]` or
            `a_lex[i] = perm_array[a_vtk[i]]`.

    Note:
        The VTK numbering is referred to the Arbitrary-order Lagrange
        Finite Elements as defined in
        https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit
        These are the cells with cell id 72 (3D) and not the linear
        (with id 12) or quadratic (29) ones.
    """

    assert 1 <= degree, "Invalid degree."

    order = degree + 1

    def _tensor_to_flat(u: int, v: int, w: int) -> int:
        return u + v * order + w * order * order

    vtk_to_lex = []

    # Vertices
    vtk_to_lex.append(_tensor_to_flat(0, 0, 0))
    vtk_to_lex.append(_tensor_to_flat(degree, 0, 0))
    vtk_to_lex.append(_tensor_to_flat(degree, degree, 0))
    vtk_to_lex.append(_tensor_to_flat(0, degree, 0))
    vtk_to_lex.append(_tensor_to_flat(0, 0, degree))
    vtk_to_lex.append(_tensor_to_flat(degree, 0, degree))
    vtk_to_lex.append(_tensor_to_flat(degree, degree, degree))
    vtk_to_lex.append(_tensor_to_flat(0, degree, degree))

    # Edges
    for u in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(u, 0, 0))
    for v in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(degree, v, 0))
    for u in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(u, degree, 0))
    for v in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(0, v, 0))
    for u in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(u, 0, degree))
    for v in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(degree, v, degree))
    for u in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(u, degree, degree))
    for v in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(0, v, degree))
    for w in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(0, 0, w))
    for w in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(degree, 0, w))
    for w in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(degree, degree, w))
    for w in range(1, degree):
        vtk_to_lex.append(_tensor_to_flat(0, degree, w))

    # Faces
    for w in range(1, degree):
        for v in range(1, degree):
            vtk_to_lex.append(_tensor_to_flat(0, v, w))
    for w in range(1, degree):
        for v in range(1, degree):
            vtk_to_lex.append(_tensor_to_flat(degree, v, w))
    for w in range(1, degree):
        for u in range(1, degree):
            vtk_to_lex.append(_tensor_to_flat(u, 0, w))
    for w in range(1, degree):
        for u in range(1, degree):
            vtk_to_lex.append(_tensor_to_flat(u, degree, w))
    for v in range(1, degree):
        for u in range(1, degree):
            vtk_to_lex.append(_tensor_to_flat(u, v, 0))
    for v in range(1, degree):
        for u in range(1, degree):
            vtk_to_lex.append(_tensor_to_flat(u, v, degree))

    # Internal
    for w in range(1, degree):
        for v in range(1, degree):
            for u in range(1, degree):
                vtk_to_lex.append(_tensor_to_flat(u, v, w))

    return np.array(vtk_to_lex, dtype=np.int32)


def lexicg_to_VTK_nodes(dim: int, degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from lexicographical to VTK
    nodes ordering.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from lexicographical to
            VTK such that `a_lex[i] = a_vtk[perm_array[i]]` or
            `a_vtk[i] = perm_array[a_lex[i]]`.

    Note:
        The VTK numbering is referred to the Arbitrary-order Lagrange
        Finite Elements as defined in
        https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit
        These are the cells with cell ids 68 (1D), 70 (2D), or 72 (3D),
        and not the linear ones (3, 9, 12, respec.) or quadratic (21,
        28, 29, respec.).
    """

    vtk_to_lex = VTK_to_lexicg_nodes(dim, degree)
    return _invert_map(vtk_to_lex)  # type: ignore


def DOLFINx_to_lexicg_nodes(dim: int, degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from DOLFINx to
    lexicographical nodes ordering.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from DOLFINx to
            lexicographical, i.e., such that `a_dlf[i] = a_lex[perm_array[i]]` or
            `a_lex[i] = perm_array[a_dlf[i]]`.
    """

    assert 1 <= dim <= 3, "Invalid dimension."
    assert 1 <= degree, "Invalid degree."

    if dim == 1:
        return _DOLFINx_to_lexicg_1D_nodes(degree)
    elif dim == 2:
        return _DOLFINx_to_lexicg_2D_nodes(degree)
    else:  # if dim == 3:
        return _DOLFINx_to_lexicg_3D_nodes(degree)


def _DOLFINx_to_lexicg_1D_nodes(degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from DOLFINx to
    lexicographical nodes ordering for 1D cells.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from DOLFINx to
            lexicographical, i.e., such that `a_dlf[i] = a_lex[perm_array[i]]` or
            `a_lex[i] = perm_array[a_dlf[i]]`.
    """

    assert 1 <= degree, "Invalid degree."

    # Vertices
    dlf_to_lex = [0, degree]

    # Internal points.
    for i in range(1, degree):
        dlf_to_lex.append(i)

    return np.array(dlf_to_lex, dtype=np.int32)


def _DOLFINx_to_lexicg_2D_nodes(degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from DOLFINx to
    lexicographical nodes ordering for 2D cells.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from DOLFINx to
            lexicographical, i.e., such that `a_dlf[i] = a_lex[perm_array[i]]` or
            `a_lex[i] = perm_array[a_dlf[i]]`.
    """

    assert 1 <= degree, "Invalid degree."

    order = degree + 1

    def _tensor_to_flat(u: int, v: int) -> int:
        return u + v * order

    dlf_to_lex = []

    # Vertices
    dlf_to_lex.append(_tensor_to_flat(0, 0))
    dlf_to_lex.append(_tensor_to_flat(degree, 0))
    dlf_to_lex.append(_tensor_to_flat(0, degree))
    dlf_to_lex.append(_tensor_to_flat(degree, degree))

    # Edges
    for u in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(u, 0))
    for v in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(0, v))
    for v in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(degree, v))
    for u in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(u, degree))

    # Internal
    for v in range(1, degree):
        for u in range(1, degree):
            dlf_to_lex.append(_tensor_to_flat(u, v))

    return np.array(dlf_to_lex, dtype=np.int32)


def _DOLFINx_to_lexicg_3D_nodes(degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from DOLFINx to
    lexicographical nodes ordering for 3D cells.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from DOLFINx to
            lexicographical, i.e., such that
            `a_dlf[i] = a_lex[perm_array[i]]` or
            `a_lex[i] = perm_array[a_dlf[i]]`.
    """

    assert 1 <= degree, "Invalid degree."

    order = degree + 1

    def _tensor_to_flat(u: int, v: int, w: int) -> int:
        return u + v * order + w * order * order

    dlf_to_lex = []

    # Vertices
    dlf_to_lex.append(_tensor_to_flat(0, 0, 0))
    dlf_to_lex.append(_tensor_to_flat(degree, 0, 0))
    dlf_to_lex.append(_tensor_to_flat(0, degree, 0))
    dlf_to_lex.append(_tensor_to_flat(degree, degree, 0))
    dlf_to_lex.append(_tensor_to_flat(0, 0, degree))
    dlf_to_lex.append(_tensor_to_flat(degree, 0, degree))
    dlf_to_lex.append(_tensor_to_flat(0, degree, degree))
    dlf_to_lex.append(_tensor_to_flat(degree, degree, degree))

    # Edges
    for u in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(u, 0, 0))
    for v in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(0, v, 0))
    for w in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(0, 0, w))

    for v in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(degree, v, 0))
    for w in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(degree, 0, w))
    for u in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(u, degree, 0))

    for w in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(0, degree, w))
    for w in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(degree, degree, w))

    for u in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(u, 0, degree))
    for v in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(0, v, degree))
    for v in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(degree, v, degree))
    for u in range(1, degree):
        dlf_to_lex.append(_tensor_to_flat(u, degree, degree))

    # Faces
    for v in range(1, degree):
        for u in range(1, degree):
            dlf_to_lex.append(_tensor_to_flat(u, v, 0))
    for w in range(1, degree):
        for u in range(1, degree):
            dlf_to_lex.append(_tensor_to_flat(u, 0, w))
    for w in range(1, degree):
        for v in range(1, degree):
            dlf_to_lex.append(_tensor_to_flat(0, v, w))
    for w in range(1, degree):
        for v in range(1, degree):
            dlf_to_lex.append(_tensor_to_flat(degree, v, w))
    for w in range(1, degree):
        for u in range(1, degree):
            dlf_to_lex.append(_tensor_to_flat(u, degree, w))
    for v in range(1, degree):
        for u in range(1, degree):
            dlf_to_lex.append(_tensor_to_flat(u, v, degree))

    # Internal
    for w in range(1, degree):
        for v in range(1, degree):
            for u in range(1, degree):
                dlf_to_lex.append(_tensor_to_flat(u, v, w))

    return np.array(dlf_to_lex, dtype=np.int32)


def lexicg_to_DOLFINx_nodes(dim: int, degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from lexicographical to
    DOLFINx nodes ordering.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from lexicographical to
            DOLFINx, i.e., such that `a_lex[i] = a_dlf[perm_array[i]]` or
            `a_dlf[i] = perm_array[a_lex[i]]`.
    """

    dlf_to_lex = DOLFINx_to_lexicg_nodes(dim, degree)
    return _invert_map(dlf_to_lex)  # type: ignore


def DOLFINx_to_VTK_nodes(dim: int, degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from DOLFINx to VTK nodes
    ordering.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from DOLFINx to VTK
            such that `a_dlf[i] = a_vtk[perm_array[i]]` or
            `a_vtk[i] = perm_array[a_dlf[i]]`.

    Note:
        The VTK numbering is referred to the Arbitrary-order Lagrange
        Finite Elements as defined in
        https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit
        These are the cells with cell ids 68 (1D), 70 (2D), or 72 (3D),
        and not the linear ones (3, 9, 12, respec.) or quadratic (21,
        28, 29, respec.)."""

    dlf_to_lex = DOLFINx_to_lexicg_nodes(dim, degree)
    lex_to_vtk = lexicg_to_VTK_nodes(dim, degree)

    return np.array([lex_to_vtk[dlf_to_lex[i]] for i in range(dlf_to_lex.size)], dtype=np.int32)


def VTK_to_DOLFINx_nodes(dim: int, degree: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from VTK to DOLFINx nodes
    ordering.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).
        degree (int): Cell's degree. It must be greater than 0.

    Returns:
        npt.NDArray[np.int32]: Permutation array from VTK to DOLFINx
            such that `a_vtk[i] = a_dlf[perm_array[i]]` or
            `a_dlf[i] = perm_array[a_vtk[i]]`.

    Note:
        The VTK numbering is referred to the Arbitrary-order Lagrange
        Finite Elements as defined in
        https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit
        These are the cells with cell ids 68 (1D), 70 (2D), or 72 (3D),
        and not the linear ones (3, 9, 12, respec.) or quadratic (21,
        28, 29, respec.).
    """

    vtk_to_lex = VTK_to_lexicg_nodes(dim, degree)
    lex_to_dlf = lexicg_to_DOLFINx_nodes(dim, degree)
    return np.array([lex_to_dlf[vtk_to_lex[i]] for i in range(vtk_to_lex.size)], dtype=np.int32)

    # dlf_to_vtk = create_DOLFINx_to_VTK(dim, degree)
    # return _invert_map(dlf_to_vtk)


@staticmethod
def lexicg_to_DOLFINx_faces(dim: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from lexicographical to
    DOLFINx faces ordering.

    See https://github.com/FEniCS/basix/#supported-elements

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).

    Returns:
        npt.NDArray[np.int32]:  Permutation array from lexicographical
            to DOLFINx such that `faces_dlf[i] = faces_lex[perm_array[i]]`
            or `faces_lex[i] = perm_array[faces_dlf[i]]`.
    """

    assert 1 <= dim <= 3
    if dim == 1:
        return np.array([0, 1], dtype=np.int32)
    elif dim == 2:
        return np.array([2, 0, 1, 3], dtype=np.int32)
    else:  # if dim == 3:
        return np.array([4, 2, 0, 1, 3, 5], dtype=np.int32)


@staticmethod
def DOLFINx_to_lexicg_faces(dim: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from DOLFINx to
    lexicographical faces ordering.

    See https://github.com/FEniCS/basix/#supported-elements

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).

    Returns:
        npt.NDArray[np.int32]:  Permutation array from DOLFINx to
            lexicographical such that `faces_lex[i] = faces_dlf[perm_array[i]]` or
            `faces_dlf[i] = perm_array[faces_lex[i]]`.
    """

    assert 1 <= dim <= 3
    if dim == 1:
        return np.array([0, 1], dtype=np.int32)
    elif dim == 2:
        return np.array([1, 2, 0, 3], dtype=np.int32)
    else:  # if dim == 3:
        return np.array([2, 3, 1, 4, 0, 5], dtype=np.int32)


@staticmethod
def lexicg_to_VTK_faces(dim: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from lexicographical to
    VTK faces ordering.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).

    Returns:
        npt.NDArray[np.int32]:  Permutation array from lexicographical
            to VTK such that `faces_vtk[i] = faces_lex[perm_array[i]]`
            or `faces_lex[i] = perm_array[faces_vtk[i]]`.
    """

    assert 1 <= dim <= 3
    if dim == 1:
        return np.array([0, 1], dtype=np.int32)
    elif dim == 2:
        return np.array([2, 1, 3, 0], dtype=np.int32)
    else:  # if dim == 3:
        return np.array([2, 4, 0, 1, 3, 5], dtype=np.int32)


@staticmethod
def VTK_to_lexicg_faces(dim: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from VTK to
    lexicographical faces ordering.

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).

    Returns:
        npt.NDArray[np.int32]:  Permutation array from VTK to
            lexicographical such that `faces_lex[i] = faces_vtk[perm_array[i]]` or
            `faces_vtk[i] = perm_array[faces_lex[i]]`.
    """

    assert 1 <= dim <= 3
    if dim == 1:
        return np.array([0, 1], dtype=np.int32)
    elif dim == 2:
        return np.array([3, 1, 0, 2], dtype=np.int32)
    else:  # if dim == 3:
        return np.array([2, 3, 0, 4, 1, 5], dtype=np.int32)


@staticmethod
def lexicg_to_DOLFINx_edges(dim: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from lexicographical to
    DOLFINx edges ordering.

    See https://github.com/FEniCS/basix/#supported-elements

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).

    Returns:
        npt.NDArray[np.int32]:  Permutation array from lexicographical
            to DOLFINx such that `edges_dlf[i] = edges_lex[perm_array[i]]`
            or `edges_lex[i] = perm_array[edges_dlf[i]]`.
    """

    assert 1 <= dim <= 3
    if dim < 3:
        return lexicg_to_DOLFINx_faces(dim)
    else:  # if dim == 3:
        return np.array([8, 4, 0, 5, 2, 9, 1, 3, 10, 6, 7, 11], dtype=np.int32)


@staticmethod
def DOLFINx_to_lexicg_edges(dim: int) -> npt.NDArray[np.int32]:
    """Creates the permutation array to map from DOLFINx to
    lexicographical edges ordering.

    See https://github.com/FEniCS/basix/#supported-elements

    Args:
        dim (int): Parametric dimension of the cells (1D, 2D, or 3D).

    Returns:
        npt.NDArray[np.int32]:  Permutation array from DOLFINx to
            lexicographical such that `edges_lex[i] = edges_dlf[perm_array[i]]` or
            `edges_dlf[i] = perm_array[edges_lex[i]]`.
    """

    assert 1 <= dim <= 3
    if dim < 3:
        return DOLFINx_to_lexicg_faces(dim)
    else:  # if dim == 3:
        return np.array([2, 6, 4, 7, 1, 3, 9, 10, 0, 5, 8, 11], dtype=np.int32)


if qugar.has_FEniCSx:
    from mpi4py import MPI

    import dolfinx
    import dolfinx.cpp.graph

    def create_identity_partitioner(
        comm: MPI.Comm,
        n_parts: int,
        dim: int,
        cells: dolfinx.cpp.graph.AdjacencyList_int64,
    ) -> dolfinx.cpp.graph.AdjacencyList_int32:
        """(Dummy) mesh partitioner for leaving cells on the current
        rank.

        Args:
            comm (MPI.Comm): Mesh's MPI communicator.
            n_parts (int): Number of part in which the mesh will be
                partitioned. Not really used as the number of parts is
                determined by the number of processes.
            dim (int): Parametric dimension of the mesh.
            cells (dolfinx.cpp.graph.AdjacencyList_int64): List of cells
                to distribute.

        Returns:
            dolfinx.cpp.graph.AdjacencyList_int32: Adjaceny list
                assigning to every cell in the list cells, the MPI rank of
                destination. In this case, the destination rank will be the
                current one.
        """

        try:
            from mpi4py import MPI

            import dolfinx.cpp.graph
        except ImportError as e:
            raise ValueError(e)

        assert isinstance(comm, MPI.Comm)
        assert isinstance(cells, dolfinx.cpp.graph.AdjacencyList_int64)

        rank_dest = np.full(cells.num_nodes, comm.rank, dtype=np.int32)

        return dolfinx.cpp.graph.AdjacencyList_int32(rank_dest)

    def create_cells_to_facets_map(
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

    def map_cells_and_local_facets_to_facets(
        mesh: dolfinx.mesh.Mesh,
        cells: npt.NDArray[np.int32],
        local_facets: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """Given cell ids and local face ids referred to those cells,
        returns the facet ids.

        Args:
            mesh (dolfinx.mesh.Mesh): Mesh to which the facets belong
                to.
            cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to the current process) to be transformed.
            local_facets (npt.NDArray[np.int32]): Array of local facets
                referred to the `cells`. They follow the FEniCSx
                convention. See
                https://github.com/FEniCS/basix/#supported-elements

        Returns:
            npt.NDArray[np.int32]: Computed DOLFINx facet ids (local to
            the current process).
        """

        assert cells.size == local_facets.size
        cells_to_facets = create_cells_to_facets_map(mesh)
        return cells_to_facets[cells, local_facets]

    def map_facets_to_cells_and_local_facets(
        mesh: dolfinx.mesh.Mesh, facets: npt.NDArray[np.int32]
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Given the ids of facets, returns the cells and local facet
        ids corresponding to those facets.

        Note:
            Interior facets belong to more whan one cell, in those cases
            only one cell (and local facet) is returned for that
            particular facet. The one chosen depends on the way in which
            that information is stored in the mesh connectivity.

        Args:
            mesh (dolfinx.mesh.Mesh): Mesh to which the facets belong
                to.
            facets (npt.NDArray[np.int32]): Array of DOLFINx facets
                (local to the current process) to be transformed.

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: Cells
                and local facet ids the associated to the facets. The first
                entry of the tuple corresponds to the cells and the second
                to the the local facets.

                The local facets indices follow the FEniCSx convention. See
                https://github.com/FEniCS/basix/#supported-elements
        """

        # TODO: this implementation can be likely improved.
        # Checking in dolfinx code.

        topology = mesh.topology
        tdim = topology.dim
        topology.create_connectivity(tdim - 1, tdim)
        conn = topology.connectivity(tdim - 1, tdim)

        cells_to_facets = create_cells_to_facets_map(mesh)

        cells = np.zeros_like(facets)
        local_facets = np.zeros_like(facets)

        for i, facet in enumerate(facets):
            # Only the first cell is considered
            cell = conn.links(facet)[0]
            ind = np.where(cells_to_facets[cell] == facet)[0]
            assert ind.size == 1

            cells[i] = cell
            local_facets[i] = ind[0]

        return cells, local_facets
