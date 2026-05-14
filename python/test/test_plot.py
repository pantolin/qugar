# i --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for quadrature point generation and assemblers."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import hashlib
from typing import Any, cast

from mpi4py import MPI

import numpy as np
import pytest
import pyvista as pv
from utils import (
    dtypes,  # type: ignore
)

import qugar.cpp
import qugar.impl
from qugar.impl import ImplicitFunc
from qugar.mesh import create_unfitted_impl_Cartesian_mesh


def create_domain(
    dom_func: ImplicitFunc,
    n_cells_dir: int,
    dtype: type[np.float32 | np.float64] = np.float64,
):
    """
    Creates an unfitted domain based on an implicit function and a Cartesian mesh.

    Args:
        dom_func (ImplicitFunc): The implicit function defining the domain geometry.
        n_cells_dir (int): The number of cells in each direction of the Cartesian mesh.
        dtype (type[np.float32 | np.float64], optional): The data type for the domain
            coordinates. Defaults to np.float64.

    Returns:
        Domain: An unfitted implementation domain created using the provided implicit
        function and Cartesian mesh.
    """
    dim = dom_func.dim
    comm = MPI.COMM_WORLD
    n_cells = [n_cells_dir] * dim
    xmin = np.zeros(dim, dtype)
    xmax = np.ones(dim, dtype)
    return create_unfitted_impl_Cartesian_mesh(comm, dom_func, n_cells, xmin, xmax)


def create_quad_PyVista_grid_hash(grid) -> str:
    """Return an ordering-invariant SHA-256 hash of a PyVista grid.

    The previous implementation hashed ``str(grid.points)`` and
    ``str(grid.cell_connectivity)`` directly. That is fragile: DOLFINx
    0.10.0 sometimes reorders vertices when constructing meshes (notably
    the 1D interval meshes used for the reparameterization wirebasket),
    which yields a different byte representation for the *same* physical
    mesh and therefore a different hash.

    The canonical form built here is invariant to:

    * vertex permutations within a cell -- vertices inside each cell
      are sorted lexicographically by their physical coordinates.
    * cell reordering -- cells are sorted lexicographically by their
      canonical vertex tuple.
    * point reordering of the underlying point array.

    Coordinates are emitted via ``np.array2string`` with a fixed
    precision so the hash is also stable across minor changes in
    numpy's default print formatting.
    """
    pts = np.asarray(grid.points)
    conn = np.asarray(grid.cell_connectivity)
    offsets = np.asarray(grid.offset)  # n_cells+1 entries; cell i = conn[offsets[i]:offsets[i+1]]

    # Per-cell canonical blob: sort that cell's vertex coordinates lex.
    n_cells = len(offsets) - 1
    canon_cell_blobs = []
    for i in range(n_cells):
        vert_idx = conn[offsets[i] : offsets[i + 1]]
        cell_pts = pts[vert_idx]  # (n_verts, gdim)
        cell_pts_sorted = cell_pts[np.lexsort(cell_pts.T)]
        canon_cell_blobs.append(cell_pts_sorted)

    # Canonicalize cell order: sort cells by their canonical content.
    canon_cell_blobs.sort(key=lambda c: c.tobytes())

    # Canonicalize the raw point set so isolated points (if any) still
    # contribute to the hash deterministically.
    pts_canon = pts[np.lexsort(pts.T)]

    combined = hashlib.sha256()
    combined.update(np.array2string(pts_canon, precision=10, separator=",").encode("utf-8"))
    for blob in canon_cell_blobs:
        combined.update(np.array2string(blob, precision=10, separator=",").encode("utf-8"))
    return combined.hexdigest()


def create_quad_PyVista_multiblock_hash(quad: pv.MultiBlock) -> str:
    """
    Generates a SHA-256 hash for a PyVista MultiBlock object.

    This function computes a combined hash by processing all components of the
    MultiBlock object. Each grid in the block is hashed individually
    and then combined into a single hash.

    Args:
        quad (pv.MultiBlock): PyVista MultiBlock to hash.

    Returns:
        str: A hexadecimal string representing the SHA-256 hash of the combined
        MultiBlock components.
    """

    combined = hashlib.sha256()
    for key in quad.keys():
        combined.update(create_quad_PyVista_grid_hash(quad.get(cast(str, key))).encode("utf-8"))

    return combined.hexdigest()


def create_quadrature_and_reparameterization_hashes(
    dom_func: ImplicitFunc,
    n_cells_dir: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64] = np.float64,
) -> tuple[str, str]:
    """
    Computes the SHA-256 hashes for the quadrature and reparameterization PyVista grids.

    This function creates a domain using the provided implicit function and creates
    a quadrature and reparameterization representations using `qugar.plot.quadrature_to_PyVista`,
    and `qugar.plot.reparam_mesh_to_PyVista`, respectively.

    Args:
        dom_func (ImplicitFunc): The implicit function defining the domain.
        n_cells_dir (int): Number of cells in each direction for the domain grid.
        n_quad_pts (int): Number of quadrature points per direction.
        dtype (type[np.float32 | np.float64], optional): Data type for the domain
            and quadrature points. Defaults to `np.float64`.

    Returns:
        tuple[str, str]: A tuple containing the SHA-256 hash strings for the quadrature
        and reparameterization grids.


    """
    domain = create_domain(dom_func, n_cells_dir, dtype)

    quad = qugar.plot.quadrature_to_PyVista(domain, n_pts_dir=n_quad_pts)

    reparam = qugar.reparam.create_reparam_mesh(domain, degree=n_quad_pts - 1, levelset=False)
    reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)

    quad_hash = create_quad_PyVista_multiblock_hash(quad)
    reparam_hash = create_quad_PyVista_multiblock_hash(reparam_pv)

    return quad_hash, reparam_hash


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_disk(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using a 2D disk.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    radius = 0.5
    center = np.array([0.51, 0.45], dtype=dtype)

    func = qugar.impl.create_disk(radius=radius, center=center, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    targets = {}
    targets[(8, 5, np.float32, False, False)] = (
        "eca442ff7f2169972bc168e3e22d0b01d0ee0333613b1199af14454849cc554e",
        "4e24bf8435b8819bc62939b457d837097ede525b77accc427a00f2c99e514fae",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "df6f224d05c5c3ddb26f900f814673581f1930db1fc44cc4fd307ba7ec86aff5",
        "a4948b097fe082880ad0375615df896a81ae33bf6fa3f034aa9b1d227e262405",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "2a16161050975698a70e2b3255c1dc7b02d5519c96defa72964abe39ade4b4d0",
        "259496eb53c02d39d3c225c7615bca09e8cc16734a5d7bc5321f4b13ea81fdcd",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "6e9a929c069e754859686430475afa1f56f32a4ed7aab14088dd4758d2790519",
        "911e2410ed77108d5a6a5c7805e70a1da98796ffeed11fa9d46e0f39471d9562",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "1d567c670ac736abc6f4788f81064911e2566e20849e849de49d0e20c3a78ff0",
        "84ea9c7aac1001334956f75cc99756c6f1c6485235b410aa769c65541f1e72f8",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "b1c3c8cc45e9a1c4977a558a8bc48d8e155c42f6fd220838503234ffbca8f5da",
        "bc996548a9649a8135bf4f1ab5251fc4107e69b822a286c621309f861748fbe2",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "7d82466a375ba8d1ec7a4a1c1092f94860136ed8c1909cf0daba0cd20cc26b11",
        "70b2f8a698c617d53c8822fa18892e4d750423ae5c6edcb4bfa0a520e2c59c11",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "21c2a643d330e167ad82d869eb572d38a174ee9defd877ccaa75b7ce73057302",
        "ad6a4481c40f7cc59b522042e34d38f94a9bd4819e5e2b7c78d0ce2484d0894c",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f"    targets[({n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {use_bzr}, {negative})] = "
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_sphere(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using a 3D sphere.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """
    radius = 0.8
    center = np.array([0.5, 0.45, 0.35], dtype=dtype)

    func = qugar.impl.create_sphere(radius=radius, center=center, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    targets = {}
    targets[(8, 5, np.float32, False, False)] = (
        "ae2bc2843a9bb38319753d54204153f31474f0026f90763da78ef730a026e624",
        "f715e7a2cbba74127d9559073f6590fecd52bc545dcb459ce37b4dab6b0571e3",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "74965505b8ad91766c6d67e3e13c5075206245e0a4afd43deedf514672895fee",
        "4a514e890099c6345c15d15920484fa5cd8a7a17adfaca11f0bc05fad93e3503",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "325225f2c648cb6a0607148eab065b62ba773113f692ad53c767536f0d6ee0e9",
        "3ff3fe4ca0d1871b18cf324509ad58a228d9206e3edb51170f6e3f51a4716f3c",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "45917b7836dff00a35d0bbfda9c109c4b2cbfd066f89503313c0a7bb6b9d7021",
        "5946936b0fd8381bf15062d5d7e262c7d6e2a508887fbbf3ac6f99f9829a00bf",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "9f9ae5792ca01570b74dbca4f2ac317e3a5f647930f03e6c3ef4c9aeff388712",
        "8685a9790102f17031dffd98e8226d21cf57e02ea6a10c04d3776657ef3e355e",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "4d90d553c8c25c4c9beb1cc384939addf7402b6b4deeed771d8b7ec2fd692505",
        "9e647023ef4500cae6f0f0a9c845e05bb6c1ffd2c08f3835411a264c31327352",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "9547bd10dad7c84741e1b2e7cace42508d38ea62c252f7d5177f57d59c8618f1",
        "fd82beb6f89f4f35e855eeba81cc24f4cf984379f41cc71766813e1e88b5b20a",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "5233a4e7cf1f8e53dca68a84c98dc39b11ec8c1da88e53ea778b6c2d66e4ecb8",
        "34fac0f28d3a9e50cb42019a96eaebe81130806649d839273cfabf0bca9b9bde",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f"    targets[({n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {use_bzr}, {negative})] = "
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_line(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using a
    2D straight line.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    a = 0.2
    origin = np.array([0.5, 0.5], dtype=dtype)
    normal = np.array([1.0, a], dtype=dtype)

    func = qugar.impl.create_line(origin, normal, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    targets = {}
    targets[(8, 5, np.float32, False, False)] = (
        "7c03b1ba0fe35788d6d5f51de2ddcfa210ef25d74896fd6d8e6c0906ba1bc8f5",
        "f0ef2cfb1e5aa7bd118ad507f2831eb4a8ee9da0319b5909b70bb964b2b41740",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "825ab009ed58d704f56af1343d7421a3f5e821c5dbad423563a2df644163b705",
        "ba470c5ac0faeda439e366d01e1f85672676e35c81104ff0cb0f1f2331b1ab52",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "e56cd012b103653f471fee1e560c6f0d8114dd49af93215deae25cbe4239c88c",
        "cbaf9264bd30254eda97982adb09623a1418fdff384068659d2c1731acd9986d",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "2968aa18e25fb71a61ba7106ed5bd714bc1d6455edcbbd49e8e91d6a6c353b64",
        "ba470c5ac0faeda439e366d01e1f85672676e35c81104ff0cb0f1f2331b1ab52",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "6a4456f7613f135426ee0a96938ea9fbe73d3a3eb4bf44099b39d204aa620798",
        "6e29c70f4d06eed8dbd8a047aac076e5d4146216e16b6469a798d8f26a958524",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "adb79517d8bd99a8ae284813d4bc9cbcd8e47bea5e6dc075e31d9f4219d89afa",
        "d29ea5710f08befc1be1ce7d69a97b7e185047774202bd51ef455a55d7fe9ffa",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "ea69b823aac71987425889f649e92ecd13a26ad89531f15c8c01ae3e6d6790d9",
        "092ac3395b1abbd266923484062c5813d5d8b95daa1d3650d600777a4a2e50f3",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "0fd21d5b15d02991d4347b4450d87368698d4c9b9cf41a7a3b78d5ed10bae3b6",
        "108cedf0306eb86e165c45a00b2c66ea13ec22c7c79942423d43c9de6f82e334",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f"    targets[({n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {use_bzr}, {negative})] = "
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_plane(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using a 3D plane.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    origin = np.array([0.3, 0.47, 0.27], dtype=dtype)
    normal = np.array([1.0, 0.3, -1.0], dtype=dtype)

    func = qugar.impl.create_plane(origin, normal, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    targets = {}
    targets[(8, 5, np.float32, False, False)] = (
        "b03d4511de48b6d463d108850caebc8a03d75992a6ebe10d583ef7f919929101",
        "12215d40851f962779bb3afa95ddda0091f95d7c597f4ee6332d57df99d492d5",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "c548922f9c6ab53dd8b419485a65d80e7155cb00cb235772308f81950040712d",
        "66c16b76ba617c10df6e23a04845cf21e65f0b915994e0fd3fc55232685529e3",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "d72b29c42078adcb120a1323dc631e6b2db3aeb37046c860ef96c8e0f760e293",
        "2690d3c69ee713cb5a60f42a6a63ba1d3fb1d5ee642154baa090e0ccc5fa1e19",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "0b696693a970f20871db49be477e317ae19291bad25853ebb6bbe5d6e35b603a",
        "99a3f29b6f50f2cdae52dab37ce2a6c9ba3dc29768b33bf2052642e27db1bdfa",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "9af26aaf7000a7217d66ec8803bb24ea270a66c142b56e8ed2c5b2fe06a851cb",
        "bbc386563fae24f3d20529c652c38c97869812584e47c21be6b8171323eef31c",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "c385c70beda2575063733dcd7f3981c03b7e4f74d1dd167bd66a3386debaf205",
        "be3f7c1b1a25480efe5bcf339813ed54409923def3fe337e13f4367a0398b029",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "027e49e3dadcb5f231d2cd503356360c6fcafd138de088ff26db534c1cb9ff94",
        "efa5029492e60eb4fe5e245322857b860c4a1227bef2c2b6a20f3bf4fa5d6972",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "3ebf6f8ed88fbf016294b4a4e697405c36fbf8b8cd42f91bd20f0171b163b8c0",
        "9002c9747c0191986e7568c31646a93972385fc28308bd58eef4a64c4e90a8a8",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f"    targets[({n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {use_bzr}, {negative})] = "
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_cylinder(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using a 3D cylinder.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    origin = np.array([0.55, 0.45, 0.47], dtype=dtype)
    axis = np.array([1.0, 0.9, -0.95], dtype=dtype)
    radius = 0.4

    func = qugar.impl.create_cylinder(radius=radius, origin=origin, axis=axis, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    targets = {}
    targets[(8, 5, np.float32, False, False)] = (
        "326a30a90af4d64d307b9708989d43d73dce12cc4d167be62409456c3b719e92",
        "8fc5db77052aa184bdc7cd7656eea12b2cbeea29dabd02d451f5a81add73490f",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "e162a1a7c445e873e9f037e21ddbdf550c9f0668327e8d86f2c074c024267242",
        "666ecaa9d22e1e3d318b598fd24532f9388f2c47f31b1fd53cb80fcdf94808bb",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "562dcb17830a5715ee320e7ca9bb0cfb68e01b9674ab5ad1ee821e0bc7a634f4",
        "b112cddbf476b21736401a6c408645a358a06c4d118338d6fdc1ce526666d380",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "96ec85db7edb12f449546f6ee0aa5aad1ad1ebe7dff9b61c73ac670d9133eb2c",
        "8a0ec3302a89c915c48a3478f63b44c0db0c9f885413ac8d19aa7c74df65a9a1",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "a4efffe9b39fb52e1380bda02ba4142ab2df277b05b0d40e0b9a6b5877826807",
        "3e3bf302f31fb2c106cadd89adf63e52c9f195334953c56860354411287f5412",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "574b25414a3246aea40b62fb63ba001193501eed6545d730b104953a61c030ee",
        "b1e596fa68529518195f4d4208c2c33f05e7c51702c4c3bebc0a85b265fd4831",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "7193d39c1f7fd6c670687f4ff3937fe8d69a0190e97b00618e8e7abd357ca6ae",
        "7999bcc62fe4edd9f8dd8ce670588055ba9f58b583808745d51cdf13de6e9201",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "7417f04070c3a6722c1d5eba8ed13bd15f75c8188e69ea1bab5e5a044f3842cb",
        "3ca091492ef3a077d5b3f243cbee4db2a3c9d5a564cfdd4c73850f093a5527fd",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f"    targets[({n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {use_bzr}, {negative})] = "
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


# TODO: Rewrite test to avoid using plot hashes
@pytest.mark.skip(reason="Temporarily skipping plot hash tests")
@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_annulus(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using a 3D annulus.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    center = np.array([0.55, 0.47], dtype=dtype)
    outer_radius = 0.75
    inner_radius = 0.2

    func = qugar.impl.create_annulus(inner_radius, outer_radius, center, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    targets = {}
    targets[(8, 5, np.float32, False, False)] = (
        "f7e9d76a94f2ab394188b2fe5db8a7884949c928a20bb1d8654b8a4a4df16990",
        "47830eb46e25e4f99071cece17947081608e6a4fea31aa9c818eeb1d99fec0fd",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "506255478fbc969fa69b20d5941f31d090b3e9929045b8be4ad9d42be2464a25",
        "12178b172db7da84a0b5d7905f5ba9f8cdddf65f412c6c53ebe450a2e02eb6e3",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "ec06dc7a644f11924b9057bdf61a711b6bf3ec9053c77b5ab08307ec03250e3d",
        "ced2b2131c5fc1bdb45ac2545f4a62a57a4910bf27df400fae080c1f0733f175",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "8ed702e5b89f8fbcef34396ad50843bc449cfd1ed59e7a3d93125c15d442285e",
        "c9350e18b9cf2f3d37e0e41e7980f9020f52e983d935422ce2672c5e1d4b6510",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "67a7ab21a51ccadf2bd910f83970fa8b974c7f4e0b09d54969a48d7e432acc7d",
        "307d1ffc712fcac13966115587cae7c27aac25737418f88546c40f26048da569",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "bd1bcc9da26625e245ce21efc898c0edfd410c240130835642e46c0c75ec82a1",
        "b825a4872704c0dee2e90831069e2db385e6c1d3a297175cb93bd983e98fc70f",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "614c45a6f6d67bb4a11707d08670928e137f17bdb414e50b30b3b5be2fd5bc60",
        "5d5f52eb7a26555f6d8ede48b1b2bfee9201cfbf51b439981da1acd4650f41b9",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "b41b3e2c2bff1be81df8e6138cffe740196807f839658eca776dc53a9d42a340",
        "5da89df0d21a3a5cf01af1984b48b5811f75fb465f72199abf9127eb7b178e4d",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f"    targets[({n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {use_bzr}, {negative})] = "
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_ellipse(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using a 2D ellipse.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    center = np.array([0.5, 0.5], dtype=dtype)
    x_axis = np.array([1.0, 1.0], dtype=dtype)
    ref_system = qugar.cpp.create_ref_system(center, x_axis)
    semi_axes = np.array([0.7, 0.5], dtype=dtype)

    func = qugar.impl.create_ellipse(semi_axes=semi_axes, ref_system=ref_system, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    targets = {}
    targets[(8, 5, np.float32, False, False)] = (
        "be21c36261e6eb6c670d36ff6575377344e9ea36cb27a7a0fe633b85d490553f",
        "a832acc2710457b4f9f696a56e1b3f40ca919ae8e9d8b4c9337a13d24d693dc8",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "18f1ffdb0e55d9074b21306513706b336ef1a0ee0f5ad24e448f13bdf8b3505a",
        "866afe6e0abc30a392229a22a2439d6a650729e328d9fc3dfdfd849f1f501fd1",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "463d00ab1fcb333655454fd9fd4e66e67f16ae0ee9a1871cf6c73265510d330e",
        "d0145486c3a092538e7446b18347f56a3ccf0887a1ec460b98a0a326822aebce",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "8e5fde7f976bd5263085005225bcc6021a8d7034fef1980853cde8eaa88b99af",
        "1f1a05ffb266959df85afb6fb76a71ef0cf1d3fae0349041c65918dc85ccd76d",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "5af027dcc2016bca92f3ebbec6c415c282182172d9edc2bf53b21b95e1859d37",
        "0d99422b7d40ce2f856d697062b6c8d129c26f4d02455dc762f6abe479bc3c51",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "d0486e558d15ba9cb56210e4742d9391d2461206903b672463ff9c348dd9882f",
        "62519291c78179de96bd27e097d02a8f16deb445ce967f47b5ce7c5f31a4e4d9",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "d2d73940c51c93fb56915d29f9c7a1de154573093b14583ea78b6661d25439bd",
        "b9e125e8787dd67343418c3bbe1c745aca153e98f643dc7fa95959ccad042328",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "e8730640d99b358b1a4982404a8fdbf7b057c2c36863b566b2eb4e09b0e9bea1",
        "eb20f05bd76f18de039d9c729f26ca7ee835d92aaad174333a1695b17eed6d1c",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f"    targets[({n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {use_bzr}, {negative})] = "
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_ellipsoid(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using a 3D
    ellipsoid.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    center = np.array([0.5, 0.5, 0.5], dtype=dtype)
    x_axis = np.array([1.0, 1.0, 1.0], dtype=dtype)
    y_axis = np.array([-1.0, 1.0, 1.0], dtype=dtype)
    ref_system = qugar.cpp.create_ref_system(center, x_axis, y_axis)
    semi_axes = np.array([0.7, 0.45, 0.52], dtype=dtype)

    func = qugar.impl.create_ellipsoid(semi_axes=semi_axes, ref_system=ref_system, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    targets = {}
    targets[(8, 5, np.float32, False, False)] = (
        "b0fe360a45e14f382c450d258f7b28ef194fea5d7bf70b7b0c7620fa78aa9f74",
        "5e2455c76a09a52392d880ea685acba8611356f8613c83fa6d16129b1bd56279",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "b1fd2beb39f4fc35e164b5e918372b188a0b66d991b57f191082273ea0784b7d",
        "959212b7cab2557170d6e2f2ce7705f3daec895e556647f6402d0145599e1d8e",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "a6e8c1ecf16952830a8ae29153ebe24edaaf61fd0b439aa7d5f4c70eff893dc9",
        "1c1c3db23cfab5c37bbc278d2eef9b5a94b708afc2e81dbe843999d73d8eed3a",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "0bb9c098c6aae883d836c474c4eaac1a370ae8a90bcfd8b8ad5c6aa169568cc4",
        "cd12c34231a3e438c97cd5162be0e40efedea6705289d472a5a2b18ce0653f45",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "357cc724744c8dd6062bdfa5204e1df4e6e25b48eb713a44e403d3bb20375ec6",
        "27fed929be64306115ff060a86445b5f6b0a0f56f43599e32757cead463248a7",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "5d43d13b0aab3810007de8581ec5e5a97967f537d0becdb317989a4ed9ddc865",
        "cc8035d41e876345942d6f77ab78a1aab3aeed068b2e5c9ac00179381c124863",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "392334e06a341777cfc478de9beb1f81865d2f2f4497cffec2381e81f2d206df",
        "3621658fc34e4a8959e47d18339c39158e99203c2e34e58a6a405f5b2b87a0d9",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "6e2476b0553253ab2a86e5ad02b0dc232c3a2dabd7fdfb16dd0336722ab878f3",
        "82909ca6414f03ee0d0fc46bfe4d057f391e6d8176f09264cc3fbae642dfb4a9",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f"    targets[({n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {use_bzr}, {negative})] = "
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_torus(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using a 3D torus.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    center = np.array([0.55, 0.47, 0.51], dtype=dtype)
    axis = np.array([1.0, 0.9, 0.8], dtype=dtype)
    major_radius = 0.77
    minor_radius = 0.35

    func = qugar.impl.create_torus(major_radius, minor_radius, center, axis, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    targets = {}
    targets[(8, 5, np.float32, False, False)] = (
        "b7d5b19e8ccfa800b4ca68ccc48aff6c179214564bdf7287280833856cbd3c28",
        "30a2dc1d662ac270775336911167a091f0817feedf98d459fa1a46cf58f62f4c",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "cdaf4dfc011692017e7601afe6ef35014da0249de23ff9b27338ff9555c16076",
        "cba4730f880bb408e51a094eb1d724436ca85be45a82c004f90f2240ee32d87b",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "9ab8203a7d0e8860ce4606e517695a9fe9eaaeb3a2827804e585afadc075d79b",
        "385fbb1ad20cab1ae143f1390ae7000ba15576defb1be7b0af88b1a0a3a48e45",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "e3239f9d5774d5b5e1477fae92fec35429c209c045c732d88473647472bc314c",
        "2c28ccfdbf1be982bab3211340561d1c43296562d1bbb002e96f70f06e68ef82",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "df97b869fd0228803e08a8e80abbc5fe93c979cf061ea8622658cd457e656ef0",
        "2dca5f3ddf85b3fc65b3ded67edc60e3aa39896945ec5be60393e077f4e03795",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "85363149a7a1091600d2a73576d8721d938da9c85e2522c4303c501a908c325b",
        "78359e2a5a72c62fb8ad380341c5a3433c7c534963cf86bcb0a2a5d64ca6a97a",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "4cb32723e2c884817faf8421be601c249049a468c15c9d7b9252919d28410ac9",
        "f57e26ff148f9bdfd6df03e533926dd35fb6ac2e5dce608ef03f5a4cba01f875",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "f637d4a6ee97de980a2523e6d1c70c77ca2e269ce95d0de497059d3256afe5a6",
        "70207bc8c227468d5be7959da7e91cc1fd7cb0a8e0504c9c696e49d14239679d",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f"    targets[({n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {use_bzr}, {negative})] = "
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


impl_functors = [
    (qugar.impl.create_Schoen, "Schoen"),
    (qugar.impl.create_Schoen_IWP, "Schoen_IWP"),
    (qugar.impl.create_Schoen_FRD, "Schoen_FRD"),
    (qugar.impl.create_Fischer_Koch_S, "Fischer_Koch_S"),
    (qugar.impl.create_Schwarz_Diamond, "Schwarz_Diamond"),
    (qugar.impl.create_Schwarz_Primitive, "Schwarz_Primitive"),
]


# TODO: Rewrite test to avoid using plot hashes
@pytest.mark.skip(reason="Temporarily skipping plot hash tests")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n_cells", [11, 12])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("impl_functor_str", impl_functors)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("negative", [False, True])
def test_tpms(
    dim: int,
    n_cells: int,
    n_quad_pts: int,
    impl_functor_str: Any,
    dtype: type[np.float32 | np.float64],
    negative: bool,
):
    """
    Test the generation of quadrature and reparameterization grids with PyVista using 2D
    and 3D gyroids of different families.

    Note:
        Asserts raise if the generated grids do not match the expected ones
        (through SHA-256 hashes).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        impl_functor_str (tuple): List of tuples containing the function to create the
            implicit function and string name of the implicit function.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    targets = {}
    targets[("Schoen", 2, 11, 5, np.float32, False)] = (
        "19ef93e88fba2dd5bde2b2f7a3b51fb6dcfd2545566b5d87e77d165f32ba0b67",
        "d6a78ccc38c07015ea827ef051e1f305661bb631bd78591822e5f986ea7b06f3",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, False)] = (
        "47a0fbf624bcffe769a554dacc36b74c0325cd37106f48e7387fb788e85391aa",
        "742843f0963b88e015e1b116ab96be361097603dea20fbef42dea9e975037840",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, False)] = (
        "4bcc9505b58a95f960c2e4ebcbf95aca5607cf6bfb2c85cb74f12abb8b99ffa3",
        "837fc25e7b2859abcc29f256cb1ad0e3661662d59d535bdd20c8dfd7d25b4950",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, False)] = (
        "b98f3558987c2b739a7210fbdb13fb0a6238c8f2318467f0164d986445d2958e",
        "42ef74ac38e645f51ee247a5c6911267f2eb5a6eebc4c5b1c211288d8820d128",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, False)] = (
        "5db4a2286e1620e73c7e61901a0e698cbee86715f53b61dcfcd25200981cb53e",
        "776d4fb03b57d2dc5471ef4ac48831a31392db708ce961e4f8c88ed8ba113691",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, False)] = (
        "2d4c1ebe71a5d3a617c62779f5bdffb8d650b072052c052beca75bb7578bdbe0",
        "dbea762a4825b2de458df3d14da9be674e2d366ef08671b78bf644cb28a10f10",
    )
    targets[("Schoen", 2, 11, 5, np.float32, True)] = (
        "ef8e2844e499e1dcdf651e3298f8cb48be050736d8cc5434402e2ac0c9a47064",
        "41a059f5c386de0fc61e3770c4ad5420e2b7741e5ef87b185b4de3f56133b603",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, True)] = (
        "5a25b19df9fe6baa67579822e083fa6d109187948feb88b110971c6b64fab70b",
        "22264d79bcb75e26869526b8f57312f119e7f30e2408416441e7f786d596f023",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, True)] = (
        "f0fa7ede3ce297b1a887a08c54b735cfceac4f5617d3c934211572ce2669b3be",
        "cc5af9d88904e947b7c18c93ef75f609bbe2bc9b4bb38952c5f8cf2ec0c4feed",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, True)] = (
        "67ffc08aceaea9f52381a12662ff6ac9fe0c5c2ea90a09f2e98d3ac3ce5ba86b",
        "abe99a78d27f9aedecbf99e5bc3f92c5a84f2d6b910ad829230b5a5c13440f8b",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, True)] = (
        "ef13a9d678fd5d6358c7a7d6f0ee4a8d4ba65c877e45c482a9985e3672c93f51",
        "cfe5f68083be1de7a98470f4c5b223ce75b2398c23195b9a592d64c24431e73e",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, True)] = (
        "edbd33921ae7b21936525d75c7dce31a1dc9ff6be434a37e11578a50a41a023e",
        "e1cf27d3f437a08690bf685a07b989119ad6133434d7ed0fac807318fb789f11",
    )
    targets[("Schoen", 2, 11, 5, np.float64, False)] = (
        "a6a958173fc33875cb71dd540e07ffcc0a6755e0e26cf6b3753392f6269ef06a",
        "4e145d5178a7d770aab2d1f85d13547d3a3d444c14a914f8bc39b418c120d7b5",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, False)] = (
        "b73803b66940c97b4d7580c42d93ee82b9d0ecbdc382bc637877962787ae9e28",
        "5de24f0beca96ad49e3633b9e3c1c9a43632b7387a3f09f4b2cc05fe439ff5e1",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, False)] = (
        "74b69b13aaf11c09e273a5175a2ac7699fca0a33688531d4a6988750e8cf95f3",
        "f1f366ecbb8fdaacb42e00fc47b68d074e711786119aeb98faf8d5917c0eb27c",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, False)] = (
        "d8d9733d365c690f42fa72c17a4e6d11a2ae37c3bf0149489e72854fec7dd630",
        "eac100fd9df2c18ebd51855b2f4dccb0f0e02e869ea3ec3ed7636f7c303feca9",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, False)] = (
        "7e5cffc7cef12fbdd2ec88a2273e619101c9fd4843663d81a8cc41c11efa9ee2",
        "5fc63317bf99107ccecd845640341c40b5ae447e46cb40c9482176c41a86f9fe",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, False)] = (
        "9ebff09efda06489b495e0dfbd4b8b080143c1f14e6c76fdb574dded2bc630b6",
        "2bab682095e00ab3d6dc47c7398a93d9b8536daf14fc275c72ec37e6444ca90d",
    )
    targets[("Schoen", 2, 11, 5, np.float64, True)] = (
        "0849cea01ae1e23cb29b4b01bf6031cb7bb7c5672cea2aee5e1c0551dc38afcc",
        "c4082f50709bcdab297f41058da1d31292abdf5a3cb5326e986c09a9072c0a04",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, True)] = (
        "90aac46313a7226d347928bce53232d91eb83363d0addab550579e2e0d2f5eab",
        "3653d1f10c9a30c83337983e399bc6ef3a7acd128bc62df669b29ab228d3774f",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, True)] = (
        "4b9b6ce53d45697d0d43195f3611dd295af9294b6b99a5aa073393b1f14f45a2",
        "5bd1318bdab4051c24838b43c6998842ef3749bcd22e9796d25283f313236675",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, True)] = (
        "7d3003e0a6b3892d807e86d9354c8970c1f11c5cb0dda39c6a8e80a2abf2fd23",
        "cbd2a09715784d676fe092af60fb95e86f343756e4e7bab1444a25d432256936",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, True)] = (
        "663d438e0f9913cdd155824c400ef52d3f44934f3a76f302f8ed0af77fd68da7",
        "9766ab27e6efd45ae18fd08672fce737b0374b64c14f3342119fdca194885100",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, True)] = (
        "4dcbf88c8f81a4e164b3cb3a21f075ca9f3b73372ba3ae7f75a7f5da10142931",
        "7c873f5fa366b484728d72c277ed4edd2e256c085fe21c8ee6f8709ebee8a324",
    )
    targets[("Schoen", 2, 12, 5, np.float32, False)] = (
        "13da4df3cf6cef59b78b73385e024e8d459cadb5fcc9c65b5d716f2c7a3945ce",
        "c9b89fea8b8dcf4db2e9304ee64820992f50c7606048ea97be4ec1586ef48fe3",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, False)] = (
        "cf66a7b22390eeaf0fd74a318bbac1179d72d356b51353da5c1d4dad511eea40",
        "263eebbf7852d7397dbe956816c13313ca2cfdabb0c23151f933bf0c77f39720",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, False)] = (
        "dae4e85421947d88c9b09b6bf6b5edaf758cd728234ce53d85458eb96dc8bd72",
        "f5d48ed617389a7e47364b4f9989bfc6302344ed361a512008d3687861fa118c",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, False)] = (
        "ec6f6490f58e0fdf52cde7eb284fc9cc437a0ce28fdea6d5c0d8c86d807c4bff",
        "3597d56072d65edf4ef6077f40b5566e89106f9f2851f1db44933444addd9c96",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, False)] = (
        "dc85c157834d330b5459183c3c708f80ff17c5b543447025bc25608a7a6cae98",
        "95ec208b95fcf01e0a0717eb429b802035202e683d06a9ec99ffb0014f2b2642",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, False)] = (
        "c20a8be83f82a416147a37d802271bffcbc1b94526ced7cf2c29f1a8874fbcf2",
        "da0c88ea8e25fcf9199660e57c95ff843c78fe64c75876ab375d024fb1f56657",
    )
    targets[("Schoen", 2, 12, 5, np.float32, True)] = (
        "8d6fce2d7a56d6524166e374ba4a2a62398b5ad7d801bebfce805e9d3a6512a0",
        "34c885a96972281444a64e7a4806234ea38ed6fd0f34cfa00be2a961425f5ee1",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, True)] = (
        "c6ab1030d8acdd4023d62998ba64951c7c4d8884f17b3d25430f12a9206cf8d0",
        "8ca7b1d8c205dfdd1901d5129204ddf48850a62ab6b6223f9bf6cd328ca90fc0",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, True)] = (
        "9871cb44c845d105babaf09dfe7c15a0e5609a9146b5fb7de00ae9ec718d01d3",
        "27228399cab7341b867176170b22b33954ebf0a3775176e9f3f3c9c261e86391",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, True)] = (
        "7a5e57c4d95184deb57baa3abf3f9ab1ae5de2c910b65a25772f12b12a8b315d",
        "e175d86388c45a212a5a9909b87415dc9a56304406153bd771410d186684d635",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, True)] = (
        "233ac98f4bd00c84f7420b393ad179f92ff24a8a2a46fa6433563bd85e4803d5",
        "0ab3d65462ecc5e707e218f49c5960f0692e98fdcd1ed7161c2be1f03eb37421",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, True)] = (
        "88098aab8e06a69d6f5c6a09e7d48e3bb3ced4c2e7e08f4a3bc3e07ea4fef257",
        "eb43dbdc84244733aff248c8c1ba25bf075fd0c2b6f77dcbe2b846e5ed6132c0",
    )
    targets[("Schoen", 2, 12, 5, np.float64, False)] = (
        "18cf48b18c5cb4bbdaba95bc857c0c32603fa7b8661dbe1d541b20510a64ea79",
        "77b607183c54275413a2a5dd0291beef32e54cce61f63cba14a641573f8c983b",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, False)] = (
        "efe04ad4734e30213a8e0c383e6b32f1afa3e62a18b922b1181b50b6124eb392",
        "9e63975237830021ccc77677239c177a5db8aa34c586151862d26597a60402f9",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, False)] = (
        "f9b2fec2e5555dc63617eb7d6282c9c10cf54db43fbdb1b4362eef44bc351cd8",
        "9579ea23af87f255c0c7edb709441d2e53777dec0d7b5e86e36885da4a449e7d",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, False)] = (
        "70c05f2980bcde7f56312ff0805e76ffaf260016357a2746bfc3d855c95c2899",
        "74b6cace25160d9efd7ff4c57e38d2fe5dc59d04ef3dc8fdb897662aa46680ff",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, False)] = (
        "15bf3f2d2d065220651446793fab4fabd7080dc4008ec19d20089ed3df0d93bb",
        "d51edeca276bd94d228bb2fe76d897134e171e24c4c70ed1de1f58dfea80e7fb",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, False)] = (
        "a1de6d8d48eb20baa4b356e55c296d32d86fc4a7877a46731dba0d306f9d2a4e",
        "67ef28b2117f9ceca5ff0fd6ae3ceee5d452ae238475f2b27b5c5e9c3243baaf",
    )
    targets[("Schoen", 2, 12, 5, np.float64, True)] = (
        "cbff011a6f4bd7cb03489a6d694913074bae703c0dc174c650b550183dd9bee5",
        "f5f04b46b8159dab98a645e7056eb87178605877e6f76812c6168a1a791199bf",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, True)] = (
        "98f43de6273e0f7e5a84a5c407c8d358bb5de3c1cc01f4be2ad85e67e3b7ea9f",
        "96341e81e3b70be4e7f5b2803ef3a0d970375e0cab0242b5f3ab553d393e95b2",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, True)] = (
        "36bd3dd92537751fe7a33c91eb48a5567b636b9d8d8a0658d79bf853301d891c",
        "002c0c3715efb7cb63a32a8e324762d8ad53704cd16f791d01668343a2ebcd29",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, True)] = (
        "babd22f191ca37436ab6eff3d45f312ffa90417f69c49cc8e3e7bc7f06d738d4",
        "f1b927befa322c4c22b9fa4749478a234e4c5a1c5ddcc086b22fe0d65005739b",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, True)] = (
        "337bee2cb415a16c657294da5268c87915c63da922ace498629f7c7820ed7c5a",
        "e2e4a378504b0cf43ba26952a07510c4b544c6f6ca1aa56d76ab8ac956c06f81",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, True)] = (
        "4297104343fbd5177b3cb690e28f3e88bcc551b452b67b84da382054e0ca001e",
        "39759d355d62f137216e1837770bce193cec693b541a556807cab60314b90641",
    )
    targets[("Schoen", 3, 11, 5, np.float32, False)] = (
        "d1d78262194cff06232251351eda8455ab31cce58dd85bd4cbca65e0c56cbc08",
        "4ce2ed9bcbfb8c874a1ddb56ab025a33b79fe003c92cb22553a1cf2d9ae4f39e",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, False)] = (
        "afb51643d51313b96b56da33f6bd686464781e7cd812d3815ea9acb9454d81e2",
        "e2ee3081ac44ce2affbb7c1f85340e4ebab7acd05583cbc0e279cd8cf902406b",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, False)] = (
        "18c06349a81694be528397c82b3c9b20ccabe0be86cf8618322ef2b4b66c604f",
        "05616bf7f5afb68bf450dcf081db0e7e6cfdae7fee528e3d8ad3b254f911dafd",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, False)] = (
        "695f4df814c16df11816977fc9b23c02c3e9cd776398cf637f929401062a6fd0",
        "8e12488f1e92c2d8a9db2003f9b6ba553cf22dc999da4ff443aec75331b53af3",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, False)] = (
        "092249ed36762054851efefcdbff29130576aac7d5d79d429fe7099ccc0c221b",
        "6099625c1c9fa1812f1b1d0930bde20838ab037c2e000164631bc3ae9d5533bc",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, False)] = (
        "be4ffa35a72adc1c16e5e1a88acf36a7b822b418528b2f636f35b378f2d7d5fb",
        "837c7baec7ade03558eae6dd5344409b97b0b5627931b7d8174b2505cd341136",
    )
    targets[("Schoen", 3, 11, 5, np.float32, True)] = (
        "c756e20041b294b19ed7b29d54eec3a823d4393ef74a291a42a10a2646c9c8a4",
        "4b9054cda32f1c29a6078454a9c5ac634ad0189c2831b9b10648d58d866fb1eb",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, True)] = (
        "b663e0b8b0b0bfa5c5746afad4d27137a663370c495fb0794d74ea78acfa8d92",
        "b40794541b388f1fba49ee454a83ff9560f862233fc0c8a5e898c54b991c9103",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, True)] = (
        "5ab0058325cb97fb5930663e21dcfd0c33e433b25fb216051d66a1f569d1a55d",
        "29da8cf9484fa67786f1cc2912077d1dff620c5304a2172ff1829296f124eda6",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, True)] = (
        "30e5729ed7bb8a3b437f44bab01d64f980fb37825d21dda5cae6e3095050bb0d",
        "b378b925fd41ff979e46cfe21c3a04a14ff73af5eb5dee11b48a5f31a8a2f91e",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, True)] = (
        "ec9998a3107ce18ec85ac7fae69ab36b44de20f4802768432e637ac7dbfe189d",
        "66a9a0936980a1a68f20110fbffb30683943450cc8a0160f8b37ef2e414d0171",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, True)] = (
        "9045020cc52d9d260ea200bae661e0f361868493a289699770b526b351e910fa",
        "d77143577683045b27c1e5d71f864bdceefe7efa859c5110daed4a5c32e32ae5",
    )
    targets[("Schoen", 3, 11, 5, np.float64, False)] = (
        "913a42d2b001ce6b9fb6786671fd055bbe3fed7cc40dd5aea0ffa739f4bd4e16",
        "f57ef9e49e21391b1f919a46f4f4265b7ea4d57c26af199e64a2195e1626789a",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, False)] = (
        "2c153a5b4381c8c89d89de4c2b31ecc1ddda0a59b06b54471d4a935c0bd9518b",
        "86e57999f8f1251738f89d69b74febd968c36ff37e1a296ed41e0f7f77caeba1",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, False)] = (
        "071a9b38c2aff67de8a8f019a87ea317ca2ae4707e72f045e1d87fe7ae73adf1",
        "0c224bf5f0ff62c7717616243f70d94fe9c545a5febb16db283cd242e4d45d42",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, False)] = (
        "3b2e75426434f4948c3fbe91e2d5ac995c6c966cb2f5f6eec66b0588fd9ee5cb",
        "0d22325111659d3a909bc3e8ee043bc70c056f997909ec05712c9bc4114e69b7",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, False)] = (
        "1890dce517cf9b1ea5fd171a5e53c5f3e785b37dd9e3ad0f60ae49ea3f404927",
        "c5c49be0fa373a356bb1213b4558c068151fbc85ab3834f87a19822a74ec8237",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, False)] = (
        "7ca67171ded3174ff1dc2ac6da3bc6b7256ee942ebb308f1ee38078885b0719e",
        "99d15e86c0da760ba8eadfec72e218f581e19d1812bda821748742ef8fc4e197",
    )
    targets[("Schoen", 3, 11, 5, np.float64, True)] = (
        "adc8dbaf396e15e0706a53da9e7e21bb18cb472b19d65a729a629e6b4c3733d5",
        "8e7d19ee6ebd4a8e5e0e63f208536192f451fa4cb5288e5beeda3ed14e8efa8c",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, True)] = (
        "c3876b685ac737d63ca24ba2a35eb835d19d5a6febf2b01cf87e48bf01562ea6",
        "a6ca4caddd30d6103b6bf47662202a687e73770478d51cd1c272c2bdaddac5f7",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, True)] = (
        "db4c4d87be808dd383e72cf63f5795624eb8bca330891c917547d51b8782279d",
        "39ec1126c90ab95669e970060bbd1835995fa6918dbd506dece24e2e9384e925",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, True)] = (
        "84bd201ad5de322f0c2362df0d095bca3dbe9652d055d1c94801e92e69f55932",
        "89c5e24dd12ca0b89b128df40926513d646df1a8b39dd6588e6921ee6c301624",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, True)] = (
        "80f6d48df3b76a6734a72dbf041a6ee008030e84e335974bb8978bb8ebbe0fe7",
        "36e79e626be97e3520b7b8f329b9940e49e11b9c8b243058a856f893dfc84e04",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, True)] = (
        "da30763ba9342db29ac3bd641ebe92e6a808c3759a7054289493e1b4670e120d",
        "37d6af404abbfe3a7f41c12bf80ea0a2ab27f258aaa2675460dddaafd8b13954",
    )
    targets[("Schoen", 3, 12, 5, np.float32, False)] = (
        "1d0643d262baf3674d1c9c62f0c2a084804004c29477d9b606283f1c147291de",
        "5dd30a4c147b5fe28115138405162b8bcb0e46ee84c6357af6a2ffc3eef66ed8",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, False)] = (
        "7ffe9b0f1c1288df0a2a54d60c1a21f5d8fe77aa211fb71563bba4bf84b33797",
        "575c937e29c1bfdece72107e4fedfb2e69e714adb232a906e38e955434ef58fa",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, False)] = (
        "425998d41afd0736c3cedfdf8f51d88ee2b0f250c588262a16cada97d1e62f0a",
        "c11381d27e0b8d08d05302ff51bf370ef92e3a381b123ac689ab6d1d30dd69a4",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, False)] = (
        "035935cdae861f80dff2224f0e908177625a26f4f967d72af810c61aaed4b3fe",
        "4cd5cfd71a354f8609200d8a79b814dae8b862d8eb14abea1081b8c8e64bd180",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, False)] = (
        "a7672272840484d04f6985f68914d8c805c1905e203e6af823ab881c8c878a8f",
        "e463272fd01db877b2757e378f375b2fdd82d9fb453d32087b4d1c0e660f7f1d",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, False)] = (
        "485015aba9c82542f49e440a06e31eb5c44b2d00e1b0adf800f3655ef901f040",
        "6bd95f9fdb42d24029643624fef421bdee927d162f37c22453b6490ea56c9961",
    )
    targets[("Schoen", 3, 12, 5, np.float32, True)] = (
        "6c91775260b628a786279c9480d0641ff5c72003c05617f632de9582a755efd6",
        "3beba72c2219069b36687a5e7d40354a89bb720370a626a1e2e7c342499bb46b",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, True)] = (
        "b2af3ff0bb8793746d274ea3ed581624a75a1fd7f37124c4b864c1ca9a15b528",
        "5919a406bceb00b5e1fc8ae479ca48e55d1dd636dac86f55faa92aba086a331b",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, True)] = (
        "e8b33ce8be788d363d387fbe6cd0befba37773f12183b3adc9929bfb63614915",
        "09ccbac2b71f528ae43f3280c52a83bcca27731da681580f7cbeecd24d7bba29",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, True)] = (
        "98133d54af8b850a1868c950df923615047d8e3a92209fe046a57c6b5414eaea",
        "654527f2fec096ae7df0b6dedf80c8932943f8ad456b06078461b5ff9cbb4f11",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, True)] = (
        "73931adb8bd6e0e3f79c3e92a3fb576b94108a991f3f687c86df75f0de3c5b6c",
        "a9d9b1bd308d690c72062a9776b2daaeb321355c4cdf6d06662f14dc9e5dd774",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, True)] = (
        "04a633808c8b4e3765fdbc61bcf8be097316a51b5aed71c7b90edafb55baaa40",
        "9df20661c3240a15881f1f1a0b445a48a9fb69119e165e2c1ce69c6234df36c9",
    )
    targets[("Schoen", 3, 12, 5, np.float64, False)] = (
        "8d38619660d53af3a9c6dc63735186ccc9a9b4619dd0575e2b53e83f15075d65",
        "caf7b542a55dbd59cf68dda1df997b8ae7682b4616330320b6f9aaaeb3b6dd09",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, False)] = (
        "690e050034717cf1b4924ce5e3fd101df6da881efa098ee258fe35e370dcfdf0",
        "e6883582b2c49d513463711039444c2a561ae71cb44fe7f6c1d1ea1fbebef422",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, False)] = (
        "e646d2fcb5791636fd43b0c0740fa2f2c3b74c389e957c315dea90069e8b9068",
        "166a86fa079809b26a90b1004be41fb844cc07c1a5aed13414e745af68ecd906",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, False)] = (
        "7e78e5d498963b3aef869ed1c473efa9a22a5f198ecb41f4a368d08f29b70d6d",
        "85a85d6f2349a552286c345ba2db347c13b8ddf2a19839bafdb6c1fa25c37fe8",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, False)] = (
        "e762aba0411d216dff88eb9329be496a7c7cd3bc598ad770da93d16d7039ff64",
        "46dacbf815e61d69f02899d5c043428c57b40a010bd2079d91106308506d7793",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, False)] = (
        "f5065f46b31c5b86b72937ab287ec55009d50c054c7654cc84be5f8370b8b878",
        "49186be84307f062e2ee8e33784ce3e58ca37f000ce7627fa298fbd3112ee175",
    )
    targets[("Schoen", 3, 12, 5, np.float64, True)] = (
        "3e389d1f46eaf63fe2ea47a347a896916bc614e122eac62dc526cf8b52624b14",
        "dc5ea85e4b8f7382eb6e07ed0cd69da170e2802c7e6b2c2129c8e417af69acf6",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, True)] = (
        "2cbd3a6cbbd61c710d76358726f24ba7980a3ae8224868bf584d582e7de55b8a",
        "e06710f6e8f6ec1468989666382ec900d733748c27d345e44db2a9b0c74639c8",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, True)] = (
        "92525637b84da9ec68665e6035c8e1eca2ca8fdf960b2212eb6c67bf65c9a386",
        "ca8574b386c6285781aacc09795a21ca5a4090d8a361cf43fabfd077cd566e72",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, True)] = (
        "04a6a1c744731ef8a25330fe30140e02a9e0a52f6da9486d57a6abff550ed1f5",
        "24b84549994885fcbdc778e483ff5c83b7cae0a9adfee2fdf268a84e5fc92dab",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, True)] = (
        "a003c555d93292b30d467606eaca0ecd5ad1f7444f37968d26ee552da8004495",
        "9367c1dcfbdb694bf61459e35e53f80a83483725013e613a8e2a4648d7aae8bc",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, True)] = (
        "9ad2ef82a715dfef334ff1a042680fadc074671979893fde411094820a2efd97",
        "b9cdae830a0af2698c3826c99e2721e6794ecf5ed91cf8db738637d3439daabd",
    )

    periods = np.ones(dim, dtype=dtype)
    functor = impl_functor_str[0]
    functor_str = impl_functor_str[1]

    func = functor(periods)
    if negative:
        func = qugar.impl.create_negative(func)

    info = (functor_str, dim, n_cells, n_quad_pts, dtype, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

    # print(
    #     f'    targets[("{functor_str}", {dim}, {n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {negative})] = '
    #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
    # )

    assert computed_hashes == target_hashes


# if __name__ == "__main__":
    # test_disk(8, 5, np.float32, False, False)
    # test_disk(8, 5, np.float64, False, False)
    # test_disk(8, 5, np.float32, True, False)
    # test_disk(8, 5, np.float64, True, False)
    # test_disk(8, 5, np.float32, False, True)
    # test_disk(8, 5, np.float64, False, True)
    # test_disk(8, 5, np.float32, True, True)
    # test_disk(8, 5, np.float64, True, True)

    # test_sphere(8, 5, np.float32, False, False)
    # test_sphere(8, 5, np.float64, False, False)
    # test_sphere(8, 5, np.float32, True, False)
    # test_sphere(8, 5, np.float64, True, False)
    # test_sphere(8, 5, np.float32, False, True)
    # test_sphere(8, 5, np.float64, False, True)
    # test_sphere(8, 5, np.float32, True, True)
    # test_sphere(8, 5, np.float64, True, True)

    # test_line(8, 5, np.float32, False, False)
    # test_line(8, 5, np.float64, False, False)
    # test_line(8, 5, np.float32, True, False)
    # test_line(8, 5, np.float64, True, False)
    # test_line(8, 5, np.float32, False, True)
    # test_line(8, 5, np.float64, False, True)
    # test_line(8, 5, np.float32, True, True)
    # test_line(8, 5, np.float64, True, True)

    # test_plane(8, 5, np.float32, False, False)
    # test_plane(8, 5, np.float64, False, False)
    # test_plane(8, 5, np.float32, True, False)
    # test_plane(8, 5, np.float64, True, False)
    # test_plane(8, 5, np.float32, False, True)
    # test_plane(8, 5, np.float64, False, True)
    # test_plane(8, 5, np.float32, True, True)
    # test_plane(8, 5, np.float64, True, True)

    # test_cylinder(8, 5, np.float32, False, False)
    # test_cylinder(8, 5, np.float64, False, False)
    # test_cylinder(8, 5, np.float32, True, False)
    # test_cylinder(8, 5, np.float64, True, False)
    # test_cylinder(8, 5, np.float32, False, True)
    # test_cylinder(8, 5, np.float64, False, True)
    # test_cylinder(8, 5, np.float32, True, True)
    # test_cylinder(8, 5, np.float64, True, True)

    # test_annulus(8, 5, np.float32, False, False)
    # test_annulus(8, 5, np.float64, False, False)
    # test_annulus(8, 5, np.float32, True, False)
    # test_annulus(8, 5, np.float64, True, False)
    # test_annulus(8, 5, np.float32, False, True)
    # test_annulus(8, 5, np.float64, False, True)
    # test_annulus(8, 5, np.float32, True, True)
    # test_annulus(8, 5, np.float64, True, True)

    # test_ellipse(8, 5, np.float32, False, False)
    # test_ellipse(8, 5, np.float64, False, False)
    # test_ellipse(8, 5, np.float32, True, False)
    # test_ellipse(8, 5, np.float64, True, False)
    # test_ellipse(8, 5, np.float32, False, True)
    # test_ellipse(8, 5, np.float64, False, True)
    # test_ellipse(8, 5, np.float32, True, True)
    # test_ellipse(8, 5, np.float64, True, True)

    # test_ellipsoid(8, 5, np.float32, False, False)
    # test_ellipsoid(8, 5, np.float64, False, False)
    # test_ellipsoid(8, 5, np.float32, True, False)
    # test_ellipsoid(8, 5, np.float64, True, False)
    # test_ellipsoid(8, 5, np.float32, False, True)
    # test_ellipsoid(8, 5, np.float64, False, True)
    # test_ellipsoid(8, 5, np.float32, True, True)
    # test_ellipsoid(8, 5, np.float64, True, True)

    # test_torus(8, 5, np.float32, False, False)
    # test_torus(8, 5, np.float64, False, False)
    # test_torus(8, 5, np.float32, True, False)
    # test_torus(8, 5, np.float64, True, False)
    # test_torus(8, 5, np.float32, False, True)
    # test_torus(8, 5, np.float64, False, True)
    # test_torus(8, 5, np.float32, True, True)
    # test_torus(8, 5, np.float64, True, True)

    # TODO: Skipping this batch because it's hash dependent
    # for functor_str in impl_functors:
    #     test_tpms(2, 11, 5, functor_str, np.float32, False)
    #     test_tpms(2, 11, 5, functor_str, np.float32, True)
    #     test_tpms(2, 11, 5, functor_str, np.float64, False)
    #     test_tpms(2, 11, 5, functor_str, np.float64, True)
    #     test_tpms(2, 12, 5, functor_str, np.float32, False)
    #     test_tpms(2, 12, 5, functor_str, np.float32, True)
    #     test_tpms(2, 12, 5, functor_str, np.float64, False)
    #     test_tpms(2, 12, 5, functor_str, np.float64, True)
    #
    #     test_tpms(3, 11, 5, functor_str, np.float32, False)
    #     test_tpms(3, 11, 5, functor_str, np.float32, True)
    #     test_tpms(3, 11, 5, functor_str, np.float64, False)
    #     test_tpms(3, 11, 5, functor_str, np.float64, True)
    #     test_tpms(3, 12, 5, functor_str, np.float32, False)
    #     test_tpms(3, 12, 5, functor_str, np.float32, True)
    #     test_tpms(3, 12, 5, functor_str, np.float64, False)
    #     test_tpms(3, 12, 5, functor_str, np.float64, True)
