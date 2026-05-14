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
        "9aa6582ec48c855716a54cbdbb2d697f251c242daa8516fe58848c14feac3b76",
        "e3067c90146df8a076f3d69fbaf51e16d4779bdea3ae1a7cd3671b6291aaa83d",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "7bf6a55f8817cb38b736f078aa210dba5a4f2be2b58126f2943672bdf187c947",
        "5f632b125e3d5cad466a5ef3d8486ab3fb1f9450e1bc74c460da81f0641a7f43",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "691639ba18aa80e761b00e904b1c97ecc7790598436d19b2c594aad8bc74a5bf",
        "0019b0e8c0d4a12cc8b6c9b8bcce78a624faba5582924dda9fc87014f9504daa",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "ac7ac28436af7e00c2faddc359cf36f9e70fb89128696ca72eeb90f4cdc9a6ae",
        "5274fbf8a298bd156d0aec259096c4381944be4295907a21faab62dfce399a98",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "9ad298f46be497cb1544ea8031e47870dbd84e35ef2f72c7d5c4ff103646ef33",
        "d11bf799a7da7c2ce9be73c1c503036f71067f07cecc028a0cc93a3dcb4a43ff",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "b91a95574a7f009a4aee92a838996e220fccc1ca21450b87d4de52a3fc8521bc",
        "e5fcb0944d26df47113d4f5951b9a7a454c335776e88dbb675fcbe772a948790",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "ea13981528f836b653f19ab9843466af465eab8ce7913a127aafc1e6d1048302",
        "e5d9d3aa024b18f3a2fd35c3e80ba226cd8d16cdda47d42b2754fc04e951e667",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "96dedd8f8747139847667e6bef390e9e111862c82c58d20cedc97f5cff7c091c",
        "baa19487aee8e73e7416ae31cb40667bc5830a48bc47f10788c9134f3b57e32f",
    )

    info = (n_cells, n_quad_pts, dtype, use_bzr, negative)
    target_hashes = targets[info]
    computed_hashes = create_quadrature_and_reparameterization_hashes(
        func, n_cells, n_quad_pts, dtype
    )

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
        "5db54ae36a727926c9eede4425862bdb13e75eedc9d259de7662f3a5bdc0b5b6",
        "7693b5becea4543b2309220be0760e93620dea6113cf3ee01ba4142b4f19feb7",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, False)] = (
        "5ab35aaf539a1331c6f60cedf496eeef9e4fb4912e144a36001d8d5e69577958",
        "623925223749b76c06b47a72821256e8713369d63cff441c1c5dca941430a122",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, False)] = (
        "4a1a944ad43e53559526e0055111e788ca65018fff46dcec80103ea5f75d631c",
        "e8b62c05efecc691946f65aec73559e1384d50a589c76aa6efe834eab6322bed",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, False)] = (
        "b519e3a47fa64dbc2b5c687026f5a0875b366a643283222a1969f4d5657ee994",
        "a482b5541383cb9df64f580e4c328cb31b289789e2bc479c4a60f4af4fa64756",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, False)] = (
        "fc9fb87fe0eb420f678f6e32c7f6ad49dd6d820579314cb830a7be049d0634cf",
        "35b7170edb4e313a2a10a489bbb24d965a2e59fa54a4d7f17304a59ee72f6d7d",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, False)] = (
        "dff8a662a15195520278cdd6f0020083f166538fdb43d87ecf387be1f6b02649",
        "754e5e8ad985bbfb491016c6072545594c7c032dcbcb2e39e554a5741c6b5d1b",
    )
    targets[("Schoen", 2, 11, 5, np.float32, True)] = (
        "99e60a4227fe4eb2a99979b0d6e053d0e212e25973290e52ff9486c432143c56",
        "6f644f0942377921d8d70d0b99edd89b36e7aa663aba810c803f0cad373122d1",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, True)] = (
        "c34b08b1bc080014cd74c12027e890e465611950965555006b11098e7e0c3b9f",
        "23eebe32cb90aeef0a06edaa5b4c12a6b018146d0fb647169d1a341828da0ef3",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, True)] = (
        "1a00bef633faaf2c60a7af99771fa42d42558ac8b5b803e8deb09f3368a009e4",
        "9f028fe11673fbfe1e35d1a53bb81a20c9e9cb0c8996115940aa15a89976a851",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, True)] = (
        "f098a2692cca5eacafa436ae4ad683394eb556dbb4affb9ca372dbb564d19fb0",
        "4cc3444c7f7dd7eb4e3420b36677fa975e5954da36da6427923a96d7ffb4c7bc",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, True)] = (
        "bdc69c3bbe489ea150783b90044057695933de361353ce600b5eacfad37cd7a7",
        "6e2cf86feb103b7b94e746f1bfc09d4591975438a11f36dd1903bfef75a35e4e",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, True)] = (
        "328489eaf65829f877dc4e0177333572c55d11a14aa2418600a1755db3a12e38",
        "4cbbfb4ebfba7fd219f20955bc4a9a43a1dcadec3a43c8a96d5b20341643de86",
    )
    targets[("Schoen", 2, 11, 5, np.float64, False)] = (
        "079f54bd2960fb7b8fd2cbd8da367624ebf5be1dc47891ef2b4c86aef69f5a80",
        "cb157a128254f1b5ee4a8ed982981b3cb7005ee38e5a786d2cfd424ef99cb72f",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, False)] = (
        "7ed86580dbd993deffc4f73086a4f24e2c04b2e3df3995569dd83cb0a46edb76",
        "e5080b28c180fb4d920a41902b5d85921d6b940742ad63416924a6c0bdf383b7",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, False)] = (
        "33eab6f7097bcd27646dd2376a8bb5f483a5b4349f5fe466bad8b994b5c8acd9",
        "cddf8807fdf30a19f68d2795613d5f73f1fd02f24c44750470e47731abc30fb2",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, False)] = (
        "da06f0cc6b08f23fdc35bcadc6fa685acf7a5e3bd913f84a519fae259c27557c",
        "30d9599e760b1cde27a0962945f8e0775b1ab7d1bf7545ac52dd0ec4eb075974",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, False)] = (
        "fab59988cffa1ab499c14cb086e8ce6759b2b37772517629fb9aa76e92151d0d",
        "5b5191587e2e195f68fd6b92b6fde03f202f60901a84ef3e602b8512a94a86df",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, False)] = (
        "b1e15c1a37757a2357ef43feeb806e431c3baa4b60c3aeec4a5997e64b4b5f2b",
        "95b426e48c4b3206cea450f86f04735c0c38112e1cd337ef65445b7a4b127298",
    )
    targets[("Schoen", 2, 11, 5, np.float64, True)] = (
        "9654a1e4216737c7de5de23288d3539c4f2db1b82b128f12059dcc6b82caca99",
        "06915115a69640216bd5ae4d98a6abc929f7c5bc4c95e11b7e009a684774edad",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, True)] = (
        "c54d937a94aa5da0a035f77da3a24edc33ee2f0ab865eb390e631ca86d288b90",
        "206389569f33ceaf6730b09767af80fc109897916c2d661f2c792a5c013682b3",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, True)] = (
        "ff03379e37684d0b1d2aa519e8a0b3a5579b434860849dea2a5253342d9502b4",
        "2f2efda54d674a38a20f307e8219b7c08203b8bfc47a2a204a2d656ca7684866",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, True)] = (
        "645d8957a33d369024a6fc0c08cec1ae9c3f8262afa6dafe40a25d1c5708f838",
        "da3aa40c9d7df40c532629d04190afe23612b9a8d3043db2060adefeefce4014",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, True)] = (
        "77398df8ce7ff6e526dbb48d20e444966f859575f218402a9da4f8b80a9d6f23",
        "84a860c0ac9eb2e1d799c51fe097cbe9df5128f88a58215c76af17a5905e59c0",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, True)] = (
        "9064b5e07e25a5600a3791ffda9bfd2002b935ec3e81db2973525cdcfedf00b8",
        "9e981e19af26a982f6b9c2f966e5aaf50d6054c692bc6c6e0c4f1f2399183425",
    )
    targets[("Schoen", 2, 12, 5, np.float32, False)] = (
        "395f7bfc216003d683174c21a4efcad800c1dc144a2f976ebc0495a45b3183ea",
        "6d3482e58211e8b3fa590492d4f48d15d805ab8ad3d1a15abc0465a8bf611feb",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, False)] = (
        "e64124272d1b1a79d5a1a737780b537f2901e480deb946dd58f5db46469e7b62",
        "dbc066fe3519360d2eb7e54e2f10205920915d7cad3f1a0232427d58fafec607",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, False)] = (
        "e56ffcea15600d8713ed3d0a5613a708bf2b4b93f5b9ee0769e633d70ca34e85",
        "e7067e8fd113bfcabb0d6f6676fe9d56d19593bf162d2c10a8bd85bb0e864308",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, False)] = (
        "16fc0b4c88c2ef4d17f04c54bc7aec1843f055a497ff3c1998ebc19985ca5024",
        "2a0d9809b448cae48f34b0b75f478b27fa76213e4253d7539e3b6e13c9672f9c",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, False)] = (
        "d82bf1f0eb6bb5edf1e7d03f71740c9b6c6348e1245663a445a43a1f32d9d8b4",
        "fef5418ce216ae0be66b2e720bc55cd9933de941ed29d7f2360f2766965786cf",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, False)] = (
        "0bcd2b30c50dbadc7917a379f46b4606b55a4a3e7796f69294b02cebf39b22dc",
        "159ab54ac859cfc9f6d2cd2e487645cf49e5b527a39365bd4cd71667a0cea655",
    )
    targets[("Schoen", 2, 12, 5, np.float32, True)] = (
        "f887c53856668f452f099558ec3e749db9ba9604dad754cb626b472d4e8d10b4",
        "fc25f539f4bbbb271fa9d87f26fbb397c0978a0b1972c79ac2503e243405c936",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, True)] = (
        "931fbda3013996b8b8e984054098886b57e67f80b9ef8d4cacace6091a295de7",
        "163d6c2ed916b8116b4f0cf9ef11bec031cf67f4c395ef0a465174af1c85c8ca",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, True)] = (
        "2c3c7a4b22ab0a8307544008cf3f1051b9d4c6029196b30bb0ea4cdf854d1104",
        "ddd797ea627a6f433f2b30d456b1739b3c36e427da38c2b35ad297024fe8c0d4",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, True)] = (
        "b728442d90ad1334cb183eb917892ba9be324bb26d37d4d622a0adf8d9cee442",
        "dc83146ab98aa0e3849bde7c7e501d56f8134a0327db199d85ce51118f23db3b",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, True)] = (
        "d82bf1f0eb6bb5edf1e7d03f71740c9b6c6348e1245663a445a43a1f32d9d8b4",
        "227bc0984dc96d5556ac4a7637f3b159972eb36c443724394864aaf7fdb12a70",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, True)] = (
        "7593cb6fd6c9a5df96019ea4de76860df1db834109266da8d4337c66759ae19f",
        "9c317c8551a39fde25c3b63837ce19b39830f7e572dfdbdca39672c1f3a2ef07",
    )
    targets[("Schoen", 2, 12, 5, np.float64, False)] = (
        "bc9f02cfbdd88a4da4e46c9ce09f82de1ab9f38cb6226b6259c05c343f40b7e2",
        "50b2304d85d8cf5b7673049a1f84e95c75428bdc0f0545db29fcc842a736b89e",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, False)] = (
        "09ae27cf723bfd6ae9d69131c2f9a0677a26b8d8386f7203dbed5d48bd4448a0",
        "07d2e7b9afadc90abcd832c331adb64e6f829899f1af0fa3306d6a3820377be6",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, False)] = (
        "9467334ed139244a1e1b9bd210f45fe8a4b5f94ef60bc5a4986b5c85c83769d6",
        "07a85a50ec17af026496f0e8bb211128db94fe456cceb2e2601c4c2b5f396d93",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, False)] = (
        "a8cf68b6d448cb07914ecc939cafb8a5244b186aba4207f820d07d16a7e7298a",
        "ff0c2214921464b44c509dc7c6c2870c9254458046bc0762b712d3f376ab4c95",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, False)] = (
        "990e6ce72bf91ea90bb85b57b218928edbb9207a5dcf9f2be01d825d67bac3e3",
        "41b6db2c5dc6c958f99e41ce7eb04a6962b1f95a869c00598257e95ea91a886e",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, False)] = (
        "3084def0b82afbfbcc103cb6464b017a8fec332a0c3f336fc2f32d1e94d42ce8",
        "816ae70585ded7c6008b6ce31e3fc7594e02224e429210463298d9893b39d572",
    )
    targets[("Schoen", 2, 12, 5, np.float64, True)] = (
        "3afc0623411c71d654f7f73a6d37fd25a2d7c32daec6bddd802a7a7a1ea44cda",
        "4761625e95c779138401276b45fd0c1e7d75f6c60ad7fc31a1cf02018162858d",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, True)] = (
        "09fcb32b9df12415babb8efe9903ad69f7805fc48b3fe0c020263c02c9ec4a47",
        "63724596bf15ce91b3ca7a53e06a09505a162a8ea07d2bf666c76f7a0f7730a9",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, True)] = (
        "0dbeb43323be6f5fac91084f806d8085751b3d490760618ae55f23c4be605b7f",
        "dee15650602a42721188c2f742c6a5641b3b07cf6f5e85735cef44a17b14880b",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, True)] = (
        "d47e2733837afc02b7c68927bc3a5891c16a07ffaaa8d0b7305ddbcd176c05bf",
        "73f330a620407bec290371f70673104ebd7ed7ba4f7c0146c7aaed036e0c1e49",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, True)] = (
        "990e6ce72bf91ea90bb85b57b218928edbb9207a5dcf9f2be01d825d67bac3e3",
        "4c315168ae4cbdc9190022a943b0a99d0c72a26e8b219637564a16be6d93bc8c",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, True)] = (
        "739b13b5981fc47d94a1a8152dff3b7d2f0634aa70c07bc1dc77b2401a471257",
        "2b721a6b7267e1d1c8c44294f6970c69c98de99d5ac3133f41efb34071d4a859",
    )
    targets[("Schoen", 3, 11, 5, np.float32, False)] = (
        "1248eb35bc06e64a9284ff5105359b8f903beecc155de1b4101b77fbe04c9d64",
        "4c09c04de29ed7f3b767cc34a37fa6af13458de2d2bfb3e84e6b0950934e53f7",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, False)] = (
        "8c370a8bad3c377824e8e78e5c7319203e22fefb650d3a41cd92d207d82c68e2",
        "8df9db5d718ce57a9d178f873e0d562c20e38e23d13355988209b3e5634c55f5",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, False)] = (
        "af8ce14b4e01a672b3594e28042eaee45779c411e1cef24072a1c7e4320821ca",
        "2eee79f21fcbcb6da718cebf71b5cbfbd09bdaa27198c3a3d8cd9be257d740c0",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, False)] = (
        "8529e7ef8cab3f0a6de6e4f5d56ce135aed50decc84f88f601cce2f2e678e095",
        "0d4963cff37c8f955811d9087235ebcb2ef2bfc18125b2df2096c59703f244ef",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, False)] = (
        "13b3c2c5bd21ce9c8b220e104053e7b257c57f0ef5c7b1fd7e7810b81a9a02b3",
        "a4fb11477f987fea03b710135d8300b14e69637fcc9d4206eaa0457fc8da9ba6",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, False)] = (
        "eefa08c56328fabadfb9a7c00c642014a045c74cdc1dcd4ead1203fff439110f",
        "0f07c527cf476028f8bc4e58ce8a19477541b803c3855e8b59fd9cca78c128e0",
    )
    targets[("Schoen", 3, 11, 5, np.float32, True)] = (
        "afaa95715524f1783620f5d1405cefe5fb9666138cd0c07e4a674b48b358947b",
        "01a80d65020c812a4d1ca6255d587cded30c6508843134f42278b9018125dbd5",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, True)] = (
        "fac0e1cc37c04b34e115d0dbe66e2ed10613a760c898caef608000df01a790c7",
        "dde24a1a014aa1cbf075383397e7912eed553f0f27de753efa2074a2cc48fadb",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, True)] = (
        "29b4a0e9920ab3278706b5355a5fbf9b17c0a1718b819098bb4d9a5ad0c5812b",
        "df3b06b91c481276bf5cd7f7d50e59b8acf3314c1ebdf4edba65d8f804b57e0f",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, True)] = (
        "25d28604ba1cce3364352b33e78bcd06e54debf730be86a996477263bd271487",
        "e3c592e4ba4265fd4d1e32d4374449f9aa421ba94f16b92f43836f3cdffebc30",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, True)] = (
        "2547a245ddd765ca63cea9d1b2cd5b902d88427bab4180a9fb2055eb0c650bbd",
        "6ce1e51a85f0a7b7fba0b43f7e388989495944e642e010de17463c5679c2aad8",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, True)] = (
        "ed4be96256dc274872b47ff941c687ebe122977837768babe27c1f2cd955b6de",
        "be49effa7a12df17ca3e754603942dd052b28ad679d33bffb5c728fbeddfa7d6",
    )
    targets[("Schoen", 3, 11, 5, np.float64, False)] = (
        "5ecb976c840da3a98afce1b4c8f6153945b91bc9571b1bd22ce46c7958d416e2",
        "f981665cb28fcbd76d1a09beca28074aafbee5897626ccf2d01909b26e845663",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, False)] = (
        "ea34c9974a35b878e689c4d2a5a68a7fc8736c68da7db9cc4538e734d1b1748d",
        "fd918a8e5360a6b296322d16c3c749e31c20e5f683594a8fd32e94b2d77ba161",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, False)] = (
        "edafb6f9a6e12f0c3dc8ea55285c02dd1749f6bc20046bcc2b88d27fff461105",
        "dc54773544d5ea3ae83a99e8864f380c6fb931e96839d705c612624751769724",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, False)] = (
        "4001ecfc1f8b6b0b11d33ac73d19f409202e30401c6ff3bd64edc1415e0f5f11",
        "94b39d579e7aea8de2bbd5a4ac975e539b97c073890d23997e346aaa2f317b41",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, False)] = (
        "e0763c8e6739fca31389d4c40717f6ac6eeae00f4c4754dc2c91d946eebdd8cb",
        "9c963dbf17bc7f9fcfe99b1df19befb476786241a519b528079d9df249f6dc41",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, False)] = (
        "18c37b9314a0bf8e864515b685f044f01238c0ee5242d5517502cfe393574311",
        "d1f642edd20e0f575f1a0704fff79257d394a1e506598d14d43cf787571b7ceb",
    )
    targets[("Schoen", 3, 11, 5, np.float64, True)] = (
        "e15e5c2256ff3a7e2d76150246870fb94d0c6618b474e57cb1f616ae2604514a",
        "61595b163a00b73b98e107bb99d3afd6971731ff357ca99cb468c082d8e255fd",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, True)] = (
        "a3107fed450a1399b63b657be6aac0453ae4ce5a5fb55612452aa8aacd7cec03",
        "b70559f6b77b44fb50700a6b37eb2b254396702e60177970d90346c6dc87b17d",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, True)] = (
        "e933ebce84a4f7b7c6f47e499c58e678fac59bb09c74c41cf4617b70095324a0",
        "1beb600087e65c535c9b27a363d154db601dc03f8face5932e00db5c60b22b88",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, True)] = (
        "53c55b4b28c2ae5d5b852c2ef273c110a0dff7b7689891bec6adee869c365d80",
        "6e676811fc2dd6dfd1700c29f10b634a64d7ed0870ce05c5de42a9dbcb44db0f",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, True)] = (
        "2eeaec5c896ee0e97e8160236e5b0d5f419b8310d50c09f30dadcc4425cda4fb",
        "7c80347b6ee57db158adaf59dd8804cfa6c96e72bb376db912313f17cf9dd356",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, True)] = (
        "15f6a7e5bed18d6858ab8c12edacb7b87f85f542fbb3bebf068254000e6fe57f",
        "123328f5c835d8cb7be66352ac76204ccd5e5fbf5c57608d88c0244fdef9c00e",
    )
    targets[("Schoen", 3, 12, 5, np.float32, False)] = (
        "877c7355712276671d70435289a1224ef74f5279ba525aa90a708abfe3d471c5",
        "e35c1c3079ac56c9bd9102a01b1336456f6744d74a33c2725397d4cf6b802582",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, False)] = (
        "3005ef9f3e5a089f37c0fa76954e246a0c62acec023dd0cacf07273a3307ff9e",
        "991a8054aa0b8fe68f62bf57fedc87ab00473d10430ba968085027aafb272d2e",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, False)] = (
        "d9dc6c1caea87b93d273f77e6af3fbad664e1a1f2c1ff5dfc92cf39b82886cf1",
        "9f550a20f46552e4300704918d9fa81cba000e28e78c9bc060b13ce039702c4f",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, False)] = (
        "5334fa12da255cfb585701d1f812ff2b1287833be2fb9de1c2d0e2fd1460587e",
        "fe7741fe6c36c8e88af5d9b4c22fef7425d1385597ef58e4c5b8b156f05cfa02",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, False)] = (
        "708001e660e267bc5decbcad2963ae9b18f1b87e1a655a70767df6da28cc6e4d",
        "5bf481d7f5c60b2b081902b74fd8dc1b0aa04b5bbe6ec275b34cd61238b5db49",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, False)] = (
        "4a1ade1f4b6eb16a5f60ed1972014cd0d8566044f12bddc4c9f8476e6fc57082",
        "4ad8d191baf01fa7c1e4a0eeac6523bcab4eca69f08bb0acc121cf705b870af3",
    )
    targets[("Schoen", 3, 12, 5, np.float32, True)] = (
        "30ba81b9d33698d61f63b73ccbf1c87b8330c9888c29448409c80ec5b359e311",
        "d8eff5b35fee77f9ce5847afdfcf03097f4c28926b8c33bb5110b32021194dde",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, True)] = (
        "8dfda83de6768f4b815f6c5ec2eef627a75ca92a47cb17bd9bfd9c2a1d1d941d",
        "c555add53078bb6a2f96bd20ecda80ead58c37166ee854c585dbe7f81bab48c4",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, True)] = (
        "1e9cf86889819a7afa71475cb03325d38a2a75bfffce90918f2afbbdd5e5f55a",
        "cf66043c138c65ebe0a032aca733bf3c818cbee665c5f070a1a3c68154a42c4d",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, True)] = (
        "d398f961f35089932ba2fb4a66182211195ed3957b71a40ef3a08d57658415b6",
        "4942e3592bf5c41da789523a7114a3c14cd9c75478deb2981da4668b958ac1c2",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, True)] = (
        "dfc39d9020b8b50aabb09c80efd15f10632164dd462af8adbe8145f7677897da",
        "a3323ca58f933c33c4ed3239cd352a12f0d06a00f6324bae8569645ab9a0a3e7",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, True)] = (
        "c3d91393eb3c086f918c1ac4b43345b29f982795a780b444834dd0f552d2e430",
        "a1775b6c71e64d2f7e05467beeaa436b73478ae28809a9cf2a9d4a7320e460b0",
    )
    targets[("Schoen", 3, 12, 5, np.float64, False)] = (
        "982c532cb7e64f40f17fc0d14b5f18cada6ab439281689a2f22a7e888688db94",
        "e8ec73ad4fceadf6d29bb0da3ecf102ddd463c0bd32261599cc5279a1b1069e7",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, False)] = (
        "1ee2dc6a7cfb7b68fcc4853ae76f66865dc29722e0c0dc98c139ff72a10631eb",
        "6b73860a765c187d321642e141cc4eaac7ef1bd7f8b0d2c1eae4933129241125",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, False)] = (
        "598cacf3a22212eab8ec3efc8ff3837dcefe23bb9cbb24e71900dabbfce35ca1",
        "5b10cb12ce3fa9c95eef5afe497298992b4ddc8c1b53dea50b052ef98621f60e",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, False)] = (
        "c0947c71e47efb4fb4393b0283a0bb074e41b9c6eb402c6959ed9bf62c89d26c",
        "1eba6505f2d7c131130ad56eec0c53076c1e8bf1833ae0977e34fe926162c515",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, False)] = (
        "253c849679a78c7ac7854a72dbf3693769d934d50cd041e0eeed6bfb87fc9208",
        "c4c94a48f892998571d04213b3f05f8e81fd65167ec668b402602400a3d4318d",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, False)] = (
        "4dfc6a33f75cfb071468948c71a5e035e4ff7470ef236a40e649581acedbbac3",
        "4ca262521aae2a5edd5893ca05098c850a0e3fc185f351b4bd77d7c7ba684398",
    )
    targets[("Schoen", 3, 12, 5, np.float64, True)] = (
        "1d06129afe6ec2232d6eb7edae765fc4ab1438b1a220bedd775e9f043e51d19e",
        "2f5e4088e90c40541b22846c17e4eafbe7c368e2569a303e2374cfddc23b4ea0",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, True)] = (
        "532eda382cdadca268a934ce057e3fe22a0ee51d46fcf9884a8934dfd6d91397",
        "014142c45edcf563dbb2d8563f886e65a6afb96688e39478a3b933b3f679e57a",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, True)] = (
        "00f8686c0b908d4d972cafad7bfbfa3b958688ab3185818f3a02b0e86a5a360f",
        "80fcbeabfc8ed941af10449fef586c27b54e1b6264ccf84f08175b9f64c3309c",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, True)] = (
        "2731dda8cff661cf9f8f8c08dfe2a5c42c66895f521d2844cd979ee6301ace6e",
        "c6861133c15c9538a49890468da4aada718753f98b67e14a0be8753855fd1151",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, True)] = (
        "2d3d33ecb2bd4b76169771f03fa7352e4791c1eda68e9f0bf8533606a75e4ea2",
        "093973ce7b9c5bccee28bb2edddd1b062dc443862d3dac5b2c99c6e929b2a4c2",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, True)] = (
        "cd27ba2346db42cffa7a5c9e6f4a61c3c91afe10ff74dcb45fc96863af3e8334",
        "5c42e7766b8261d21df5172138c785e5b772f53f70a5598b73efd74ef595a387",
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
