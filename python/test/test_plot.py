# --------------------------------------------------------------------------
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
from typing import cast

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
from qugar.mesh import create_Cartesian_mesh


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
    cart_mesh = create_Cartesian_mesh(
        comm,
        n_cells,
        xmin,
        xmax,
    )
    return qugar.impl.create_unfitted_impl_domain(dom_func, cart_mesh)


def create_quad_PyVista_grid_hash(grid) -> str:
    """
    Generates a unique hash for a PyVista grid based on its points and cell connectivity.

    This function computes SHA-256 hashes for the grid's points and cell connectivity
    (string representations), combines them, and returns a final SHA-256 hash representing the grid.

    Args:
        grid: A PyVista grid object containing `points` and `cell_connectivity` attributes.

    Returns:
        str: A SHA-256 hash string uniquely identifying the grid.
    """
    pts_hash = hashlib.sha256(str(grid.points).encode("utf-8")).hexdigest()
    conn_hash = hashlib.sha256(str(grid.cell_connectivity).encode("utf-8")).hexdigest()

    combined = hashlib.sha256()
    combined.update(pts_hash.encode("utf-8"))
    combined.update(conn_hash.encode("utf-8"))

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

    reparam = qugar.reparam.create_reparam_mesh(domain, n_pts_dir=n_quad_pts, levelset=False)
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
    targets[(8, 5, np.float32, True, False)] = (
        "556f1aa59997b3583f8aa1d663ef01877ea89429ef6b7c218e5914d5599089a6",
        "201df42fc91e4537eb9732ff0cfdfcd65176d75a672cb15deab98dba884f8354",
    )
    targets[(8, 5, np.float32, False, False)] = (
        "556f1aa59997b3583f8aa1d663ef01877ea89429ef6b7c218e5914d5599089a6",
        "36cbd689151b2c0b4c75161aca6e2725e13ad4f5f9d4d9e517a80f7895aa07e2",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "9ba26f11f291f51e9c631ecbc6d41958d132886acf5839a70fd08ada25d05dda",
        "56cb2ee82b0574f7586b30f1ddc2e85fcee6bf3930a43f3118f1130a1005ba33",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "9ba26f11f291f51e9c631ecbc6d41958d132886acf5839a70fd08ada25d05dda",
        "1c9acb16a619078e8f48ee51c8b5568b68c9708cce1cfe42a8034863dd6f1c4d",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "d3a096e4e2481330c1a299c0fe118ad2a4b2c0f40d67abd8bd17cdc7ba777a3b",
        "b54fe1e2d98a330ed0b12c7cfcef93af974a9c2a747a813bf90912a3866f9c99",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "d3a096e4e2481330c1a299c0fe118ad2a4b2c0f40d67abd8bd17cdc7ba777a3b",
        "3963fb0a8fabcd8be59f8fa87407c54727eccd2a068a3eda09064fda481d73dc",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "ac9cbbc7f9045f1baeec4edcd9d30bce59c89bd60a8a1d009ae3d2053b0f3b52",
        "1bbdef1dc42db22808c0b1c7b0584a239a87067e1dbde04b6d9bf061ae98e68e",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "ac9cbbc7f9045f1baeec4edcd9d30bce59c89bd60a8a1d009ae3d2053b0f3b52",
        "1c9acb16a619078e8f48ee51c8b5568b68c9708cce1cfe42a8034863dd6f1c4d",
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
    targets[(8, 5, np.float32, True, False)] = (
        "163885e693ac263bf96c5b3c5b07ca00effad68fc6f4e101f55cef37496a66e2",
        "97757ca9254be516099cc74f612df7483201ecdbeca8f2c78c5a194ecabfe70a",
    )
    targets[(8, 5, np.float32, False, False)] = (
        "163885e693ac263bf96c5b3c5b07ca00effad68fc6f4e101f55cef37496a66e2",
        "cf9c6a06d77dc4cdb32b155f96a98066bd2a6b6babf2f9fb206df23188009148",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "83d7815b7e6662491f2762d5985232cbb45497c1307fbcedf6a1b38054674ba2",
        "8a2bc3c8bb3e583c3402912a5ad1ab1cf009b87bb804faa494934cc75d9eb512",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "83d7815b7e6662491f2762d5985232cbb45497c1307fbcedf6a1b38054674ba2",
        "0d16d62b8683203b8b4f0b451083fd6965f75806a447f35569168e9c9844f0c5",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "c387b317de210cb53e1317a6515b2ff2e85b2a8be6ac56139f665b9df717f0fa",
        "e495d602598effe2c75c4762e07e236dce3dac91fe66a59137d300b37ac4a287",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "c387b317de210cb53e1317a6515b2ff2e85b2a8be6ac56139f665b9df717f0fa",
        "0dd2e165082a6e4f17768dd26b407062516bbba0814f3682ed3a132f865cc43b",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "899776667a4bc8f47fc86e1d82cb7d8f7ab3664b8c680c565a023b6940f3cbe8",
        "f218bab84de2a5058c01b772aa4766480c2995decfa54bee43fa78f1be4d06c1",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "899776667a4bc8f47fc86e1d82cb7d8f7ab3664b8c680c565a023b6940f3cbe8",
        "0d5602871a0f7872fa0ad225f60cb4197a811135e53f8450693a9be06105c1bc",
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
    targets[(8, 5, np.float32, True, False)] = (
        "a96aff523e1d6263ac93b174806aa4abf72a61427a8de4c2dfc330848a7ec8a0",
        "a5d80388562215e3c075e21afcc302e6ef05cb6e4bea1e4a56b1c21aa16e6e8a",
    )
    targets[(8, 5, np.float32, False, False)] = (
        "fc6f0c5de937b7489da0554658516645e25cee15af1e2e9ccb924ed23d541547",
        "fd5d04b802cf1927cc75949724871d149a396a4044c71be5dc6f9524d525fb7b",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "52e0e3cfeffc117b8bb289028ec4688195ef8d82cea811afb5e2e1035cc3c07f",
        "1f7e5182531e6476b43e3a1ed1ff2ce799bc1203dab238d2881a82e0b7a3eaaf",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "55153c915dfdd1af3ff348c824b924610f7f015b69f93d5b4d2d09774e358724",
        "d1e3b00648ac6cffe7b1e172a563a7121e98b9921895a3c0f974dd937f0f70e9",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "08d1f77114073e5dda6d9f82dfaa1286b494f79b3fbda95d26868df8328f4405",
        "3939138f7506df1c995a6130d79a15d33bfdbc7140d9e1bb4c82916b39bbdec4",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "08d1f77114073e5dda6d9f82dfaa1286b494f79b3fbda95d26868df8328f4405",
        "fd5d04b802cf1927cc75949724871d149a396a4044c71be5dc6f9524d525fb7b",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "597c06c6ad59507f351a5fd9ff6ff23d745dda19cf7cb69700f6a98503a5d122",
        "eab363eaa44fb62aef6e1bd6d07bbfa8aaae43caae3028b2a8e479459741809b",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "2d4cb9b12701d1ead396424ab53eb1bd43835ae88297c7531cc9c03fd8bb8261",
        "d1e3b00648ac6cffe7b1e172a563a7121e98b9921895a3c0f974dd937f0f70e9",
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
    targets[(8, 5, np.float32, True, False)] = (
        "7cb83d5d39f29913ab4f566c0f6b91f746af969a73f6d368324e07601213b719",
        "2751654f4fc245b588a26568db89e393fc048f84af6dcf43d48c084d0849d9c5",
    )
    targets[(8, 5, np.float32, False, False)] = (
        "79ddf2c940e91b4eeb1e818f6c3f0420845f55f2b674420f54f27b2e677dc34c",
        "15b4bfc657a013d89bba7aacd22e6ecec9a10b2b9f96115fdb35ceccadb374b7",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "ee3fac1386e74f0c0411bb0c3422d8543ebdbb77def316a1c1c9f6c74ac87a83",
        "4b00ba170199fb2c547c8d58b21c73723f15cc3db610dbe3e0b089ab18df1343",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "51b826372baae72824bb145e5c7ebf1d53161c61321011bc9f8b4c27251dd80e",
        "6a7358c8bebb876c47e75548c2b6a911cb5d30db05203d17980fbe8b5072a743",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "52b234691d9256183d81a0ac114dcb29d86f8d8bab2a516b22cc3404c55fc9d8",
        "e080f576a4a536354d9e92808e2b5859d9a6f2ef7f58968872a32f860f9b0df6",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "4cd8ee0e8679c32403f6ec666a81ff3fcfa2f12a4159ee977e91ba9187a3ae6a",
        "f86a8a49ed4114ab1198acd0efc17e8d1baf037745ec423b316f35b35ee178ca",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "2c5a1822b05eca30e9b7b4e97937d2db6157ab2fada42feaeb762675be6afbb1",
        "0763cad35231f75b7898d33dce89036c827c700fdd93ba9deb765479b688ab55",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "f66468255abe6c93b3eb9d9e26c727de09ad40a7dda5644046032cb474f17fbf",
        "9bb3c5efb87575b39265cab20b6a6461003c81093af0be0a7f30351d28bbd854",
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
    targets[(8, 5, np.float32, True, False)] = (
        "39fa10cc0d1505593083719886f0f40ae2ab8708f9c5e314e0e55bdb38884ea7",
        "6a4bbb75b8e35d146b9d941cd8a2c57f31aeb696321967b36302406cf6a90c9c",
    )
    targets[(8, 5, np.float32, False, False)] = (
        "39fa10cc0d1505593083719886f0f40ae2ab8708f9c5e314e0e55bdb38884ea7",
        "3bdb32cab6ab204a04e4c8f42e2dff00c205dac3618425bf2f7f5faa98bddfba",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "5c944f50eda7160212a8a79aeff9917cfe4e86c479c86e19d30cd5594f4b0b89",
        "af12369ee43f41c8e447ea76a23d4b09869b8a7809162617406a645c163bb72a",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "5c944f50eda7160212a8a79aeff9917cfe4e86c479c86e19d30cd5594f4b0b89",
        "153b78652c98c294728a62a9a9f7f938aa74243812c30af681fb2128edfe0221",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "768e8268b6e7b12d69a07af58c9380c178c633c1ca698ca0d3983573991904ae",
        "b10f74ae7ce373cb17b43441154607b5ff7bbb513ec0b0dfa3f94800ce77bd56",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "768e8268b6e7b12d69a07af58c9380c178c633c1ca698ca0d3983573991904ae",
        "190cff8a7525811933dd604f2948992c5798f28d2b7c4be6dac5e0df00cd56bd",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "7b991b18a9c1cfea9dafb4751d24b8612de55acd6e77bf7be3add1f608f60a0e",
        "92f3977a3cb3b90db0b5427663438d2577fcaf29ca1c73ae51c1a9c14e9a745b",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "7b991b18a9c1cfea9dafb4751d24b8612de55acd6e77bf7be3add1f608f60a0e",
        "c6e97366a5eb34b105192ae3ea3f3aff8f2eced09efaa8b1f74e253612f9005b",
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
    targets[(8, 5, np.float32, True, False)] = (
        "ec06dc7a644f11924b9057bdf61a711b6bf3ec9053c77b5ab08307ec03250e3d",
        "ccb4e1024d1012f2b4b7f6f22dd40ab6459a8735e102fc60f633ecbb962850e9",
    )
    targets[(8, 5, np.float32, False, False)] = (
        "f7e9d76a94f2ab394188b2fe5db8a7884949c928a20bb1d8654b8a4a4df16990",
        "f7e1c688ae4c020c3c6eeb8d8b7e9f98704da363a73cb88db703c364e37fedbc",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "614c45a6f6d67bb4a11707d08670928e137f17bdb414e50b30b3b5be2fd5bc60",
        "47cf68901cf46294f1bf456790b92e2e6919b2e0bafd834e269be0d3f7c876f0",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "67a7ab21a51ccadf2bd910f83970fa8b974c7f4e0b09d54969a48d7e432acc7d",
        "2bb8a099966bd213dd08b2cc40cbf7c74c862b8037d712fed0dda0b5cebe5195",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "8ed702e5b89f8fbcef34396ad50843bc449cfd1ed59e7a3d93125c15d442285e",
        "94453fb05f299f3547bf233ed7d98cb9a309e4cd29cba31e40fb6a6e717f8395",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "506255478fbc969fa69b20d5941f31d090b3e9929045b8be4ad9d42be2464a25",
        "8eefe3aa4032eb836e6d73867a2f4a7ba26a72d0d4ecdf8e86159ccdf37656ad",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "3028c194b42118fdb6d68eefc1071ca27e8db0640a926e39bc14d9fbc3032221",
        "e613c7eed5d4473098b27316e5708ad61028d55d476ce78e97cc134465deba28",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "bd1bcc9da26625e245ce21efc898c0edfd410c240130835642e46c0c75ec82a1",
        "ade27ac42be3bc2114c357dd44e9115d7362308508abb9710a6db0ef3b2e04fe",
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
    targets[(8, 5, np.float32, True, False)] = (
        "b1ed5dc926cd073c57df02dec54f6ab9d3096f539cbc979489e3ff084b27ff07",
        "8f29086103364db76a8271bbdbfb531dc23830a4910f8ba5d4c233e0b4725af3",
    )
    targets[(8, 5, np.float32, False, False)] = (
        "4ea320753a3bb39b1f36d1994745663948f4188d668cecfec192f3aaaa544cbc",
        "da07f08a3cf63bb93f8c9e0a1fdb5b310e70a5100fac85f5ccc93effa07a01e5",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "f56169f8136fec826fe5f98e1a6e0fa4ea7ef08ffb774e87fe795c509ba11cf6",
        "8513becb7578f80c57047b09dd445222963a2c1921271a71c24c853755ef8dbe",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "4829af442a74c78e48bc1dd40a103dff5827b3d4aed55616ee6ce9ac74beb965",
        "65fc87569184d3ea42d95e03ac8f68a4739a6743d475380e5c4418fc49bb969d",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "965602616cc51aea3366f1140e0c539b50fd2100e96d353b9ecb1e6001934989",
        "c875b80671650b77a3974265c4e9edd102f0c70be00363c4d530a69a05b670e7",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "d4d6953fdc86f0269c2eb4785eb0308b49b6e49cb6d7d4fe075320a0779e3757",
        "ea0451a72ebeb1079fbba401edaf1faf44279fa63eccc32e64b5c2eb9ab62a95",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "d200ef76efac824f075eeddc7d3a32bef8239d52675971829a0161635c76074b",
        "384b5cfeb0d2ed48f3e599567f8449517212cb87b33cfcfe51079bc6b7040946",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "aa1b6e860a8e16d93a3648db27dbd1966adae9bdd23e9f24387a224dfb8f8447",
        "38d401b535258f29580c2c59ce5cffe8694e48a168302e41f0f9428303ace958",
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
    targets[(8, 5, np.float32, True, False)] = (
        "bc6db0268d4637d05a2722dccba4a671f594db1f562dcf79e9b34fc6075c121d",
        "d9a168bb49458fa2bde837547983616dcc86e9d6caa5c898838b38b8b6fb7d15",
    )
    targets[(8, 5, np.float32, False, False)] = (
        "8d3b4667d1a7259188aa211f111e513eb362d9ba2a357989c1515da8953196a6",
        "a45493441ee29059b13dbe0dc7538d3183cbfe5c147dd72e90951c3c4b4cbfc8",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "8f75c12f3feae8e8ac48d5803819931589df271edd5266398e1441127515fb5c",
        "eb8444b22cf898e6ff94a75e1a3bc7d5a2355fa8e145372042869e42690111fc",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "274d9d178980ba72d0ca939ddbb90403de29c071533f5f0b75d82afeb3dd74ee",
        "40dacd5b2686f2de10e2580b861f2a2d957bebdf3c6713ea29ba8f24b54b66b6",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "76cc5c95868bd8e0c874f5bb6146cf3cf1db91eeff332c47fb70570cf341ef88",
        "d8dcc8d10180bc83ffdc3ed69fd96eb8b8d8e10dce5fb13b813e07d0748b0da0",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "9c3a575f67f990053666723454c81df59a80afc3336da5573db45541d8f0abb2",
        "f0ecdf17bd533f37be3228210e978af39fdaddf1e97259edf0cc4d83e53fb3dc",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "793b48fb6213c38d60cfeeedca72f92bfd3a2e8d8e5f2bf969804f5cca04a770",
        "a0d3143fdf3896254d3f332b7420ef722467e954f14735566f056260e8ec8187",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "ad804dbce0368acc2a1f92b6d82ca6fd5306309656aecc9af2429edee2eaee2c",
        "d87939e176c9267783023c06a95640bc9e3d98650e0d65b7ece38ba7b9d1c00d",
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
    targets[(8, 5, np.float32, True, False)] = (
        "b10622500d30867ebf04f3175f86eef7cea58cdd9d34d824f8f2fdbe0d263548",
        "082f7111736634d050f2012e87353e79d533eb10fe148746e862633b10df338d",
    )
    targets[(8, 5, np.float32, False, False)] = (
        "e5baccf69719a966c325dc414e6a3c57bbb9d9232aa003cd9479121f05373adf",
        "68588cf5a41b764b2ebc4d35c772aca7b260f98edccdc9fb9f90e07acc14d444",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "b80b305d3837ef4ebaae936e3bae5d74dfa3d00733ad3977cb7f6ec727735e88",
        "c3ec4659d658f2b3275281705b4389830e600df5437e82ef0bdf865b35c4709c",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "e65e36b6efef6712c1b0272522e98f02fea4b58d4341d293370051a0c697c353",
        "e086f6b6b99a8c5260e161038e0b1f6520a532de6f0abf8bf430103e6a457cf6",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "f0a13ea776737d58be588a9637885dae67d5812a26c30e2a851a7790f5b2e055",
        "42f3d40e04878baa087839cadffc0b2d9c7f7930edcd04098e43068d800c18d2",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "19ba7dc3b9ce975ab20452274197c8d208b3fa706e30904bb2c71af8fc914ad5",
        "e16bd094214d2c164967b770635e2c33cc6134ed7e4e981ad7665e109570907f",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "8b43b25f9f6675dbe14f0ca80544148c48bcd607c696150a3d24313cd3fbebcd",
        "ac2167f8de670490c26225d8070f04f135d14d95e731aab0c36cce7cb1d34364",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "f959343ad14d461202f972e9c6183418abda303555ced74302fca731797e9720",
        "689bf62c89745e0ea1068c617e7a6f7e4cf5b5af8b7eb890b8bf35c4ed4fbabb",
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


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n_cells", [11, 12])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("negative", [False, True])
def test_tpms(
    dim: int,
    n_cells: int,
    n_quad_pts: int,
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
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function should be
            used.
    """

    targets = {}
    targets[("Schoen", 2, 11, 5, np.float32, True)] = (
        "13440c38eb94f6731eb51b6b7ac8153eb16bc41e2d4df935fe970653b2c436ce",
        "a30605c8649e7cc856885ac3c642b13a4e37e5763c28965dbd3debd7ae3fd5bc",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, True)] = (
        "6a61c13041f27bce40c1bd7e09c48d9999cbd8f2810c45368acdfe986cbce39c",
        "1260981354a3efcf44c778fb36f843715e70bce96f76b4f111d0a2f6144626d2",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, True)] = (
        "cac72c131dfa0eca19429e910f4914b7255e10b57bb273dffc9610ecc079e742",
        "5458362d027a7ab20763f75564b229f28c8ac989ca295a260e611958aef33af8",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, True)] = (
        "9853b8473df333c5313a5032d24fbc6341a76ff47ce512c207c719503ad28f05",
        "8c79173d67555a147125e00ad13dfa46bc762a662467bd230ac905af26b7012d",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, True)] = (
        "700a816158cf4ce4a20bb3fba89dc71cf45433fdf8558f681acf117df2d1df88",
        "86522fd79398a08c227ca35520a099608c902c06cef5ae976b6896581ae6ebee",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, True)] = (
        "edbd33921ae7b21936525d75c7dce31a1dc9ff6be434a37e11578a50a41a023e",
        "6fc04a1b4725f2d386da00cdab660368e6c884e29e6508d955d1a843057eebea",
    )
    targets[("Schoen", 2, 11, 5, np.float32, False)] = (
        "6b32371644495ac702ab7ec3063b133aa5c6f907703e556aae3f13f3162f32c9",
        "05f494b367617bb45ca3809f433189271bfbb94b223ede19e0a9456bedadb765",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, False)] = (
        "63dcf4a67fb589f03eed1084adb5e76e79968608bdd681b0e551111e7282e4a3",
        "21990023b3b74f9672a54983e24766783def86f3949d84dc2e37b1f68b76bdcc",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, False)] = (
        "bc305aac799bea4351547c313659116684abef9fc6bcdb38b33a95d7a9e67556",
        "73b7ffaf03cc04f81e88e9abc402cc1c022ab88829ed06a2766382f2dbe99958",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, False)] = (
        "7beefaf3a8a900cc79adaa4e6e5047c92e90c30ef6b57dc4c91cf98b60103cdc",
        "0f372be9524e88df1a7c8d5dd455342959b9a907c8238b0b591ea2615b5541e8",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, False)] = (
        "7552b69430081cc72feb003a0a49e501c4c2d5aa16174e0ad3bba82a23eb6b66",
        "71ba2c58431dd3cb1681ffa5f94b2857427f6d6c398a4f9436676836ecd68376",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, False)] = (
        "2d4c1ebe71a5d3a617c62779f5bdffb8d650b072052c052beca75bb7578bdbe0",
        "633452040c428abd27dc58b9d1c4d1a1d7e0aa6c3ca265b2f6d8a1bade50d845",
    )
    targets[("Schoen", 2, 11, 5, np.float64, True)] = (
        "f7b6e3bbaf6ef39d5bda6554f90a03b8a3146948b20b48ce92ebb88dca0b8f19",
        "044e3f5ceec25f9f18b46fb21054c5c89df13f2a5edf5c03b9a2de487ba53732",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, True)] = (
        "cded91d9446bdc001b41ccc4fcdcbc98322510142396d90e2ab26225235de2ee",
        "2e73eace5480167448425194cb2f899331c85535bf47e3524c599891558d8e7a",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, True)] = (
        "d61c8aa6939c3efcd470255296bd4e31be7a5c8618f53ae7557980f0489dac9f",
        "6134f79c6808b11e30d5b4dd40fe42371cc232491c355cc7829440961a12e401",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, True)] = (
        "3c46fc9fe51813114ab45294040800ca56b3c590fd788ea7f59900512c7717ae",
        "9a3deec4d4dfa392134ef30cd2f47ccf6a17585a5ae29bd2682c9129371c67d5",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, True)] = (
        "d0f82148b35210a336f2e1eff3177458075b51b33944aafb57e1f4ddf771682b",
        "c79b7ce275ef0093d554c944a2a12e6d39053e214cedf61137462c60d98a82da",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, True)] = (
        "4dcbf88c8f81a4e164b3cb3a21f075ca9f3b73372ba3ae7f75a7f5da10142931",
        "b100b87f5423c72c915b5778f82ea33ca891ef552322d78f9b500f282620e22f",
    )
    targets[("Schoen", 2, 11, 5, np.float64, False)] = (
        "73ef0e54317f555e62db0cc7414b77de04616e04bcdd85dd18601417c48539e3",
        "1d3b10119097c54d0d6a5b61d65b67fe5de132a4a37a9b4944d8348448eee233",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, False)] = (
        "41907abf736de3e6821716e4487cb2f36eb0ddc62847d69c1fb2f2ad2576d157",
        "40520ee1ba11c18bca1c98a0869c62aa42b40c3afea638ad7824d66827031268",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, False)] = (
        "32fc5ea90663fb1218051a768b21de7660b22f054853f4ba0e49b389c7da2548",
        "e10ddc1735981deebdfc7adc93407d1b22e1a4a1ea7fdcb464a4117bbe8ae7b8",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, False)] = (
        "2463ae82476ce2df24d90b277b858b09c5b9e57a0afd50a793e8b156eac9fc2b",
        "5190aea8cb99708210bca8395688a182c75422a006562c8d618441f70a6777f9",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, False)] = (
        "8dbff578a9cec00acb5abf3daa9900a9a8b759ecaa8940d132707a4ec31714f1",
        "8cd4901fe736741f43fadfce9beeee8ee9e431a9e4070df43a8f6adef427e972",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, False)] = (
        "9ebff09efda06489b495e0dfbd4b8b080143c1f14e6c76fdb574dded2bc630b6",
        "c6c508e7e885a5e45010cf6cd717ccfeb6544d7cb3bb40bfec7e64eddda6bd22",
    )
    targets[("Schoen", 2, 12, 5, np.float32, True)] = (
        "8d6fce2d7a56d6524166e374ba4a2a62398b5ad7d801bebfce805e9d3a6512a0",
        "7d5dd11a454f8d7212e77487b71de9a574600666392197529eb42c2cb6c436d1",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, True)] = (
        "aa6f35b1b9f0767055b83ab15649ef2b32019b75c66035f1f130f37232f8c698",
        "58f1021fdbdc1d0bbda2acd47af238871354337fa4099484f98a11c98d30a748",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, True)] = (
        "02e2de12c4d6fdd5117fc8935295586aa030b3dad8cfde0e5c920a80bd9a81af",
        "b16de04a28c511bec770fa0ec794d8dad1afb84d10fcb3ec848ad1c0d170efd6",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, True)] = (
        "7a5e57c4d95184deb57baa3abf3f9ab1ae5de2c910b65a25772f12b12a8b315d",
        "a15c321be38d1bc0bbcdbc089675f88f312177d181280e65239a188ac72a017c",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, True)] = (
        "490192179a4b027f4de5e3a6100b5070c26d1cf06dce41c7020fba21dfeef485",
        "4db069722fbe6baf603e5f5fcd628232a2ff2ac54e58baee1ee48b00e13d8868",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, True)] = (
        "88098aab8e06a69d6f5c6a09e7d48e3bb3ced4c2e7e08f4a3bc3e07ea4fef257",
        "12384b8c3c12fae0bafd7af6ee27db89d78c4499de8fb8b1567a3b77210809eb",
    )
    targets[("Schoen", 2, 12, 5, np.float32, False)] = (
        "13da4df3cf6cef59b78b73385e024e8d459cadb5fcc9c65b5d716f2c7a3945ce",
        "327aa1e2c10cb29cf6326f3b5226992b518c232ed74f861873755fbd6d276f81",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, False)] = (
        "a1cddb7734a152a7574430193cb1b5151b74facf04b02bb1390b323a90016cf4",
        "5d23947bda5ec9a82b699a130702a5def86d889ca77b752255aed7ef9bf49877",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, False)] = (
        "12f9ccc9b9038fc3fe4b97c721f4a65d542d22aad24282cfd216adf12039e4e2",
        "684891d082d21b9e67a5c004af02c490c13733e7eebb96cee0c7a2c8ad37f873",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, False)] = (
        "ec6f6490f58e0fdf52cde7eb284fc9cc437a0ce28fdea6d5c0d8c86d807c4bff",
        "e8c0aed8cd2408156143e35c7f726846ef4e76890cf19e9a488709ecc0c28053",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, False)] = (
        "490192179a4b027f4de5e3a6100b5070c26d1cf06dce41c7020fba21dfeef485",
        "7b597dbb82c1587fa1c4762ef3999e8cd67d2ebcb226ea9bc560d65c5a67bce4",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, False)] = (
        "3f636f5e8076d80afd6955d03c7865c5e6d5378595c4edbbffccaa3913fa3a64",
        "ce27fa6a1b60d0174c20d275de587d5a56c4f055be6a1654de23a9ec658ad2f6",
    )
    targets[("Schoen", 2, 12, 5, np.float64, True)] = (
        "cbff011a6f4bd7cb03489a6d694913074bae703c0dc174c650b550183dd9bee5",
        "5216f3b0d41562c3b55af453eeafafbf9c3238fa4404af619a50904fae8c6515",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, True)] = (
        "28ffb299c8c591b38db5f8c5a762418a4f5a9bb6877bc30e094a4be1421330c3",
        "944230c1a8f4ea77f8dfe42f89f01a08e316fd79ecab35ceaa4b99e2c9a06cb1",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, True)] = (
        "032bfda476685f0e629c59838c83d8f4a38e7dae35cdffa98dae3f99a0b8654e",
        "ac4a648fc5cc9d37f0705f2370a3fb572cb3b89d1bf9be3f238e0fac5e28ce41",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, True)] = (
        "babd22f191ca37436ab6eff3d45f312ffa90417f69c49cc8e3e7bc7f06d738d4",
        "0d407f5cd8d014eb77527429f76481d85e7c3718e22e9fe78fc97d9649e74152",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, True)] = (
        "490192179a4b027f4de5e3a6100b5070c26d1cf06dce41c7020fba21dfeef485",
        "5934f6551c6996029787e92b4aa94d0191d425ec7cf6adc55af7e7d777304f68",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, True)] = (
        "4297104343fbd5177b3cb690e28f3e88bcc551b452b67b84da382054e0ca001e",
        "b255f36671efa52eeba22e3725859ee36f43bd9fd3c28fa024eb962b6b49ba50",
    )
    targets[("Schoen", 2, 12, 5, np.float64, False)] = (
        "18cf48b18c5cb4bbdaba95bc857c0c32603fa7b8661dbe1d541b20510a64ea79",
        "651806bca6831fb5c23edf5f334c7dfc3563025bcce0fe2cde5815cff861ab06",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, False)] = (
        "53c4465cf4b6fd95ba0f24c741d5e970107721787baf2917bd1e9c7c076df796",
        "31e06bbc86bb343c262b7f25c8183cdffa603b7f28846938ad955970c0855afe",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, False)] = (
        "cbfaa21ad69fa892ccbef5433c805e5509fc22b23e1bacdbb8ba90abaa030cf4",
        "35f64cf45ed065f600c954857ebfb98ec3b262246fd08d7dcb7aa435650d024f",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, False)] = (
        "70c05f2980bcde7f56312ff0805e76ffaf260016357a2746bfc3d855c95c2899",
        "49169c99629b60941669364a19153f927b0ccd69d1028741a106af2af8a61f95",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, False)] = (
        "490192179a4b027f4de5e3a6100b5070c26d1cf06dce41c7020fba21dfeef485",
        "253c1117ea7bac90dd0124456cb80fa667ee82b11ccdfcd5e953c6ade3549fee",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, False)] = (
        "c5ccaa09d772a7a3ed5d028c97e7f4fddc6a680a6e29fb0f7941518c8b814169",
        "f1636a1eb3a4c0db49162d93b8746f050377d25300ab6fd63012a4bb3b322b01",
    )
    targets[("Schoen", 3, 11, 5, np.float32, True)] = (
        "6b583e965b978be46a577f4d3b3fda6dcd804b492b74f17288066e82db18b9f0",
        "00823034fefe6d7f8fa032a9f90818684863310d86375ecd4250f7139763ddb4",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, True)] = (
        "ab1f09492caae376b8daa7a508bfd7b19e7fd9e6b5c2660b433045dcd7e61ff2",
        "078bc6e761feffe572ccb1e18c83ab4c904e770a7cd0f7d5f6378c06a9ba192e",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, True)] = (
        "b923725e31524f9e9dc8e2832a1bbbd56e0f4e6c7d2b264ad65e33a8efa9fdc4",
        "54b6236abedc523edc2f143ca77978ed33dc83bbb454171a543c8c3aaf8f10ca",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, True)] = (
        "2b650ea9987206e2785ab83725384ae8116a1da3bd22c9240196e626d7900e44",
        "6ff607892ae2692935ce464a1499d573f026469fec9eacf6991f58dca1b430b4",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, True)] = (
        "61dbaa1631edcb0e4cb6427b7108b1185ff521f216736287be11162829f9c750",
        "cee885514c8190064ea7b4c13ee762b85fa86fdeaa28dbc248e00c66a6ab01bc",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, True)] = (
        "51997404d2d0d72bfc23c7e38280e40f21b51fb7cd2a31b9c235314f60da952d",
        "ccb92e80bb3970e98210babe4d80f9ddf693283f9fad8893b6dcaacff9811be2",
    )
    targets[("Schoen", 3, 11, 5, np.float32, False)] = (
        "458e64804572ada8172b0ffb187e60bd7a2825d15e2d37aafe4854f5a06b2bba",
        "d9492918885c80f649287a16ada6672b09f220133188d80cce4fa3ee56794723",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, False)] = (
        "12a096d40b132501860f5e19c09e23e9d6a0c9b8a58ea19a3b8be34da9316df7",
        "4eaefd724e3bcbb9501a5241d4d740d03b0baff14a73287ca7901ab13d7c9cf1",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, False)] = (
        "367f253babb86c4944a0ff73b68bf2d20f23ae6bf60ee0b4099e186da6cf34de",
        "2e179bd3c914a9ed54d63437d59c73a4a97fbd7a0c87eed6bab1571b939fc0cc",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, False)] = (
        "9a7fd7f78239b23257415b1eb647b1b2ea0390d8a415264b3767ac1913ca66ee",
        "c8a4a356b140752e799f77526667f02a11bd9872a53e8f12fab31c776b0b0bb8",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, False)] = (
        "ccd831f51499c0c5e6f6305acf28bdd105f98157617393b9a728896f57e0a56f",
        "890b1e50bbb5fc1a38db6a4278235ce433e0a94cf2ab7107bbb2fc15360633bc",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, False)] = (
        "654c927ba2a88a429e47bc711e6c73a204285e2dbac0cd280aaa0328bccc367e",
        "f0a61458abebb6477d97986b9fb1372bc4ef34817cfeea9a2f85fef3e21b7010",
    )
    targets[("Schoen", 3, 11, 5, np.float64, True)] = (
        "5c5ca86070af404a99bded2f0ccedcb473003b3f7feec47b6f5e44608b3bda26",
        "018ffc5721eee8298f75fc4ab9bf2477c630fb88c3f94b046b347127d5738d17",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, True)] = (
        "37b1fb3ba4a9a5ef1c5b587d7da1eaefb794db1035e98695d2d5d4c73bdf625c",
        "e7fa02e71ede1889a6335fd8fa040712302bee2a2c3ad2c72acad43182fe1ef4",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, True)] = (
        "4c3d4c3c545127fed65b5523ac0d645fb64b8a46a905a2adcfba49222b90e29d",
        "6719dcb8d8fbfc2316cc232c6b490018cc3eb1cae69d95b855c0ba7d459008ea",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, True)] = (
        "858fa59bd50f6836b837a5cd03ba8e99291a06fe9ebc32b7182a07a9415b5374",
        "8df375c7b2ef9cc02e5089225781ee46253d5b3119f7b8a38c4075ec035ac716",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, True)] = (
        "f32dc89cbe322f23024ea0c2f38c85c7f348f89633aff330d5d4fd6785506b1b",
        "91e75fa41994c3b5e3f624e28014641f709ca99cc22489d1217d836d59e12c3d",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, True)] = (
        "11d5970fe648bac493bde745394ed06adc005e2bcaaee521a003a71c4f55acf6",
        "b390c2022dc85009f010398c42b3a672171fcd1d912f9b7555b71662b1da0b34",
    )
    targets[("Schoen", 3, 11, 5, np.float64, False)] = (
        "7b36875b202c5186902d2a88bc02e24deb65b5a66e7f362a531bc6fd418b022b",
        "f20ab94ccee2539e81437fad0f28bb95c84e3aa4c8b8fe40bbfc92a4aef60b03",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, False)] = (
        "2141fc390565459d713cb38a3bfac7724a4d429c1c5bc53a58a392228e03804d",
        "40fe1e5d3cc372877f62ad3c29901a4d68e6be8b891c89481f3be5f455b75ed5",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, False)] = (
        "7c97ec7913ab62d062ea30ed089b048825e8f5200a690fabc593eeccb53abda3",
        "23dc9e61516aee4e69917f331ca145aaa4f902051fba60114e42aedc1b0401df",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, False)] = (
        "5b96572be43c2a0ab3d94dc0fbe448cad82a2d220e4e869175a690819a4d4e91",
        "9749e9d972184e2b8c8d765dd023703733db00ed8f108e9deb6717c912854d2f",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, False)] = (
        "3c4ae3ada1827835ee76eddec2794f4527aacf94b3cfe9404a5538ddea5e1261",
        "6b9f361476486bd449863d38da133598b570a282132d4d4afac6fef550716ba4",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, False)] = (
        "1ff98bb29a8d1eb13b00d425b2495de033065c4591e39324fb5d781d92258cbd",
        "528845238e8d83d138ab6e5d965b5672a1ab0bb3ae1960cc928afdde71d4ec10",
    )
    targets[("Schoen", 3, 12, 5, np.float32, True)] = (
        "d3e9ff1d98b1add10bc1dcea47b150e2df209497c3efe4553f14132a88c093c2",
        "f3cb5fdcf0a09eee2aa75e2d3b08e40cb0fd1b145932e1f3d73e77357650f42c",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, True)] = (
        "bd0dc3798ee6938efdbfd58081729bc5e5ed2f68e3a4c938633a40fa35c76c49",
        "f3b0f54e611da66c9173aa9bac67842c71a3a797576c104419d99858d597adf0",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, True)] = (
        "895b48e3d47092516e652669e87422bec701d0127eb567a1327676ef100a9553",
        "60fa12ff436701278ff6a303b83d7d16566b11fdfdaa497c6a43afa4496bdbce",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, True)] = (
        "5b496c4f906db06a9985f03f293fab7352f7208a61f8c9bc6d5e525e129826ff",
        "d744d9d702df31a510603200c56b95a48da62a8bb8c739e3be9ac9e0bb5e1918",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, True)] = (
        "918327cb7412460bc74164bfa825bab60c8eb41f1aaa1f7ca07cd10da0e82f5c",
        "c37ff7c028179224774d7637f6ed248ab5583d420d4d827fd6a8cd09e8eeafdc",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, True)] = (
        "6e62bd67c8490e4d4b353a7b829dde7f27a243cb7fff52c2ee188fb632bdc62d",
        "6ef664fab57e2c10564cfb2b3f1e7b0f1bc1b7fcbef8f14b5b09076436997608",
    )
    targets[("Schoen", 3, 12, 5, np.float32, False)] = (
        "7176a0010ded4ecb71844eb39b318af3f04410578134bcbe80d8afd03fd86582",
        "e682ba1b1a84fbcafb8cc672b69f01e4cea905df23c73f64f520f717757617dc",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, False)] = (
        "85c9e65a30cb32dcec7e4126a933914569f4a4296adabb545952cc3b93c13711",
        "e451f11ea5288b206a66e0158806fe10c8425600c1b967f6ed20de29019135ec",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, False)] = (
        "c063e36f72ab2b82730cad8d3697a87ce58d56c1d6ff005901dec131dc1a0494",
        "8b23aae3a000a0e79aaf609230e3c7bd126f25f389a9d430025b84ceb7cfcbbe",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, False)] = (
        "33b1d1154336f3b98c9e01837b8ad43239f72100b325f63dbce7de0971d692fe",
        "3762557175af225315da7e8970969645354bf67691b0b3196e289e2a344adb82",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, False)] = (
        "80da25acd1ae901534c8f052d4b55a91b76207d22d69af8389698df70040506f",
        "a237859eea2e1d21fe9c8a34ca4108747e07ff1cb21929a4eb0ce4fe388e83a9",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, False)] = (
        "5c610e1968af403773b88c3563efe9968d56c2a940e3ebb547fd80956588eaaf",
        "cbf7400172e5bfcfccdf9460dc5a8629be9614ee1c90b272ab66471c0e8f4921",
    )
    targets[("Schoen", 3, 12, 5, np.float64, True)] = (
        "bd4d4336c7fb76cfa4b4b1f9194fa735dbe2bfa56e1ec55a0dd120d701baff79",
        "d7b4ff58ce7cb8ab002418b36b4e0a235ea9c1f7dc9062331563e38f70b71e91",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, True)] = (
        "febbff573ddae0fa0b085c51ae05bf9f80fc4a5951e30f725f8e0695b6ef1964",
        "c1a81d381c75377b2edfb3de05b9384367b5d036e1af6807730830bfd6e573f1",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, True)] = (
        "f523984907df7aa6f4448eb4ba99be64c20d2162ae2723ca83a56f915b4b37e8",
        "136dce63de886c278e9ce75e7916d9b725f7644b45bc5ce3050b73df208b2b78",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, True)] = (
        "1a560a2ab5f86cc2498a2abf303cdd1aa09a7b3eda796e9e57715687619f54a2",
        "9686ce7b8e593ff360d641f002750eb942fd1cae75cf8d20fb3cfd4a83bb20d0",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, True)] = (
        "cf4f197e25d8127bbe6280bbd6dd3c1947da22527a4f246f87e4f24deea6ee56",
        "041b6828f122755ec7cf636a8dec36993555b51de44f1dc3f040ae5729533093",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, True)] = (
        "72cc6b87fe2ea90e6a3ac5dce78d67bff1882fadb722b23f86647857defb893d",
        "bc55538ff68823aab12016fe6e3f3f69a89e00b054b5f386c2aedcf2669ac91e",
    )
    targets[("Schoen", 3, 12, 5, np.float64, False)] = (
        "664d56b1c9df20a540fc833be423ad1834ef0388e6d46466a462e94167d1e18b",
        "5df18540fbf34fd9cc329ab5089d1375f25fb2de39cddc23f15f1d8ec22d7845",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, False)] = (
        "b93197e70a0731325da354b6eb7bdee7823f439f4f35cb57dc70728a46a71968",
        "beada2601a4804c1ef29c76c62bd90529bc15e906d45a2cd9d527e3b11e20f5d",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, False)] = (
        "9f126e88fea441ac58878bbaeba2b50488d844c4d929012bcc3831e7e384c8bd",
        "ad495e1d6030475158942470a1648f08a690cff8e225ccf16ba83d919d4b4ef8",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, False)] = (
        "34b1c9e3e5bfd525499c7e2a6d649aa80026193dd57f6ccf6d562fd28d63236e",
        "a76843f2d76cdc98d9cc8fc787144d10ca00db047b6fb53d42355867c87815e4",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, False)] = (
        "3705cc5a5e394ff4eb35f4c144f673f766934e13d9139b48f5190591ab97ac8f",
        "78f40ff11118dc53e6943d93b072f147ade027368c81ca47eb5ca032442cf1ee",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, False)] = (
        "133d3dac45c4954acf237647acac02fc6763976a582159da9d0f5d9514128903",
        "7f639cfef39587f5780008ac8b2d2d90f69d4114b8475170774e80370b785d5d",
    )

    periods = np.ones(dim, dtype=dtype)
    for functor, functor_str in [
        [qugar.impl.create_Schoen, "Schoen"],
        [qugar.impl.create_Schoen_IWP, "Schoen_IWP"],
        [qugar.impl.create_Schoen_FRD, "Schoen_FRD"],
        [qugar.impl.create_Fischer_Koch_S, "Fischer_Koch_S"],
        [qugar.impl.create_Schwarz_Diamond, "Schwarz_Diamond"],
        [qugar.impl.create_Schwarz_Primitive, "Schwarz_Primitive"],
    ]:
        func = functor(periods)
        if negative:
            func = qugar.impl.create_negative(func)

        info = (functor_str, dim, n_cells, n_quad_pts, dtype, negative)
        target_hashes = targets[info]
        computed_hashes = create_quadrature_and_reparameterization_hashes(
            func, n_cells, n_quad_pts, dtype
        )

        # print(
        #     f"    targets[({functor_str}, {dim}, {n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {negative})] = "
        #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
        # )

        assert computed_hashes == target_hashes


if __name__ == "__main__":
    test_tpms(2, 11, 5, np.float32, True)
