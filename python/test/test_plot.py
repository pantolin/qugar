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
    targets[(8, 5, np.float32, False, False)] = (
        "935bed3f7e152488c80cabac0c85e92ab55ac858d054c7897d025314a55dd8dd",
        "36cbd689151b2c0b4c75161aca6e2725e13ad4f5f9d4d9e517a80f7895aa07e2",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "935bed3f7e152488c80cabac0c85e92ab55ac858d054c7897d025314a55dd8dd",
        "201df42fc91e4537eb9732ff0cfdfcd65176d75a672cb15deab98dba884f8354",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "97608eb53302a500c215891d97e0d0bfbd99a21651104aeb7fbe06078aff14db",
        "1c9acb16a619078e8f48ee51c8b5568b68c9708cce1cfe42a8034863dd6f1c4d",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "97608eb53302a500c215891d97e0d0bfbd99a21651104aeb7fbe06078aff14db",
        "56cb2ee82b0574f7586b30f1ddc2e85fcee6bf3930a43f3118f1130a1005ba33",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "fba12cf61483d379aa6829cd82358b84c5c320184fc0878c5764919e6f709b7e",
        "3963fb0a8fabcd8be59f8fa87407c54727eccd2a068a3eda09064fda481d73dc",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "fba12cf61483d379aa6829cd82358b84c5c320184fc0878c5764919e6f709b7e",
        "b54fe1e2d98a330ed0b12c7cfcef93af974a9c2a747a813bf90912a3866f9c99",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "c270407f531f829892ed5e2f47c34ec44b001f57da3d3dbb36069b59400601cd",
        "1c9acb16a619078e8f48ee51c8b5568b68c9708cce1cfe42a8034863dd6f1c4d",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "c270407f531f829892ed5e2f47c34ec44b001f57da3d3dbb36069b59400601cd",
        "1bbdef1dc42db22808c0b1c7b0584a239a87067e1dbde04b6d9bf061ae98e68e",
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
        "d0d8d501301b635fca61f6a2214a954d82aea844859756d74e9822f2daa5bd50",
        "cf9c6a06d77dc4cdb32b155f96a98066bd2a6b6babf2f9fb206df23188009148",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "d0d8d501301b635fca61f6a2214a954d82aea844859756d74e9822f2daa5bd50",
        "97757ca9254be516099cc74f612df7483201ecdbeca8f2c78c5a194ecabfe70a",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "d18e4f95fe39d68062e7455311bdf545f94368ff34c4ccbea55e50ea88ad0fcc",
        "0d16d62b8683203b8b4f0b451083fd6965f75806a447f35569168e9c9844f0c5",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "d18e4f95fe39d68062e7455311bdf545f94368ff34c4ccbea55e50ea88ad0fcc",
        "8a2bc3c8bb3e583c3402912a5ad1ab1cf009b87bb804faa494934cc75d9eb512",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "4a3ff2788529c9cd5d91665cf0b88c394d6a4c83cbf393f33fd7f0a3c42a6cbf",
        "0dd2e165082a6e4f17768dd26b407062516bbba0814f3682ed3a132f865cc43b",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "4a3ff2788529c9cd5d91665cf0b88c394d6a4c83cbf393f33fd7f0a3c42a6cbf",
        "e495d602598effe2c75c4762e07e236dce3dac91fe66a59137d300b37ac4a287",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "9ad82d23c15760e242c7c55e2110d5a2a6d2303e051a2abbe5347b3f4b7f88cf",
        "0d5602871a0f7872fa0ad225f60cb4197a811135e53f8450693a9be06105c1bc",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "9ad82d23c15760e242c7c55e2110d5a2a6d2303e051a2abbe5347b3f4b7f88cf",
        "f218bab84de2a5058c01b772aa4766480c2995decfa54bee43fa78f1be4d06c1",
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
        "fc6f0c5de937b7489da0554658516645e25cee15af1e2e9ccb924ed23d541547",
        "fd5d04b802cf1927cc75949724871d149a396a4044c71be5dc6f9524d525fb7b",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "a96aff523e1d6263ac93b174806aa4abf72a61427a8de4c2dfc330848a7ec8a0",
        "a5d80388562215e3c075e21afcc302e6ef05cb6e4bea1e4a56b1c21aa16e6e8a",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "55153c915dfdd1af3ff348c824b924610f7f015b69f93d5b4d2d09774e358724",
        "d1e3b00648ac6cffe7b1e172a563a7121e98b9921895a3c0f974dd937f0f70e9",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "52e0e3cfeffc117b8bb289028ec4688195ef8d82cea811afb5e2e1035cc3c07f",
        "1f7e5182531e6476b43e3a1ed1ff2ce799bc1203dab238d2881a82e0b7a3eaaf",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "08d1f77114073e5dda6d9f82dfaa1286b494f79b3fbda95d26868df8328f4405",
        "fd5d04b802cf1927cc75949724871d149a396a4044c71be5dc6f9524d525fb7b",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "08d1f77114073e5dda6d9f82dfaa1286b494f79b3fbda95d26868df8328f4405",
        "3939138f7506df1c995a6130d79a15d33bfdbc7140d9e1bb4c82916b39bbdec4",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "2d4cb9b12701d1ead396424ab53eb1bd43835ae88297c7531cc9c03fd8bb8261",
        "d1e3b00648ac6cffe7b1e172a563a7121e98b9921895a3c0f974dd937f0f70e9",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "597c06c6ad59507f351a5fd9ff6ff23d745dda19cf7cb69700f6a98503a5d122",
        "eab363eaa44fb62aef6e1bd6d07bbfa8aaae43caae3028b2a8e479459741809b",
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
        "658cfc2438a92bd82890f9174066b4fefa94252ef777ce56ef778e462278be4b",
        "15b4bfc657a013d89bba7aacd22e6ecec9a10b2b9f96115fdb35ceccadb374b7",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "c867208199b95ca5494e181109a4c3b783c7300d82573c03ffe07864cadd6213",
        "2751654f4fc245b588a26568db89e393fc048f84af6dcf43d48c084d0849d9c5",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "b11279697aa3047c7fc4b3d7cc52f340847dff19969be8b3d451e9ee4f61700a",
        "6a7358c8bebb876c47e75548c2b6a911cb5d30db05203d17980fbe8b5072a743",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "eaf2826ba0f64e2215cf3b414ed314e9c4bfcca7a4514d07d416d0032459d081",
        "4b00ba170199fb2c547c8d58b21c73723f15cc3db610dbe3e0b089ab18df1343",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "2ad1c282cb29fc0aa40edaf9475003a0c55f52f482f0324c47109dba724ab454",
        "f86a8a49ed4114ab1198acd0efc17e8d1baf037745ec423b316f35b35ee178ca",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "8d5408803135a09dc89ca9c4c9b9a6920f29ef6933424baedba90be36253ef07",
        "e080f576a4a536354d9e92808e2b5859d9a6f2ef7f58968872a32f860f9b0df6",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "3e3398123d238530194067b4d4f796390dde429b5cb326dd2bba35e42d5eab07",
        "9bb3c5efb87575b39265cab20b6a6461003c81093af0be0a7f30351d28bbd854",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "42c08fea0fc32dcc94ab693402ce373bca220353380cf00248648039205318c6",
        "0763cad35231f75b7898d33dce89036c827c700fdd93ba9deb765479b688ab55",
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
        "c1f6bc4e10e9477dcb807a721ad2539e2d00ed1e62e69c5dbc796b369fcb6809",
        "3bdb32cab6ab204a04e4c8f42e2dff00c205dac3618425bf2f7f5faa98bddfba",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "c1f6bc4e10e9477dcb807a721ad2539e2d00ed1e62e69c5dbc796b369fcb6809",
        "6a4bbb75b8e35d146b9d941cd8a2c57f31aeb696321967b36302406cf6a90c9c",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "717655e8aa24293a5f1b52c9bf030c3627c8b0da0a33344044941e831a1865f2",
        "153b78652c98c294728a62a9a9f7f938aa74243812c30af681fb2128edfe0221",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "717655e8aa24293a5f1b52c9bf030c3627c8b0da0a33344044941e831a1865f2",
        "af12369ee43f41c8e447ea76a23d4b09869b8a7809162617406a645c163bb72a",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "eb41c5bcd99a66a34242ad7e47c4ef463f142a6ed0dc3296a1891e7321211a14",
        "190cff8a7525811933dd604f2948992c5798f28d2b7c4be6dac5e0df00cd56bd",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "eb41c5bcd99a66a34242ad7e47c4ef463f142a6ed0dc3296a1891e7321211a14",
        "b10f74ae7ce373cb17b43441154607b5ff7bbb513ec0b0dfa3f94800ce77bd56",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "8584263066cf71793204f8bed7a1ee7f0d8191901534665efa7543f3b19b74ec",
        "c6e97366a5eb34b105192ae3ea3f3aff8f2eced09efaa8b1f74e253612f9005b",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "8584263066cf71793204f8bed7a1ee7f0d8191901534665efa7543f3b19b74ec",
        "92f3977a3cb3b90db0b5427663438d2577fcaf29ca1c73ae51c1a9c14e9a745b",
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
        "f7e9d76a94f2ab394188b2fe5db8a7884949c928a20bb1d8654b8a4a4df16990",
        "f7e1c688ae4c020c3c6eeb8d8b7e9f98704da363a73cb88db703c364e37fedbc",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "ec06dc7a644f11924b9057bdf61a711b6bf3ec9053c77b5ab08307ec03250e3d",
        "ccb4e1024d1012f2b4b7f6f22dd40ab6459a8735e102fc60f633ecbb962850e9",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "67a7ab21a51ccadf2bd910f83970fa8b974c7f4e0b09d54969a48d7e432acc7d",
        "2bb8a099966bd213dd08b2cc40cbf7c74c862b8037d712fed0dda0b5cebe5195",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "614c45a6f6d67bb4a11707d08670928e137f17bdb414e50b30b3b5be2fd5bc60",
        "47cf68901cf46294f1bf456790b92e2e6919b2e0bafd834e269be0d3f7c876f0",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "506255478fbc969fa69b20d5941f31d090b3e9929045b8be4ad9d42be2464a25",
        "8eefe3aa4032eb836e6d73867a2f4a7ba26a72d0d4ecdf8e86159ccdf37656ad",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "8ed702e5b89f8fbcef34396ad50843bc449cfd1ed59e7a3d93125c15d442285e",
        "94453fb05f299f3547bf233ed7d98cb9a309e4cd29cba31e40fb6a6e717f8395",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "bd1bcc9da26625e245ce21efc898c0edfd410c240130835642e46c0c75ec82a1",
        "ade27ac42be3bc2114c357dd44e9115d7362308508abb9710a6db0ef3b2e04fe",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "3028c194b42118fdb6d68eefc1071ca27e8db0640a926e39bc14d9fbc3032221",
        "e613c7eed5d4473098b27316e5708ad61028d55d476ce78e97cc134465deba28",
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
        "35045a899098ff135a5bb5780bb5e1812513589438f5767fba386f1fe193b2b2",
        "da07f08a3cf63bb93f8c9e0a1fdb5b310e70a5100fac85f5ccc93effa07a01e5",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "2c2b889cb3df9ff8c906b2bc859822f22dcd77cd7c4ec336aea1139fb623066b",
        "8f29086103364db76a8271bbdbfb531dc23830a4910f8ba5d4c233e0b4725af3",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "ead88c8a6e4c36d7234f4530ee2501853e1fc9c8f36e23a02671f8fcd46269d1",
        "65fc87569184d3ea42d95e03ac8f68a4739a6743d475380e5c4418fc49bb969d",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "25d57833895051068535a8d8ab1ac3b3edae6bc2f7c1af6e230da9245337f532",
        "8513becb7578f80c57047b09dd445222963a2c1921271a71c24c853755ef8dbe",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "6990a3b45def317fd891fdf18bc3f30b6e58776415af0a59b5c3ff88a9bcddf7",
        "ea0451a72ebeb1079fbba401edaf1faf44279fa63eccc32e64b5c2eb9ab62a95",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "77dca8b602f64db7c97a1ce7f88be189ffe20de55bb74f11925943c8089c687d",
        "c875b80671650b77a3974265c4e9edd102f0c70be00363c4d530a69a05b670e7",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "c81fd013162edc9ad220b83a0eab2e34c65337f1f73f0de2695b867f12bce04a",
        "38d401b535258f29580c2c59ce5cffe8694e48a168302e41f0f9428303ace958",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "90075c152a7b6a0eb8d0a95291bd10f8299088c8ef313c6dd700e49caf19a7f7",
        "384b5cfeb0d2ed48f3e599567f8449517212cb87b33cfcfe51079bc6b7040946",
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
        "0ba3e7a4d4cb8389845ff9e3fe450ba336483f6631002b50656a1ff0409e640d",
        "a45493441ee29059b13dbe0dc7538d3183cbfe5c147dd72e90951c3c4b4cbfc8",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "bb93bab3cd869beba530e41e8d6e2703d4866320d5737adc99644e33ac5e9f59",
        "d9a168bb49458fa2bde837547983616dcc86e9d6caa5c898838b38b8b6fb7d15",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "4b93c3acdbd44b79930219f9c20b591477bd8eb15f908e98d29c72787063260e",
        "40dacd5b2686f2de10e2580b861f2a2d957bebdf3c6713ea29ba8f24b54b66b6",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "8bf8c40668356cefe9638a1b33cf36f42ceaadb9d469fe1b41724f795fcc8c79",
        "eb8444b22cf898e6ff94a75e1a3bc7d5a2355fa8e145372042869e42690111fc",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "3cf39d70aa253bb9f493b07b39a8ddb047d4ed32687d66fb6bffed1f54e3c3ba",
        "f0ecdf17bd533f37be3228210e978af39fdaddf1e97259edf0cc4d83e53fb3dc",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "40f0c4c1f4b131bb7d3438d96ae2817c5e9af3fa64c3000036eda0aaee39ca07",
        "d8dcc8d10180bc83ffdc3ed69fd96eb8b8d8e10dce5fb13b813e07d0748b0da0",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "da7d4237084044eea8f3054b0dc71085d0ea07a7ff85a0bdf988fbe4041556ef",
        "d87939e176c9267783023c06a95640bc9e3d98650e0d65b7ece38ba7b9d1c00d",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "75ccba9ed34285b357550845d9d3eaeafe4373ab3ae7481166292d4ad6a7b451",
        "a0d3143fdf3896254d3f332b7420ef722467e954f14735566f056260e8ec8187",
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
        "302ab3095440b29920fde98ffa50ab68fc6bb8a37a686e9db07196764fe1d388",
        "68588cf5a41b764b2ebc4d35c772aca7b260f98edccdc9fb9f90e07acc14d444",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "159d4da4f0134dc56090e1a048c11670e2b7ac8aa9753a6d75961c8d0585ca4d",
        "082f7111736634d050f2012e87353e79d533eb10fe148746e862633b10df338d",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "c790dced3a5c5f366bf327f76beae9414f187c378f6d61b9380b760d6fe79c6a",
        "e086f6b6b99a8c5260e161038e0b1f6520a532de6f0abf8bf430103e6a457cf6",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "50c60ff0d708d4f08e8e2ed33dd229160e7c8397e634afe5151cb94969d29054",
        "c3ec4659d658f2b3275281705b4389830e600df5437e82ef0bdf865b35c4709c",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "f49859eb4e79ee6db67ff5a4083a21f8dec7d9cd62b515ded2942421606c8301",
        "e16bd094214d2c164967b770635e2c33cc6134ed7e4e981ad7665e109570907f",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "c14383773ede2fdf7a878dd147c56380a714672f514c033b8ffa25e608139116",
        "42f3d40e04878baa087839cadffc0b2d9c7f7930edcd04098e43068d800c18d2",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "3ebe0211663ed011bf062cdfb8a1d618df307a3cc4f5ce583f7b59a79fc746dc",
        "689bf62c89745e0ea1068c617e7a6f7e4cf5b5af8b7eb890b8bf35c4ed4fbabb",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "ce28f5a9d8f14937fd02c568e463820303436b6564b447f3075ed9b8ac4caf86",
        "ac2167f8de670490c26225d8070f04f135d14d95e731aab0c36cce7cb1d34364",
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
    targets[("Schoen", 2, 11, 5, np.float32, False)] = (
        "19ef93e88fba2dd5bde2b2f7a3b51fb6dcfd2545566b5d87e77d165f32ba0b67",
        "05f494b367617bb45ca3809f433189271bfbb94b223ede19e0a9456bedadb765",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, False)] = (
        "47a0fbf624bcffe769a554dacc36b74c0325cd37106f48e7387fb788e85391aa",
        "21990023b3b74f9672a54983e24766783def86f3949d84dc2e37b1f68b76bdcc",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, False)] = (
        "4bcc9505b58a95f960c2e4ebcbf95aca5607cf6bfb2c85cb74f12abb8b99ffa3",
        "73b7ffaf03cc04f81e88e9abc402cc1c022ab88829ed06a2766382f2dbe99958",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, False)] = (
        "b98f3558987c2b739a7210fbdb13fb0a6238c8f2318467f0164d986445d2958e",
        "0f372be9524e88df1a7c8d5dd455342959b9a907c8238b0b591ea2615b5541e8",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, False)] = (
        "5db4a2286e1620e73c7e61901a0e698cbee86715f53b61dcfcd25200981cb53e",
        "71ba2c58431dd3cb1681ffa5f94b2857427f6d6c398a4f9436676836ecd68376",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, False)] = (
        "2d4c1ebe71a5d3a617c62779f5bdffb8d650b072052c052beca75bb7578bdbe0",
        "633452040c428abd27dc58b9d1c4d1a1d7e0aa6c3ca265b2f6d8a1bade50d845",
    )
    targets[("Schoen", 3, 11, 5, np.float32, False)] = (
        "d1d78262194cff06232251351eda8455ab31cce58dd85bd4cbca65e0c56cbc08",
        "d9492918885c80f649287a16ada6672b09f220133188d80cce4fa3ee56794723",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, False)] = (
        "afb51643d51313b96b56da33f6bd686464781e7cd812d3815ea9acb9454d81e2",
        "4eaefd724e3bcbb9501a5241d4d740d03b0baff14a73287ca7901ab13d7c9cf1",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, False)] = (
        "18c06349a81694be528397c82b3c9b20ccabe0be86cf8618322ef2b4b66c604f",
        "2e179bd3c914a9ed54d63437d59c73a4a97fbd7a0c87eed6bab1571b939fc0cc",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, False)] = (
        "695f4df814c16df11816977fc9b23c02c3e9cd776398cf637f929401062a6fd0",
        "c8a4a356b140752e799f77526667f02a11bd9872a53e8f12fab31c776b0b0bb8",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, False)] = (
        "092249ed36762054851efefcdbff29130576aac7d5d79d429fe7099ccc0c221b",
        "890b1e50bbb5fc1a38db6a4278235ce433e0a94cf2ab7107bbb2fc15360633bc",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, False)] = (
        "be4ffa35a72adc1c16e5e1a88acf36a7b822b418528b2f636f35b378f2d7d5fb",
        "f0a61458abebb6477d97986b9fb1372bc4ef34817cfeea9a2f85fef3e21b7010",
    )
    targets[("Schoen", 2, 12, 5, np.float32, False)] = (
        "13da4df3cf6cef59b78b73385e024e8d459cadb5fcc9c65b5d716f2c7a3945ce",
        "327aa1e2c10cb29cf6326f3b5226992b518c232ed74f861873755fbd6d276f81",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, False)] = (
        "cf66a7b22390eeaf0fd74a318bbac1179d72d356b51353da5c1d4dad511eea40",
        "5d23947bda5ec9a82b699a130702a5def86d889ca77b752255aed7ef9bf49877",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, False)] = (
        "dae4e85421947d88c9b09b6bf6b5edaf758cd728234ce53d85458eb96dc8bd72",
        "684891d082d21b9e67a5c004af02c490c13733e7eebb96cee0c7a2c8ad37f873",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, False)] = (
        "ec6f6490f58e0fdf52cde7eb284fc9cc437a0ce28fdea6d5c0d8c86d807c4bff",
        "e8c0aed8cd2408156143e35c7f726846ef4e76890cf19e9a488709ecc0c28053",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, False)] = (
        "dc85c157834d330b5459183c3c708f80ff17c5b543447025bc25608a7a6cae98",
        "7b597dbb82c1587fa1c4762ef3999e8cd67d2ebcb226ea9bc560d65c5a67bce4",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, False)] = (
        "3f636f5e8076d80afd6955d03c7865c5e6d5378595c4edbbffccaa3913fa3a64",
        "ce27fa6a1b60d0174c20d275de587d5a56c4f055be6a1654de23a9ec658ad2f6",
    )
    targets[("Schoen", 3, 12, 5, np.float32, False)] = (
        "f6846d2fc757ad9b6de7a249e81370351b7dd11744434945f530a4bafc3a78cf",
        "e682ba1b1a84fbcafb8cc672b69f01e4cea905df23c73f64f520f717757617dc",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, False)] = (
        "7ffe9b0f1c1288df0a2a54d60c1a21f5d8fe77aa211fb71563bba4bf84b33797",
        "e451f11ea5288b206a66e0158806fe10c8425600c1b967f6ed20de29019135ec",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, False)] = (
        "425998d41afd0736c3cedfdf8f51d88ee2b0f250c588262a16cada97d1e62f0a",
        "8b23aae3a000a0e79aaf609230e3c7bd126f25f389a9d430025b84ceb7cfcbbe",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, False)] = (
        "13be62628d024a1acb43dce4a2763e40f88c0d92870758448cf881212503a365",
        "3762557175af225315da7e8970969645354bf67691b0b3196e289e2a344adb82",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, False)] = (
        "ce9b415bdf3c859d74eee90eb314022694d02fa53639a8dfd570de941010d73f",
        "a237859eea2e1d21fe9c8a34ca4108747e07ff1cb21929a4eb0ce4fe388e83a9",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, False)] = (
        "eeb559bd3e1ea301f1e2b3e2514b54185ba2688788aeb9df3057295072f6e96e",
        "cbf7400172e5bfcfccdf9460dc5a8629be9614ee1c90b272ab66471c0e8f4921",
    )
    targets[("Schoen", 2, 11, 5, np.float64, False)] = (
        "a6a958173fc33875cb71dd540e07ffcc0a6755e0e26cf6b3753392f6269ef06a",
        "1d3b10119097c54d0d6a5b61d65b67fe5de132a4a37a9b4944d8348448eee233",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, False)] = (
        "b73803b66940c97b4d7580c42d93ee82b9d0ecbdc382bc637877962787ae9e28",
        "40520ee1ba11c18bca1c98a0869c62aa42b40c3afea638ad7824d66827031268",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, False)] = (
        "74b69b13aaf11c09e273a5175a2ac7699fca0a33688531d4a6988750e8cf95f3",
        "e10ddc1735981deebdfc7adc93407d1b22e1a4a1ea7fdcb464a4117bbe8ae7b8",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, False)] = (
        "d8d9733d365c690f42fa72c17a4e6d11a2ae37c3bf0149489e72854fec7dd630",
        "5190aea8cb99708210bca8395688a182c75422a006562c8d618441f70a6777f9",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, False)] = (
        "7e5cffc7cef12fbdd2ec88a2273e619101c9fd4843663d81a8cc41c11efa9ee2",
        "8cd4901fe736741f43fadfce9beeee8ee9e431a9e4070df43a8f6adef427e972",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, False)] = (
        "9ebff09efda06489b495e0dfbd4b8b080143c1f14e6c76fdb574dded2bc630b6",
        "c6c508e7e885a5e45010cf6cd717ccfeb6544d7cb3bb40bfec7e64eddda6bd22",
    )
    targets[("Schoen", 3, 11, 5, np.float64, False)] = (
        "913a42d2b001ce6b9fb6786671fd055bbe3fed7cc40dd5aea0ffa739f4bd4e16",
        "f20ab94ccee2539e81437fad0f28bb95c84e3aa4c8b8fe40bbfc92a4aef60b03",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, False)] = (
        "2c153a5b4381c8c89d89de4c2b31ecc1ddda0a59b06b54471d4a935c0bd9518b",
        "40fe1e5d3cc372877f62ad3c29901a4d68e6be8b891c89481f3be5f455b75ed5",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, False)] = (
        "071a9b38c2aff67de8a8f019a87ea317ca2ae4707e72f045e1d87fe7ae73adf1",
        "23dc9e61516aee4e69917f331ca145aaa4f902051fba60114e42aedc1b0401df",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, False)] = (
        "3b2e75426434f4948c3fbe91e2d5ac995c6c966cb2f5f6eec66b0588fd9ee5cb",
        "9749e9d972184e2b8c8d765dd023703733db00ed8f108e9deb6717c912854d2f",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, False)] = (
        "1890dce517cf9b1ea5fd171a5e53c5f3e785b37dd9e3ad0f60ae49ea3f404927",
        "6b9f361476486bd449863d38da133598b570a282132d4d4afac6fef550716ba4",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, False)] = (
        "7ca67171ded3174ff1dc2ac6da3bc6b7256ee942ebb308f1ee38078885b0719e",
        "528845238e8d83d138ab6e5d965b5672a1ab0bb3ae1960cc928afdde71d4ec10",
    )
    targets[("Schoen", 2, 12, 5, np.float64, False)] = (
        "18cf48b18c5cb4bbdaba95bc857c0c32603fa7b8661dbe1d541b20510a64ea79",
        "651806bca6831fb5c23edf5f334c7dfc3563025bcce0fe2cde5815cff861ab06",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, False)] = (
        "efe04ad4734e30213a8e0c383e6b32f1afa3e62a18b922b1181b50b6124eb392",
        "31e06bbc86bb343c262b7f25c8183cdffa603b7f28846938ad955970c0855afe",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, False)] = (
        "f9b2fec2e5555dc63617eb7d6282c9c10cf54db43fbdb1b4362eef44bc351cd8",
        "35f64cf45ed065f600c954857ebfb98ec3b262246fd08d7dcb7aa435650d024f",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, False)] = (
        "70c05f2980bcde7f56312ff0805e76ffaf260016357a2746bfc3d855c95c2899",
        "49169c99629b60941669364a19153f927b0ccd69d1028741a106af2af8a61f95",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, False)] = (
        "15bf3f2d2d065220651446793fab4fabd7080dc4008ec19d20089ed3df0d93bb",
        "253c1117ea7bac90dd0124456cb80fa667ee82b11ccdfcd5e953c6ade3549fee",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, False)] = (
        "c5ccaa09d772a7a3ed5d028c97e7f4fddc6a680a6e29fb0f7941518c8b814169",
        "f1636a1eb3a4c0db49162d93b8746f050377d25300ab6fd63012a4bb3b322b01",
    )
    targets[("Schoen", 3, 12, 5, np.float64, False)] = (
        "2a19861727280609b5ab67b3159cd58335b83a8f46f442da3599421c87fd3da1",
        "5df18540fbf34fd9cc329ab5089d1375f25fb2de39cddc23f15f1d8ec22d7845",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, False)] = (
        "a6e5db2bb164d28904e677cfa83e8bb34893a329a77c758a5ea8c7d0ecb661cd",
        "beada2601a4804c1ef29c76c62bd90529bc15e906d45a2cd9d527e3b11e20f5d",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, False)] = (
        "e646d2fcb5791636fd43b0c0740fa2f2c3b74c389e957c315dea90069e8b9068",
        "ad495e1d6030475158942470a1648f08a690cff8e225ccf16ba83d919d4b4ef8",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, False)] = (
        "2899931d1cdc89a01cd5ed2239431f86a79d409a0dbccc2a3061153d2fe373f3",
        "a76843f2d76cdc98d9cc8fc787144d10ca00db047b6fb53d42355867c87815e4",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, False)] = (
        "bcd1c2028dfc94865c0b16834929fa14f2975c869576c4b33e1e409c1d594406",
        "78f40ff11118dc53e6943d93b072f147ade027368c81ca47eb5ca032442cf1ee",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, False)] = (
        "08c4103c7d6b2c05764ceb988761d0d7f87284c0fecb6c153919975a06dc42ec",
        "7f639cfef39587f5780008ac8b2d2d90f69d4114b8475170774e80370b785d5d",
    )
    targets[("Schoen", 2, 11, 5, np.float32, True)] = (
        "ef8e2844e499e1dcdf651e3298f8cb48be050736d8cc5434402e2ac0c9a47064",
        "a30605c8649e7cc856885ac3c642b13a4e37e5763c28965dbd3debd7ae3fd5bc",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, True)] = (
        "5a25b19df9fe6baa67579822e083fa6d109187948feb88b110971c6b64fab70b",
        "1260981354a3efcf44c778fb36f843715e70bce96f76b4f111d0a2f6144626d2",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, True)] = (
        "f0fa7ede3ce297b1a887a08c54b735cfceac4f5617d3c934211572ce2669b3be",
        "5458362d027a7ab20763f75564b229f28c8ac989ca295a260e611958aef33af8",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, True)] = (
        "67ffc08aceaea9f52381a12662ff6ac9fe0c5c2ea90a09f2e98d3ac3ce5ba86b",
        "8c79173d67555a147125e00ad13dfa46bc762a662467bd230ac905af26b7012d",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, True)] = (
        "ef13a9d678fd5d6358c7a7d6f0ee4a8d4ba65c877e45c482a9985e3672c93f51",
        "86522fd79398a08c227ca35520a099608c902c06cef5ae976b6896581ae6ebee",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, True)] = (
        "edbd33921ae7b21936525d75c7dce31a1dc9ff6be434a37e11578a50a41a023e",
        "6fc04a1b4725f2d386da00cdab660368e6c884e29e6508d955d1a843057eebea",
    )
    targets[("Schoen", 3, 11, 5, np.float32, True)] = (
        "c756e20041b294b19ed7b29d54eec3a823d4393ef74a291a42a10a2646c9c8a4",
        "00823034fefe6d7f8fa032a9f90818684863310d86375ecd4250f7139763ddb4",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, True)] = (
        "b663e0b8b0b0bfa5c5746afad4d27137a663370c495fb0794d74ea78acfa8d92",
        "078bc6e761feffe572ccb1e18c83ab4c904e770a7cd0f7d5f6378c06a9ba192e",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, True)] = (
        "5ab0058325cb97fb5930663e21dcfd0c33e433b25fb216051d66a1f569d1a55d",
        "54b6236abedc523edc2f143ca77978ed33dc83bbb454171a543c8c3aaf8f10ca",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, True)] = (
        "30e5729ed7bb8a3b437f44bab01d64f980fb37825d21dda5cae6e3095050bb0d",
        "6ff607892ae2692935ce464a1499d573f026469fec9eacf6991f58dca1b430b4",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, True)] = (
        "ec9998a3107ce18ec85ac7fae69ab36b44de20f4802768432e637ac7dbfe189d",
        "cee885514c8190064ea7b4c13ee762b85fa86fdeaa28dbc248e00c66a6ab01bc",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, True)] = (
        "9045020cc52d9d260ea200bae661e0f361868493a289699770b526b351e910fa",
        "ccb92e80bb3970e98210babe4d80f9ddf693283f9fad8893b6dcaacff9811be2",
    )
    targets[("Schoen", 2, 12, 5, np.float32, True)] = (
        "8d6fce2d7a56d6524166e374ba4a2a62398b5ad7d801bebfce805e9d3a6512a0",
        "7d5dd11a454f8d7212e77487b71de9a574600666392197529eb42c2cb6c436d1",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, True)] = (
        "c6ab1030d8acdd4023d62998ba64951c7c4d8884f17b3d25430f12a9206cf8d0",
        "58f1021fdbdc1d0bbda2acd47af238871354337fa4099484f98a11c98d30a748",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, True)] = (
        "9871cb44c845d105babaf09dfe7c15a0e5609a9146b5fb7de00ae9ec718d01d3",
        "b16de04a28c511bec770fa0ec794d8dad1afb84d10fcb3ec848ad1c0d170efd6",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, True)] = (
        "7a5e57c4d95184deb57baa3abf3f9ab1ae5de2c910b65a25772f12b12a8b315d",
        "a15c321be38d1bc0bbcdbc089675f88f312177d181280e65239a188ac72a017c",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, True)] = (
        "233ac98f4bd00c84f7420b393ad179f92ff24a8a2a46fa6433563bd85e4803d5",
        "4db069722fbe6baf603e5f5fcd628232a2ff2ac54e58baee1ee48b00e13d8868",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, True)] = (
        "8befd7797912dbd803c8dd7052b217a8cb406d1c08d4a9bdc6fa41cb9bfc8147",
        "12384b8c3c12fae0bafd7af6ee27db89d78c4499de8fb8b1567a3b77210809eb",
    )
    targets[("Schoen", 3, 12, 5, np.float32, True)] = (
        "dba690db832247dab1f09e4e95ac43c14d89884e8262d6d90fb3d735f95aefbb",
        "f3cb5fdcf0a09eee2aa75e2d3b08e40cb0fd1b145932e1f3d73e77357650f42c",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, True)] = (
        "b2af3ff0bb8793746d274ea3ed581624a75a1fd7f37124c4b864c1ca9a15b528",
        "f3b0f54e611da66c9173aa9bac67842c71a3a797576c104419d99858d597adf0",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, True)] = (
        "e8b33ce8be788d363d387fbe6cd0befba37773f12183b3adc9929bfb63614915",
        "60fa12ff436701278ff6a303b83d7d16566b11fdfdaa497c6a43afa4496bdbce",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, True)] = (
        "8403eb47a46dee167d8175edbe6ed24911ad5f5ebdfabe5c7564d0c2731c2583",
        "d744d9d702df31a510603200c56b95a48da62a8bb8c739e3be9ac9e0bb5e1918",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, True)] = (
        "2b95090a18dd3b8dc81a8861ceaf5f0df328ea53daf06673673a4d37f4eceac2",
        "c37ff7c028179224774d7637f6ed248ab5583d420d4d827fd6a8cd09e8eeafdc",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, True)] = (
        "ad4ed0810d07256d284e6e92a73de80665edb68c1d4a8ad0cd0e13c26047ca4d",
        "6ef664fab57e2c10564cfb2b3f1e7b0f1bc1b7fcbef8f14b5b09076436997608",
    )
    targets[("Schoen", 2, 11, 5, np.float64, True)] = (
        "0849cea01ae1e23cb29b4b01bf6031cb7bb7c5672cea2aee5e1c0551dc38afcc",
        "044e3f5ceec25f9f18b46fb21054c5c89df13f2a5edf5c03b9a2de487ba53732",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, True)] = (
        "90aac46313a7226d347928bce53232d91eb83363d0addab550579e2e0d2f5eab",
        "2e73eace5480167448425194cb2f899331c85535bf47e3524c599891558d8e7a",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, True)] = (
        "4b9b6ce53d45697d0d43195f3611dd295af9294b6b99a5aa073393b1f14f45a2",
        "6134f79c6808b11e30d5b4dd40fe42371cc232491c355cc7829440961a12e401",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, True)] = (
        "7d3003e0a6b3892d807e86d9354c8970c1f11c5cb0dda39c6a8e80a2abf2fd23",
        "9a3deec4d4dfa392134ef30cd2f47ccf6a17585a5ae29bd2682c9129371c67d5",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, True)] = (
        "663d438e0f9913cdd155824c400ef52d3f44934f3a76f302f8ed0af77fd68da7",
        "c79b7ce275ef0093d554c944a2a12e6d39053e214cedf61137462c60d98a82da",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, True)] = (
        "4dcbf88c8f81a4e164b3cb3a21f075ca9f3b73372ba3ae7f75a7f5da10142931",
        "b100b87f5423c72c915b5778f82ea33ca891ef552322d78f9b500f282620e22f",
    )
    targets[("Schoen", 3, 11, 5, np.float64, True)] = (
        "adc8dbaf396e15e0706a53da9e7e21bb18cb472b19d65a729a629e6b4c3733d5",
        "018ffc5721eee8298f75fc4ab9bf2477c630fb88c3f94b046b347127d5738d17",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, True)] = (
        "c3876b685ac737d63ca24ba2a35eb835d19d5a6febf2b01cf87e48bf01562ea6",
        "e7fa02e71ede1889a6335fd8fa040712302bee2a2c3ad2c72acad43182fe1ef4",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, True)] = (
        "db4c4d87be808dd383e72cf63f5795624eb8bca330891c917547d51b8782279d",
        "6719dcb8d8fbfc2316cc232c6b490018cc3eb1cae69d95b855c0ba7d459008ea",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, True)] = (
        "84bd201ad5de322f0c2362df0d095bca3dbe9652d055d1c94801e92e69f55932",
        "8df375c7b2ef9cc02e5089225781ee46253d5b3119f7b8a38c4075ec035ac716",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, True)] = (
        "80f6d48df3b76a6734a72dbf041a6ee008030e84e335974bb8978bb8ebbe0fe7",
        "91e75fa41994c3b5e3f624e28014641f709ca99cc22489d1217d836d59e12c3d",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, True)] = (
        "da30763ba9342db29ac3bd641ebe92e6a808c3759a7054289493e1b4670e120d",
        "b390c2022dc85009f010398c42b3a672171fcd1d912f9b7555b71662b1da0b34",
    )
    targets[("Schoen", 2, 12, 5, np.float64, True)] = (
        "cbff011a6f4bd7cb03489a6d694913074bae703c0dc174c650b550183dd9bee5",
        "5216f3b0d41562c3b55af453eeafafbf9c3238fa4404af619a50904fae8c6515",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, True)] = (
        "98f43de6273e0f7e5a84a5c407c8d358bb5de3c1cc01f4be2ad85e67e3b7ea9f",
        "944230c1a8f4ea77f8dfe42f89f01a08e316fd79ecab35ceaa4b99e2c9a06cb1",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, True)] = (
        "36bd3dd92537751fe7a33c91eb48a5567b636b9d8d8a0658d79bf853301d891c",
        "ac4a648fc5cc9d37f0705f2370a3fb572cb3b89d1bf9be3f238e0fac5e28ce41",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, True)] = (
        "babd22f191ca37436ab6eff3d45f312ffa90417f69c49cc8e3e7bc7f06d738d4",
        "0d407f5cd8d014eb77527429f76481d85e7c3718e22e9fe78fc97d9649e74152",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, True)] = (
        "337bee2cb415a16c657294da5268c87915c63da922ace498629f7c7820ed7c5a",
        "5934f6551c6996029787e92b4aa94d0191d425ec7cf6adc55af7e7d777304f68",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, True)] = (
        "8f11dcffaa6324afa1e995af9d7414c5340bfa16bddec6874dfe6112b783d5fe",
        "b255f36671efa52eeba22e3725859ee36f43bd9fd3c28fa024eb962b6b49ba50",
    )
    targets[("Schoen", 3, 12, 5, np.float64, True)] = (
        "6bd859e5259b21e5f2014f57fc4b5f28bbc38ff5329103c2659fd8a19b5ebd27",
        "d7b4ff58ce7cb8ab002418b36b4e0a235ea9c1f7dc9062331563e38f70b71e91",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, True)] = (
        "c8fa04f7ed89f3f22b7655fa77f10dc39c23031b9e2707a50c84623955b60ded",
        "c1a81d381c75377b2edfb3de05b9384367b5d036e1af6807730830bfd6e573f1",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, True)] = (
        "92525637b84da9ec68665e6035c8e1eca2ca8fdf960b2212eb6c67bf65c9a386",
        "136dce63de886c278e9ce75e7916d9b725f7644b45bc5ce3050b73df208b2b78",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, True)] = (
        "6f9b806a4c20722394d0c328c1a96bbcdf7de9e1d49ef647f93e93080e08520d",
        "9686ce7b8e593ff360d641f002750eb942fd1cae75cf8d20fb3cfd4a83bb20d0",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, True)] = (
        "38fb1d23bb2573550b97bbbfc351685648bb64e1a3307d149f25758362f654be",
        "041b6828f122755ec7cf636a8dec36993555b51de44f1dc3f040ae5729533093",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, True)] = (
        "4de909d9e192433a8979fb44123623083233560d563328ecb8d8509f82c54cae",
        "bc55538ff68823aab12016fe6e3f3f69a89e00b054b5f386c2aedcf2669ac91e",
    )

    periods = np.ones(dim, dtype=dtype)
    for functor, functor_str in [
        # [qugar.impl.create_Schoen, "Schoen"],
        # [qugar.impl.create_Schoen_IWP, "Schoen_IWP"],
        # [qugar.impl.create_Schoen_FRD, "Schoen_FRD"],
        # [qugar.impl.create_Fischer_Koch_S, "Fischer_Koch_S"],
        [qugar.impl.create_Schwarz_Diamond, "Schwarz_Diamond"],
        # [qugar.impl.create_Schwarz_Primitive, "Schwarz_Primitive"],
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
        #     f'    targets[("{functor_str}", {dim}, {n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {negative})] = '
        #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
        # )

        assert computed_hashes == target_hashes


if __name__ == "__main__":
    test_tpms(2, 12, 5, np.float32, False)
    # test_tpms(2, 12, 5, np.float32, True)
    # test_tpms(2, 12, 5, np.float64, False)
    # test_tpms(2, 12, 5, np.float64, True)
