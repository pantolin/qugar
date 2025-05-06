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
        "935bed3f7e152488c80cabac0c85e92ab55ac858d054c7897d025314a55dd8dd",
        "027b1d06c53c189df97749ee1a3b98a1f89b4b9d90f511cb4e7842cf246feee0",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "fba12cf61483d379aa6829cd82358b84c5c320184fc0878c5764919e6f709b7e",
        "781c780944122307b4267bb0c9489623558c6cb64bd21653adc9e62235e6e223",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "935bed3f7e152488c80cabac0c85e92ab55ac858d054c7897d025314a55dd8dd",
        "027b1d06c53c189df97749ee1a3b98a1f89b4b9d90f511cb4e7842cf246feee0",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "fba12cf61483d379aa6829cd82358b84c5c320184fc0878c5764919e6f709b7e",
        "781c780944122307b4267bb0c9489623558c6cb64bd21653adc9e62235e6e223",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "97608eb53302a500c215891d97e0d0bfbd99a21651104aeb7fbe06078aff14db",
        "cf4d49002f233df2980dba1ba6565e89100e6e3be4cb03022aef073a81afe54e",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "c270407f531f829892ed5e2f47c34ec44b001f57da3d3dbb36069b59400601cd",
        "cf4d49002f233df2980dba1ba6565e89100e6e3be4cb03022aef073a81afe54e",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "97608eb53302a500c215891d97e0d0bfbd99a21651104aeb7fbe06078aff14db",
        "cf4d49002f233df2980dba1ba6565e89100e6e3be4cb03022aef073a81afe54e",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "c270407f531f829892ed5e2f47c34ec44b001f57da3d3dbb36069b59400601cd",
        "cf4d49002f233df2980dba1ba6565e89100e6e3be4cb03022aef073a81afe54e",
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
        "ac5064288ffb72b2f5adb23e68312b434b09c9f71fdf2332e88f31dbfc1e2f13",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "4a3ff2788529c9cd5d91665cf0b88c394d6a4c83cbf393f33fd7f0a3c42a6cbf",
        "2a5dc697f43bb84d10e9ef82f3cb24bc2d5fc88412b61ca20f9cc0072a6c6830",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "d0d8d501301b635fca61f6a2214a954d82aea844859756d74e9822f2daa5bd50",
        "107ecb8b5edeb329d683917e255802ecfea311c664729e4e54327966d1b81e64",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "4a3ff2788529c9cd5d91665cf0b88c394d6a4c83cbf393f33fd7f0a3c42a6cbf",
        "83c70435220c2dc9612914069286d204c81a28cc51fc917676a62611190c31d8",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "d18e4f95fe39d68062e7455311bdf545f94368ff34c4ccbea55e50ea88ad0fcc",
        "4c93567cf036f3b39502b2a0a60eb00a3e2c302b6954bee9e91c238325d8f70a",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "9ad82d23c15760e242c7c55e2110d5a2a6d2303e051a2abbe5347b3f4b7f88cf",
        "4e58bcbc8d7b47bb51cb51797ee874c2e0bd7c6fa4e98047735b2a696ff350d0",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "d18e4f95fe39d68062e7455311bdf545f94368ff34c4ccbea55e50ea88ad0fcc",
        "2b5c1e9355ad6b17049224ba5c05abe62c3d0169cdc26959a93930f16ba0f116",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "9ad82d23c15760e242c7c55e2110d5a2a6d2303e051a2abbe5347b3f4b7f88cf",
        "83ff05a68697831be83e0c875231ffaf6ce7ce01e7ce5ba8b89dc2fcf254afe2",
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
        "adefc720f389400f18e4c671fd1d07af969aba0d203fcf325abfc99315709680",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "08d1f77114073e5dda6d9f82dfaa1286b494f79b3fbda95d26868df8328f4405",
        "adefc720f389400f18e4c671fd1d07af969aba0d203fcf325abfc99315709680",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "a96aff523e1d6263ac93b174806aa4abf72a61427a8de4c2dfc330848a7ec8a0",
        "0fc1cfe96c100218e34d3259dede0ee88e190fa28955c3c38e8e4fd23a850e6a",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "08d1f77114073e5dda6d9f82dfaa1286b494f79b3fbda95d26868df8328f4405",
        "adefc720f389400f18e4c671fd1d07af969aba0d203fcf325abfc99315709680",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "55153c915dfdd1af3ff348c824b924610f7f015b69f93d5b4d2d09774e358724",
        "7728d100acfc1956c2494e44fa43f57d95de6ace2b87a7acc523ef9185c84faa",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "2d4cb9b12701d1ead396424ab53eb1bd43835ae88297c7531cc9c03fd8bb8261",
        "7728d100acfc1956c2494e44fa43f57d95de6ace2b87a7acc523ef9185c84faa",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "52e0e3cfeffc117b8bb289028ec4688195ef8d82cea811afb5e2e1035cc3c07f",
        "7728d100acfc1956c2494e44fa43f57d95de6ace2b87a7acc523ef9185c84faa",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "597c06c6ad59507f351a5fd9ff6ff23d745dda19cf7cb69700f6a98503a5d122",
        "b38e6ae468f9a9108f8162659d2ba66711e974e79c909be446a4d92bbb890e4c",
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
        "6ede698552d9ea95d5d213265e0256252ab5da7023091f09400a7ff9d3134068",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "2ad1c282cb29fc0aa40edaf9475003a0c55f52f482f0324c47109dba724ab454",
        "6ede698552d9ea95d5d213265e0256252ab5da7023091f09400a7ff9d3134068",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "c867208199b95ca5494e181109a4c3b783c7300d82573c03ffe07864cadd6213",
        "459a8513994fe838abc87b85c128ca7015ea1a877641e0e8133e8fb687b7081a",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "8d5408803135a09dc89ca9c4c9b9a6920f29ef6933424baedba90be36253ef07",
        "2ad604a83455632979f3b9bca54242c4f3d97b8b5e12d573c01a081fd75495eb",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "b11279697aa3047c7fc4b3d7cc52f340847dff19969be8b3d451e9ee4f61700a",
        "f8502639121dd92bef1a36b58deda779861548676946d23e5c13c8594e3b216c",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "3e3398123d238530194067b4d4f796390dde429b5cb326dd2bba35e42d5eab07",
        "6dc903aa3b0a8d4adcd82d316c2eb804e0fced2d646007dad9264b76f73f03b5",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "eaf2826ba0f64e2215cf3b414ed314e9c4bfcca7a4514d07d416d0032459d081",
        "866afd884a14def4970a89781bc7920ea63a0e2fc4529736e55e85dd387dd5a7",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "42c08fea0fc32dcc94ab693402ce373bca220353380cf00248648039205318c6",
        "5349ba93a62128d232342c927716902fae1da69c59f264809e5e90d3f8348b0f",
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
        "516267d4c1c183c3df07bb1dad013237c7d395cc84116057b8a979c49c5c1dbd",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "eb41c5bcd99a66a34242ad7e47c4ef463f142a6ed0dc3296a1891e7321211a14",
        "b0825f9739429a233d5e58804445e54234c187874ec0debc744a6d456fbfda0b",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "c1f6bc4e10e9477dcb807a721ad2539e2d00ed1e62e69c5dbc796b369fcb6809",
        "87678712f57e7cff55b4156e2fd59761ce225e1d95b892305fed8c1d7dc071a1",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "eb41c5bcd99a66a34242ad7e47c4ef463f142a6ed0dc3296a1891e7321211a14",
        "8707a67ab72b3fdf059f5c26bbba351c819488652bde7e2242593f4145ab499a",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "717655e8aa24293a5f1b52c9bf030c3627c8b0da0a33344044941e831a1865f2",
        "556fc81339e902aea1391d55b450efffd0edbe3d0fd9056733c1ba750d683527",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "8584263066cf71793204f8bed7a1ee7f0d8191901534665efa7543f3b19b74ec",
        "556fc81339e902aea1391d55b450efffd0edbe3d0fd9056733c1ba750d683527",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "717655e8aa24293a5f1b52c9bf030c3627c8b0da0a33344044941e831a1865f2",
        "db1e4d01720f36ed6a27190045cca4a528014456af516265675e38ff8bbee55f",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "8584263066cf71793204f8bed7a1ee7f0d8191901534665efa7543f3b19b74ec",
        "db1e4d01720f36ed6a27190045cca4a528014456af516265675e38ff8bbee55f",
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
        "009ed8bfdf5958aa517bb46aa8caa9fba71f751045da8affffda3a9863fe0bf1",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "506255478fbc969fa69b20d5941f31d090b3e9929045b8be4ad9d42be2464a25",
        "009ed8bfdf5958aa517bb46aa8caa9fba71f751045da8affffda3a9863fe0bf1",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "ec06dc7a644f11924b9057bdf61a711b6bf3ec9053c77b5ab08307ec03250e3d",
        "9c6c3569bd2b9a7545d38477eeecf7a6532a4384cf78d5e5ef3c46ef37015aa3",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "8ed702e5b89f8fbcef34396ad50843bc449cfd1ed59e7a3d93125c15d442285e",
        "86051528246c63e3d0f2d5df5b632d5775a3e1d3bbcacebb75d028a072cc9018",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "67a7ab21a51ccadf2bd910f83970fa8b974c7f4e0b09d54969a48d7e432acc7d",
        "091834cb46bb83f986805ea37c49a2ec17a5a66077c117a7dcdd04bfbd7ed6bb",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "bd1bcc9da26625e245ce21efc898c0edfd410c240130835642e46c0c75ec82a1",
        "4d3fe308986384cee610da346aea6f8b19d30bfa49cf9a4527a0fb81986ba653",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "614c45a6f6d67bb4a11707d08670928e137f17bdb414e50b30b3b5be2fd5bc60",
        "c836f7c515a1d0b6e94858ff8302d13870550d0fac44aaa2cf007513076315fc",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "b41b3e2c2bff1be81df8e6138cffe740196807f839658eca776dc53a9d42a340",
        "15214fb77d71dcfc293700b72f67fdcec389f69b18996bd4c12138f42a3646a2",
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
        "273a95c40e7ebed1b22bed470ab25261fd60401143568f1b4dad722798d6fd3e",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "6990a3b45def317fd891fdf18bc3f30b6e58776415af0a59b5c3ff88a9bcddf7",
        "e0bdcb73f3bfa0e4c5d89b2925c30d8c6a72424d8a0c527cf53e60f4014514a5",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "2c2b889cb3df9ff8c906b2bc859822f22dcd77cd7c4ec336aea1139fb623066b",
        "24ac09f93e596ba1c6a0d1dc223eaa389d59831a0a5bd095d106910800bdd350",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "77dca8b602f64db7c97a1ce7f88be189ffe20de55bb74f11925943c8089c687d",
        "4ab462a5d67cc05f2448bb20463e3babeea116e2c01a25216b4f0a0ca746cfb4",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "ead88c8a6e4c36d7234f4530ee2501853e1fc9c8f36e23a02671f8fcd46269d1",
        "44c89c20695d8997283261b2cdc79ca4dfdd90f34e78a980f696191202311ba7",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "c81fd013162edc9ad220b83a0eab2e34c65337f1f73f0de2695b867f12bce04a",
        "e8d9f4c6b75cfa9b30e279a669f2d6f08fb9c9ba7330f18d468d7dc714bb81f2",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "25d57833895051068535a8d8ab1ac3b3edae6bc2f7c1af6e230da9245337f532",
        "ae336bf6679ad1af79f2f1262ef4a660eb7bce059fb8278fef15c1f3bf082ff8",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "90075c152a7b6a0eb8d0a95291bd10f8299088c8ef313c6dd700e49caf19a7f7",
        "63610877a20cf7d12652d591376d3aeefb8ef26bc517b385ca0178636e848429",
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
        "cec226d8be64f9cd37af9b0a2ddbab67f827fe73a79f8afeb21af459bf3064f5",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "3cf39d70aa253bb9f493b07b39a8ddb047d4ed32687d66fb6bffed1f54e3c3ba",
        "9c302dbdd1ec4bd52605ab65e8ed0a9e45d9cecba8f74ab37929b0de7f098d8c",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "bb93bab3cd869beba530e41e8d6e2703d4866320d5737adc99644e33ac5e9f59",
        "d02d160eb6b171982928d586e5f847ca942de75eba5fb56f0e2c5ff82206c661",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "40f0c4c1f4b131bb7d3438d96ae2817c5e9af3fa64c3000036eda0aaee39ca07",
        "80b2e4aad2b32951c193cc86c5a080c056785a066a9742a14a76ca27d91ffafc",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "4b93c3acdbd44b79930219f9c20b591477bd8eb15f908e98d29c72787063260e",
        "2470bf101ce8013645b88db6d23c9abf0b75bdd19c18680cec3edab269ce4269",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "da7d4237084044eea8f3054b0dc71085d0ea07a7ff85a0bdf988fbe4041556ef",
        "2470bf101ce8013645b88db6d23c9abf0b75bdd19c18680cec3edab269ce4269",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "8bf8c40668356cefe9638a1b33cf36f42ceaadb9d469fe1b41724f795fcc8c79",
        "5b776b5d1a0efce3b836e5e5abd698ef4148a3ad35157139674dfc517c424478",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "75ccba9ed34285b357550845d9d3eaeafe4373ab3ae7481166292d4ad6a7b451",
        "5b776b5d1a0efce3b836e5e5abd698ef4148a3ad35157139674dfc517c424478",
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
        "65fb5bd5317c8dc26188ec7ee051559a05e4aac3951e3990754082a0787e612c",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "f49859eb4e79ee6db67ff5a4083a21f8dec7d9cd62b515ded2942421606c8301",
        "65fb5bd5317c8dc26188ec7ee051559a05e4aac3951e3990754082a0787e612c",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "159d4da4f0134dc56090e1a048c11670e2b7ac8aa9753a6d75961c8d0585ca4d",
        "f27ef6981002a3750bbed69aa56ab8674480da3cf1af3e2b6e133b42129abbe5",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "c14383773ede2fdf7a878dd147c56380a714672f514c033b8ffa25e608139116",
        "f27ef6981002a3750bbed69aa56ab8674480da3cf1af3e2b6e133b42129abbe5",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "c790dced3a5c5f366bf327f76beae9414f187c378f6d61b9380b760d6fe79c6a",
        "b77a253474d28340ce6387dddae8f6b4d222e611a0cc3ff0073b53a83f4399ac",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "3ebe0211663ed011bf062cdfb8a1d618df307a3cc4f5ce583f7b59a79fc746dc",
        "b77a253474d28340ce6387dddae8f6b4d222e611a0cc3ff0073b53a83f4399ac",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "50c60ff0d708d4f08e8e2ed33dd229160e7c8397e634afe5151cb94969d29054",
        "338255d43d778ca6f2bc6dd13bbc39e739a9f8097260e8385db8286819abbafd",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "ce28f5a9d8f14937fd02c568e463820303436b6564b447f3075ed9b8ac4caf86",
        "338255d43d778ca6f2bc6dd13bbc39e739a9f8097260e8385db8286819abbafd",
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
        "6cace9383039b8fd27f86a8e0b22e36c0a353ea3d26f8077c20100d5840c4de4",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, False)] = (
        "47a0fbf624bcffe769a554dacc36b74c0325cd37106f48e7387fb788e85391aa",
        "7abd730840e24a323d62f3edb6918997ef427a3a13f6fe3d6ceff4c8cf5a50cf",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, False)] = (
        "4bcc9505b58a95f960c2e4ebcbf95aca5607cf6bfb2c85cb74f12abb8b99ffa3",
        "61bfdf7d31c4d077843ceac6e11c439baf45db6e5bafc99d91c5b426a5b47302",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, False)] = (
        "b98f3558987c2b739a7210fbdb13fb0a6238c8f2318467f0164d986445d2958e",
        "99756c46a98acc2cd208bc677e24c085a3d6ace76e0e0f1953ef707da373163e",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, False)] = (
        "5db4a2286e1620e73c7e61901a0e698cbee86715f53b61dcfcd25200981cb53e",
        "71c9eca07cc0c1a548e470fef504faf9c3c67098e63b7331dfc6a325a84d6868",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, False)] = (
        "2d4c1ebe71a5d3a617c62779f5bdffb8d650b072052c052beca75bb7578bdbe0",
        "c9ed3eb81436788442ea64b2185441a3c22240cae7de1150b61620838a96735c",
    )
    targets[("Schoen", 2, 11, 5, np.float32, True)] = (
        "ef8e2844e499e1dcdf651e3298f8cb48be050736d8cc5434402e2ac0c9a47064",
        "c029434fb3f38321843030875cbc386f7fa4b02303bb11969400ff09d773c3fb",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float32, True)] = (
        "5a25b19df9fe6baa67579822e083fa6d109187948feb88b110971c6b64fab70b",
        "8c850b0830ebdc37cb3ce854b388bec9883603ac7d8b26068c4da79214bb3e55",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float32, True)] = (
        "f0fa7ede3ce297b1a887a08c54b735cfceac4f5617d3c934211572ce2669b3be",
        "db249377acf9a258db9c7cb592e27893a23bd8887c0aa860e6949076cf3f5fe1",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float32, True)] = (
        "67ffc08aceaea9f52381a12662ff6ac9fe0c5c2ea90a09f2e98d3ac3ce5ba86b",
        "8ea0fa2db5d996800db3cb2c4dfc6dbcc40872ae56fb7fb0a60659e188ee2b2f",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float32, True)] = (
        "ef13a9d678fd5d6358c7a7d6f0ee4a8d4ba65c877e45c482a9985e3672c93f51",
        "437337a135073375564b9b0a0d842569128955ab7c7ebf07921822c061e21cc3",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float32, True)] = (
        "edbd33921ae7b21936525d75c7dce31a1dc9ff6be434a37e11578a50a41a023e",
        "c07a219415a9ac43a20c9884a1efeaba68ff16e0e13b3cbe444b8f23bd7236f3",
    )
    targets[("Schoen", 2, 11, 5, np.float64, False)] = (
        "a6a958173fc33875cb71dd540e07ffcc0a6755e0e26cf6b3753392f6269ef06a",
        "1dd8fd401e31c38fc653e53ad1bc38984f87376f7c1fa7bc91ac7860ecd2010f",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, False)] = (
        "b73803b66940c97b4d7580c42d93ee82b9d0ecbdc382bc637877962787ae9e28",
        "99d4884a23ce16e7824b21c9f061fcfd544bc3e557826b538c53a98605df0667",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, False)] = (
        "74b69b13aaf11c09e273a5175a2ac7699fca0a33688531d4a6988750e8cf95f3",
        "ca9442f8c94d69d9368d6fba7f759fad766c94cf13c700c7b1bf190587f1f326",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, False)] = (
        "d8d9733d365c690f42fa72c17a4e6d11a2ae37c3bf0149489e72854fec7dd630",
        "a161e729764b6fe6031c9d008193acedbbc3cdc75951a2f320a9cc988f1bc930",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, False)] = (
        "7e5cffc7cef12fbdd2ec88a2273e619101c9fd4843663d81a8cc41c11efa9ee2",
        "deaf8f90ef498ce588b4041e0f5b472490c25beb3267464ce4f078a9bfaad9c3",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, False)] = (
        "9ebff09efda06489b495e0dfbd4b8b080143c1f14e6c76fdb574dded2bc630b6",
        "5ca8f145e6c4eb4d41e7fa4d96059dcf2d3d053c9349c75fca882cec5cea3bc6",
    )
    targets[("Schoen", 2, 11, 5, np.float64, True)] = (
        "0849cea01ae1e23cb29b4b01bf6031cb7bb7c5672cea2aee5e1c0551dc38afcc",
        "ac225a69ce938d47ddac5246563638ce20e6be33cd6df7c7173823e08f81f552",
    )
    targets[("Schoen_IWP", 2, 11, 5, np.float64, True)] = (
        "90aac46313a7226d347928bce53232d91eb83363d0addab550579e2e0d2f5eab",
        "a7d7cc2db9ef38e183a62e76a715dfd0ff722bbd2f47eced5d6d59db88f5d951",
    )
    targets[("Schoen_FRD", 2, 11, 5, np.float64, True)] = (
        "4b9b6ce53d45697d0d43195f3611dd295af9294b6b99a5aa073393b1f14f45a2",
        "11c3500ec18aa03e81f84fab4bc952544185fe6d5fc78d13628369d9311263af",
    )
    targets[("Fischer_Koch_S", 2, 11, 5, np.float64, True)] = (
        "7d3003e0a6b3892d807e86d9354c8970c1f11c5cb0dda39c6a8e80a2abf2fd23",
        "fadf334fc64c6762da07e2e462f145a9b6596fbb96e3c7bed2575b9b38b29125",
    )
    targets[("Schwarz_Diamond", 2, 11, 5, np.float64, True)] = (
        "663d438e0f9913cdd155824c400ef52d3f44934f3a76f302f8ed0af77fd68da7",
        "3c581049d72b54b05c375381528d82598447fedf7414f851acb84066d8531376",
    )
    targets[("Schwarz_Primitive", 2, 11, 5, np.float64, True)] = (
        "4dcbf88c8f81a4e164b3cb3a21f075ca9f3b73372ba3ae7f75a7f5da10142931",
        "0d14e6b8da5fdc9759f04e2aa3c204746f1fa297bb6eae10b5f28f45dab25d13",
    )
    targets[("Schoen", 2, 12, 5, np.float32, False)] = (
        "13da4df3cf6cef59b78b73385e024e8d459cadb5fcc9c65b5d716f2c7a3945ce",
        "4574742e97ea029193d59276bdbb007e85b0563e2973e006ed450edb03b0f1ef",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, False)] = (
        "cf66a7b22390eeaf0fd74a318bbac1179d72d356b51353da5c1d4dad511eea40",
        "c7a0200d6e15afc6338004105debbf015984870c77062f5af61c7e6c0241deb4",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, False)] = (
        "dae4e85421947d88c9b09b6bf6b5edaf758cd728234ce53d85458eb96dc8bd72",
        "6bb153424fcd1c5426d0f8e9f5f0ad450ec119634172047d43ec1a76aa0d8ee4",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, False)] = (
        "ec6f6490f58e0fdf52cde7eb284fc9cc437a0ce28fdea6d5c0d8c86d807c4bff",
        "048e96107d405f74a9a1c10a36e8ceabb65f3a41dfd03768bc7679a0ac1c8dd6",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, False)] = (
        "dc85c157834d330b5459183c3c708f80ff17c5b543447025bc25608a7a6cae98",
        "97c87bf9e8d4615627d304d51b394181d6ae74cf1afe25187c6974ac09313967",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, False)] = (
        "c20a8be83f82a416147a37d802271bffcbc1b94526ced7cf2c29f1a8874fbcf2",
        "1e9834ee416b0b521ecb19b41189ba25f4023fb2512bae622bd8b46620279921",
    )
    targets[("Schoen", 2, 12, 5, np.float32, True)] = (
        "8d6fce2d7a56d6524166e374ba4a2a62398b5ad7d801bebfce805e9d3a6512a0",
        "8086ae55070426454a8879d6c81ef391810fb01b105e167b123f7fc8acb1d5e9",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float32, True)] = (
        "c6ab1030d8acdd4023d62998ba64951c7c4d8884f17b3d25430f12a9206cf8d0",
        "62ea0d52e8c9c5dda5c8beb8f1838ee305e5a0372960e05ece5bca539cddbd5b",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float32, True)] = (
        "9871cb44c845d105babaf09dfe7c15a0e5609a9146b5fb7de00ae9ec718d01d3",
        "a978bace445ba14aa0f991b4ce09cf46a4e3b39a15cef5ad542b51e0993a1793",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float32, True)] = (
        "7a5e57c4d95184deb57baa3abf3f9ab1ae5de2c910b65a25772f12b12a8b315d",
        "a73fe92f771c7b278a54a9afc61e609466fbb9a4b1934a72f5f5bbaec884ca84",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float32, True)] = (
        "233ac98f4bd00c84f7420b393ad179f92ff24a8a2a46fa6433563bd85e4803d5",
        "277254bd1e0b15b54443626b082805d3e5b4d2421903c6edd79e58f8d013a8e3",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float32, True)] = (
        "88098aab8e06a69d6f5c6a09e7d48e3bb3ced4c2e7e08f4a3bc3e07ea4fef257",
        "0494e2542a967d407e37232385115746ce52351c9f47452a842c03d0b2ec8285",
    )
    targets[("Schoen", 2, 12, 5, np.float64, False)] = (
        "18cf48b18c5cb4bbdaba95bc857c0c32603fa7b8661dbe1d541b20510a64ea79",
        "10ce62edc246f1a484ba578f6ac5c6a234dd0fc99a9ace2215d4d81366b59c76",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, False)] = (
        "efe04ad4734e30213a8e0c383e6b32f1afa3e62a18b922b1181b50b6124eb392",
        "0dde93f9d4511b746ef8a6c44d527b72935833b110683454a63d89dd3a4e2bf3",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, False)] = (
        "f9b2fec2e5555dc63617eb7d6282c9c10cf54db43fbdb1b4362eef44bc351cd8",
        "ef82670f37bec2a1d904dcd5709615b850f7b3666314022cd59a734e7d6bbd12",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, False)] = (
        "70c05f2980bcde7f56312ff0805e76ffaf260016357a2746bfc3d855c95c2899",
        "86f5728a024270ba8133b20fef0e93dd598f6d804c0844f204ff15e0c43dfc4e",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, False)] = (
        "15bf3f2d2d065220651446793fab4fabd7080dc4008ec19d20089ed3df0d93bb",
        "aecabb558ac56ce247b059112ea463461829a506f494fbf2c7e816200e7d0738",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, False)] = (
        "a1de6d8d48eb20baa4b356e55c296d32d86fc4a7877a46731dba0d306f9d2a4e",
        "e0c9ea8f4f351cef69241638551c7e2d3c362fbf2a4363e25be24e30a1c67412",
    )
    targets[("Schoen", 2, 12, 5, np.float64, True)] = (
        "cbff011a6f4bd7cb03489a6d694913074bae703c0dc174c650b550183dd9bee5",
        "96c8b31e6786abaf09b204d5f56faf2e5cb2093ababb311782629f4b7622fae2",
    )
    targets[("Schoen_IWP", 2, 12, 5, np.float64, True)] = (
        "98f43de6273e0f7e5a84a5c407c8d358bb5de3c1cc01f4be2ad85e67e3b7ea9f",
        "0ab279862d2211e3955723ffdf314338acbbe5a21080ad507fbc23da72e3ab83",
    )
    targets[("Schoen_FRD", 2, 12, 5, np.float64, True)] = (
        "36bd3dd92537751fe7a33c91eb48a5567b636b9d8d8a0658d79bf853301d891c",
        "1b68d8afaaed409040005a2ca249679761a0f1fb36a7dd4aff32d7180283509e",
    )
    targets[("Fischer_Koch_S", 2, 12, 5, np.float64, True)] = (
        "babd22f191ca37436ab6eff3d45f312ffa90417f69c49cc8e3e7bc7f06d738d4",
        "398c4f211f51cfea9ff6538cebd5c418a95a5a6925512c0fa20ead86b0b60d0d",
    )
    targets[("Schwarz_Diamond", 2, 12, 5, np.float64, True)] = (
        "337bee2cb415a16c657294da5268c87915c63da922ace498629f7c7820ed7c5a",
        "ae144409787b5369ecaf648a134b82c8ae9d417eb4e05d1573a794048c9d6ff4",
    )
    targets[("Schwarz_Primitive", 2, 12, 5, np.float64, True)] = (
        "4297104343fbd5177b3cb690e28f3e88bcc551b452b67b84da382054e0ca001e",
        "57457d6ff296015eabe19f6160741997f03ca713c5575450c201e63585a24f88",
    )
    targets[("Schoen", 3, 11, 5, np.float32, False)] = (
        "d1d78262194cff06232251351eda8455ab31cce58dd85bd4cbca65e0c56cbc08",
        "84f03aec5da7c8c274c555699890dd248f0d88cdd7ecbc2900952fa5a2c9d61d",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, False)] = (
        "afb51643d51313b96b56da33f6bd686464781e7cd812d3815ea9acb9454d81e2",
        "db4e3e70a32d9e6644bb97a3fe62d09fbd3ea02f8f47d979d588e939ad996953",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, False)] = (
        "18c06349a81694be528397c82b3c9b20ccabe0be86cf8618322ef2b4b66c604f",
        "a1f3e9a76bc5d89b9cec4ce0512620d32be5d0fc50f1f89b3a7e25466d6e9a94",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, False)] = (
        "695f4df814c16df11816977fc9b23c02c3e9cd776398cf637f929401062a6fd0",
        "e04955afd1f2768e38a58d7a9b733ae83787bf8f6f7c47f86a23ab7c7d7a3385",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, False)] = (
        "092249ed36762054851efefcdbff29130576aac7d5d79d429fe7099ccc0c221b",
        "41a1bab36881deb64218e047cb01e327969bb004e7cb9ddffedd3feada9e8e1f",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, False)] = (
        "be4ffa35a72adc1c16e5e1a88acf36a7b822b418528b2f636f35b378f2d7d5fb",
        "f72ff725379d8cd7629247dccd64f667ca405215ceae2ec3345228ded5ed9580",
    )
    targets[("Schoen", 3, 11, 5, np.float32, True)] = (
        "c756e20041b294b19ed7b29d54eec3a823d4393ef74a291a42a10a2646c9c8a4",
        "2a7f3bfd7e1af9fda0d12b1973871613636467bf529ad95b00f6563d6fdd7d1f",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float32, True)] = (
        "b663e0b8b0b0bfa5c5746afad4d27137a663370c495fb0794d74ea78acfa8d92",
        "cc011e711bec05db8c738fdf4aad7871d524d39f746f35ee02a84619d31a7dd0",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float32, True)] = (
        "5ab0058325cb97fb5930663e21dcfd0c33e433b25fb216051d66a1f569d1a55d",
        "40eaeabeb1d0bba89bb518d7ba595d3d640275fe8f0fde3d41f65bd38a57c4a8",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float32, True)] = (
        "30e5729ed7bb8a3b437f44bab01d64f980fb37825d21dda5cae6e3095050bb0d",
        "43738a4301fb1bd9ddb6d19b53bc3c27f2e7e6f3d3d4b61d9e9bd3eaaf619e31",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float32, True)] = (
        "ec9998a3107ce18ec85ac7fae69ab36b44de20f4802768432e637ac7dbfe189d",
        "23f7100486e3bd32cd6582c3ae5d35c1760c463bcbfc69595f80a9a232bef057",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float32, True)] = (
        "9045020cc52d9d260ea200bae661e0f361868493a289699770b526b351e910fa",
        "15024f42754d5242affba3dea2413d406b1ea2a84f6ece97a8ec19df9c0ac950",
    )
    targets[("Schoen", 3, 11, 5, np.float64, False)] = (
        "913a42d2b001ce6b9fb6786671fd055bbe3fed7cc40dd5aea0ffa739f4bd4e16",
        "2d4880b2d95b3dafdb303ac02e29ac1cdda15ece80ad990d4d1ce36fc99f8529",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, False)] = (
        "2c153a5b4381c8c89d89de4c2b31ecc1ddda0a59b06b54471d4a935c0bd9518b",
        "f6fb1a675fc301a9e7311ba5a99cf2f2707168bd81a24143d68017cce6de0ebc",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, False)] = (
        "071a9b38c2aff67de8a8f019a87ea317ca2ae4707e72f045e1d87fe7ae73adf1",
        "1a17e49cd711bc5dfac0fcfc13341d1d583a2d5b3465e1dec508cf8ae5aec1fc",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, False)] = (
        "3b2e75426434f4948c3fbe91e2d5ac995c6c966cb2f5f6eec66b0588fd9ee5cb",
        "76c3ea1e72da171ea3a85cd2005dbeb1fef80b26105f1245e49328a999c6677b",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, False)] = (
        "1890dce517cf9b1ea5fd171a5e53c5f3e785b37dd9e3ad0f60ae49ea3f404927",
        "0f293d69e1f83ad69b7db5bd9bd77c317a67e133013fa7a5ad4b26c28d240886",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, False)] = (
        "7ca67171ded3174ff1dc2ac6da3bc6b7256ee942ebb308f1ee38078885b0719e",
        "f1917daad625fb38f67ec53c633b1d1481aa3e184758c23642283d4cef4b8b32",
    )
    targets[("Schoen", 3, 11, 5, np.float64, True)] = (
        "adc8dbaf396e15e0706a53da9e7e21bb18cb472b19d65a729a629e6b4c3733d5",
        "ce77ffd0931a4413d6f6339679653b7bba55c14723af3164ed44fabe8c95c243",
    )
    targets[("Schoen_IWP", 3, 11, 5, np.float64, True)] = (
        "c3876b685ac737d63ca24ba2a35eb835d19d5a6febf2b01cf87e48bf01562ea6",
        "930b8c50aab0b06e29fa361934fa030077fa1d250eb2b7e8086d552b3aa57f15",
    )
    targets[("Schoen_FRD", 3, 11, 5, np.float64, True)] = (
        "db4c4d87be808dd383e72cf63f5795624eb8bca330891c917547d51b8782279d",
        "a8169f221e7fbb385f445ace87cc1f7a92e8ec910729739aaf546989dc196c05",
    )
    targets[("Fischer_Koch_S", 3, 11, 5, np.float64, True)] = (
        "84bd201ad5de322f0c2362df0d095bca3dbe9652d055d1c94801e92e69f55932",
        "10f6f36742a54ed967391285fff27f7e9010a89d6c0d2de1e937823fa9612ae9",
    )
    targets[("Schwarz_Diamond", 3, 11, 5, np.float64, True)] = (
        "80f6d48df3b76a6734a72dbf041a6ee008030e84e335974bb8978bb8ebbe0fe7",
        "ab45f6c204d1c72fe185b472546cb121416cd71171df4566e96091511eeda69c",
    )
    targets[("Schwarz_Primitive", 3, 11, 5, np.float64, True)] = (
        "da30763ba9342db29ac3bd641ebe92e6a808c3759a7054289493e1b4670e120d",
        "7d1a7dc425f67529cec6649e600e803039f74839caf671f1406e3ca030637283",
    )
    targets[("Schoen", 3, 12, 5, np.float32, False)] = (
        "1d0643d262baf3674d1c9c62f0c2a084804004c29477d9b606283f1c147291de",
        "124b8a2470f6e757f7ba689c35ca9c8089ffacdc1d2a22408c8780d1cc2425a9",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, False)] = (
        "7ffe9b0f1c1288df0a2a54d60c1a21f5d8fe77aa211fb71563bba4bf84b33797",
        "7494e5aef2dea0b9b2f32d3cb720a4e9eaa8c35f182a4b4d9f5f170358614156",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, False)] = (
        "425998d41afd0736c3cedfdf8f51d88ee2b0f250c588262a16cada97d1e62f0a",
        "d3369af3c3385a3523ef0db6f465431c8647985f9e473b4610fa4b7ccdffa95f",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, False)] = (
        "035935cdae861f80dff2224f0e908177625a26f4f967d72af810c61aaed4b3fe",
        "0ec21978f6c577d615a93a66c17494b99442b6acc27f811c39ced95784a3a695",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, False)] = (
        "a7672272840484d04f6985f68914d8c805c1905e203e6af823ab881c8c878a8f",
        "f5f7531de7fa88791decd433099be0fd4c3b5bc1b0390a281bc25f0686656acd",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, False)] = (
        "485015aba9c82542f49e440a06e31eb5c44b2d00e1b0adf800f3655ef901f040",
        "b53c3a801a6b68aa25593484f7420b800580551cf67625e1e140a4593440a8d4",
    )
    targets[("Schoen", 3, 12, 5, np.float32, True)] = (
        "6c91775260b628a786279c9480d0641ff5c72003c05617f632de9582a755efd6",
        "62b65721612e21d74fa60ab42cd0e8dd2c85f2718da9677e38b14f22891ddea4",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float32, True)] = (
        "b2af3ff0bb8793746d274ea3ed581624a75a1fd7f37124c4b864c1ca9a15b528",
        "6b240dbff8042749c12b2413deec436851a99ba1b7e66d03ed8999f15d9fa615",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float32, True)] = (
        "e8b33ce8be788d363d387fbe6cd0befba37773f12183b3adc9929bfb63614915",
        "30845bad85d54a6d7e2a02cc79d651ce0c47ad0e282b837a3f5398f715efe1d9",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float32, True)] = (
        "98133d54af8b850a1868c950df923615047d8e3a92209fe046a57c6b5414eaea",
        "10e482571bb29eec42cfc4e14975ec0855b033a5ef8ba8004e68d7665b47e62f",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float32, True)] = (
        "73931adb8bd6e0e3f79c3e92a3fb576b94108a991f3f687c86df75f0de3c5b6c",
        "ac67985c1779062c057ea79c9c729146f458b62ae01de0637cd2da93c9cf30a9",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float32, True)] = (
        "04a633808c8b4e3765fdbc61bcf8be097316a51b5aed71c7b90edafb55baaa40",
        "8a6f62ca27d6b73f8865e1c88d1fbb26c8f102d90a0801a15891279b4709b4ff",
    )
    targets[("Schoen", 3, 12, 5, np.float64, False)] = (
        "8d38619660d53af3a9c6dc63735186ccc9a9b4619dd0575e2b53e83f15075d65",
        "5fc4328cc70095199b77e79c8afc7190b9dd79452061658db15aec6f6f917734",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, False)] = (
        "690e050034717cf1b4924ce5e3fd101df6da881efa098ee258fe35e370dcfdf0",
        "e26d355270026ac5f98750582a58b7763b6e078d03454b31f7e54d568942641d",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, False)] = (
        "e646d2fcb5791636fd43b0c0740fa2f2c3b74c389e957c315dea90069e8b9068",
        "e2e7ec54ad1d90f321689e55d08a497ca1eeddf7924745b5a09941df66297b02",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, False)] = (
        "7e78e5d498963b3aef869ed1c473efa9a22a5f198ecb41f4a368d08f29b70d6d",
        "8146cc757c67f95e90f865cf17105db59488964f86f2a9b7f6bf080150cc1ff3",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, False)] = (
        "e762aba0411d216dff88eb9329be496a7c7cd3bc598ad770da93d16d7039ff64",
        "dc33b8444e8f5261700e8fad37d116bef90ee990daa07967c7e55f60507644bf",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, False)] = (
        "f5065f46b31c5b86b72937ab287ec55009d50c054c7654cc84be5f8370b8b878",
        "d4c4b5880aa4bc0d5ff4288e5a79dd20cd4760f8d6c873344ed606c035ee1899",
    )
    targets[("Schoen", 3, 12, 5, np.float64, True)] = (
        "3e389d1f46eaf63fe2ea47a347a896916bc614e122eac62dc526cf8b52624b14",
        "7f1fac2d015d839f26fc9bee8fb629a4b3fc84a2bdb832e47d1bf51a62b26f2f",
    )
    targets[("Schoen_IWP", 3, 12, 5, np.float64, True)] = (
        "2cbd3a6cbbd61c710d76358726f24ba7980a3ae8224868bf584d582e7de55b8a",
        "9d785c024cfa99dcd796cb0831f9dad0a6228627ab6bc6340a80e28ae032f36e",
    )
    targets[("Schoen_FRD", 3, 12, 5, np.float64, True)] = (
        "92525637b84da9ec68665e6035c8e1eca2ca8fdf960b2212eb6c67bf65c9a386",
        "85b92d6259eacaf6593f396e0c6095839965002c5a8e7a20fe418ee785a7c66c",
    )
    targets[("Fischer_Koch_S", 3, 12, 5, np.float64, True)] = (
        "04a6a1c744731ef8a25330fe30140e02a9e0a52f6da9486d57a6abff550ed1f5",
        "5017d6fdcb83152a2b4da4355bbc2429d33f305eb66ebdf0415ae75aa3cb5c32",
    )
    targets[("Schwarz_Diamond", 3, 12, 5, np.float64, True)] = (
        "a003c555d93292b30d467606eaca0ecd5ad1f7444f37968d26ee552da8004495",
        "06f14973a91029d106ed65c54661dc00ad506cb8a1dd3ef1e30d7b4fa7716ea4",
    )
    targets[("Schwarz_Primitive", 3, 12, 5, np.float64, True)] = (
        "9ad2ef82a715dfef334ff1a042680fadc074671979893fde411094820a2efd97",
        "1e33b7a2b06818aad34b3b1f2451b1c09efdb6f5cd8828b78d0ebdc37363231c",
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
        #     f'    targets[("{functor_str}", {dim}, {n_cells}, {n_quad_pts}, np.{str(np.dtype(dtype))}, {negative})] = '
        #     f'("{computed_hashes[0]}", "{computed_hashes[1]}")'
        # )

        assert computed_hashes == target_hashes


if __name__ == "__main__":
    # test_tpms(2, 11, 5, np.float32, False)
    # test_tpms(2, 11, 5, np.float32, True)
    # test_tpms(2, 11, 5, np.float64, False)
    # test_tpms(2, 11, 5, np.float64, True)
    # test_tpms(2, 12, 5, np.float32, False)
    # test_tpms(2, 12, 5, np.float32, True)
    # test_tpms(2, 12, 5, np.float64, False)
    # test_tpms(2, 12, 5, np.float64, True)

    # test_tpms(3, 11, 5, np.float32, False)
    # test_tpms(3, 11, 5, np.float32, True)
    # test_tpms(3, 11, 5, np.float64, False)
    # test_tpms(3, 11, 5, np.float64, True)
    # test_tpms(3, 12, 5, np.float32, False)
    # test_tpms(3, 12, 5, np.float32, True)
    # test_tpms(3, 12, 5, np.float64, False)
    # test_tpms(3, 12, 5, np.float64, True)

    test_disk(8, 5, np.float32, False, False)
    test_disk(8, 5, np.float64, False, False)
    test_disk(8, 5, np.float32, True, False)
    test_disk(8, 5, np.float64, True, False)
    test_disk(8, 5, np.float32, False, True)
    test_disk(8, 5, np.float64, False, True)
    test_disk(8, 5, np.float32, True, True)
    test_disk(8, 5, np.float64, True, True)

    test_sphere(8, 5, np.float32, False, False)
    test_sphere(8, 5, np.float64, False, False)
    test_sphere(8, 5, np.float32, True, False)
    test_sphere(8, 5, np.float64, True, False)
    test_sphere(8, 5, np.float32, False, True)
    test_sphere(8, 5, np.float64, False, True)
    test_sphere(8, 5, np.float32, True, True)
    test_sphere(8, 5, np.float64, True, True)

    test_line(8, 5, np.float32, False, False)
    test_line(8, 5, np.float64, False, False)
    test_line(8, 5, np.float32, True, False)
    test_line(8, 5, np.float64, True, False)
    test_line(8, 5, np.float32, False, True)
    test_line(8, 5, np.float64, False, True)
    test_line(8, 5, np.float32, True, True)
    test_line(8, 5, np.float64, True, True)

    test_plane(8, 5, np.float32, False, False)
    test_plane(8, 5, np.float64, False, False)
    test_plane(8, 5, np.float32, True, False)
    test_plane(8, 5, np.float64, True, False)
    test_plane(8, 5, np.float32, False, True)
    test_plane(8, 5, np.float64, False, True)
    test_plane(8, 5, np.float32, True, True)
    test_plane(8, 5, np.float64, True, True)

    test_cylinder(8, 5, np.float32, False, False)
    test_cylinder(8, 5, np.float64, False, False)
    test_cylinder(8, 5, np.float32, True, False)
    test_cylinder(8, 5, np.float64, True, False)
    test_cylinder(8, 5, np.float32, False, True)
    test_cylinder(8, 5, np.float64, False, True)
    test_cylinder(8, 5, np.float32, True, True)
    test_cylinder(8, 5, np.float64, True, True)

    test_annulus(8, 5, np.float32, False, False)
    test_annulus(8, 5, np.float64, False, False)
    test_annulus(8, 5, np.float32, True, False)
    test_annulus(8, 5, np.float64, True, False)
    test_annulus(8, 5, np.float32, False, True)
    test_annulus(8, 5, np.float64, False, True)
    test_annulus(8, 5, np.float32, True, True)
    test_annulus(8, 5, np.float64, True, True)

    test_ellipse(8, 5, np.float32, False, False)
    test_ellipse(8, 5, np.float64, False, False)
    test_ellipse(8, 5, np.float32, True, False)
    test_ellipse(8, 5, np.float64, True, False)
    test_ellipse(8, 5, np.float32, False, True)
    test_ellipse(8, 5, np.float64, False, True)
    test_ellipse(8, 5, np.float32, True, True)
    test_ellipse(8, 5, np.float64, True, True)

    test_ellipsoid(8, 5, np.float32, False, False)
    test_ellipsoid(8, 5, np.float64, False, False)
    test_ellipsoid(8, 5, np.float32, True, False)
    test_ellipsoid(8, 5, np.float64, True, False)
    test_ellipsoid(8, 5, np.float32, False, True)
    test_ellipsoid(8, 5, np.float64, False, True)
    test_ellipsoid(8, 5, np.float32, True, True)
    test_ellipsoid(8, 5, np.float64, True, True)

    test_torus(8, 5, np.float32, False, False)
    test_torus(8, 5, np.float64, False, False)
    test_torus(8, 5, np.float32, True, False)
    test_torus(8, 5, np.float64, True, False)
    test_torus(8, 5, np.float32, False, True)
    test_torus(8, 5, np.float64, False, True)
    test_torus(8, 5, np.float32, True, True)
    test_torus(8, 5, np.float64, True, True)
