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
        "66d04631572faa96335036b6eff187da05726917c10f5edfb5ceaad9a36dd115",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "fba12cf61483d379aa6829cd82358b84c5c320184fc0878c5764919e6f709b7e",
        "410edf30ebe1c56758426020be009f86efaa60deb18a9d9917570da87b19e305",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "935bed3f7e152488c80cabac0c85e92ab55ac858d054c7897d025314a55dd8dd",
        "66d04631572faa96335036b6eff187da05726917c10f5edfb5ceaad9a36dd115",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "fba12cf61483d379aa6829cd82358b84c5c320184fc0878c5764919e6f709b7e",
        "410edf30ebe1c56758426020be009f86efaa60deb18a9d9917570da87b19e305",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "97608eb53302a500c215891d97e0d0bfbd99a21651104aeb7fbe06078aff14db",
        "ff6c8b0f42f522c8fd8a83952edc1d54cd52c99f7d090a79c36be5c769c3f598",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "c270407f531f829892ed5e2f47c34ec44b001f57da3d3dbb36069b59400601cd",
        "af7909dac4d08bcc63e925dc19e6067da928553e6e05beee2fec98d320d8eedb",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "97608eb53302a500c215891d97e0d0bfbd99a21651104aeb7fbe06078aff14db",
        "ff6c8b0f42f522c8fd8a83952edc1d54cd52c99f7d090a79c36be5c769c3f598",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "c270407f531f829892ed5e2f47c34ec44b001f57da3d3dbb36069b59400601cd",
        "af7909dac4d08bcc63e925dc19e6067da928553e6e05beee2fec98d320d8eedb",
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
        "08ced48a77ad8d73e3dbed6b74986cdfecfff64334c756798e346e8675b5e7b1",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "4a3ff2788529c9cd5d91665cf0b88c394d6a4c83cbf393f33fd7f0a3c42a6cbf",
        "5748c27eaf5684e3ee109e8f22f507dcb5289dc06c7425265afdb8d8706b7798",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "d0d8d501301b635fca61f6a2214a954d82aea844859756d74e9822f2daa5bd50",
        "531ad82f0f8473e4565a6a3a8519b9309afb4a9fef96b19128422869767efb41",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "4a3ff2788529c9cd5d91665cf0b88c394d6a4c83cbf393f33fd7f0a3c42a6cbf",
        "40330fd57a79d78fbbbbc16cc7a86679265dab36e9a40d89e9d3a5cd994f65ab",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "d18e4f95fe39d68062e7455311bdf545f94368ff34c4ccbea55e50ea88ad0fcc",
        "426e29c505f3fa418b120b81a0dd0d4e5758d4de8891627ec28477a89b2052cd",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "9ad82d23c15760e242c7c55e2110d5a2a6d2303e051a2abbe5347b3f4b7f88cf",
        "0c5851ed28943076db1c88da8ee62a671a011baae0e10dcde2f89722e9fc7876",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "d18e4f95fe39d68062e7455311bdf545f94368ff34c4ccbea55e50ea88ad0fcc",
        "64faff9b803840bdb570db1d1dd70f5198043b3967cc207286bd954e319fd2a1",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "9ad82d23c15760e242c7c55e2110d5a2a6d2303e051a2abbe5347b3f4b7f88cf",
        "0409b4225433396388421d5a1f4e692ed913a06245375e507096ee094179da51",
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
        "bfa3f68fd3676f00f914bb093bbf170ab99f34a007f345618ca9032e49b3e130",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "08d1f77114073e5dda6d9f82dfaa1286b494f79b3fbda95d26868df8328f4405",
        "bfa3f68fd3676f00f914bb093bbf170ab99f34a007f345618ca9032e49b3e130",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "a96aff523e1d6263ac93b174806aa4abf72a61427a8de4c2dfc330848a7ec8a0",
        "302a303881b2b6bd6b0202b94c5e10b21f81db079bbf064803e78abdefcacbcb",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "08d1f77114073e5dda6d9f82dfaa1286b494f79b3fbda95d26868df8328f4405",
        "bfa3f68fd3676f00f914bb093bbf170ab99f34a007f345618ca9032e49b3e130",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "55153c915dfdd1af3ff348c824b924610f7f015b69f93d5b4d2d09774e358724",
        "6eb8f32e6366a7a6e11ff370cb8eb6b5279a4725be5fd8cf0c7f2bf3df545ad5",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "2d4cb9b12701d1ead396424ab53eb1bd43835ae88297c7531cc9c03fd8bb8261",
        "6eb8f32e6366a7a6e11ff370cb8eb6b5279a4725be5fd8cf0c7f2bf3df545ad5",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "52e0e3cfeffc117b8bb289028ec4688195ef8d82cea811afb5e2e1035cc3c07f",
        "6eb8f32e6366a7a6e11ff370cb8eb6b5279a4725be5fd8cf0c7f2bf3df545ad5",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "597c06c6ad59507f351a5fd9ff6ff23d745dda19cf7cb69700f6a98503a5d122",
        "521c447bbf29fa5548c6b5faadcf615b294a0d30f2350dbb59f9cf63f5aa65f3",
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
        "e06d1c1f4638ccc49523ad110f0fbcdc5312b3a1b4fb40df90ff9915088abcdb",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "2ad1c282cb29fc0aa40edaf9475003a0c55f52f482f0324c47109dba724ab454",
        "2714ab3a6aef0f5d906677cde739994c79e4b694512cf5eec153370adfcc0691",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "c867208199b95ca5494e181109a4c3b783c7300d82573c03ffe07864cadd6213",
        "985c6fd6f9940ac9d6bcd41582b4913aa08da661eb5e2a9850c12815354b389e",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "8d5408803135a09dc89ca9c4c9b9a6920f29ef6933424baedba90be36253ef07",
        "92db4ae1bdcd62b915062dac4f4d958681837c313eb59a31bb7e31e8b0434170",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "b11279697aa3047c7fc4b3d7cc52f340847dff19969be8b3d451e9ee4f61700a",
        "298e49c9c517facb182fe142d4861ed17f7c358a4208f1e468da3106dbc728e6",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "3e3398123d238530194067b4d4f796390dde429b5cb326dd2bba35e42d5eab07",
        "f1cc8e49f20ca11d248dee642d450ff5d33058b7fe0cadafb7c186f33a922969",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "eaf2826ba0f64e2215cf3b414ed314e9c4bfcca7a4514d07d416d0032459d081",
        "9a26ce74cac89cbe29fdfeeeb3ae44b42e22a759673a877c7e3c91e849af5c48",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "42c08fea0fc32dcc94ab693402ce373bca220353380cf00248648039205318c6",
        "cebf15085802ad35e62a06ead4a2f33ce7afbed70ce0366c1524a9ab23fb3ec9",
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
        "1e155f63f13b11b922515e2b34fbc52f51e9bba0e57a5e4d1504e77d70cde431",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "eb41c5bcd99a66a34242ad7e47c4ef463f142a6ed0dc3296a1891e7321211a14",
        "fc1b69cd40fe8183efdd50505e6785e392bc0546068f6e20f34008130453f179",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "c1f6bc4e10e9477dcb807a721ad2539e2d00ed1e62e69c5dbc796b369fcb6809",
        "0cb63fa440399742c577bd8160e1c25fa9924552459a821af14f02271f0c12f8",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "eb41c5bcd99a66a34242ad7e47c4ef463f142a6ed0dc3296a1891e7321211a14",
        "9612f30f78753c6ce39854d14906716bfdb41398e2f372e1f30d6747d7734bdc",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "717655e8aa24293a5f1b52c9bf030c3627c8b0da0a33344044941e831a1865f2",
        "30be4f0278693a55eee9ea7c3f50d43cb450d7fe5050d468fb6fff42dda29d52",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "8584263066cf71793204f8bed7a1ee7f0d8191901534665efa7543f3b19b74ec",
        "c6e2084ef1acdc83175470b98ba5068592b016404fc85e92486264c21d595e66",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "717655e8aa24293a5f1b52c9bf030c3627c8b0da0a33344044941e831a1865f2",
        "0702acb69114725bd83f35626839fbbc46f3d46f58fcf79c4f90275ea6a79243",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "8584263066cf71793204f8bed7a1ee7f0d8191901534665efa7543f3b19b74ec",
        "42c88f6565379755e46767a7cbcad3d5f00505615801be041af901aa583272d3",
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
        "35045a899098ff135a5bb5780bb5e1812513589438f5767fba386f1fe193b2b2",
        "b2d455b6aff072a5842f2707b27a3c013ccd60d288cc57db5ad7dd47eee4a088",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "6990a3b45def317fd891fdf18bc3f30b6e58776415af0a59b5c3ff88a9bcddf7",
        "a390698db103911a59e60f11c8b78a14c6bab3faf931bc230d9cfd415e4c5515",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "2c2b889cb3df9ff8c906b2bc859822f22dcd77cd7c4ec336aea1139fb623066b",
        "e5ac6895b9c71bfc9e810c0ad2d9124c5d07aae8ce65c29f5e10cf33a27cad7d",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "77dca8b602f64db7c97a1ce7f88be189ffe20de55bb74f11925943c8089c687d",
        "7a8a6ce820a9d73210dae7bc6ac0fff620fab3b38c2bc0db0dace2585f2d2aa4",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "ead88c8a6e4c36d7234f4530ee2501853e1fc9c8f36e23a02671f8fcd46269d1",
        "add3fa52d77ab252c03468025e7eac607566a0cd4358a257ebadf726a6a4f939",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "c81fd013162edc9ad220b83a0eab2e34c65337f1f73f0de2695b867f12bce04a",
        "d0a161783270725a7884f6bf110e6a4719bf9f9c3a5efe1c504c44f55381a275",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "25d57833895051068535a8d8ab1ac3b3edae6bc2f7c1af6e230da9245337f532",
        "c9072927e8934c9e0eb9f0831dd19ba9559133adee18a828a40a0892c64d769f",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "90075c152a7b6a0eb8d0a95291bd10f8299088c8ef313c6dd700e49caf19a7f7",
        "908da56ecf57ce9365d50f40b083d2c44c98a02909215ac07f7dbf43d9e2fb71",
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
        "25e13d8d3448b22bdc836a7f2b1191b7aa1a0f9534a891b71b70d82701d70a32",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "3cf39d70aa253bb9f493b07b39a8ddb047d4ed32687d66fb6bffed1f54e3c3ba",
        "0617529de85b236083149d29da85a1556fffbc8d9aca435e878ee31aef4d460f",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "bb93bab3cd869beba530e41e8d6e2703d4866320d5737adc99644e33ac5e9f59",
        "279775a0ab0e3b377c0ea70b254d142a12710461408e7c70103aeed52f358689",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "40f0c4c1f4b131bb7d3438d96ae2817c5e9af3fa64c3000036eda0aaee39ca07",
        "2710293aed8a3317d515ebe63ea354ba8d0f830fbecd4046bdf324ee4dfb12f4",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "4b93c3acdbd44b79930219f9c20b591477bd8eb15f908e98d29c72787063260e",
        "5b74cfce2f53df1ac386b61a27f0f87a93ad96e2dac8e1f9095756283ac82f23",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "da7d4237084044eea8f3054b0dc71085d0ea07a7ff85a0bdf988fbe4041556ef",
        "4dd5f4e3e88010cdc4d049bb652c85a02c825339e13c5ad054b6f8ac47cd6d07",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "8bf8c40668356cefe9638a1b33cf36f42ceaadb9d469fe1b41724f795fcc8c79",
        "2ba8f2089350c830c8531ecb3c9c89edc7894ab6d22a5cd90b8dedd449b8b188",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "75ccba9ed34285b357550845d9d3eaeafe4373ab3ae7481166292d4ad6a7b451",
        "a224c85b8daac01666a4058d05e190f4358098e66aa014f6c9bef0b59fe909d2",
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
        "c81b37ea55f7cb7afc092541bbc7e2f6e96a0cf73987a8b290598b9e23b6d333",
    )
    targets[(8, 5, np.float64, False, False)] = (
        "f49859eb4e79ee6db67ff5a4083a21f8dec7d9cd62b515ded2942421606c8301",
        "2a443175fffabee8a2fbf6a0001927ee2dd0af1a7177608ea609388df7769209",
    )
    targets[(8, 5, np.float32, True, False)] = (
        "159d4da4f0134dc56090e1a048c11670e2b7ac8aa9753a6d75961c8d0585ca4d",
        "9ff977e8500871d57027c6c0de4e4e40afefde5dd86ca800cf269bc4936d1f0a",
    )
    targets[(8, 5, np.float64, True, False)] = (
        "c14383773ede2fdf7a878dd147c56380a714672f514c033b8ffa25e608139116",
        "93a9dbbb1ad55407a0da092174b2967eeb2843801a581214385326040b1f49a7",
    )
    targets[(8, 5, np.float32, False, True)] = (
        "c790dced3a5c5f366bf327f76beae9414f187c378f6d61b9380b760d6fe79c6a",
        "43327bef9ba0722145edf48cdb976643ff49e672b1a7632f94aca1eff9862e79",
    )
    targets[(8, 5, np.float64, False, True)] = (
        "3ebe0211663ed011bf062cdfb8a1d618df307a3cc4f5ce583f7b59a79fc746dc",
        "2b47f8ec919fb4abbe619d4c387c25fd0176a570211fa616033b24d8d9a5fa57",
    )
    targets[(8, 5, np.float32, True, True)] = (
        "50c60ff0d708d4f08e8e2ed33dd229160e7c8397e634afe5151cb94969d29054",
        "0b87d67786e09df6e22ce58eb8d9d6be368b597658487462d2ef4bcece376a22",
    )
    targets[(8, 5, np.float64, True, True)] = (
        "ce28f5a9d8f14937fd02c568e463820303436b6564b447f3075ed9b8ac4caf86",
        "a87256c6ecaf4a9570ff45ee848394c60e74c117c1b7af5cf066462eac5776c8",
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
