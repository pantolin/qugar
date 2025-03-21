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


from mpi4py import MPI

import dolfinx.fem as fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from utils import (
    dtypes,  # type: ignore
    get_Gauss_quad_degree,  # type: ignore
)

import qugar.cpp
import qugar.impl
from qugar.dolfinx import CustomForm, dx_bdry_unf, form_custom, mapped_normal
from qugar.impl import ImplicitFunc, UnfittedImplDomain
from qugar.mesh import create_Cartesian_mesh
from qugar.quad import create_quadrature_generator


def create_vector_func(dlf_mesh: dolfinx.mesh.Mesh) -> ufl.Coefficient:
    """
    Create a vector function for testing the divergence theorem.

    Args:
        dlf_mesh (dolfinx.mesh.Mesh): The input mesh for which the UFL function is created.

    Returns:
        ufl.Coefficient: The generated UFL vector field as a coefficient.
    """
    x = ufl.SpatialCoordinate(dlf_mesh)
    if dlf_mesh.geometry.dim == 2:
        return ufl.as_vector([ufl.sin(x[0]) * ufl.cos(x[1]), ufl.cos(x[0]) * ufl.sin(x[1])])  # type: ignore
    else:
        return ufl.as_vector(
            [
                ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.exp(x[2]),  # type: ignore
                ufl.cos(x[0]) * ufl.sin(x[1]) * ufl.exp(x[2]),  # type: ignore
                ufl.cos(x[0]) * ufl.cos(x[1]) * ufl.exp(x[2]),  # type: ignore
            ]
        )


def create_div_thm_volume_ufl_form(domain: UnfittedImplDomain, n_quad_pts: int):
    """
    Creates a UFL form representing the volume integral of the divergence theorem
    for a given unfitted domain.

    Args:
        domain (UnfittedImplDomain): The unfitted domain for which the UFL form is created.
        n_quad_pts (int): The number of quadrature points to be used for integration of
            cut cells.

    Returns:
        ufl.Form: The UFL form representing the volume integral of the divergence theorem.
    """
    dlf_mesh = domain.cart_mesh.dolfinx_mesh

    full_tag = 1
    cut_tag = 0
    cell_tags = domain.create_cell_subdomain_data(cut_tag=cut_tag, full_tag=full_tag)

    quad_degree = get_Gauss_quad_degree(n_quad_pts)
    dx_ = ufl.dx(
        domain=dlf_mesh,
        subdomain_data=cell_tags,
    )
    dx = dx_(subdomain_id=cut_tag, degree=quad_degree) + dx_(full_tag)

    # Note: if no specific number of quadrature points is set for the cut cells,
    # it would be enough to use a single tag for both cut and full cells.
    # and invoke dx_ only once for that tag.

    func = create_vector_func(dlf_mesh)
    div_func = ufl.div(func)

    ufl_form_vol = div_func * dx
    return ufl_form_vol


def create_div_thm_surface_ufl_form(domain: UnfittedImplDomain, n_quad_pts: int):
    """
    Creates a UFL form representing the surface integral of the divergence theorem
    for a given unfitted domain.

    Args:
        domain (UnfittedImplDomain): The unfitted domain for which the UFL form is created.
        n_quad_pts (int): The number of quadrature points to be used for integration of
            cut cells and facets.

    Returns:
        ufl.Form: The UFL form representing the surface integral of the divergence theorem.
    """
    dlf_mesh = domain.cart_mesh.dolfinx_mesh

    cut_tag = 0
    full_tag = 1
    unf_bdry_tag = 2
    cell_subdomain_data = domain.create_cell_subdomain_data(cut_tag=cut_tag, full_tag=full_tag)
    facet_tags = domain.create_exterior_facet_subdomain_data(
        cut_tag=cut_tag, full_tag=full_tag, unf_bdry_tag=unf_bdry_tag
    )

    quad_degree = get_Gauss_quad_degree(n_quad_pts)

    ds_unf = dx_bdry_unf(
        domain=dlf_mesh,
        subdomain_data=cell_subdomain_data,
        subdomain_id=cut_tag,
        degree=quad_degree,
    )

    ds_ = ufl.ds(domain=dlf_mesh, subdomain_data=facet_tags)
    ds = ds_(cut_tag, degree=quad_degree) + ds_((full_tag, unf_bdry_tag))

    # Note: if no specific number of quadrature points is set for the cut facets,
    # it would be enough to use a single tag for both cut and full facets.
    # and invoke ds_ only once for that tag.

    bound_normal = mapped_normal(dlf_mesh)
    facet_normal = ufl.FacetNormal(dlf_mesh)
    func = create_vector_func(dlf_mesh)

    ufl_form_srf = ufl.inner(func, bound_normal) * ds_unf + ufl.inner(func, facet_normal) * ds
    return ufl_form_srf


def check_div_thm(
    dom_func: ImplicitFunc,
    n_cells_dir: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64] = np.float64,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Check the divergence theorem for a given unfitted implicit domain.

    Args:
        dom_func (ImplicitFunc): The implicit function defining the unfitted domain.
        n_cells_dir (int): Number of cells in each direction.
        n_quad_pts (int): Number of quadrature points for cut cells and facets.
        dtype (type[np.float32 | np.float64], optional): Data type for computations.
            Defaults to np.float64.
        rtol (float, optional): Relative tolerance for comparison. Defaults to 1e-5.
        atol (float, optional): Absolute tolerance for comparison. Defaults to 1e-8.

    Raises:
        AssertionError: If the volume integral is not close to the surface integral
            within the specified tolerances.
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
    domain = qugar.impl.create_unfitted_impl_domain(dom_func, cart_mesh)

    ufl_form_vol = create_div_thm_volume_ufl_form(domain, n_quad_pts)
    ufl_form_srf = create_div_thm_surface_ufl_form(domain, n_quad_pts)

    form_vol = form_custom(ufl_form_vol, dtype=dtype)
    assert isinstance(form_vol, CustomForm)

    form_srf = form_custom(ufl_form_srf, dtype=dtype)
    assert isinstance(form_srf, CustomForm)

    quad_gen = create_quadrature_generator(domain)

    vol_integral = fem.assemble_scalar(form_vol, coeffs=form_vol.pack_coefficients(quad_gen))
    srf_integral = fem.assemble_scalar(form_srf, coeffs=form_srf.pack_coefficients(quad_gen))

    assert np.isclose(vol_integral, srf_integral, atol=atol, rtol=rtol)


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
    Test the divergence theorem for a 2D disk unfitted domain.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """
    radius = 0.8
    center = np.array([0.51, 0.45], dtype=dtype)

    func = qugar.impl.create_disk(radius=radius, center=center, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    check_div_thm(func, n_cells, n_quad_pts, dtype)


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
    Test the divergence theorem for a 3D sphere unfitted domain.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """
    radius = 0.8
    center = np.array([0.5, 0.45, 0.35], dtype=dtype)

    func = qugar.impl.create_sphere(radius=radius, center=center, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    check_div_thm(func, n_cells, n_quad_pts, dtype)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [6])
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
    Test the divergence theorem for a 2D domain whose boundary is a straigth line.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    a = 0.2
    origin = np.array([0.5, 0.5], dtype=dtype)
    normal = np.array([1.0, a], dtype=dtype)

    func = qugar.impl.create_line(origin, normal, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    check_div_thm(func, n_cells, n_quad_pts, dtype)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [6])
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
    Test the divergence theorem for a 3D domain whose boundary is a plane.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    origin = np.array([0.3, 0.47, 0.27], dtype=dtype)
    normal = np.array([1.0, 0.3, -1.0], dtype=dtype)

    func = qugar.impl.create_plane(origin, normal, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    check_div_thm(func, n_cells, n_quad_pts, dtype)


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
    Test the divergence theorem for a 3D cylinder.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    origin = np.array([0.55, 0.45, 0.47], dtype=dtype)
    axis = np.array([1.0, 0.9, -0.95], dtype=dtype)
    radius = 0.4

    func = qugar.impl.create_cylinder(radius=radius, origin=origin, axis=axis, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    check_div_thm(func, n_cells, n_quad_pts, dtype)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [6])
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
    Test the divergence theorem for a 2D annulus.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    center = np.array([0.55, 0.47], dtype=dtype)
    outer_radius = 0.75
    inner_radius = 0.2

    func = qugar.impl.create_annulus(inner_radius, outer_radius, center, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    check_div_thm(func, n_cells, n_quad_pts, dtype)


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
    Test the divergence theorem for a 2D ellipse.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    center = np.array([0.5, 0.5], dtype=dtype)
    x_axis = np.array([1.0, 1.0], dtype=dtype)
    ref_system = qugar.cpp.create_ref_system(center, x_axis)
    semi_axes = np.array([0.7, 0.5], dtype=dtype)

    func = qugar.impl.create_ellipse(semi_axes=semi_axes, ref_system=ref_system, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    check_div_thm(func, n_cells, n_quad_pts, dtype)


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
    Test the divergence theorem for a 3D ellipsoid.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    center = np.array([0.5, 0.5, 0.5], dtype=dtype)
    x_axis = np.array([1.0, 1.0, 1.0], dtype=dtype)
    y_axis = np.array([-1.0, 1.0, 1.0], dtype=dtype)
    ref_system = qugar.cpp.create_ref_system(center, x_axis, y_axis)
    semi_axes = np.array([0.7, 0.45, 0.52], dtype=dtype)

    func = qugar.impl.create_ellipsoid(semi_axes=semi_axes, ref_system=ref_system, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    check_div_thm(func, n_cells, n_quad_pts, dtype)


@pytest.mark.parametrize("n_cells", [12])
@pytest.mark.parametrize("n_quad_pts", [8])
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
    Test the divergence theorem for a 3D torus.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    center = np.array([0.55, 0.47, 0.51], dtype=dtype)
    axis = np.array([1.0, 0.9, 0.8], dtype=dtype)
    major_radius = 0.77
    minor_radius = 0.35

    func = qugar.impl.create_torus(major_radius, minor_radius, center, axis, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    rtol = 1e-4
    atol = 1e-8
    check_div_thm(func, n_cells, n_quad_pts, dtype, rtol, atol)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n_cells", [11, 12])
@pytest.mark.parametrize("n_quad_pts", [8])
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
    Test the divergence theorem for a series of TPMS in 2D and 3.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        dim (int): Dimension. Either 2D or 3D.
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
    """

    periods = np.ones(dim, dtype=dtype)
    for functor in [
        qugar.impl.create_Schoen,
        qugar.impl.create_Schoen_IWP,
        qugar.impl.create_Schoen_FRD,
        qugar.impl.create_Fischer_Koch_S,
        qugar.impl.create_Schwarz_Diamond,
        qugar.impl.create_Schwarz_Primitive,
    ]:
        func = functor(periods)
        if negative:
            func = qugar.impl.create_negative(func)

        rtol = 1e-5 if dim == 2 else 1e-3
        atol = 1e-8 if dim == 2 else 5.0e-3
        check_div_thm(func, n_cells, n_quad_pts, dtype, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_disk(8, 6, np.float64, True, False)
    # test_cylinder(8, 6, np.float64, True, False)
    # test_tpms(2, 12, 8, np.float32, True)
