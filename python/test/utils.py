# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tools for pytests."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import os
from pathlib import Path
from typing import Optional, Tuple

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
from dolfinx.cpp.mesh import CellType
from dolfinx.fem.forms import Form
from mock_unf_mesh import MockUnfittedMesh

from qugar.dolfinx import CustomForm, form_custom
from qugar.mesh.unfitted_domain_abc import UnfittedDomainABC

"""Path for storing DOLFINx JIT cache.
Note that fefault prefix ``~/.cache/`` can be changed using
``XDG_CACHE_HOME`` environment variable.
"""
DOLFINX_CACHE_DIR = os.getenv("XDG_CACHE_HOME", default=Path.home().joinpath(".cache")) / Path(
    "fenics"
)


"""`numpy` scalar types to be used in the tests.
"""
dtypes = (
    [np.float32, np.float64] if np.dtype(ScalarType).kind == "f" else [np.complex64, np.complex128]
)


def clean_cache(dir_name: str = str(DOLFINX_CACHE_DIR)):
    """Cleans the FEniCSx cache folder removing all the compiled
    files and their source codes.

    Args:
        dir_name (str, optional): Folder to be clean_. Defaults to str
            (DOLFINX_CACHE_DIR).
    """
    if not os.path.exists(dir_name):
        return
    test = os.listdir(dir_name)
    for item in test:
        if item.endswith(".c"):
            os.remove(os.path.join(dir_name, item))
        if item.endswith(".so"):
            os.remove(os.path.join(dir_name, item))
        if item.endswith(".o"):
            os.remove(os.path.join(dir_name, item))
        if item.endswith("c.cached"):
            os.remove(os.path.join(dir_name, item))
        if item.endswith("c.failed"):
            os.remove(os.path.join(dir_name, item))


def _create_mesh(
    dim: int,
    N: int,
    simplex_cell: bool = False,
    dtype: type[np.float32 | np.float64] = np.float64,
) -> dolfinx.mesh.Mesh:
    """Create a finite element mesh in the domain [0,1]^dim.

    Args:
        dim (int): Dimension of the mesh.
        N (int): Number of cells per direction.
        simplex_cell (bool, optional): If ``True``, creates simplices
            (triangles and tetrahedra), otherwise (if ``False``) creates
            quadrilaterals and hexahedra. Defaults to False.
        dtype (np.dtype[np.float32  |  np.float64], optional): `numpy`
            type to be used in the mesh.

    Returns:
        dolfinx.mesh.Mesh: Create mesh.
    """

    assert N > 0

    comm = MPI.COMM_WORLD

    if dim == 2:
        cell_type = CellType.triangle if simplex_cell else CellType.quadrilateral
        mesh = dolfinx.mesh.create_unit_square(comm, N, N, cell_type=cell_type, dtype=dtype)
    else:
        assert dim == 3
        cell_type = CellType.tetrahedron if simplex_cell else CellType.hexahedron
        mesh = dolfinx.mesh.create_unit_cube(comm, N, N, N, cell_type=cell_type, dtype=dtype)
    return mesh


def create_mock_unfitted_mesh(
    dim: int,
    N: int,
    simplex_cell: bool = False,
    nnz: float = 0,
    max_quad_sets: int = 3,
    dtype: type[np.float32 | np.float64] = np.float64,
) -> MockUnfittedMesh:
    """Create a finite element mesh in the domain [0,1]^dim.

    Args:
        dim (int): Dimension of the mesh.
        N (int): Number of cells per direction.
        simplex_cell (bool, optional): If ``True``, creates simplices
            (triangles and tetrahedra), otherwise (if ``False``) creates
            quadrilaterals and hexahedra. Defaults to False.
        nnz (int, optional): Ratio of entities with custom
            quadratures respect to the total number of entities.
            It must be a value in the range [0.0, 1.0].
        max_quad_sets (int, optional): Maximum number of repetitions
            of the standard quadrature in each custom cell. For each
            custom entity a random number is generated between 1 and
            `max_quad_sets`. Defaults to 3.
        dtype (np.dtype[np.float32  |  np.float64], optional): `numpy`
            type to be used in the mesh.

    Returns:
        dolfinx.mesh.Mesh: Create mesh.
    """
    return MockUnfittedMesh(_create_mesh(dim, N, simplex_cell, dtype), nnz, max_quad_sets)


def get_dtype(ufl_form) -> type[np.float32 | np.float64]:
    """Extracts the `numpy` associated to the given form.

    Args:
        ufl_form: Form from which the `numpy` scalar type is extracted.

    Returns:
        type[np.float32 | np.float64]: Extracted `numpy` scalar type. It can
        be either ``np.float32
    """
    for integral in ufl_form.integrals():
        return integral._ufl_domain.ufl_cargo().geometry.x.dtype
    assert False, "No integral found."


def get_dolfinx_forms(ufl_form, unf_domain: UnfittedDomainABC) -> tuple[CustomForm, Form]:
    """Creates DOLFINx forms from the given UFL form.
    It creates a standard DOLFINx `Form`and a `CustomForm` produced
    by QUGaR.

    Args:
        ufl_form: UFL form to be compiled.
        unf_domain (UnfittedDomainABC): Unfitted domain to be used in the
            custom form.

    Returns:
        tuple[CustomForm, Form]: Generated custom form (first), and
        standard form (second).
    """

    dtype = get_dtype(ufl_form)

    custom_form = form_custom(ufl_form, dtype=dtype)
    assert isinstance(custom_form, CustomForm)

    form: Form = dolfinx.fem.form(ufl_form, dtype=dtype)  # type: ignore

    return custom_form, form


def set_tolerances(
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    dtype: type[np.float32 | np.float64] = np.float64,
) -> Tuple[float, float]:
    """Sets the tolerances to be used for compare results coming from
    the assembly of different forms.

    Args:
        rtol (Optional[float], optional): Relative tolerance. Defaults
            to None.
        atol (Optional[float], optional): Absolute tolerance. Defaults
            to None.
        dtype (type[np.float32 | np.float64], optional): `numpy` scalar type
            to set the absolute tolerance. Default value is `np.float64`.

    Returns:
        Tuple[float, float]: Relative (first) and absolute (second)
        tolerances.
    """

    if rtol is None:
        rtol = 1.0e-5  # numpy' default value for isclose

    if atol is None:
        # 1.0e-8 seems like a good fit for double precision numbers.
        # atol = 1.0e-8  # numpy' default value for isclose
        eps = np.finfo(dtype).eps  # type: ignore
        atol = np.sqrt(eps)

    assert rtol >= 0.0 and atol >= 0.0  # type: ignore

    return rtol, atol  # type: ignore


def check_vals(
    lhs: npt.NDArray,
    rhs: npt.NDArray,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    dtype: type[np.float32 | np.float64] = np.float64,
):
    """Checks that two arrays are equal up to a tolerance

    If the two array are not close up to the given tolerance, an assert
    is raised.

    Args:
        lhs (npt.NDArray): First array to compare.
        rhs (npt.NDArray): Second array to compare.
        rtol (Optional[float], optional): Relative tolerance to be used
            in the comparison. Defaults to None.
        atol (Optional[float], optional): Absolute tolerance to be used
            in the comparison. Defaults to None.
        dtype (type[np.float32 | np.float64], optional): `numpy` scalar type
            to set the absolute tolerance. Default value is `np.float64`.
    """

    rtol, atol = set_tolerances(rtol, atol, dtype)

    assert np.allclose(lhs, rhs, rtol, atol) or np.allclose(rhs, lhs, rtol, atol), (
        "Non matching values."
    )


def run_scalar_check(
    ufl_form,
    unf_domain: UnfittedDomainABC,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
):
    """Assembles the scalar associated to the given UFL form using the
    standard DOLFINx `Form` and the custom `CustomForm`, and checks that both results
    are the same up to the given tolerance.

    If the quantities computed using both form are not equal (up to the
    given tolerance), and assert is raised.

    Args:
        ufl_form: UFL form to assemble.
        unf_domain (UnfittedDomainABC): Unfitted domain to be used in the
            custom form.
        rtol (Optional[float], optional): Relative tolerance to be used
            in the comparison. Defaults to None.
        atol (Optional[float], optional): Absolute tolerance to be used
            in the comparison. Defaults to None.
    """

    custom_form, form = get_dolfinx_forms(ufl_form, unf_domain)

    custom_coeffs = custom_form.pack_coefficients()
    custom_value = dolfinx.fem.assemble_scalar(custom_form, coeffs=custom_coeffs)

    value = dolfinx.fem.assemble_scalar(form)

    dtype = get_dtype(ufl_form)

    check_vals(custom_value, value, rtol, atol, dtype)


def run_vector_check(ufl_form, unf_domain: UnfittedDomainABC, rtol=None, atol=None):
    """Assembles the vector associated to the given UFL form using the
    standard DOLFINx `Form` and the custom `CustomForm`,
    and checks that both results
    are the same up to the given tolerance.

    If the quantities computed using both form are not equal (up to the
    given tolerance), and assert is raised.

    Args:
        ufl_form: UFL form to assemble.
        unf_domain (UnfittedDomainABC): Unfitted domain to be used in the
            custom form.
        rtol (Optional[float], optional): Relative tolerance to be used
            in the comparison. Defaults to None.
        atol (Optional[float], optional): Absolute tolerance to be used
            in the comparison. Defaults to None.
    """

    custom_form, form = get_dolfinx_forms(ufl_form, unf_domain)

    custom_coeffs = custom_form.pack_coefficients()
    custom_vector = dolfinx.fem.assemble_vector(custom_form, coeffs=custom_coeffs)

    vector = dolfinx.fem.assemble_vector(form)

    dtype = get_dtype(ufl_form)

    check_vals(custom_vector.array, vector.array, rtol, atol, dtype)


def run_matrix_check(ufl_form, unf_domain: UnfittedDomainABC, rtol=None, atol=None):
    """Assembles the matrix associated to the given UFL form using the
    standard DOLFINx `Form` and the custom `CustomForm`, and checks that
    both results are the same up to the given tolerance.

    If the quantities computed using both form are not equal (up to the
    given tolerance), and assert is raised.

    Args:
        ufl_form: UFL form to assemble.
        unf_domain (UnfittedDomainABC): Unfitted domain to be used in the
            custom form.
        rtol (Optional[float], optional): Relative tolerance to be used
            in the comparison. Defaults to None.
        atol (Optional[float], optional): Absolute tolerance to be used
            in the comparison. Defaults to None.
    """

    custom_form, form = get_dolfinx_forms(ufl_form, unf_domain)

    custom_coeffs = custom_form.pack_coefficients()
    custom_matrix = dolfinx.fem.assemble_matrix(custom_form, coeffs=custom_coeffs)

    matrix = dolfinx.fem.assemble_matrix(form)

    dtype = get_dtype(ufl_form)
    check_vals(custom_matrix.data, matrix.data, rtol, atol, dtype)


def get_Gauss_quad_degree(n_pts: int) -> int:
    """Computes the Gauss-Legendre quadrature degree of exactness for a given number of points.

    Args:
        n_pts (int): Number of quadrature points per direction.

    Returns:
        int: Quadrature's degree of exactness.
    """
    return 2 * n_pts - 1
