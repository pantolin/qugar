# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""End-to-end tests for the Dirichlet-BC + ``apply_lifting`` path of
``qugar.dolfinx.LinearProblem``.

The standard ``test_matrix`` / ``test_vector`` suites assemble
matrices and vectors without any boundary conditions. The actual
Newton / time-stepping workflows live or die on ``apply_lifting``
correctly subtracting the BC contribution from the right-hand side,
and on the matrix being correctly zeroed (rows / columns + diagonal)
on constrained DOFs.

This module solves a small Poisson problem with non-homogeneous
Dirichlet BCs on the mock unfitted mesh (where the custom quadrature
is constructed to be equivalent to the standard quadrature) and
checks that the qugar custom-form solve agrees with a plain DOLFINx
solve of the same problem.
"""

from qugar.utils import has_FEniCSx, has_PETSc

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")
if not has_PETSc:
    import pytest as _pytest

    _pytest.skip("petsc4py installation not found", allow_module_level=True)


from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from utils import check_vals, create_mock_unfitted_mesh, dtypes  # type: ignore

from qugar.dolfinx import LinearProblem

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3

_PETSC_DTYPES = [d for d in dtypes if np.dtype(d) == np.dtype(ScalarType)]


def _build_poisson_problem(unf, dtype):
    """Build the Poisson UFL forms ``(a, L)`` over the mock unfitted
    mesh, plus a Dirichlet BC fixing ``u = g`` on the whole boundary."""
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Manufactured source: u_ex = 1 + x^2 + 2y^2 -> -laplace(u_ex) = -6
    rhs = dolfinx.fem.Constant(unf, dtype(-6.0))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)
    L = -rhs * v * ufl.dx(domain=unf)

    g = dolfinx.fem.Function(V, dtype=dtype)
    g.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)  # type: ignore

    tdim = unf.topology.dim
    fdim = tdim - 1
    unf.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(unf.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(g, boundary_dofs)

    return V, a, L, bc


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_poisson_dirichlet(dim, simplex_cell, dtype):
    """Solve Poisson with non-homogeneous Dirichlet BCs through
    ``qugar.dolfinx.LinearProblem`` and compare against a plain
    DOLFINx solve of the same problem on the same mesh.

    On the mock unfitted mesh the custom quadrature is by construction
    equivalent to the standard quadrature, so the two solutions must
    agree up to numerical precision -- this covers the full
    ``assemble_matrix(bcs=...) + apply_lifting + set_bc`` pipeline
    inside ``LinearProblem.solve``.
    """
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V, a, L, bc = _build_poisson_problem(unf, dtype)

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options)
    problem.solve()
    uh_custom = problem.u

    # Reference: plain DOLFINx solve of the same problem (no qugar
    # custom path). DOLFINx 0.10.0 requires a mandatory
    # ``petsc_options_prefix`` kwarg.
    problem_std = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="qugar_test_ref_",
        petsc_options=petsc_options,
    )
    problem_std.solve()
    uh_std = problem_std.u

    check_vals(uh_custom.x.array, uh_std.x.array, dtype=dtype)


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_poisson_dirichlet_bc_honored(dim, simplex_cell, dtype):
    """The solution returned by ``LinearProblem`` must match the BC
    value on constrained DOFs to roundoff precision (catches
    regressions where ``set_bc`` is skipped or partially applied).
    """
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V, a, L, bc = _build_poisson_problem(unf, dtype)

    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    problem.solve()
    uh = problem.u

    # On the constrained DOFs the solution equals the BC values.
    bc_dofs = bc.dof_indices()[0]
    bc_vals = bc.g.x.array[bc_dofs]
    check_vals(uh.x.array[bc_dofs], bc_vals, dtype=dtype)
