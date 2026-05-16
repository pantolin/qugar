# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""End-to-end test for a real DG interior-penalty Poisson formulation
through ``qugar.dolfinx.form_custom``.

The matrix-assembly suite (``test_matrix.py``) covers ``dx``, ``ds``
and ``dS`` integrals separately but never in the same form. A
discontinuous-Galerkin Poisson formulation combines all three
integral types in one bilinear form -- volume Laplacian (``dx``),
interior-facet consistency / symmetry / penalty (``dS``), exterior
facet weakly-imposed BC penalty (``ds``) -- which is the most-likely
real form to stress qugar's custom-quadrature pipeline.

The reference comparison is qugar custom assembly versus standard
DOLFINx assembly of the same UFL form on the same mock unfitted
mesh; on the mock mesh the two assemblies must agree to numerical
precision.
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import dolfinx
import dolfinx.fem
import pytest
import ufl
from utils import create_mock_unfitted_mesh, dtypes, run_matrix_check  # type: ignore

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3


def _dg_poisson_form(unf, p, dtype):
    """Bilinear form of the symmetric interior penalty DG Poisson
    operator with weakly-imposed homogeneous Dirichlet BCs."""
    V = dolfinx.fem.functionspace(unf, ("DG", p))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(unf)
    h = ufl.CellDiameter(unf)
    h_avg = (h("+") + h("-")) / 2.0

    # Standard SIPG penalty scaling.
    alpha = dolfinx.fem.Constant(unf, dtype(10.0))

    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)
        - ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * ufl.dS(domain=unf)
        - ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS(domain=unf)
        + alpha / h_avg * ufl.inner(ufl.jump(u, n), ufl.jump(v, n)) * ufl.dS(domain=unf)
        - ufl.inner(ufl.grad(u), v * n) * ufl.ds(domain=unf)
        - ufl.inner(u * n, ufl.grad(v)) * ufl.ds(domain=unf)
        + alpha / h * u * v * ufl.ds(domain=unf)
    )
    return a


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("p", [1, 2])
def test_dg_poisson_matrix(dim, p, simplex_cell, dtype):
    """Custom-form matrix assembly of the SIPG DG-Poisson bilinear
    form must agree with standard DOLFINx assembly to numerical
    precision (since the mock unfitted mesh's custom quadrature is
    equivalent to the standard one)."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    a = _dg_poisson_form(unf, p, dtype)
    run_matrix_check(a, unf)
