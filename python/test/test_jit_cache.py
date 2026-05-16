# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for FFCx JIT cache reuse in qugar custom forms.

``qugar.dolfinx.jit.ffcx_jit`` wraps the upstream ``dolfinx.jit``
caching path. A refactor that breaks the cache key would silently
re-compile every form (slow, but still correct), so the symptom would
only show up as a runtime regression in production. These tests
verify both:

* **Consistency**: two custom forms built from the same UFL expression
  produce numerically identical assembled values.
* **Cache effectiveness**: the second build is dramatically faster
  than the first (cache hit -> no JIT compilation).
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import time

import dolfinx
import dolfinx.fem
import numpy as np
import pytest
import ufl
from utils import check_vals, clean_cache, create_mock_unfitted_mesh, dtypes  # type: ignore

from qugar.dolfinx import form_custom

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3


def _build_form(unf, dtype):
    """A non-trivial form that requires real ffcx compilation
    (coefficient-modulated grad-grad). Returns ``(ufl_form, function
    space)``."""
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    e = dolfinx.fem.Function(V, dtype=dtype)
    e.interpolate(lambda x: 1.0 + x[0] ** 2)  # type: ignore
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    return e * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf), V


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_jit_consistency(dim, simplex_cell, dtype):
    """Two ``form_custom`` builds of the same UFL expression must
    produce identical assembled values, regardless of whether the
    second hits the cache or re-compiles."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    ufl_form, _ = _build_form(unf, dtype)

    f1 = form_custom(ufl_form, dtype=dtype)
    f2 = form_custom(ufl_form, dtype=dtype)

    M1 = dolfinx.fem.assemble_matrix(f1, coeffs=f1.pack_coefficients())
    M2 = dolfinx.fem.assemble_matrix(f2, coeffs=f2.pack_coefficients())

    check_vals(M1.data, M2.data, dtype=dtype)


@pytest.mark.parametrize("dim", [2])
def test_jit_cache_speedup(dim):
    """The second ``form_custom`` build of the same UFL form must hit
    the JIT cache and be substantially faster than the first.

    We don't tie the test to an absolute time -- the FFCx JIT
    compilation step is at least an order of magnitude slower than the
    cache-hit branch (typically ~10 s vs ~0.1 s), so requiring a 5x
    speedup is loose but catches the regression where the cache key
    changes between calls and every build re-compiles.

    The test is restricted to 2D + float64 + non-simplex (one cheap
    parametrization) because measuring compile time is the point;
    repeating it does not add signal.
    """
    dtype = np.float64
    # Clean any cached compilations from earlier test runs so the
    # first build genuinely incurs the FFCx compile cost.
    clean_cache()

    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell=False, nnz=_NNZ, max_quad_sets=_MAX_QUAD, dtype=dtype)
    # Use a degree-4 element to guarantee a unique cache key that no
    # other test has populated (just in case clean_cache misses
    # something).
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 4))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ufl_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)

    # Cold build: incurs full FFCx compile cost.
    t0 = time.perf_counter()
    form_custom(ufl_form, dtype=dtype)
    cold = time.perf_counter() - t0

    # Warm build of the same UFL: should hit the cache.
    t0 = time.perf_counter()
    form_custom(ufl_form, dtype=dtype)
    warm = time.perf_counter() - t0

    # A 5x speedup is a very generous lower bound. Cache hits are
    # typically 50-200x faster than compiles. If the second build is
    # not at least 5x faster than the first, the cache is likely
    # missing.
    assert warm < cold / 5.0, (
        f"JIT cache appears to miss on rebuild: "
        f"cold={cold * 1e3:.1f}ms, warm={warm * 1e3:.1f}ms"
    )
