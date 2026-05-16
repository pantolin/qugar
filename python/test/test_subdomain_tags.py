# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for tag-based subdomain selection in qugar custom forms.

Most of the assembly suite uses either no subdomain selection
(``dx(domain=mesh)``) or the special ``(0, 1)`` tuple from
``test_div_thm``. The qugar ``form_custom`` path that walks
``form.integrals()`` and assembles a per-``IntegralType`` subdomain-id
list (``forms.py``, around the comment "Build the per-integral-type
subdomain id list") is otherwise only stressed in demos.

This file builds forms with explicit ``MeshTags`` and verifies that:

* selecting a single tag id yields the same value as a hand-built
  integral over the marked cells;
* selecting multiple ids equals the sum of the per-id integrals;
* combining cell and facet subdomain selections in the same form
  works.
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from utils import check_vals, create_mock_unfitted_mesh, dtypes  # type: ignore

from qugar.dolfinx import form_custom

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3


def _build_cell_tags(unf, partition_fn, dtype):
    """Return a ``MeshTags`` that tags every cell of ``unf`` with either
    ``1`` or ``2`` according to ``partition_fn(midpoint) > 0.5``.

    Used so the resulting tags split the mesh into two distinct
    families of cells that each test can integrate over.
    """
    tdim = unf.topology.dim
    n_cells = unf.topology.index_map(tdim).size_local
    midpoints = dolfinx.mesh.compute_midpoints(unf, tdim, np.arange(n_cells, dtype=np.int32))

    tags = np.where(partition_fn(midpoints.T), np.int32(2), np.int32(1))
    return dolfinx.mesh.meshtags(
        unf, tdim, np.arange(n_cells, dtype=np.int32), tags
    )


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_single_tag_integral(dim, simplex_cell, dtype):
    """``dx(subdomain_id=1, subdomain_data=tags)`` integrates a
    constant over exactly the tag-1 cells; the result must equal the
    sum of cell volumes in that tag."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    cell_tags = _build_cell_tags(unf, lambda x: x[0] > 0.5, dtype)

    dx = ufl.Measure("dx", domain=unf, subdomain_data=cell_tags)
    one = dolfinx.fem.Constant(unf, dtype(1.0))

    form_a = form_custom(one * dx(1), dtype=dtype)
    val_a = dolfinx.fem.assemble_scalar(form_a, coeffs=form_a.pack_coefficients())

    form_b = form_custom(one * dx(2), dtype=dtype)
    val_b = dolfinx.fem.assemble_scalar(form_b, coeffs=form_b.pack_coefficients())

    # Tags 1 and 2 partition the mesh; their sum is the total domain
    # measure (volume in 3D, area in 2D).
    form_total = form_custom(one * ufl.dx(domain=unf), dtype=dtype)
    total = dolfinx.fem.assemble_scalar(form_total, coeffs=form_total.pack_coefficients())

    check_vals(np.array([val_a + val_b]), np.array([total]), dtype=dtype)
    # And each individually must be strictly positive (the partition
    # cannot be empty for these midpoint thresholds).
    assert val_a > 0.0 and val_b > 0.0


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_multi_id_tuple_equals_sum(dim, simplex_cell, dtype):
    """``dx((1, 2))`` (tuple of tag ids in one measure call) must
    equal ``dx(1) + dx(2)``. This is the most concise way to select
    multiple subdomains in a form."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    cell_tags = _build_cell_tags(unf, lambda x: x[1] > 0.5, dtype)

    dx = ufl.Measure("dx", domain=unf, subdomain_data=cell_tags)
    one = dolfinx.fem.Constant(unf, dtype(1.0))

    form_tuple = form_custom(one * dx((1, 2)), dtype=dtype)
    val_tuple = dolfinx.fem.assemble_scalar(
        form_tuple, coeffs=form_tuple.pack_coefficients()
    )

    form_sum = form_custom(one * dx(1) + one * dx(2), dtype=dtype)
    val_sum = dolfinx.fem.assemble_scalar(form_sum, coeffs=form_sum.pack_coefficients())

    check_vals(np.array([val_tuple]), np.array([val_sum]), dtype=dtype)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", [2, 3])
def test_cell_and_facet_tags_in_same_form(dim, dtype):
    """Forms mixing ``dx(subdomain_id=...)`` and ``ds(subdomain_id=...)``
    in the same expression. Catches regressions where qugar's
    subdomain-id walk drops the facet measure (or vice versa)."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell=False, nnz=_NNZ, max_quad_sets=_MAX_QUAD, dtype=dtype)
    cell_tags = _build_cell_tags(unf, lambda x: x[0] > 0.5, dtype)

    # Tag facets: left-half facets get 1, right-half get 2.
    fdim = unf.topology.dim - 1
    unf.topology.create_connectivity(fdim, unf.topology.dim)
    n_facets = unf.topology.index_map(fdim).size_local
    facet_mid = dolfinx.mesh.compute_midpoints(unf, fdim, np.arange(n_facets, dtype=np.int32))
    facet_tag_vals = np.where(facet_mid[:, 0] > 0.5, np.int32(2), np.int32(1))
    facet_tags = dolfinx.mesh.meshtags(
        unf, fdim, np.arange(n_facets, dtype=np.int32), facet_tag_vals
    )

    dx = ufl.Measure("dx", domain=unf, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=unf, subdomain_data=facet_tags)
    one = dolfinx.fem.Constant(unf, dtype(1.0))

    # Cell vol over tag=1 + boundary area over tag=2.
    form_combined = form_custom(one * dx(1) + one * ds(2), dtype=dtype)
    val = dolfinx.fem.assemble_scalar(
        form_combined, coeffs=form_combined.pack_coefficients()
    )

    # Same quantity assembled as two separate forms.
    form_dx = form_custom(one * dx(1), dtype=dtype)
    form_ds = form_custom(one * ds(2), dtype=dtype)
    val_ref = dolfinx.fem.assemble_scalar(
        form_dx, coeffs=form_dx.pack_coefficients()
    ) + dolfinx.fem.assemble_scalar(form_ds, coeffs=form_ds.pack_coefficients())

    check_vals(np.array([val]), np.array([val_ref]), dtype=dtype)
