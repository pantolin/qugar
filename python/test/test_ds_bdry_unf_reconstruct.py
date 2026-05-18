# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Regression test for issue #3: ``ds_bdry_unf`` losing its subclass
identity (and therefore the Nanson correction in ``__rmul__``) when
restricted via ``ds(subdomain_id)`` or summed via ``ds(a) + ds(b)``.

The bug is purely in UFL's subclassing contract: ``ufl.Measure.__call__``
delegates to ``ufl.Measure.reconstruct`` which hard-codes
``Measure(...)`` rather than ``type(self)(...)``. The fix is a thin
``reconstruct`` override in :py:class:`ds_bdry_unf`.
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import dolfinx
import dolfinx.mesh
import numpy as np
import ufl

from qugar.dolfinx import ds_bdry_unf


def _mesh():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)


def test_call_preserves_subclass():
    """``ds(id)`` must return a ``ds_bdry_unf``, not a plain Measure."""
    mesh = _mesh()
    ds = ds_bdry_unf(domain=mesh)

    restricted = ds(2)

    assert isinstance(restricted, ds_bdry_unf), (
        f"Expected ds_bdry_unf, got {type(restricted).__name__}"
    )
    assert restricted.subdomain_id() == 2


def test_call_preserves_nanson_correction():
    """Restricting carries over ``_measure_complement`` (Nanson term)."""
    mesh = _mesh()
    ds = ds_bdry_unf(domain=mesh)

    restricted = ds(2)

    # Same domain → identical Nanson term (same UFL object).
    assert restricted._measure_complement is ds._measure_complement
    assert restricted._integral_type_mod == ds._integral_type_mod
    assert restricted._measure_name == ds._measure_name


def _integrand_contains(integrand, target):
    """Return True if ``target`` appears as a subexpression of
    ``integrand``. Used to confirm the Nanson term is injected."""
    if integrand is target:
        return True
    return any(_integrand_contains(op, target) for op in integrand.ufl_operands)


def test_rmul_applies_nanson_after_restriction():
    """``f * ds(id)`` must still inject ``_measure_complement`` and
    the Constant identity must be preserved (cache stability)."""
    mesh = _mesh()
    ds = ds_bdry_unf(domain=mesh)

    f = dolfinx.fem.Constant(mesh, 1.0)
    form = f * ds(3)
    integrals = form.integrals()
    assert len(integrals) == 1
    integral = integrals[0]
    assert integral.subdomain_id() == 3
    assert _integrand_contains(integral.integrand(), ds._measure_complement), (
        "Nanson `_measure_complement` term missing from integrand after ds(3)"
    )

    # Re-calling ds(3) must produce a form with the same signature
    # (FFCx JIT cache stability).
    form_again = f * ds(3)
    assert form.signature() == form_again.signature()


def test_tuple_subdomain_id_preserves_subclass():
    """Tuple subdomain ids are spread inside ``Measure.__rmul__`` via
    ``self.reconstruct(subdomain_id=d)`` — every per-id integrand must
    contain the Nanson term."""
    mesh = _mesh()
    ds = ds_bdry_unf(domain=mesh)

    f = dolfinx.fem.Constant(mesh, 1.0)
    form = f * ds((1, 2))
    integrals = form.integrals()
    assert len(integrals) == 2
    assert {integ.subdomain_id() for integ in integrals} == {1, 2}
    for integ in integrals:
        assert _integrand_contains(integ.integrand(), ds._measure_complement), (
            f"Nanson term missing from tuple-id integrand "
            f"(subdomain_id={integ.subdomain_id()})"
        )


def test_measure_sum_preserves_nanson():
    """``ds(1) + ds(2)`` builds a MeasureSum whose elements must remain
    ``ds_bdry_unf`` so that each integral gets the Nanson term."""
    mesh = _mesh()
    ds = ds_bdry_unf(domain=mesh)

    measure_sum = ds(1) + ds(2)
    assert isinstance(measure_sum, ufl.measure.MeasureSum)
    for m in measure_sum._measures:
        assert isinstance(m, ds_bdry_unf), (
            f"MeasureSum element should be ds_bdry_unf, got {type(m).__name__}"
        )

    f = dolfinx.fem.Constant(mesh, 1.0)
    form = f * measure_sum
    integrals = form.integrals()
    assert len(integrals) == 2
    for integ in integrals:
        assert _integrand_contains(integ.integrand(), ds._measure_complement), (
            f"Nanson term missing from MeasureSum integrand "
            f"(subdomain_id={integ.subdomain_id()})"
        )


def test_subdomain_data_is_carried_through():
    """``ds(id)`` with ``subdomain_data`` set on the original measure
    keeps the data — this is the natural call pattern from issue #3."""
    mesh = _mesh()
    cell_indices = np.array([0, 1], dtype=np.int32)
    cell_values = np.array([7, 7], dtype=np.int32)
    tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cell_indices, cell_values)

    ds = ds_bdry_unf(domain=mesh, subdomain_data=tags)
    restricted = ds(7)

    assert isinstance(restricted, ds_bdry_unf)
    assert restricted.subdomain_data() is tags
