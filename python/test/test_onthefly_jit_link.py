# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""End-to-end JIT-link smoke test for the on-the-fly path.

Drives ``qugar.dolfinx.jit.ffcx_jit`` on a P2-triangle cell form: the
modified kernel is generated, run through CFFI, and linked against the
basix tabulation shim. If this completes without error, codegen + JIT +
shim linking are coherent. Element-tensor correctness is covered by the
broader assembly tests.
"""

from __future__ import annotations

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import basix.ufl
import numpy as np
import ufl


def test_jit_link_smoke(tmp_path, monkeypatch):
    """Compiling a small P2-triangle form through ffcx_jit must succeed,
    proving the kernel can be linked against the shim."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("QUGAR_SHIM_CACHE", str(tmp_path / "qugar_shim"))

    # Import after env vars are set so dolfinx/qugar pick them up.
    from qugar.dolfinx.jit import ffcx_jit

    el = basix.ufl.element("Lagrange", "triangle", 2)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(domain, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    form = (ufl.inner(u, v) + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx

    itg_data, compiled, module, (header, impl) = ffcx_jit(
        MPI.COMM_WORLD,
        form,
        form_compiler_options={"scalar_type": np.float64},
    )
    assert len(itg_data) >= 1
    assert module is not None
    # Shim symbol must appear in the generated kernel source (i.e. the
    # custom kernel actually calls into the shim).
    assert "qugar_tabulate_f64" in impl
