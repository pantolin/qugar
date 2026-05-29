"""M1 isolated validation: basix-block -> FFCx-table repack mapping.

For a real FFCx-generated cell-integral kernel, every point-varying FE table is
re-derived from an on-the-fly basix tabulation (via the shim) and compared to the
values FFCx baked in. This proves the (derivative, component) index mapping
between basix's full block and FFCx's separated tables -- the #1 risk of M1 --
before any change to the live code generator.

Mapping under test:
    FFCx_table[perm=0][entity=0][pt][dof]
        == basix_block[ basix.index(*derivatives) ][pt][dof][value_axis]
where value_axis = 0 for blocked elements (scalar core, value_size 1) and the
FFCx component for native-vector elements (e.g. RT).

Run with the qugar-0.10.0 env python.
"""

from __future__ import annotations

import sys
from pathlib import Path

import basix
import basix.ufl
import ffcx.options
import numpy as np
import ufl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shim  # noqa: E402

from qugar.dolfinx.compiler import compile_ufl_objects  # noqa: E402


def make_form(element, cell="triangle", with_grad=True):
    gdim = {"triangle": 2, "quadrilateral": 2, "tetrahedron": 3,
            "hexahedron": 3}[cell]
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell, 1, shape=(gdim,)))
    V = ufl.FunctionSpace(domain, element)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    form = ufl.inner(u, v) * ufl.dx
    if with_grad:
        form = form + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    return form


def validate_form(lib, label, element, cell="triangle", with_grad=True):
    print(f"\n=== {label} ({cell}) ===")
    form = make_form(element, cell, with_grad)
    options = ffcx.options.get_options()
    options["scalar_type"] = np.float64
    _h, _c, itg_data_list = compile_ufl_objects([form], options)

    npass = nfail = nskip = 0
    for itg in itg_data_list:
        for quad_data, tables in itg.quad_data_FE_tables.items():
            pts = np.asarray(quad_data.rule.points)
            for t in tables:
                if t.is_constant_for_pts():
                    nskip += 1
                    continue
                dtype = t.dtype
                gdim = pts.shape[1]
                derivs = list(t.derivatives) if t.derivatives else []
                derivs = (derivs + [0] * gdim)[:gdim]
                didx = basix.index(*derivs)

                block = shim.tabulate(lib, t.element, dtype, sum(derivs), pts)
                _, _block_size = block.shape[-1], None
                bsz = getattr(t.element, "block_size", 1)
                vaxis = 0 if bsz > 1 else t.component
                got = block[didx, :, :, vaxis]              # (npts, ndofs)
                ref = np.asarray(t.values)[0, 0, :, :]        # (npts, funcs)

                ok = ref.shape == got.shape and np.allclose(
                    ref, got, atol=1e-12 if dtype == np.float64 else 1e-5)
                maxerr = (float(np.max(np.abs(ref - got)))
                          if ref.shape == got.shape else float("nan"))
                npass += ok
                nfail += (not ok)
                print(f"  [{'OK ' if ok else 'FAIL'}] {t.name:24s} "
                      f"deriv={tuple(derivs)} didx={didx} comp={t.component} "
                      f"bsz={bsz} ref{ref.shape} got{got.shape} maxerr={maxerr:.2e}")
    print(f"  -> pass={npass} fail={nfail} skipped(constant)={nskip}")
    return nfail == 0


def main() -> int:
    lib = shim.load()
    P = lambda c, d: basix.ufl.element("Lagrange", c, d)  # noqa: E731
    Pvec = lambda c, d, g: basix.ufl.element("Lagrange", c, d, shape=(g,))  # noqa: E731
    RT = lambda c, d: basix.ufl.element("RT", c, d)  # noqa: E731

    all_ok = True
    all_ok &= validate_form(lib, "scalar P1", P("triangle", 1))
    all_ok &= validate_form(lib, "scalar P2", P("triangle", 2))
    all_ok &= validate_form(lib, "scalar P3", P("triangle", 3))
    all_ok &= validate_form(lib, "scalar P2 tet", P("tetrahedron", 2),
                            cell="tetrahedron")
    all_ok &= validate_form(lib, "vector P2 (blocked)", Pvec("triangle", 2, 2))
    all_ok &= validate_form(lib, "RT1", RT("triangle", 1))

    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
