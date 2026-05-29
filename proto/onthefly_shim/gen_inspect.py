"""Generate the modified custom kernel C for a cell form and syntax-check it.

Validates that the live codegeneration.py on-the-fly path produces compilable C
(structure + syntax), prior to full end-to-end assembly validation.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import basix.ufl
import ffcx.codegeneration
import ffcx.options
import numpy as np
import ufl

from qugar.dolfinx.compiler import compile_ufl_objects

PREFIX = Path(sys.prefix)


def main() -> int:
    el = basix.ufl.element("Lagrange", "triangle", 2)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(domain, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    form = (ufl.inner(u, v) + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx

    options = ffcx.options.get_options()
    options["scalar_type"] = np.float64
    _h, code_c, _itg = compile_ufl_objects([form], options)

    # Show the on-the-fly bits.
    print("===== load_points + tabulate/repack snippets =====")
    for line in code_c.splitlines():
        if any(k in line for k in ("load_points", "qugar_register", "qugar_tabulate",
                                   "_buf[", "block_", "extern int qugar")):
            print(line)

    # Syntax check.
    cxx_inc = ffcx.codegeneration.get_include_path()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "kernel.c"
        f.write_text(code_c)
        cc = next(PREFIX.joinpath("bin").glob("clang*"), None)
        cmd = [str(PREFIX / "bin" / "clang"), "-std=c17", "-fsyntax-only",
               f"-I{cxx_inc}", str(f)]
        print("\n===== syntax check =====\n", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout)
        print(r.stderr)
        print("SYNTAX", "OK" if r.returncode == 0 else "FAIL")
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
