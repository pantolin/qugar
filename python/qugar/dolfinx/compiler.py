# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# This file is a modification of the original ``ffcx/compiler.py`` file.
# See copyright below.
#
# Copyright (C) 2007-2020 Anders Logg and Michal Habera
#
# This file is part of FFCx.(https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# --------------------------------------------------------------------------

"""Main interface for compilation of forms.

Breaks the compilation into several sequential stages.
The output of each stage is the input of the next stage.

Note:
   This file is a modification of the original ``ffcx.compiler`` file.
   It modifies the code generation stage calling a custom code
   generator, and return one extra argument (the generated integrals
   data, needed for generating custom coefficients to be used by
   integrals at runtime).

Compiler stages
---------------

0. Language, parsing

   - Input:  Python code or .ufl file
   - Output: UFL form

   This stage consists of parsing and expressing a form in the UFL form
   language. This stage is handled by UFL.

1. Analysis

   - Input:  UFL form
   - Output: Preprocessed UFL form and FormData (metadata)

   This stage preprocesses the UFL form and extracts form metadata. It
   may also perform simplifications on the form.

2. Code representation

   - Input:  Preprocessed UFL form and FormData (metadata)
   - Output: Intermediate Representation (IR)

   This stage examines the input and generates all data needed for code
   generation. This includes generation of finite element basis
   functions, extraction of data for mapping of degrees of freedom and
   possible precomputation of integrals. Most of the complexity of
   compilation is handled in this stage.

   The IR is stored as a dictionary, mapping names of UFC functions to
   data needed for generation of the corresponding code.

3. Code generation

   - Input:  Intermediate Representation (IR)
   - Output: C code, list of integrals data

   This stage examines the IR and generates the actual C code for the
   body of each UFC function.

   The code is stored as a dictionary, mapping names of UFC functions to
   strings containing the C code of the body of each function.

   This step has been modified for calling the custom code generator.

4. Code formatting

   - Input:  C code
   - Output: C code files

   This stage examines the generated C++ code and formats it according
   to the UFC format, generating as output one or more .h/.c files
   conforming to the UFC format.

"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import logging
import typing
from time import time

import numpy.typing as npt
from ffcx.analysis import analyze_ufl_objects
from ffcx.formatting import format_code
from ffcx.ir.representation import compute_ir

# from ffcx.codegeneration.codegeneration import generate_code
import qugar.dolfinx.codegeneration
from qugar.dolfinx.integral_data import IntegralData

logger = logging.getLogger("ffcx")


def _print_timing(stage: int, timing: float):
    logger.info(f"Compiler stage {stage} finished in {timing:.4f} seconds.")


def compile_ufl_objects(
    ufl_objects: list[typing.Any],
    options: dict[str, int | float | npt.DTypeLike],
    object_names: dict[int, str] | None = None,
    prefix: str | None = None,
    visualise: bool = False,
) -> tuple[str, str, list[IntegralData]]:
    """Generate UFC code for a given UFL objects.


    Note:
        This function is just a copy of
        ``ffcx.compiler.compile_ufl_objects`` with a few modifications.
        Namely:

        - replacing ``ffcx.codegeneration.codegeneration.generate_code``
          with ``qugar.dolfinx.codegeneration.generate_code``.
        - enriching the return with on extra argument
          (``list[IntegralData]``).

    Args:
        ufl_objects: Objects to be compiled. Accepts elements, forms,
          integrals or coordinate mappings.
        object_names: Map from object Python id to object name
        prefix: Prefix
        options: Options
        visualise: Toggle visualisation
    """
    _object_names = object_names if object_names is not None else {}
    _prefix = prefix if prefix is not None else ""

    # Stage 1: analysis
    cpu_time = time()
    analysis = analyze_ufl_objects(ufl_objects, options["scalar_type"])  # type: ignore
    _print_timing(1, time() - cpu_time)

    # Stage 2: intermediate representation
    cpu_time = time()
    ir = compute_ir(analysis, _object_names, _prefix, options, visualise)
    _print_timing(2, time() - cpu_time)

    # Stage 3: code generation
    cpu_time = time()
    # code = ffcx.codegeneration.codegeneration.generate_code(ir, options) # noqa
    code, itg_data = qugar.dolfinx.codegeneration.generate_code(analysis, ir, options)
    _print_timing(3, time() - cpu_time)

    # Stage 4: format code
    cpu_time = time()
    code_h, code_c = format_code(code)
    _print_timing(4, time() - cpu_time)

    # return code_h, code_c
    return code_h, code_c, itg_data
