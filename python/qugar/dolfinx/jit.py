# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# This file is a modification of the original ``dolfinx/python/dolfinx/jit.py`` file.
# See copyright below.
#
# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# --------------------------------------------------------------------------

"""JIT compiling funcionalities. This file is just an adaptation of
jit.py in DOLFINx."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import io
import logging
import os
import sys
import tempfile
import time
import typing
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import cffi
import ffcx
import ffcx.codegeneration
import ffcx.naming
import ffcx.options
import numpy as np
import ufl
import ufl.core.expr
from dolfinx.jit import get_options, mpi_jit_decorator
from ffcx.codegeneration.C.file_template import libraries as _libraries
from ffcx.codegeneration.jit import (
    UFC_FORM_DECL,
    UFC_HEADER_DECL,
    UFC_INTEGRAL_DECL,
    _compilation_signature,
    _compute_option_signature,
    _load_objects,
    get_cached_module,
    logger,
    root_logger,
)

from qugar.dolfinx.compiler import compile_ufl_objects
from qugar.dolfinx.integral_data import IntegralData


def _compile_objects(
    decl,
    ufl_objects,
    object_names,
    module_name,
    options,
    cache_dir,
    cffi_extra_compile_args,
    cffi_verbose,
    cffi_debug,
    cffi_libraries,
    visualise: bool = False,
):
    """
    Note:
        This function is just a copy of
        ``ffcx.codegeneration.jit._compile_objects`` with some small
        modifications. Namely:

        - replacing ``ffcx.compiler`` with ``qugar.dolfinx.compiler``.
        - return one extra value: a list of ``IntegralData`` to be used
          for generating custom coefficients at runtime.
    """

    libraries = _libraries + cffi_libraries if cffi_libraries is not None else _libraries

    # JIT uses module_name as prefix, which is needed to make names of
    # all struct/function
    # unique across modules
    _, code_body, itg_data = compile_ufl_objects(
        ufl_objects, prefix=module_name, options=options, visualise=visualise
    )

    # Raise error immediately prior to compilation if no support for C99
    # _Complex. Doing this here allows FFCx to be used for complex
    # codegen onWindows.
    if sys.platform.startswith("win32"):
        if np.issubdtype(options["scalar_type"], np.complexfloating):
            raise NotImplementedError("win32 platform does not support C99 _Complex numbers")
        elif isinstance(options["scalar_type"], str) and "complex" in options["scalar_type"]:
            raise NotImplementedError("win32 platform does not support C99 _Complex numbers")

    # Compile in C17 mode
    if sys.platform.startswith("win32"):
        cffi_base_compile_args = ["-std:c17"]
    else:
        cffi_base_compile_args = ["-std=c17"]

    cffi_final_compile_args = cffi_base_compile_args + cffi_extra_compile_args

    ffibuilder = cffi.FFI()

    ffibuilder.set_source(
        module_name,
        code_body,
        include_dirs=[ffcx.codegeneration.get_include_path()],
        extra_compile_args=cffi_final_compile_args,
        libraries=libraries,
    )

    ffibuilder.cdef(decl)

    c_filename = cache_dir.joinpath(module_name + ".c")
    ready_name = c_filename.with_suffix(".c.cached")

    # Compile (ensuring that compile dir exists)
    cache_dir.mkdir(exist_ok=True, parents=True)

    logger.info(79 * "#")
    logger.info("Calling JIT C compiler")
    logger.info(79 * "#")

    t0 = time.time()
    f = io.StringIO()
    # Temporarily set root logger handlers to string buffer only
    # since CFFI logs into root logger
    old_handlers = root_logger.handlers.copy()
    root_logger.handlers = [logging.StreamHandler(f)]
    with redirect_stdout(f):
        ffibuilder.compile(tmpdir=cache_dir, verbose=True, debug=cffi_debug)
    s = f.getvalue()
    if cffi_verbose:
        print(s)

    logger.info(f"JIT C compiler finished in {time.time() - t0:.4f}")

    # Create a "status ready" file. If this fails, it is an error,
    # because it should not exist yet.
    # Copy the stdout verbose output of the build into the ready file
    fd = open(ready_name, "x")
    fd.write(s)
    fd.close()

    # Copy back the original handlers (in case someone is logging into
    # root logger and has custom handlers)
    root_logger.handlers = old_handlers

    return code_body, itg_data


def compile_forms(
    forms: list[ufl.Form],
    options: dict = {},
    cache_dir: Path | None = None,
    timeout: int = 10,
    cffi_extra_compile_args: list[str] = [],
    cffi_verbose: bool = False,
    cffi_debug: bool = False,
    cffi_libraries: list[str] = [],
    visualise: bool = False,
):
    """Compile a list of UFL forms into UFC Python objects.

    Note:
        This function is just a copy of
        ``ffcx.codegeneration.jit.compile_forms`` with some small
        modifications. Namely:

        - adding ``custom`` option to ``options`` dictionary.
        - calling ``qugar.dolfinx.compiler.compile_ufl_objects`` even when
          the module is cached and does not need to be recompiled.
        - return one extra value: a list of ``IntegralData`` to be used
          for generating custom coefficients at runtime.

    Args:
        forms: List of ufl.form to compile.
        options: Options
        cache_dir: Cache directory
        timeout: Timeout
        cffi_extra_compile_args: Extra compilation args for CFFI
        cffi_verbose: Use verbose compile
        cffi_debug: Use compiler debug mode
        cffi_libraries: libraries to use with compiler
        visualise: Toggle visualisation
    """

    p = ffcx.options.get_options(options)

    # This options is introduced just for modifying the module name
    # respect to the non-custom version.
    p["custom"] = True
    module_name = "libffcx_forms_" + ffcx.naming.compute_signature(
        forms,  # type: ignore
        _compute_option_signature(p) + _compilation_signature(cffi_extra_compile_args, cffi_debug),
    )

    form_names = [ffcx.naming.form_name(form, i, module_name) for i, form in enumerate(forms)]

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        obj, mod = get_cached_module(module_name, form_names, cache_dir, timeout)
        if obj is not None:
            # Calling compile_ufl_objects, even if result is cached, to
            # obtain itg_data.
            decl, impl, itg_data = compile_ufl_objects(
                forms, prefix=module_name, options=p, visualise=visualise
            )

            return obj, mod, (decl, impl), itg_data
    else:
        cache_dir = Path(tempfile.mkdtemp())

    try:
        decl = (
            UFC_HEADER_DECL.format(np.dtype(p["scalar_type"]).name)  # type: ignore
            + UFC_INTEGRAL_DECL
            + UFC_FORM_DECL
        )

        form_template = "extern ufcx_form {name};\n"
        for name in form_names:
            decl += form_template.format(name=name)

        impl, itg_data = _compile_objects(
            decl,
            forms,
            form_names,
            module_name,
            p,
            cache_dir,
            cffi_extra_compile_args,
            cffi_verbose,
            cffi_debug,
            cffi_libraries,
            visualise=visualise,
        )

    except Exception as e:
        try:
            # remove c file so that it will not timeout next time
            c_filename = cache_dir.joinpath(module_name + ".c")
            os.replace(c_filename, c_filename.with_suffix(".c.failed"))
        except Exception:
            pass
        raise e

    obj, module = _load_objects(cache_dir, module_name, form_names)
    return obj, module, (decl, impl), itg_data


@mpi_jit_decorator
def ffcx_jit(
    ufl_object,
    form_compiler_options: Optional[dict] = None,
    jit_options: Optional[dict] = None,
) -> tuple[list[IntegralData], typing.Any, typing.Any, tuple[str, str]]:
    """Compile UFL object with FFCx and CFFI.

    Note:
        This function is just a copy of ``dolfinx.jit.ffcx_jit``
        with some small modifications. Namely:

        - It only compiles forms. If an UFL expression or element is
          passed as input, it will raise an exception.
        - return one extra value: a list of ``IntegralData`` to be used
          for generating custom coefficients at runtime.

    Args:
        ufl_object: Object to compile, e.g. ``ufl.Form``.
        form_compiler_options (Optional[dict]): Options used in FFCx
            compilation of this form. Run ``ffcx --help`` at the
            command line to see all available options. Takes priority
            over all other option values.
        jit_options (Optional[dict]): Options used in CFFI JIT
            compilation of C code generated by FFCx. See
            ``python/dolfinx/jit.py`` for all available options.
            Takes priority over all other option values.

    Returns:
        (integral data, compiled object, module, (header code, implementation code))

    Note:
        Priority ordering of options controlling DOLFINx JIT
        compilation from highest to lowest is:

        -  **jit_options** (API)
        -  **$PWD/dolfinx_jit_options.json** (local options)
        -  **$XDG_CONFIG_HOME/dolfinx/dolfinx_jit_options.json**
           (user options)
        -  **DOLFINX_DEFAULT_JIT_OPTIONS** in `dolfinx.jit`

        Priority ordering of options controlling FFCx from highest to
        lowest is:

        -  **form_compiler_optionss** (API)
        -  **$PWD/ffcx_options.json** (local options)
        -  **$XDG_CONFIG_HOME/ffcx/ffcx_options.json** (user options)
        -  **FFCX_DEFAULT_OPTIONS** in `ffcx.options`

        `$XDG_CONFIG_HOME` is `~/.config/` if the environment variable
            is not set.

        The contents of the `dolfinx_options.json` files are cached
        on the first call. Subsequent calls to this function use this
        cache.

        Example `dolfinx_jit_options.json` file:

            ``{ "cffi_extra_compile_args": ["-O2", "-march=native" ],
            "cffi_verbose": True }``

    """
    p_ffcx = ffcx.options.get_options(form_compiler_options)
    p_jit = get_options(jit_options)

    if not isinstance(ufl_object, ufl.Form):
        raise ValueError("Functinality only implemented for Form objects.")

    # Switch on type and compile, returning cffi object
    ufl_forms = [ufl_object]
    r = compile_forms(ufl_forms, options=p_ffcx, **p_jit)
    return (r[3], r[0][0], r[1], r[2])
