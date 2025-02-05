# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# This file is a modification of the original ``dolfinx/python/dolfinx/fem/forms.py`` file.
# See copyright below.
#
# Copyright (C) 2017-2024 Chris N. Richardson, Garth N. Wells, Michal Habera and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# --------------------------------------------------------------------------


from __future__ import annotations

import collections.abc
import types
import typing
from dataclasses import dataclass
from itertools import chain

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import ffcx.options
import numpy as np
import numpy.typing as npt
import ufl
import ufl.algorithms.analysis
from dolfinx import cpp as _cpp  # type: ignore
from dolfinx import default_scalar_type, jit
from dolfinx.fem import IntegralType
from dolfinx.fem.forms import (
    Form,
    _ufl_to_dolfinx_domain,
    form_cpp_class,
    form_cpp_creator,
    get_integration_domains,
)

from qugar.dolfinx.custom_coefficients import CustomCoeffsPacker
from qugar.dolfinx.integral_data import IntegralData
from qugar.dolfinx.jit import ffcx_jit
from qugar.quad import QuadGenerator

if typing.TYPE_CHECKING:
    from mpi4py import MPI

    from dolfinx.fem import function
    from dolfinx.mesh import Mesh


class CustomForm(Form):
    """Form for custom integrals.

    It derives from dolfinx.forms.Form just adding an extra
    functionality for computing the required custom coefficients at
    runtime (the method ``pack_coefficients``).

    """

    def __init__(
        self,
        form: typing.Union[
            _cpp.fem.Form_complex64,
            _cpp.fem.Form_complex128,
            _cpp.fem.Form_float32,
            _cpp.fem.Form_float64,
        ],
        itg_data: list[IntegralData],
        ufcx_form=None,
        code: str | None = None,
        module: typing.Optional[types.ModuleType] = None,
    ):
        """A custom finite element form.

        Note:
            CustomForms should normally be constructed using
            :func:`form_custom` and not using this class initialiser.
            This class is combined with different base classes that
            depend on the scalar type used in the Form.

        Args:
            form: Compiled form object.
            ufcx_form: UFCx form.
            code: Form C++ code.
            module: CFFI module.
        """
        super().__init__(form, ufcx_form, code, module)
        self._coeffs_packer = CustomCoeffsPacker(self, itg_data)

    def pack_coefficients(
        self, quad_gen: QuadGenerator
    ) -> dict[
        tuple[IntegralType, int],
        npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128],
    ]:
        """Function for generating the coefficients needed for computing
        custom integrals at runtime.

        Note:
            This function mimics the behaviour of
            ``dolfinx.cpp.fem.pack_coefficients``.

        Args:
            quad_gen (QuadGenerator): Custom quadrature generator.

        Returns:
            dict[ tuple[IntegralType, int], npt.NDArray[np.float32 |
            np.float64 | np.complex64 | np.complex128]]:
            Generated custom coefficients.
        """
        return self._coeffs_packer.pack_coefficients(quad_gen)


def form_custom(
    form: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
    dtype: npt.DTypeLike = default_scalar_type,
    form_compiler_options: typing.Optional[dict] = None,
    jit_options: typing.Optional[dict] = None,
    entity_maps: typing.Optional[dict[Mesh, npt.NDArray[np.int32]]] = None,
):
    """Creates a CustomForm or a list of CustomForm to be integrated
    over custom domains.

    Note:
        This function is just a copy of
        ``dolfinx.fem.forms.form`` with some small
        modifications. Namely:

        - replacing ``dolfinx.jit.ffcx_jit`` with
          ``qugar.dolfinx.jit.ffcx_jit``.
        - creating a new form class ``CustomForm`` (that derives from
          ``dolfinx.forms.Form`` and adds a new functionalitiy for
          computing the required custom coefficients at runtime).

    Args:
        form: A UFL form or list(s) of UFL forms.
        dtype: Scalar type to use for the compiled form.
        form_compiler_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`.
        entity_maps: If any trial functions, test functions, or
            coefficients in the form are not defined over the same mesh
            as the integration domain, `entity_maps` must be supplied.
            For each key (a mesh, different to the integration domain
            mesh) a map should be provided relating the entities in the
            integration domain mesh to the entities in the key mesh e.g.
            for a key-value pair (msh, emap) in `entity_maps`, `emap[i]`
            is the entity in `msh` corresponding to entity `i` in the
            integration domain mesh.

    Returns:
        CustomForm | list[CustomForm]: Compiled finite element `CustomForm`.

    Note:
        This function is responsible for the compilation of a UFL form
        (using FFCx) and attaching coefficients and domains specific
        data to the underlying C++ form. It dynamically create a
        :class:`Form` instance with an appropriate base class for the
        scalar type, e.g. :func:`_cpp.fem.Form_float64`.
    """

    if form_compiler_options is None:
        form_compiler_options = dict()

    form_compiler_options["scalar_type"] = dtype
    ftype = form_cpp_class(dtype)

    def _form(form) -> CustomForm:
        """Compile a single UFL form."""
        # Extract subdomain data from UFL form
        sd = form.subdomain_data()
        (domain,) = list(sd.keys())  # Assuming single domain

        # Check that subdomain data for each integral type is the same
        for data in sd.get(domain).values():
            assert all([d is data[0] for d in data])

        mesh = domain.ufl_cargo()
        if mesh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")
        itg_data, ufcx_form, module, code = ffcx_jit(
            mesh.comm,
            form,
            form_compiler_options=form_compiler_options,  # type: ignore
            jit_options=jit_options,
        )

        # For each argument in form extract its function space
        V = [arg.ufl_function_space()._cpp_object for arg in form.arguments()]

        # Prepare coefficients data. For every coefficient in form take
        # its C++ object.
        original_coeffs = form.coefficients()
        coeffs = [
            original_coeffs[ufcx_form.original_coefficient_positions[i]]._cpp_object
            for i in range(ufcx_form.num_coefficients)
        ]
        constants = [c._cpp_object for c in form.constants()]

        # Make map from integral_type to subdomain id
        subdomain_ids = {type: [] for type in sd.get(domain).keys()}
        for integral in form.integrals():
            if integral.subdomain_data() is not None:
                # Subdomain ids can be strings, its or tuples with
                # strings and ints
                if integral.subdomain_id() != "everywhere":
                    try:
                        ids = [sid for sid in integral.subdomain_id() if sid != "everywhere"]
                    except TypeError:
                        # If not tuple, but single integer id
                        ids = [integral.subdomain_id()]
                else:
                    ids = []
                subdomain_ids[integral.integral_type()].append(ids)

        # Chain and sort subdomain ids
        for itg_type, marker_ids in subdomain_ids.items():
            flattened_ids = list(chain.from_iterable(marker_ids))
            flattened_ids.sort()
            subdomain_ids[itg_type] = flattened_ids

        # Subdomain markers (possibly empty list for some integral
        # types)
        subdomains = {
            _ufl_to_dolfinx_domain[key]: get_integration_domains(
                _ufl_to_dolfinx_domain[key], subdomain_data[0], subdomain_ids[key]
            )
            for (key, subdomain_data) in sd.get(domain).items()
        }

        if entity_maps is None:
            _entity_maps = dict()
        else:
            _entity_maps = {msh._cpp_object: emap for (msh, emap) in entity_maps.items()}

        f = ftype(
            module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form)),
            V,
            coeffs,
            constants,
            subdomains,
            _entity_maps,
            mesh,
        )
        return CustomForm(f, itg_data, ufcx_form, code, module)

    def _flatten_list(lst):
        new_lst = []
        for i in lst:
            if isinstance(i, list):
                new_lst.extend(_flatten_list(i))
        return new_lst

    def _create_form(form) -> CustomForm | None | list[CustomForm | None]:
        """Recursively convert ufl.Forms to dolfinx.fem.Form.

        Args:
            form: UFL form or list of UFL forms to extract DOLFINx forms
                from.

        Returns:
            A ``dolfinx.fem.Form`` or a list of ``dolfinx.fem.Form``.
        """
        if isinstance(form, ufl.Form):
            if form.empty():
                return None
            else:
                return _form(form)
        elif isinstance(form, collections.abc.Iterable):
            forms = _flatten_list(list(map(lambda sub_form: _create_form(sub_form), form)))
            return forms
        else:
            raise ValueError("Not implemented case.")

    return _create_form(form)


@dataclass
class CompiledCustomForm:
    """Compiled UFL form without associated DOLFINx data."""

    ufl_form: ufl.Form  # The original ufl form
    itg_data: list[IntegralData]  # Integral data
    ufcx_form: typing.Any  # The compiled form
    module: typing.Any  # The module
    code: str  # The source code
    dtype: npt.DTypeLike  # data type used for the `ufcx_form`


def compile_form_custom(
    comm: MPI.Intracomm,
    form: ufl.Form,
    form_compiler_options: typing.Optional[dict] = {"scalar_type": default_scalar_type},
    jit_options: typing.Optional[dict] = None,
) -> CompiledCustomForm:
    """Compile UFL form without associated DOLFINx data.

    Args:
        comm: The MPI communicator used when compiling the form
        form: The UFL form to compile
        form_compiler_options: See
        :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`.
    """
    p_ffcx = ffcx.options.get_options(form_compiler_options)
    p_jit = jit.get_options(jit_options)
    itg_data, ufcx_form, module, code = ffcx_jit(comm, form, p_ffcx, p_jit)  # type: ignore # noqa

    return CompiledCustomForm(
        form,
        itg_data,
        ufcx_form,
        module,
        code,
        p_ffcx["scalar_type"],  # type: ignore
    )


def create_form_custom(
    form: CompiledCustomForm,
    function_spaces: list[function.FunctionSpace],
    mesh: Mesh,
    subdomains: dict[IntegralType, list[tuple[int, np.ndarray]]],
    coefficient_map: dict[ufl.FunctionSpace, function.Function],
    constant_map: dict[ufl.Constant, function.Constant],
    entity_maps: dict[Mesh, npt.NDArray[np.int32]] | None = None,
) -> CustomForm:
    """
    Create a CustomForm object from a data-independent compiled form.

    Args:
        form: Compiled ufl form custom
        function_spaces: List of function spaces associated with the
            form. Should match the number of arguments in the form.
        mesh: Mesh to associate form with
        subdomains: A map from integral type to a list of pairs, where
            each pair corresponds to a subdomain id and the set of of
            integration entities to integrate over. Can be computed with
            {py:func}`dolfinx.fem.compute_integration_domains`.
        coefficient_map: Map from UFL coefficient to function with data.
        constant_map: Map from UFL constant to constant with data.
        entity_map: A map where each key corresponds to a mesh different
            to the integration domain `mesh`. The value of the map is an
            array of integers, where the i-th entry is the entity in the
            key mesh.
    """
    if entity_maps is None:
        _entity_maps = {}
    else:
        _entity_maps = {m._cpp_object: emap for (m, emap) in entity_maps.items()}

    _subdomain_data = subdomains.copy()
    for _, idomain in _subdomain_data.items():
        idomain.sort(key=lambda x: x[0])

    # Extract name of ufl objects and map them to their corresponding
    # C++ object
    ufl_coefficients = ufl.algorithms.extract_coefficients(form.ufl_form)
    coefficients = {
        f"w{ufl_coefficients.index(u)}": uh._cpp_object for (u, uh) in coefficient_map.items()
    }
    ufl_constants = ufl.algorithms.analysis.extract_constants(form.ufl_form)
    constants = {f"c{ufl_constants.index(u)}": uh._cpp_object for (u, uh) in constant_map.items()}

    ftype = form_cpp_creator(form.dtype)
    f = ftype(
        form.module.ffi.cast("uintptr_t", form.module.ffi.addressof(form.ufcx_form)),
        [fs._cpp_object for fs in function_spaces],
        coefficients,
        constants,
        _subdomain_data,
        _entity_maps,
        mesh._cpp_object,
    )
    return CustomForm(f, form.itg_data, form.ufcx_form, form.code)
