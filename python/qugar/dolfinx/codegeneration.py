# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Functionalities for modifying FFCx generated code for integrals."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import math
import re

import basix
import basix.ufl
import ffcx.codegeneration.codegeneration
import numpy as np
import numpy.typing as npt
from ffcx.analysis import UFLData
from ffcx.codegeneration.codegeneration import CodeBlocks
from ffcx.ir.representation import DataIR, IntegralIR

from qugar.dolfinx.boundary import ParamNormal
from qugar.dolfinx.fe_table import FETable
from qugar.dolfinx.integral_data import IntegralData, extract_integral_data
from qugar.dolfinx.parsing_utils import dtype_to_C_str, get_pairing_brackets, parse_dtype_C


def _modify_header(code_blocks: CodeBlocks) -> CodeBlocks:
    """Adds stddef.h include in the header of the given `code_blocks`.

    This include may be required for ``ptrdiff_t`` type used in the
    modified integral code.

    It modifies the header of the C code associated to a form to be
    computed by appending the new include before ``ufcx.h``

    Args:
        code_blocks (CodeBlocks): Original code blocks generated
            by the FFCx library.

    Returns:
        CodeBlocks: Modified code blocks including extra header.
    """

    assert len(code_blocks[0]) == 1

    header = code_blocks[0][0][1]
    match = re.search(r"#include <ufcx.h>", header)
    assert match

    new_header = header[: match.start()]
    new_header += "#include <stddef.h>\n"
    new_header += header[match.start() :]

    code_blocks[0][0] = (code_blocks[0][0][0], new_header)

    return code_blocks


class _IntegralModifier:
    """Class for modifying a C code corresponding a tabulate tensor
    integral function, allowing the use of custom quadrature rules
    changing cell by cell, for a subset of cells.

    The main use of this class after initialization is to call its
    public method `create_new_code` to create the modified integral
    code. This new code replaces the original C code with a new version
    that can invoke two possible functions: the original version with
    static definitions for weights and FE tables (to be used when all
    the cells have a static quadrature defined at compile time), and a
    new version where those values are loaded dynamically at runtime
    for every cell from provided custom coefficients. The latter also
    allows to use normals at custom unfitted boundaries that do not
    coincide with cells' facets.

    Parameters:
        _ir (IntegralIR): FFCx Intermediate Representation of the
            integral.
        _data (IntegralData): Data structure containing information
            associated to the quadrature, as integral type, subdomain
            ids, quadratures, and FE tables.
        _func_signt (str): Block of C code containing the signature of
            original tabulate tensor function
        _func_impl (str): Block of C code containing the implementation
            of original tabulate tensor function
        _before_func (str): Block of C code before the original tabulate
            tensor function.
        _after_func (str): Block of C code after the original tabulate
            tensor function.
        _coeffs_dtype (type[np.float32 | np.float64 | np.complex64 |
            np.complex128]): Type of the function's coefficients and
            constants.
    """

    def __init__(self, code: str, itg_ir: IntegralIR, itg_data: IntegralData) -> None:
        """Constructor.

        Args:
            code (str): Original C code associated to the tabulate
                tensor integral function.
            itg_ir (IntegralIR): ffcx Intermediate Representation of the
                integral to be modified.
            data (IntegralData): Data structure containing information
                associated to the quadrature, as integral type,
                subdomain ids, quadratures, and FE tables.
        """

        self._ir = itg_ir
        self._data = itg_data
        self._parse(code)

    def _parse(self, code: str) -> None:
        """Parses all the information from the C code with the help
        of the information stored in ``self._ir`` and ``self._data``.

        Args:
            code (str): C code of the original function to be parsed.
        """
        self._split_code(code)
        self._parse_coeffs_dtype()
        self._check_FFCx_input()

    def _compute_coeffs_offset(self) -> int:
        """Computes the offfset of the position where the
        custom information starts in the coefficients array pointer
        ``w``.

        Note that this offset corresponds to all the coefficients of all
        the integrands for a single integral, according to the way it is
        computed in DOLFINx.
        See https://github.com/FEniCS/dolfinx/blob/5a20e2ba3907c1f108cb0af45931f46e32250351/cpp/dolfinx/fem/utils.h#L957C1-L987C2

        Returns:
            int: Offset value.
        """

        offsets = list(self._ir.expression.coefficient_offsets.values())
        if len(offsets) == 0:
            return 0
        else:
            coeff = list(self._ir.expression.coefficient_offsets.keys())[-1]
            element = coeff.ufl_function_space().element  # type: ignore
            return offsets[-1] + element.space_dimension

    def _split_code(self, code: str) -> None:
        """Splits in blocks the given C `code`. It extracts its
        signature and implementation body, as well as the code included
        before and after the function.

        These code blocks are stored in ``self._func_signt``,
        ``self._func_impl``, ``self._before_func``, and
        ``self._after_func``, ``self._func_signt``, respectively.

        Args:
            code (str): Original C code associated to the tabulate
                tensor integral function to be split in blocks.
        """

        itg_pattern = r"void\s*tabulate_tensor_integral_(\w+)\s*\([\w|\s|\,|*]*\)"
        match = re.search(itg_pattern, code)
        assert match, "Integral not found."

        self._before_func = code[: match.start()]
        self._func_signt = match.group(0)
        # In FEniCSx 0.10.0, the kernel name is "integral_<hash>_<cell_type>"
        # (e.g. "_hexahedron"). self._data.name only carries the hash part, so
        # we read the full suffix-aware name from the actual function signature.
        self._integral_name = match.group(1)

        sub_code = code[match.end() :]

        # The start-end indices could be obatained as
        # start,end = get_pairing_brackets(sub_code)
        # But this may be very slow in some situations.
        match = re.search("{", sub_code)
        assert match
        start = match.start()

        # This assumes that the function does not end with }; but just }
        match = re.search("\n}", sub_code[::-1])
        assert match
        end = len(sub_code) - match.start() + 1

        self._func_impl = sub_code[start:end]

        self._after_func = sub_code[end:]

    def _check_FFCx_input(self) -> None:
        """Checks the validity of the integral to be modified.
        If the integral type is not supported, an exception is raised.

        Right now, this function only supports integrals to be computed
        over cells, exterior and interior facets. Vertices are not
        supported. It does not support either the use of
        sum-factorization quadratures.

        It also checks that some assumptions about the
        `entity_local_index` and `quadrature_permutation` arrays on
        which the implementation relies upon, are valid.

        Raises:
            ValueError: An exception is raised if the integral is not
                performed in a cell or facet.
            ValueError: An exception is raised is thrown if the
                the hypotheses about `entity_local_index` array are
                wrong.
            ValueError: An exception is raised is thrown if the
                the hypotheses about `quadrature_permutation` array are
                wrong.
            ValueError: An exception is raised is thrown if the
                integral is to be performed using sum-factorization.
        """

        # Checking integral and entity types.

        if self._ir.expression.entity_type not in ["cell", "facet"]:
            raise ValueError("Only cell and facet entity types are supported.")

        if self._data.integral_type not in [
            "cell",
            "exterior_facet",
            "interior_facet",
        ]:
            raise ValueError(
                "Only cell, exterior and interior facets integral types are supported."
            )

        # Checking that is the integral presents mixed dimension,
        # then it must be an integral over exterior facets.
        if self._data.is_mixed_dim:
            assert self._data.integral_type == "exterior_facet", "Not implemented."

        # Checking that sum-factorization is disabled.

        for _cell_type, quad in self._ir.expression.integrand.keys():
            if quad.has_tensor_factors:  # type: ignore
                raise ValueError(
                    "Sum-factorization quadrature it is incompatible with "
                    "custom quadratures. Toogle off sum-factorization in the "
                    "ffcx options passed when building the form."
                )

        # Checking the indices of `entity_local_index` being accessed.
        # It should be none for cell entities, 0 for exterior facets,
        # and 0 and 1 for interior facets.

        indices = set()
        for match in re.finditer(r"entity_local_index\[(\w+)\]", self._func_impl):
            index_str = match.group(1)
            assert index_str.isnumeric()
            indices.add(int(index_str))

        target_indices = {
            "cell": set(),
            "exterior_facet": set([0]),
            "interior_facet": set([0, 1]),
        }
        if indices != target_indices[self._data.integral_type]:
            raise ValueError(
                "Implementation error: Invalid indices for 'entity_local_index' array."
            )

        # Checking the indices of `quadrature_permutation` being
        # accessed. It should be none for cells and exterior facets,
        # 0 and 1 for interior facets, and 0 for mixed dimensions.

        indices = set()
        for match in re.finditer(r"quadrature_permutation\[(\w+)\]", self._func_impl):
            index_str = match.group(1)
            assert index_str.isnumeric()
            indices.add(int(index_str))

        target_indices = {
            "cell": set(),
            "exterior_facet": set([0]),
            "interior_facet": set([0, 1]),
        }
        if len(indices) > 0 and indices != target_indices[self._data.integral_type]:
            raise ValueError(
                "Implementation error: Invalid indices for 'quadrature_permutation' array."
            )

    def _parse_coeffs_dtype(self) -> None:
        """Parses the scalar `numpy` type associated to the integral
        coefficients and constants. It is ``np.float32``,
        ``np.float64``, ``np.complex64``, or ``np.complex128``, and it
        is stored in ``self._coeffs_dtype``.
        """

        # Extracting coefficients (and constants) type.
        match = re.search(r"const\s+([^\*]+)\*\s+restrict\s+w", self._func_signt)
        assert match
        self._coeffs_dtype = parse_dtype_C(match.group(1))

        # Checking types compabilitiy between _coeffs_dtype and
        # integral's dtype.
        if self._coeffs_dtype in [np.float32, np.complex64]:
            assert self._data.dtype == np.float32
        elif self._coeffs_dtype in [np.float64, np.complex128]:
            assert self._data.dtype == np.float64
        else:
            assert False

    def _modify_FE_tables_accesses(self, code: str) -> str:
        """Finds all the accesses to the static 4-dimensional arrays of
        FE tables and replaces them with accesses to a plain (1D) array
        that is dynamically loaded.

        This is done for all FE tables, except for the ones whose values
        are the same for all the quadrature points. In that cases, the
        tables are statically defined and the 4D accesses are kept.

        Args:
            code (str): Original C code to modify.

        Returns:
            str: Newly generated C code.
        """

        bracket_pattern = r"\[\s*(\w+\[\d+\]|[^\]]*)\s*\]\s*"
        FE_pattern = r"FE([\w|\_]+)\s*" + bracket_pattern * 4

        has_permutations = self._data.integral_type == "interior_facet"

        new_code = ""

        def get_FE_table(table_name: str) -> FETable:
            for tables in self._data.quad_data_FE_tables.values():
                if table := next(filter(lambda table: table.name == table_name, tables), None):
                    return table
            assert False, "FE table not found."

        pos = 0
        for match in re.finditer(FE_pattern, code[:]):
            table_name = f"FE{match.group(1)}"
            FE_table = get_FE_table(table_name)

            if FE_table.is_constant_for_pts():
                access_code = match.group(0)  # 4D access is kept.
            else:
                indices = tuple(match.group(i) for i in range(2, 6))
                access_code = FE_table.create_new_access_code(indices, has_permutations)

            new_code += code[pos : match.start()] + access_code
            pos = match.end()

        new_code += code[pos:]

        return new_code

    @staticmethod
    def _find_quad_name_in_loop(code: str) -> str:
        """Find the quadrature name in the given code."""
        match = re.search(r"_Q(\w\w\w)[\[|_e]", code)
        if match is None:
            match = re.search(r"weights_(\w\w\w)", code)
        assert match
        quad_name = match.group(1)
        return quad_name

    def _modify_points_loops(self, code: str) -> str:
        """Finds all the for loops along quadrature points in the given
        C code and transforms them from static (with a compile time
        defined upper bound) to dymamic (with an upper bound defined as
        the number of quadrature points loaded at runtime for that
        particular integrand).

        Args:
            code (str): Original C code to modify.

        Returns:
            str: Newly generated C code.
        """

        loop_pattern = r"for\s*\(\s*int\s+iq\s*=\s*0\s*;\s*iq\s*<\s*(\d+)\s*;\s*\+\+iq\s*\)"

        new_code = code[:]
        while match := re.search(loop_pattern, new_code):
            quad_name = _IntegralModifier._find_quad_name_in_loop(new_code[match.end() :])

            new_loop = f"for (int iq = 0; iq < n_pts_Q{quad_name}; ++iq)"

            new_code = new_code[: match.start()] + new_loop + new_code[match.end() :]

        return new_code

    def _get_constant_normal_offsets(self) -> tuple[np.int64, ...]:
        """Finds the indices of the (fake) constants associated to
        unfitted boundary normals.

        Returns:
            tuple[np.int64, ...]: Offsets of the constants associated
            to unfitted boundary normals.
        """

        constant_offsets = self._ir.expression.original_constant_offsets

        offsets = tuple(
            np.int64(offset)
            for constant, offset in constant_offsets.items()
            if isinstance(constant, ParamNormal)
        )
        return offsets

    def _substitute_normals(self, code_block: str, quad_name: str) -> str:
        """This function replaces the used (fake) constants by their
        corresponding boundary normals varying at every quadrature
        point.

        Args:
            code_block (_type_): C code block in which normals are
                replaced.
            quad_name (_type_): Name of quadrature associated to the
                `code_block`.

        Returns:
            str: Newly generated C code.
        """

        for ct_offset in self._get_constant_normal_offsets():
            for i in range(self._data.tdim):
                constant_pattern = rf"([\*|\+|\-\/\%|\s|\(])c\[{ct_offset + i}\]"

                new_norm_ind = f"{self._data.tdim} * iq"
                if i > 0:
                    new_norm_ind += f" + {i}"
                normal_str = f"normals_{quad_name}[{new_norm_ind}]"

                while match := re.search(constant_pattern, code_block):
                    code_block = code_block.replace(match.group(0), f"{match.group(1)}{normal_str}")

        return code_block

    def _has_unfitted_boundary(self) -> bool:
        """Checks if the integral has unfitted boundaries.

        Returns:
            bool: True if the integral has unfitted boundaries,
                false otherwise.
        """

        return any(
            quad_data.unfitted_boundary for quad_data in self._data.quad_data_FE_tables.keys()
        )

    def _modify_normal_constants(self, code: str) -> str:
        """In the case of unfitted custom boundaries, this function
        replaces the used (fake) constants by their corresponding
        boundary normals varying at every quadrature point.

        Args:
            code (str): Original C code to modify.

        Returns:
            str: Newly generated C code.
        """

        if not self._has_unfitted_boundary():
            return code  # No normals to subsitute.

        loop_pattern = r"for\s*\(\s*int\s+iq\s*=\s*0\s*;\s*iq\s*<\s*(\w+)\s*;\s*\+\+iq\s*\)"

        # Finding the start of the first quadrature loop.
        match = re.search(loop_pattern, code)
        assert match is not None
        first_loop = match.start()

        # Note that in cases as for linear triangles, quantities as
        # the jacobians are constant at all the quadrature points and
        # therefore FFCx defines them out of the quadrature loops.
        # However, in the case of unfitted boundaries, these quantities
        # get multiplied by boundary normals that vary from point
        # to point and become non constant.
        # Thus, we take those constant definitions and move them to the
        # quadrature loops, to introduce the normals variation later on.
        # Unfortunately, including these constant variables in the loops
        # may cause the compiler complaining about unused variables in
        # the case there is more than one quadrature loop (integrand).
        # This is due to the fact that some of the variables are shared
        # by several loops but not all of them. We don't know which
        # ones, so we copy everything at the risk of some variables
        # being unused.

        assert code[:2] == "{\n"
        cnt_vars_block = code[2:first_loop]
        code = code[:2] + code[first_loop:]

        ind = 0
        new_code = ""
        for match in re.finditer(loop_pattern, code):
            quad_name = match.group(1).split("Q")[1]
            all_quad_datas = list(self._data.quad_data_FE_tables.keys())
            quad_data = next(filter(lambda qd: qd.name == quad_name, all_quad_datas))

            ids = get_pairing_brackets(code[match.start() :])
            i0 = ids[0] + match.start()
            i1 = ids[1] + match.start()

            assert code[i0 : i0 + 2] == "{\n"
            block = "{\n" + cnt_vars_block + code[i0 + 2 : i1]

            if quad_data.unfitted_boundary:
                block = self._substitute_normals(block, quad_name)

            new_code += code[ind:i0] + block
            ind = i1

        new_code += code[ind:]

        return new_code

    def _erase_static_declarations(self, code: str) -> str:
        """Erases in the given C `code` all static declarations of
        weights array and FE tables (including the associated comments).

        Args:
            code (str): C code containing static weights arrays and FE
                tables to be removed.

        Returns:
            str: New code without weights and FE table static
            declarations.
        """

        # Finding block where weights are defined.
        quad_pattern = r"// Quadrature rules"
        match = re.search(quad_pattern, code)
        assert match
        start = match.start()

        # Finding block where FE tables are defined.
        precmp_pattern = r"// Precomputed values of basis functions"
        match = re.search(precmp_pattern, code)
        assert match

        FE_pattern = r"static\s+const\s+\w+\s+FE[^;]*;\s*\n"
        for match in re.finditer(FE_pattern, code):
            continue
        end = match.end()

        return code[:start] + code[end:]

    def _shim_suffix(self) -> str:
        """Returns the shim entry-point suffix for the real dtype."""
        return "f64" if self._data.dtype == np.float64 else "f32"

    def _create_shim_decls(self) -> str:
        """Emits the ``extern`` declarations of the basix tabulation shim
        entry points used by the on-the-fly custom kernel."""
        suffix = self._shim_suffix()
        dtype_str = dtype_to_C_str(self._data.dtype)
        return (
            f"extern int qugar_register_element_{suffix}"
            "(int, int, int, int, int, int);\n"
            f"extern int qugar_tabulate_{suffix}"
            f"(int, int, const {dtype_str}*, int, int, {dtype_str}*, long);\n\n"
        )

    @staticmethod
    def _resolve_mixed_component(element, flat_component: int):
        """For a ``basix.ufl._MixedElement``, drill into the sub-element
        that owns ``flat_component`` and return
        ``(sub_element, component_within_sub)``. Non-mixed elements pass
        through unchanged. Recursive for nested mixed elements.
        """
        if not isinstance(element, basix.ufl._MixedElement):
            return element, flat_component
        acc = 0
        for sub in element.sub_elements:
            if flat_component < acc + sub.reference_value_size:
                return _IntegralModifier._resolve_mixed_component(
                    sub, flat_component - acc)
            acc += sub.reference_value_size
        raise ValueError(
            f"flat_component {flat_component} out of range for mixed element "
            f"with reference_value_size {element.reference_value_size}"
        )

    @staticmethod
    def _table_codegen_info(table: FETable) -> dict:
        """Computes the constants needed to repack a basix tabulation
        block into a single FFCx FE table.

        Returns a dict with the core-element ``create_element`` parameters
        and the indices used by the repack: ``didx`` (basix derivative
        index), ``vaxis`` (value-axis index), ``ndofs`` and ``vs`` (basix
        block dims), and ``gdim`` (points dimension).

        For mixed elements (``basix.ufl._MixedElement``), drills into the
        sub-element that owns ``table.component`` so the basix block is
        tabulated for the right sub-element.
        """
        el, local_c = _IntegralModifier._resolve_mixed_component(
            table.element, table.component)
        block_size = getattr(el, "block_size", 1)
        scalar = el.sub_elements[0] if block_size > 1 else el
        be = scalar.basix_element
        params = (
            int(be.family),
            int(be.cell_type),
            be.degree,
            int(be.lagrange_variant),
            int(be.dpc_variant),
            int(be.discontinuous),
        )
        gdim = table.element_dim
        vs = 1 if block_size > 1 else int(el.reference_value_size)
        derivs = list(table.derivatives) if table.derivatives else []
        derivs = (derivs + [0] * gdim)[:gdim]
        return {
            "params": params,
            "gdim": gdim,
            "vs": vs,
            "ndofs": table.funcs,
            "didx": basix.index(*derivs),
            "vaxis": 0 if block_size > 1 else local_c,
            "maxnd": sum(derivs),
        }

    def _create_custom_data_loaders(self) -> str:
        """Creates C functions that load, per quadrature/integrand, the
        custom quadrature *points* and *weights* (and unfitted-boundary
        normals) from the custom coefficients array.

        Unlike the previous design, FE table *values* are NOT read here:
        they are computed on the fly by tabulating the basix elements at
        the loaded points (see ``_create_custom_data_callers``). The
        loader only advances the ``w_custom`` pointer past the per-cell
        points/weights/normals and returns the number of points.

        Returns:
            str: C code for the generated loader functions.
        """

        dtype_str = dtype_to_C_str(self._data.dtype)
        name = self._data.name
        gdim = self._data.tdim
        mixed_dim = self._data.is_mixed_dim
        interior_facet = self._data.integral_type == "interior_facet"
        fdim = gdim - 1  # facet-reference coordinate dimension

        code = ""

        for quad_data, _FE_tables in self._data.quad_data_FE_tables.items():
            quad_name = quad_data.name

            func_name = f"load_points_{name}_Q{quad_name}"
            points_name = f"points_{quad_name}"
            points_side1_name = f"points_side1_{quad_name}"
            points_facet_name = f"points_facet_{quad_name}"
            weights_name = f"weights_{quad_name}"
            normals_name = f"normals_{quad_name}"
            n_pts_name = f"n_pts_Q{quad_name}"

            has_normals = quad_data.unfitted_boundary

            indent = " " * len(f"int {func_name}(")
            code += f"\nint {func_name}(const {dtype_str}* restrict *w_custom"
            code += f",\n{indent}const {dtype_str}* restrict *{points_name}"
            if interior_facet:
                code += f",\n{indent}const {dtype_str}* restrict *{points_side1_name}"
            if mixed_dim:
                code += f",\n{indent}const {dtype_str}* restrict *{points_facet_name}"
            code += f",\n{indent}const {dtype_str}* restrict *{weights_name}"
            if has_normals:
                code += f",\n{indent}const {dtype_str}* restrict *{normals_name}"
            code += ")\n{\n"

            code += f"const int {n_pts_name} = (int) **w_custom;\n*w_custom += 1;\n\n"
            code += f"*{points_name} = *w_custom;\n"
            code += f"*w_custom += {gdim} * {n_pts_name};\n\n"
            if interior_facet:
                code += f"*{points_side1_name} = *w_custom;\n"
                code += f"*w_custom += {gdim} * {n_pts_name};\n\n"
            if mixed_dim:
                code += f"*{points_facet_name} = *w_custom;\n"
                code += f"*w_custom += {fdim} * {n_pts_name};\n\n"
            code += f"*{weights_name} = *w_custom;\n"
            code += f"*w_custom += {n_pts_name};\n\n"
            if has_normals:
                code += f"*{normals_name} = *w_custom;\n"
                code += f"*w_custom += {self._data.tdim} * {n_pts_name};\n\n"

            code += f"return {n_pts_name};\n}}\n\n\n"

        return code

    def _create_custom_data_callers(self) -> str:
        """Creates the code for calling the functions loading
        the weights arrays and FE tables from extra coefficients.

        This code is intended to be called inside the new generated
        function that uses dynamic (instead of static) data.
        It also declare the needed pointers.

        Returns:
            str: C code for calling dynamic data loaders.
        """

        dtype_str = dtype_to_C_str(self._data.dtype)
        suffix = self._shim_suffix()

        itg_name = self._data.name
        tdim = self._data.tdim
        mixed_dim = self._data.is_mixed_dim

        is_interior_facet = self._data.integral_type == "interior_facet"

        call_code = ""
        for quad_data, FE_tables in self._data.quad_data_FE_tables.items():
            quad_name = quad_data.name

            has_normals = quad_data.unfitted_boundary

            func_name = f"load_points_{itg_name}_Q{quad_name}"
            points_name = f"points_{quad_name}"
            points_side1_name = f"points_side1_{quad_name}"
            points_facet_name = f"points_facet_{quad_name}"
            weights_name = f"weights_{quad_name}"
            normals_name = f"normals_{quad_name}"
            n_pts_name = f"n_pts_Q{quad_name}"

            # Constant-for-points tables stay statically defined.
            for table in FE_tables:
                if table.is_constant_for_pts():
                    call_code += table.code + "\n"

            # Per-cell points/weights (+side-1, +facet points, +normals) decls.
            call_code += f"const {dtype_str}* restrict {points_name};\n"
            if is_interior_facet:
                call_code += f"const {dtype_str}* restrict {points_side1_name};\n"
            if mixed_dim:
                call_code += f"const {dtype_str}* restrict {points_facet_name};\n"
            call_code += f"const {dtype_str}* restrict {weights_name};\n"
            if has_normals:
                call_code += f"const {dtype_str}* restrict {normals_name};\n"

            # Loader execution (returns the runtime number of points).
            aux = f"const int {n_pts_name} = {func_name}("
            indent = " " * len(aux)
            call_code += f"{aux}&w_custom,\n{indent}&{points_name}"
            if is_interior_facet:
                call_code += f",\n{indent}&{points_side1_name}"
            if mixed_dim:
                call_code += f",\n{indent}&{points_facet_name}"
            call_code += f",\n{indent}&{weights_name}"
            if has_normals:
                call_code += f",\n{indent}&{normals_name}"
            call_code += ");\n\n"

            # On-the-fly tabulation + repack, grouped per element so each
            # element is tabulated once (one basix block feeds all of its
            # component/derivative tables). For interior-facet integrals
            # with perm>1 tables, two basix blocks are produced (one per
            # side) and perm>1 tables get an FE..[2] array of per-side
            # buffers; perm<=1 tables stay single-buffer (their values
            # agree on both sides).
            varying = [t for t in FE_tables if not t.is_constant_for_pts()]

            groups: dict[tuple, list[tuple[FETable, dict]]] = {}
            for table in varying:
                info = self._table_codegen_info(table)
                groups.setdefault(info["params"], []).append((table, info))

            for gi, (params, items) in enumerate(groups.items()):
                fam, cell, deg, lv, dv, disc = params
                info0 = items[0][1]
                gdim, ndofs, vs = info0["gdim"], info0["ndofs"], info0["vs"]
                maxnd = max(info["maxnd"] for _t, info in items)
                nderiv = math.comb(maxnd + gdim, gdim)

                handle = f"h_{quad_name}_{gi}"
                blk0 = f"block_{quad_name}_{gi}"
                blk1 = f"block_side1_{quad_name}_{gi}"
                blk_size = f"{nderiv} * {n_pts_name} * {ndofs} * {vs}"
                stride = f"{ndofs} * {vs}"

                # In mixed-dim integrals, facet-dim elements (gdim < tdim)
                # must be tabulated at facet-reference points; cell-dim
                # elements at the cell-mapped points.
                pts_var = (points_facet_name
                           if (mixed_dim and gdim != tdim)
                           else points_name)
                any_perm = is_interior_facet and any(
                    t.permutations > 1 for t, _ in items)

                call_code += (
                    f"const int {handle} = qugar_register_element_{suffix}"
                    f"({fam}, {cell}, {deg}, {lv}, {dv}, {disc});\n"
                )
                call_code += f"{dtype_str} {blk0}[{blk_size}];\n"
                call_code += (
                    f"qugar_tabulate_{suffix}({handle}, {maxnd}, {pts_var}, "
                    f"{n_pts_name}, {gdim}, {blk0}, (long)({blk_size}));\n"
                )
                if any_perm:
                    call_code += f"{dtype_str} {blk1}[{blk_size}];\n"
                    call_code += (
                        f"qugar_tabulate_{suffix}({handle}, {maxnd}, "
                        f"{points_side1_name}, {n_pts_name}, {gdim}, "
                        f"{blk1}, (long)({blk_size}));\n"
                    )
                call_code += "\n"

                for table, info in items:
                    funcs = table.funcs
                    didx, vaxis = info["didx"], info["vaxis"]
                    two_sided = is_interior_facet and table.permutations > 1

                    def _repack(buf_name: str, src: str) -> str:
                        r = f"{dtype_str} {buf_name}[{n_pts_name} * {funcs}];\n"
                        r += f"for (int iq = 0; iq < {n_pts_name}; ++iq)\n"
                        r += f"  for (int kd = 0; kd < {funcs}; ++kd)\n"
                        r += (
                            f"    {buf_name}[iq * {funcs} + kd] = {src}"
                            f"[{didx} * {n_pts_name} * {stride} + iq * {stride}"
                            f" + kd * {vs} + {vaxis}];\n"
                        )
                        return r

                    if two_sided:
                        buf0 = f"{table.name}_buf_0"
                        buf1 = f"{table.name}_buf_1"
                        call_code += _repack(buf0, blk0)
                        call_code += _repack(buf1, blk1)
                        call_code += (
                            f"const {dtype_str}* restrict {table.name}[2];\n"
                        )
                        call_code += f"{table.name}[0] = {buf0};\n"
                        call_code += f"{table.name}[1] = {buf1};\n\n"
                    else:
                        buf = f"{table.name}_buf"
                        call_code += _repack(buf, blk0)
                        call_code += (
                            f"const {dtype_str}* restrict {table.name} = "
                            f"{buf};\n\n"
                        )

        return call_code

    def _modify_unfitted_boundary_original_integrals(self) -> str:
        """Modifies the original function's implementation code for
        unfitted boundary integrals by deactivating the loops
        associated to the quadrature points.

        This is needed in the case the original function is called
        for a non-custom cell in which such integral has no sense.

        This is done by including a break statement inside the loop,
        which will prevent the loop from being executed.
        The compiler will optimize out the full loop, since it is
        never executed.

        Returns:
            str: New modified C code.
        """

        assert self._has_unfitted_boundary()

        loop_pattern = r"for\s*\(\s*int\s+iq\s*=\s*0\s*;\s*iq\s*<\s*(\d+)\s*;\s*\+\+iq\s*\)"

        # Finding the start of the first quadrature loop.
        match = re.search(loop_pattern, self._func_impl)
        assert match is not None
        first_loop = match.start()

        # Note that in cases as for linear triangles, quantities as
        # the jacobians are constant at all the quadrature points and
        # therefore FFCx defines them out of the quadrature loops.
        # However, in the case of unfitted boundaries, these quantities
        # get multiplied by boundary normals that vary from point
        # to point and become non constant.
        # Thus, we take those constant definitions and move them to the
        # quadrature loops, to introduce the normals variation later on.
        # Unfortunately, including these constant variables in the loops
        # may cause the compiler complaining about unused variables in
        # the case there is more than one quadrature loop (integrand).
        # This is due to the fact that some of the variables are shared
        # by several loops but not all of them. We don't know which
        # ones, so we copy everything at the risk of some variables
        # being unused.

        assert self._func_impl[:2] == "{\n"
        cnt_vars_block = self._func_impl[2:first_loop]
        code = self._func_impl[:2] + self._func_impl[first_loop:]

        all_quad_datas = list(self._data.quad_data_FE_tables.keys())

        ind = 0
        new_code = ""
        for match in re.finditer(loop_pattern, code):
            quad_name = _IntegralModifier._find_quad_name_in_loop(code[match.end() :])
            quad_data = next(filter(lambda qd: qd.name == quad_name, all_quad_datas))
            ids = get_pairing_brackets(code[match.start() :])
            i0 = ids[0] + match.start()
            i1 = ids[1] + match.start()

            assert code[i0 : i0 + 2] == "{\n"
            block = "{\n"

            if quad_data.unfitted_boundary:
                # This prevents the loop from being executed.
                # The compiler will optimize out the full loop.
                block += "break;\n"

            block += cnt_vars_block + code[i0 + 2 : i1]

            new_code += code[ind:i0] + block
            ind = i1

        new_code += code[ind:]

        return new_code

    def _create_non_custom_quad_function(self) -> str:
        """Creates a copy of the original function (signature +
        implementation), but appending the suffix ``_original`` to the
        function's name.

        Returns:
            str: New generated C code.
        """

        name = self._integral_name
        new_code = self._func_signt.replace(name, f"{name}_original")

        if self._has_unfitted_boundary():
            new_code += self._modify_unfitted_boundary_original_integrals()
        else:
            new_code += self._func_impl

        new_code += "\n\n"
        return new_code

    def _create_custom_function(self) -> str:
        """Creates a new version of the original function (signature +
        implementation), in which the static versions of weigths and FE
        tables are replaced by values dynamically loaded from the
        coefficients array ``w_custom``.

        The accesses to FE tables are transformed from accesses to a 4D
        (static) array, 1D array accesses.
        The upper bounds of the quadrature points loops are modified to
        accomodate the number of quadrature points for every
        particular cell. And, in the case of unfitted custom boundaries,
        the used (fake) constants are replaced by their corresponding
        boundary normals.

        Returns:
            str: New generated code.
        """

        # Creating new function signature.
        name = self._integral_name
        new_signt = self._func_signt.replace(name, f"{name}_custom")

        coeff_dtype_str = dtype_to_C_str(self._coeffs_dtype)
        dtype_str = dtype_to_C_str(self._data.dtype)

        # Adding w_custom to the list of arguments in the signature.
        match = re.search(
            rf"(\s*)const\s+{coeff_dtype_str}\*\s+restrict\s+w\s*,\s*\n",
            self._func_signt,
        )  # Search w array in function's signature.
        assert match is not None

        line = match.group(0)[:-1]
        indent = match.group(1)
        new_line = f"{indent}const {dtype_str}* restrict w_custom,"
        new_signt = new_signt.replace(line, line + new_line)

        # Create new function implementation.
        new_impl = self._erase_static_declarations(self._func_impl)
        new_impl = self._modify_FE_tables_accesses(new_impl)
        new_impl = self._modify_points_loops(new_impl)
        new_impl = self._modify_normal_constants(new_impl)

        custom_data_calls = self._create_custom_data_callers()
        assert new_impl[:2] == "{\n"
        new_impl = new_impl[:2] + custom_data_calls + new_impl[2:]

        return new_signt + new_impl + "\n\n"

    def _create_new_function(self) -> str:
        """Creates a new function for replacing the original one.

        This function decides at runtime if the static (original)
        version of the function should be called, or the new dynamic
        one. This decision is made based of the additional information
        appended to the coefficients array.

        Returns:
            str: New generated code.
        """

        code_impl = "{\n"

        dtype_str = dtype_to_C_str(self._data.dtype)
        coeffs_offset = self._compute_coeffs_offset()
        code_impl += (
            "const ptrdiff_t w_custom_offset = *((const ptrdiff_t *) " + f"&w[{coeffs_offset}]);\n"
        )

        code_impl += "const bool is_custom = w_custom_offset > 0;\n"
        code_impl += "const bool is_full   = w_custom_offset < 0;\n\n"
        code_impl += "if (is_custom)\n"
        code_impl += "{\n"

        code_impl += f"  const {dtype_str}* restrict w_custom = "
        offset_str = "w + w_custom_offset"
        if self._data.dtype == self._coeffs_dtype:
            code_impl += offset_str
        else:
            code_impl += f"(const {dtype_str} *) ({offset_str})"
        code_impl += ";\n"

        name = self._integral_name
        code_impl += (
            f"  tabulate_tensor_integral_{name}_custom(A, w, w_custom, c, "
            "coordinate_dofs, entity_local_index, quadrature_permutation, custom_data);\n"
        )
        code_impl += "}\nelse if (is_full)\n{\n"
        code_impl += (
            f"  tabulate_tensor_integral_{name}_original(A, w, c, "
            "coordinate_dofs, entity_local_index, quadrature_permutation, custom_data);\n"
        )
        code_impl += "}\n}\n"

        return self._func_signt + "\n" + code_impl

    def create_new_code(self) -> str:
        """Creates a modification of the original function's C code by
        creating the posibility of using either the statically defined
        weights and FE tables, or loading them dynamically at certain
        cells (with weights and FE tables that are different for each
        cell or facet).

        Returns:
            str: Newly generated code.
        """

        new_code = ""
        new_code += self._before_func
        new_code += self._create_shim_decls()
        new_code += self._create_non_custom_quad_function()
        new_code += self._create_custom_data_loaders()
        new_code += self._create_custom_function()
        new_code += self._create_new_function()
        new_code += self._after_func

        return new_code


def generate_code(
    ufl_data: UFLData,
    ir: DataIR,
    ffcx_options: dict[str, int | float | npt.DTypeLike],
) -> tuple[CodeBlocks, list[IntegralData]]:
    """Generates code blocks from FFCx intermediate representation and
    modifies integrals code for accommodating the use of custom
    quadrature rules defined cell by cell (or facet by facet) at
    runtime.

    It also returns a list of integral data containing the required
    information for generating the custom coefficients required at
    runtime by the created integrals.

    Note:
        This function is an alternative to the original FFCx function
        `ffcx.codegeneration.codegeneration.generate_code`.
        Indeed, it first calls that function and then modifies the
        generated code for the integrals.

    Args:
        ufl_data (UFLData): UFL (analysis) data structure holding,
            among others, the UFL form data objects.
        ir (DataIR): FFCx Intermediate Representation that contains
            elements, forms, coordinate mappings, and integrals.
        ffcx_options (dict[str, int | float | npt.DTypeLike]): FFCx
            options for generating the code.

    Returns:
        tuple[CodeBlocks, list[IntegralData]]: First: generated
        blocks of code. Second: integral data containing the required
        information for generating the custom coefficients required at
        runtime by the created integrals.
    """

    code_blocks = ffcx.codegeneration.codegeneration.generate_code(ir, ffcx_options)
    code_blocks = _modify_header(code_blocks)

    # In FEniCSx 0.10.0, ffcx emits one code block per (integral, cell_type)
    # pair, so the lists may diverge when an integral spans multiple cell
    # types. qugar currently supports a single cell type per integral, so the
    # lengths must match for the index pairing below to be valid.
    assert len(code_blocks.integrals) == len(ir.integrals), (
        "qugar requires one cell type per integral; multi-cell-type integrals "
        "(e.g. prism/pyramid facets) are not yet supported."
    )

    itg_datas = []
    for i, (header, impl) in enumerate(code_blocks.integrals):
        # Note that the integrals follow the same ordering as the
        # code blocks. See
        # ffcx.codegeneration.codegeneration.generate_code
        itg_ir = ir.integrals[i]
        itg_data = extract_integral_data(ufl_data, ir, itg_ir, ffcx_options, impl)
        itg_datas.append(itg_data)

        itg_mod = _IntegralModifier(impl, itg_ir, itg_data)
        new_impl = itg_mod.create_new_code()

        code_blocks.integrals[i] = (header, new_impl)

    return code_blocks, itg_datas
