# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Data structure for storing finite element basis functions (and
derivatives) values, plus associated information."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import hashlib
import re

import numpy as np
import numpy.typing as npt
import ufl
from basix.ufl import _BasixElement as BasixElement
from ffcx.ir.analysis.modified_terminals import (
    analyse_modified_terminal,
    is_modified_terminal,
)
from ffcx.ir.elementtables import (
    default_atol,
    default_rtol,
    equal_tables,
    get_modified_terminal_element,
)
from ffcx.ir.representation import IntegralIR

from qugar.dolfinx.parsing_utils import parse_real_dtype_from_C
from qugar.dolfinx.quadrature_data import QuadratureData


class FETable:
    """Class for storing a table of finite element basis values (or
    derivatives) as defined in a tabulate tensor integral.

    Parameters:
        pattern (str): Regex used for recognizing the definition of
            basis functions tables in a C code string.
        _code (str): C code containing at least one definition of a
            table.
        _dtype (type[np.float32 | np.float64]): `numpy` data type of the
            basis functions.
        _FE_id (int): Id of the table. A non negative number.
        _component (int): Basis component of the basis function the
            table refers to. A non negative number.
        _derivatives (tuple[int,...]): Order of the derivatives along
            the parametric direction of the basis function values in the
            table. If 0, no derivative is computed along that specific
            direction.
        _avg (Optional[str]): String indicating if the basis functions
            were averaged. If None, no average was computed. Otherwise,
            it can be 'cell', 'facet', or 'vertex' (the latter is not
            supported).
        _table_type (str): Type of table. Possible values are 'zeros',
            'ones', 'quadrature', 'fixed', 'piecewise', 'uniform', or
            'varying'. So far, only 'fixed', 'uniform', and 'varying'
            are supported.
        _integral_type (str): Integral type. It may be 'exterior_facet',
            'interior_facet', or 'cell'.
        _entity_type (str): Entity type. It may be 'vertex', 'facet', or
            'cell'. 'vertex' is not supported.
        _name (str): Name of the table following ffcx notation.
            Something like ``FE#_C#_D###[_AC|_AF|][_F|V][_Q#]``.
            See ``_parse_FE_extra_options`` documentation for further
            details.
        _quad_data (QuadratureData): Data of the quadrature used for
            creating the table.
        _permutations (int): Number of permutations in the table.
            Size of the table along the first axis.
        _entities (int): Number of entities in the table.
            Size of the table along the second axis.
        _points (int): Number of points in the table.
            Size of the table along the third axis.
        _funcs (int): Number of basis functions in the table.
            Size of the table along the fourth axis.
        _values (npt.NDArray[np.float32 | np.float64]): Array of basis
            functions values. It is a 4-dimensional (permutations,
            entities, points, funcs) with type `self._dtype`.
        _element (Element): Basix element used for creating the table.
    """

    pattern: str = (
        r"static\s+const\s+(.+)\s+FE(\d+)_C(\d+)(.*)_Q(\w+)"
        + r"\s*\[\s*(\d+)\s*\]" * 4
        + r"\s*=\s*{([^;]*)}\s*;"
    )

    def __init__(
        self,
        code: str,
        itg_tdim: int,
        integral_type: str,
        entity_type: str,
        is_mixed_dim: bool,
        table_types: dict[str, str],
        all_quads_data: dict[str, QuadratureData],
    ):
        """Constructor.

        Given a piece of C `code` containing the definition of at least
        one basis functions table, finds the first table occurrence and
        extracts the associated information.

        Args:
            code (str): C code containing at least one definition of a
                basis functions table.
            itg_tdim (int): Topological dimension of the integral
                to which the table refers to. E.g., dimension 2 for
                integrals over triangles or quadrilaterals, and
                dimension 3 in the case of tetrahedra, hexahedra, etc.
            integral_type (str): Integral type. It must be 'cell',
                'exterior_facet', or 'interior_facet'.
            entity_type (str): Entity type. It must be 'cell' or
                'facet'. 'vertex' is not supported.
            is_mixed_dim (bool): Flag indicating if the integral to
                which the element table is associated to presents mixed
                dimensions or not.
            table_types (dict[str, str]): Dictionary mapping the
                table names to the table types.
            all_quads_data (dict[str, QuadratureData]): Dictionary of
                quadrature data (quadrature names to quadrature data
                objects).
        """

        assert 1 <= itg_tdim and itg_tdim <= 3

        assert integral_type in ["cell", "exterior_facet", "interior_facet"]

        assert entity_type in ["cell", "facet"]

        self._integral_type = integral_type
        self._entity_type = entity_type
        self._avg = None

        self._parse_code(code)

        self._table_type = table_types.get(self._name)
        assert self._table_type in [
            "uniform",
            "varying",
            "fixed",
            "piecewise",
        ]

        self._quad_data = all_quads_data[self._quad_name]

        self._values = self._extract_values()
        # self._element = self._find_element(candidate_elements, is_mixed_dim, itg_tdim)

    def _parse_code(self, code: str):
        """Parses the first occurrence of a FE table in the given
        C `code`.

        It initializes the class members `_code`, `_name`, `_quad_name`,
        `_dtype`, `_FE_id`, `_component`, `_permutations`, `_entities`,
        `_points`, `_funcs`, `_avg`, `_derivatives`, and `_entity_type`.

        Args:
            code (str): C code containing at least one definition of a
                basis functions table to be parsed.
        """

        match = re.search(FETable.pattern, code)
        assert match

        self._code = match.group(0)

        self._name = self._parse_FE_name()
        self._quad_name = match.group(5)
        self._dtype = parse_real_dtype_from_C(match.group(1))

        self._FE_id = int(match.group(2))
        self._component = int(match.group(3))

        other_opts = match.group(4)
        self._parse_FE_extra_options(other_opts)

        self._permutations = int(match.group(6))
        self._entities = int(match.group(7))
        self._points = int(match.group(8))
        self._funcs = int(match.group(9))

    def _parse_FE_name(self) -> str:
        """Parses the FE table name from the stored C `code`.

        Returns:
            str: FE table name with a format like
            ``FE#_C#_D###[_AC|_AF|][_F|V][_Q#]``. See
            ``FETable._parse_FE_extra_options`` documentation for further
            details.
        """

        name_pattern = r"const\s+\w+\s+FE([\w|\_]*)\s*\["
        match = re.search(name_pattern, self.code)
        assert match

        name = "FE" + match.group(1)
        return name

    def _extract_values(self) -> npt.NDArray[np.float32 | np.float64]:
        """Extracts the table values from `self._code`.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Extracted FE table
            values stored in a 4-dimensional array.
        """

        match = re.search(FETable.pattern, self._code)
        assert match

        values_str = match.group(10).replace(r"{", "").replace("}", "").replace(",", " ")

        shape = (self._permutations, self._entities, self._points, self._funcs)
        vals = [self.dtype(i) for i in values_str.split()]  # type: ignore
        values = np.array(vals, dtype=self.dtype).reshape(shape)
        return values

    @property
    def dtype(self) -> type[np.float32 | np.float64]:
        """Returns the `numpy` number type of FE table values.

        Returns:
            type[np.float32 | np.float64]: `numpy` dtype of FE table values.
        """
        return self._dtype

    def is_constant_for_pts(self) -> bool:
        """Checks if the table is constant for all the points.

        I.e., it is 'piecewise' or 'fixed' (i.e., both 'piecewise' and
        'uniform'), 'zeros', or 'ones'.

        If the table is 'piecewise' or 'fixed', the values are constant
        for all the points in each entity or for all the points of all
        the entities, respectively. If the type is 'zeros' or 'ones',
        all the values in the table are either 0 or 1, respectively.
        Therefore, from the point of view of custom quadratures, the
        values of these tables will not change.

        Returns:
            bool: Whether the table is constant for all the points.
        """
        n_quad_pts = self._quad_data.rule.points.shape[0]  # type: ignore
        return n_quad_pts > 1 and self._table_type in [
            "fixed",
            "piecewise",
            "zeros",
            "ones",
        ]

    @property
    def FE_id(self) -> int:
        """Returns the id of the table."""
        return self._FE_id

    @property
    def name(self) -> str:
        """Returns the full name of the table."""
        return self._name

    @property
    def quad_name(self) -> str:
        """Returns the name of the quadrature associated to the
        table."""
        return self._quad_data.name

    @property
    def quad_data(self) -> QuadratureData:
        """Returns the data of the quadrature associated to the
        table."""
        return self._quad_data

    @property
    def entity(self) -> str:
        """Returns the entity type of the table."""
        return self._entity_type

    @property
    def integral_type(self) -> str:
        """Returns the integral type of the table."""
        return self._integral_type

    @property
    def derivatives(self) -> tuple[int, ...]:
        """Returns the order of the basis function derivatives along the
        parametric directions. If 0, no derivative is computed along
        that specific direction.
        """
        return self._derivatives

    @property
    def avg(self) -> str | None:
        """Returns the average type of the table, if any."""
        return self._avg

    @property
    def element(self) -> BasixElement:
        """Returns the element associated to the FE table."""
        return self._element

    @property
    def element_dim(self) -> int:
        """Returns the dimension of the element associated to the FE
        table."""
        return self.element.cell.topological_dimension()

    @property
    def component(self) -> int:
        """Returns the component of the table's element."""
        return self._component

    @property
    def code(self) -> str:
        """Returns the original C code of the table."""
        return self._code

    @property
    def permutations(self) -> int:
        """Returns the number of permutations in the table."""
        return self._permutations

    @property
    def entities(self) -> int:
        """Returns the number of entities in the table."""
        return self._entities

    @property
    def points(self) -> int:
        """Returns the number of points in the table."""
        return self._points

    @property
    def funcs(self) -> int:
        """Returns the number of functions in the table."""
        return self._funcs

    @property
    def is_permuted(self) -> bool:
        """Returns whether if the values table present permutations.
        I.e., its first dimension is greater than 1."""
        return self._permutations > 1

    @property
    def values(self) -> npt.NDArray:
        """Returns the table values."""
        return self._values

    def _parse_FE_extra_options(self, opts_str: str):
        """Extracts part of the information encoded in the name of a FE
        basis table.

        The format of the basis table naming is
        ``FE#_C#[_D###][_AC|_AF|][_F|V]_Q#``, where ``#`` will be an
        integer value (or characters). Specifically:

        - FE is a simple counter to distinguish the various bases, it
          will be assigned in an arbitrary fashion (not preserved among
          Python sessions).
        - C is the component number if any (this does not yet take into
          account tensor valued functions).
        - D is the number of derivatives along each parametric
          direction. If not present, the table is associated to the
          basis functions values, and not to any derivative.
        - AC marks that the element values are averaged over the cell.
          If not present, no averages are computed.
        - AF marks that the element values are averaged over the facet.
          If not present, no averages are computed.
        - F marks that the first array dimension enumerates facets on
          the cell. If not present, the table corresponds to the cell
          interior.
        - V marks that the first array dimension enumerates vertices on
          the cell. If not present, the table corresponds to the cell
          interior.
        - Q unique ID of quadrature rule, to distinguish between tables
          in a mixed quadrature rule setting. It is a string of 3
          alphanumeric characters.

        This method sets the members `_derivative`, `_avg`, and
        `_entity_type`, if found.

        Args:
            opts_str (str): String containing the part of the name
                associated to ``[_D###][_AC|_AF|][_F|V]``. If empty,
                the associated values are set to default. Otherwise, it
                is assumed that the string starts with ``_``.
        """

        if len(opts_str) == 0:
            return

        assert opts_str[0] == "_"

        for opt in opts_str[1:].split("_"):
            if opt[0] == "D":
                self._derivatives = tuple(int(i) for i in opt[1:])
            elif opt == "AC":
                self._avg = "cell"
                assert self._entity_type == "cell"
            elif opt == "AF":
                self._avg = "facet"
                assert self._entity_type == "facet"
            elif opt == "F":
                assert self._entity_type == "facet"
            elif opt == "V":
                assert self._entity_type == "vertex"
            else:
                assert False, f"Invalid finite element option: {opt}."

    def _set_element(self, element: BasixElement) -> None:
        """Sets the element associated to the table.

        This method is used for setting the element after the table
        has been created. If not previously set, it also sets the
        derivates of the table to 0.

        Args:
            element (BasixElement): Element associated to the table.
        """
        self._element = element
        elem_dim = element.cell.topological_dimension()
        self._set_derivatives(elem_dim)

    def _set_derivatives(self, elem_dim: int) -> None:
        """Sets the derivatives of the table.
        If the derivatives are not set, it sets them to 0.
        Args:
            elem_dim (int): Element dimension.
        """
        if not hasattr(self, "_derivatives") or all(der == 0 for der in self._derivatives):
            self._derivatives = (0,) * elem_dim

    def _find_element(
        self, candidate_elements: list[BasixElement], is_mixed_dim: bool, itg_tdim: int
    ) -> BasixElement:
        """Finds the element associated to the current FE basis table
        among a list of `candidate_elements`.

        Among all the possible candidates, it makes a short list of the
        ones whose dimension (number of basis functions) and number of
        components match the requirements of the current table.

        Using this short list, new tables are created using the table's
        quadrature. The values of the generated tables are compared to
        ``self._values`` until a match is found.

        Args:
            candidate_elements (list[BasixElement]): List of possible
                elements that were used for creating the table.
            is_mixed_dim (bool): Flag indicating if the integral to
                which the element table is associated to presents mixed
                dimensions or not.
            itg_tdim (int): Topological dimension of the integral
                to which the table refers to. E.g., dimension 2 for
                integrals over triangles or quadrilaterals, and
                dimension 3 in the case of tetrahedra, hexahedra, etc.

        Returns:
            BasixElement: Element associated to the FE table.
        """
        assert candidate_elements

        short_list = set()
        for element in candidate_elements:
            if (
                self._component < element.reference_value_size
                and element.dim / element.block_size == self._funcs
            ):
                short_list.add(element)
        assert short_list

        def set_derivatives(elem_dim: int) -> None:
            if not hasattr(self, "_derivatives") or all(der == 0 for der in self._derivatives):
                self._derivatives = (0,) * elem_dim

        if len(short_list) == 1:
            element = short_list.pop()
            elem_dim = element.cell.topological_dimension()
            set_derivatives(elem_dim)
            assert len(self._derivatives) == elem_dim

            return element

        for element in short_list:
            from qugar.dolfinx.fe_table_eval import evaluate_FE_table

            elem_dim = element.cell.topological_dimension()
            codim = itg_tdim - elem_dim
            assert 0 <= codim <= 1

            set_derivatives(elem_dim)

            if len(self._derivatives) != elem_dim:
                continue

            new_table = evaluate_FE_table(self, element, is_mixed_dim, codim)

            if equal_tables(new_table, self._values, default_rtol, default_atol):
                return element

        assert False, "No valid element found."

    def create_new_access_code(self, indices: tuple[str, ...], has_permutations: bool) -> str:
        """Transforms the 4 indices access to the statically defined FE
        4D table to the access to a plain 1D array (for values
        dynamically loaded from a custom coefficients array passed to a
        custom integral).

        An access like ``FE_C0_D10_Q48f[i][j][k][l]`` will be
        transformed to an access like ``FE_C0_D10_Q48f[index]`` where
        the new ``FE_C0_D10_Q48f`` is the flattened dynamic version of
        the original array.

        In the case the table presents permutations (for interior
        facet integrals), the access will be something like
        ``FE_C0_D10_Q48f[i][index]``.

        Args:
            indices (tuple[str]): Four indices for accessing the
                4-dimensional FE table.
            has_permutations (bool): Flat indicating if permutations
                must be considered (mostly, for interior facets).

        Returns:
            str: Generated C code for accessing the dynamic array.
        """

        assert not self.is_constant_for_pts()

        perm_index = None
        if has_permutations and indices[0] != "0":
            assert self._permutations > 1
            match = re.search(r"quadrature_permutation\[(\d+)\]", indices[0])
            assert match
            perm_index = int(match.group(1))

        def transform_index(index: str) -> int | str:
            index = index.replace(" ", "")
            if index.isdecimal():
                index = int(index)  # type: ignore
            elif not index.isalpha() and index not in ["iq", "ic"]:
                index = "(" + index + ")"
            return index

        terms = []

        # Permutation and entity indices are discarded.
        offsets = [f"{self._funcs} * ", ""]
        for index, offset in zip(indices[2:], offsets):
            new_index = transform_index(index)
            if new_index != 0:
                terms.append(f"{offset}{index}")

        plain_index = " + ".join(terms)

        new_access = f"{self._name}"
        if perm_index is not None:
            new_access += f"[{perm_index}]"
        new_access += f"[{plain_index}]"

        return new_access


def _sort_FE_tables(FE_tables: list[FETable]) -> list[FETable]:
    """Sorts the given list of FE tables such that their ordering
    is preserved between Python sessions independently of their FE
    id (that may not be preserved).

    The sorting is done by hashing the element associated to the table.

    Args:
        FE_tables (list[FETable]): List of tables to sort.

    Returns:
        list[FETable]: Sorted tables list.
    """

    # Creating id to sorted map.
    ids_to_hashes = {}
    for table in FE_tables:
        h = hashlib.sha1(str(table.element).encode())
        ids_to_hashes[table.FE_id] = int(h.hexdigest(), 32)

    sorted_to_id = np.argsort(list(ids_to_hashes.values())).tolist()
    id_to_sorted = {}
    for i, id in enumerate(ids_to_hashes):
        id_to_sorted[id] = sorted_to_id.index(i)

    def tables_sort(table: FETable):
        return id_to_sorted[table.FE_id]

    FE_tables.sort(key=tables_sort)

    return FE_tables


def _parse_quad_name(code: str) -> str:
    """Parses the name of the quadrature associated to the
    FE table defined in the given C `code`.

    The quadrature rule name is extracted from the FE table's
    name.

    Args:
        code (str): C code containing at least one definition of a
            basis functions table.

    Returns:
        str: Quadrature rule name (three alphanumeric characters).
    """
    table_name = _parse_FE_name(code)
    quad_name = table_name.split("Q")[-1]
    return quad_name


def _extract_elements_in_integral(
    ir_itg: IntegralIR,
) -> dict[str, list[BasixElement]]:
    """Extracts all the elements associated to the given integral.

    The main purpose of this function is to obtain all the elements
    associated to the integral, for later on building a one-to-one map
    between the FE tables in the integral function and these elements.

    Returns:
        dict[str, list[BasixElement]]: Dict mapping every quadrature
        (name) used in the integral to the elements using that
        quadrature.
    """

    elements = {}
    for quad, integrand in ir_itg.expression.integrand.items():
        # This code was partially copied from the function
        # ffcx.ir.integral.compute_integral_ir

        mod_terminals = [
            get_modified_terminal_element(analyse_modified_terminal(node["expression"]))
            for node in integrand["factorization"].nodes.values()
            if is_modified_terminal(node["expression"])
        ]

        all_elements = [mt[0] for mt in mod_terminals if mt]
        elements[quad.id()] = ufl.algorithms.sort_elements(  # type: ignore
            set(ufl.algorithms.analysis.extract_sub_elements(all_elements))  # type: ignore # noqa
        )
    return elements


def _check_mixed_dim(tdim, elements) -> bool:
    """Checks if an integral presents mixed dimensions.

    I.e., if the integral domain's dimension do not match with any
    of the elements used in the integral.

    Args:
        tdim (int): Topological dimension of the integral's domain.
            E.g., dimension 2 if the integral is performed inside
            (or on the facet of) triangles or quadrilaterals.
            Dimension 3 in the case of tetrahedra, hexahedra, etc.
        elements (dict[str, list[BasixElement]]): Elements used in the
            integral. It is a dictionary mapping every quadrature (name)
            used in the integral to the elements using that quadrature.

    Returns:
        bool: _description_
    """

    for elems_list in elements.values():
        for elem in elems_list:
            if elem.cell.topological_dimension() != tdim:
                return True

    return False


def _find_element(
    FE_table: FETable, candidate_elements: list[BasixElement], itg_tdim: int, is_mixed_dim: bool
) -> BasixElement:
    """Finds the element associated to the given FE table.

    Args:
        FE_table (FETable): FE table to which the element is
            associated.
        candidate_elements (dict[str, list[BasixElement]]): Elements
            used in the integral. It is a dictionary mapping every
            quadrature (name) used in the integral to the elements
            using that quadrature.
        is_mixed_dim (bool): Flag indicating if the integral to
            which the element table is associated to presents mixed
            dimensions or not.

    Returns:
        BasixElement: Element associated to the FE table.
    """

    assert candidate_elements

    short_list = set()
    for element in candidate_elements:
        if FE_table.component < element.reference_value_size:
            short_list.add(element)
    assert short_list

    if len(short_list) == 1:
        element = short_list.pop()
        elem_dim = element.cell.topological_dimension()
        FE_table._set_derivatives(elem_dim)
        assert len(FE_table.derivatives) == elem_dim

        return element

    for element in short_list:
        from qugar.dolfinx.fe_table_eval import evaluate_FE_table

        elem_dim = element.cell.topological_dimension()
        codim = itg_tdim - elem_dim
        assert 0 <= codim <= 1

        FE_table._set_derivatives(elem_dim)

        if len(FE_table.derivatives) != elem_dim:
            continue

        new_table = evaluate_FE_table(FE_table, element, is_mixed_dim, codim)

        if equal_tables(new_table, FE_table.values, default_rtol, default_atol):
            return element

    assert False, "No valid element found."


def _match_FE_IDs_to_elements(
    FE_tables: list[FETable],
    candidate_elements: dict[str, list[BasixElement]],
    itg_tdim: int,
    is_mixed_dim: bool,
) -> dict[tuple[int, str], BasixElement]:
    """Matches the FE tables to the elements used in the integral.

    The matching is done by first grouping the tables by their id and quadrature.
    Then, for each id and quadrature we keep the table with the highest component.
    Finally, we find the element associated to each table.
    The element is found by comparing the table properties to the
    properties of the candidate elements.

    Args:
        FE_tables (list[FETable]): List of FE tables to match.
        candidate_elements (dict[str, list[BasixElement]]): Elements
            used in the integral. It is a dictionary mapping every
            quadrature (name) used in the integral to the elements
            using that quadrature.
        itg_tdim (int): Topological dimension of the integral's domain.
            E.g., dimension 2 if the integral is performed inside
            (or on the facet of) triangles or quadrilaterals.
            Dimension 3 in the case of tetrahedra, hexahedra, etc.
        is_mixed_dim (bool): Flag indicating if the integral to
            which the element table is associated to presents mixed
            dimensions or not.

    Returns:
        dict[tuple[int,str], BasixElement]: Dictionary mapping the FE
        table id and quadrature name to the element associated to that
        table. The keys are tuples of the form (FE_id, quad_name),
        where FE_id is the id of the table and quad_name is the name
        of the quadrature associated to the table. The values are the
        elements associated to the tables.
    """

    # First we group tables by id.
    tables_by_id_and_quad = {}
    for table in FE_tables:
        key = (table.FE_id, table.quad_name)
        if key not in tables_by_id_and_quad:
            tables_by_id_and_quad[key] = []
        tables_by_id_and_quad[key].append(table)

    # Then, for each id and quad we keep the table with the highest component
    for id_quad, tables in tables_by_id_and_quad.items():
        tables.sort(key=lambda t: t.component, reverse=True)
        tables_by_id_and_quad[id_quad] = tables[0]

    # Finally we find the element associated to each (id,quad) pair.
    return {
        id_quad: _find_element(table, candidate_elements[id_quad[1]], itg_tdim, is_mixed_dim)
        for id_quad, table in tables_by_id_and_quad.items()
    }


def extract_FE_tables(
    code: str,
    ir_itg: IntegralIR,
    all_quads_data: dict[str, QuadratureData],
    itg_tdim: int,
) -> list[FETable]:
    """Parses all the FE tables defined in the tabulate integral
    function stored in C `code`.

    Parameters:
        code (str): C code containing the integral implementation
            from which tables will be extracted.
        ir_itg (IntegralIR): ffcx Intermediate Representation of the
            integral.
        all_quads_data (dict[str, QuadratureData]): Dictionary of
            quadrature data (quadrature names to quadrature data
            objects) for the quadratures in `ir_itg`.
        itg_tdim (int): Topological dimension of the integral's domain.
            E.g., dimension 2 if the integral is performed inside
            (or on the facet of) triangles or quadrilaterals.
            Dimension 3 in the case of tetrahedra, hexahedra, etc.

    Returns:
        list[FETable]: All FE tables found in the given `code`.
        The list ordering guarantees preserves the ordering of the
        tables (despite their non-reproducible id) between Python
        sessions.
    """

    table_types = ir_itg.expression.unique_table_types
    integral_type = ir_itg.expression.integral_type
    entity_type = ir_itg.expression.entity_type
    candidate_elements = _extract_elements_in_integral(ir_itg)

    is_mixed_dim = _check_mixed_dim(itg_tdim, candidate_elements)

    FE_tables = []
    for match in re.finditer(FETable.pattern, code):
        code_block = code[match.start() :]

        FE_tables.append(
            FETable(
                code_block,
                itg_tdim,
                integral_type,
                entity_type,
                is_mixed_dim,
                table_types,
                all_quads_data,
            )
        )

    ids_quads_to_element = _match_FE_IDs_to_elements(
        FE_tables, candidate_elements, itg_tdim, is_mixed_dim
    )

    for FE_table in FE_tables:
        FE_table._set_element(ids_quads_to_element[(FE_table.FE_id, FE_table.quad_name)])

    FE_tables = _sort_FE_tables(FE_tables)
    return FE_tables
