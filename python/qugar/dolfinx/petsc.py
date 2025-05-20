# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# This file is a modification of the original ``dolfinx/python/dolfinx/fem/petsc.py`` file.
# See copyright below.
#
# Copyright (C) 2018-2025 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# --------------------------------------------------------------------------


"""Linear problem class based in PETSc objects for variational forms
built on top of unfitted domains.
"""

import typing

from petsc4py import PETSc

# ruff: noqa: E402
import dolfinx
import numpy.typing as npt

assert dolfinx.has_petsc4py  # type: ignore

import dolfinx.fem
import dolfinx.fem.petsc

# import dolfinx.la
import ufl
from dolfinx.fem import IntegralType  # type: ignore
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.petsc import create_matrix, create_vector
from dolfinx.la import create_petsc_vector_wrap

from qugar.dolfinx import CustomForm, form_custom
from qugar.mesh import UnfittedDomain


def _pack_coefficients(
    form: CustomForm | list[CustomForm],
) -> (
    dict[tuple[IntegralType, int], npt.NDArray] | list[dict[tuple[IntegralType, int], npt.NDArray]]
):
    """Packs coefficients for the given custom forms().
    Args:
        form (CustomForm | list[CustomForm]): The custom form (or
            list of forms) whose coefficients are to be packed.
    Returns:
        dict[tuple[IntegralType, int], npt.NDArray] | list[dict[tuple[IntegralType, int], npt.NDArray]]:
            The packed coefficients.
    """  # noqa: E501

    def _pack(form):
        if isinstance(form, CustomForm):
            return form.pack_coefficients()
        else:
            return list(
                map(
                    lambda sub_form: typing.cast(
                        dict[tuple[IntegralType, int], npt.NDArray], _pack(sub_form)
                    ),
                    form,
                )
            )

    return _pack(form)


def _update_coefficients(
    form: CustomForm | list[CustomForm],
    old_coeffs: dict[tuple[IntegralType, int], npt.NDArray]
    | list[dict[tuple[IntegralType, int], npt.NDArray]],
) -> (
    dict[tuple[IntegralType, int], npt.NDArray] | list[dict[tuple[IntegralType, int], npt.NDArray]]
):
    """Packs coefficients for the given custom forms().
    Args:
        form (CustomForm | list[CustomForm]): The custom form (or
            list of forms) whose coefficients are to be packed.
        old_coeffs (dict[tuple[IntegralType, int], npt.NDArray] | list[dict[tuple[IntegralType, int], npt.NDArray]]):
            The old coefficients to be updated.
    Returns:
        dict[tuple[IntegralType, int], npt.NDArray] | list[dict[tuple[IntegralType, int], npt.NDArray]]:
            The packed coefficients.
    """  # noqa: E501

    def _update(form):
        if isinstance(form, CustomForm):
            return form.update_coefficients(old_coeffs)
        else:
            return list(
                map(
                    lambda sub_form: typing.cast(
                        dict[tuple[IntegralType, int], npt.NDArray], _update(sub_form)
                    ),
                    form,
                )
            )

    return _update(form)


class LinearProblem(dolfinx.fem.petsc.LinearProblem):
    """Class for solving a linear variational problem.

    Analogous to DOLFINx' LinearProblem, but for unfitted domains.

    Solves of the form :math:`a(u, v) = L(v) \\,  \\forall v \\in V`
    using PETSc as a linear algebra backend.
    """

    def __init__(
        self,
        a: ufl.Form,
        L: ufl.Form,
        bcs: list[DirichletBC] = [],
        u: typing.Optional[_Function] = None,
        petsc_options: typing.Optional[dict] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for a linear variational problem defined
        over an unfitted domain.

        Args:
            a (ulf.Form): A bilinear UFL form, the left hand side of the
                variational problem. This form must be defined over
                an uniftted domain.
            L (ulf.Form): A linear UFL form, the right hand side of the variational
                problem. This form must be defined over
                an uniftted domain.
            bcs (list[DirichletBC]): A list of Dirichlet boundary conditions.
            u (typing.Optional[_Function]): The solution function.
                It will be created if not provided.
            petsc_options (typing.Optional[dict]): Options that are passed
                to the linear algebra backend PETSc. For available choices
                for the 'petsc_options' kwarg, see the `PETSc documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
            form_compiler_options (typing.Optional[dict]): Options used
                in FFCx compilation of this form. Run ``ffcx --help``
                at the commandline to see all available options.
            jit_options (typing.Optional[dict]): Options used in
                CFFI JIT compilation of C code generated by FFCx
                See DOLFINx's `python/dolfinx/jit.py` for all available
                options. Takes priority over all other option values.

        Example::

            problem = LinearProblem(a, L, [bc0, bc1], petsc_options={"ksp_type": "preonly",
                                                                     "pc_type": "lu",
                                                                     "pc_factor_mat_solver_type":
                                                                       "mumps"})
        """

        funtion_space = a.arguments()[-1].ufl_function_space()  # type: ignore
        unf_domain = typing.cast(UnfittedDomain, funtion_space.mesh)

        _a = form_custom(
            a,
            dtype=PETSc.ScalarType,  # type: ignore
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        self._a = typing.cast(CustomForm | list[CustomForm], _a)

        self._A = create_matrix(self._a)  # type: ignore
        _L = form_custom(
            L,
            dtype=PETSc.ScalarType,  # type: ignore
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        self._L = typing.cast(CustomForm | list[CustomForm], _L)

        self._b = create_vector(self._L)  # type: ignore

        if u is None:
            # Extract function space from TrialFunction (which is at the
            # end of the argument list as it is numbered as 1, while the
            # Test function is numbered as 0)
            self.u = typing.cast(_Function, _Function(funtion_space))
        else:
            self.u = u

        self._x = create_petsc_vector_wrap(self.u.x)
        self.bcs = bcs

        self._solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self._solver.setOperators(self._A)

        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        opts = PETSc.Options()  # type: ignore
        opts.prefixPush(problem_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(problem_prefix)
        self._A.setFromOptions()
        self._b.setOptionsPrefix(problem_prefix)
        self._b.setFromOptions()

    def solve(self) -> _Function:
        """Solve the problem."""

        # Assemble lhs
        self._A.zeroEntries()

        A_coeffs = _pack_coefficients(self._a)
        dolfinx.fem.petsc.assemble_matrix(self._A, self._a, bcs=self.bcs, coeffs=A_coeffs)
        self._A.assemble()

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        b_coeffs = _pack_coefficients(self._L)
        dolfinx.fem.petsc.assemble_vector(self._b, self._L, coeffs=b_coeffs)

        # Apply boundary conditions to the rhs
        A_coeffs_lift = A_coeffs if isinstance(A_coeffs, list) else [A_coeffs]
        dolfinx.fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs], coeffs=A_coeffs_lift)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        for bc in self.bcs:
            bc.set(self._b.array_w)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()

        return self.u


class NonlinearProblem(dolfinx.fem.petsc.NonlinearProblem):
    """Nonlinear problem class for solving the non-linear problems.

    Solves problems of the form :math:`F(u, v) = 0 \\ \\forall v \\in V` using
    PETSc as the linear algebra backend.
    """

    def __init__(
        self,
        F: ufl.form.Form,
        u: _Function,
        bcs: list[DirichletBC] = [],
        J: ufl.form.Form = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for solving a non-linear problem using Newton's method`.

        Args:
            F: The PDE residual F(u, v)
            u: The unknown
            bcs: List of Dirichlet boundary conditions
            J: UFL representation of the Jacobian (Optional)
            form_compiler_options: Options used in FFCx
                compilation of this form. Run ``ffcx --help`` at the
                command line to see all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See ``python/dolfinx/jit.py``
                for all available options. Takes priority over all
                other option values.

        Example::

            problem = LinearProblem(F, u, [bc0, bc1])
        """

        funtion_space = F.arguments()[-1].ufl_function_space()  # type: ignore
        unf_domain = typing.cast(UnfittedDomain, funtion_space.mesh)

        _L = form_custom(
            F,
            dtype=PETSc.ScalarType,  # type: ignore
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        self._L = typing.cast(CustomForm | list[CustomForm], _L)

        # Create the Jacobian matrix, dF/du
        if J is None:
            V = u.function_space
            du = ufl.TrialFunction(V)
            J = ufl.derivative(F, u, du)

        _a = form_custom(
            J,
            dtype=PETSc.ScalarType,  # type: ignore
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        self._a = typing.cast(CustomForm | list[CustomForm], _a)

        self.bcs = bcs

        self._A_coeffs = None
        self._b_coeffs = None

    def _get_b_coeffs(
        self,
    ) -> (
        dict[tuple[IntegralType, int], npt.NDArray]
        | list[dict[tuple[IntegralType, int], npt.NDArray]]
    ):
        """Get the coefficients for the residual form."""
        if self._b_coeffs is None:
            self._b_coeffs = _pack_coefficients(self._L)
        else:
            self._b_coeffs = _update_coefficients(self._L, self._b_coeffs)
        return self._b_coeffs

    def _get_A_coeffs(
        self,
    ) -> (
        dict[tuple[IntegralType, int], npt.NDArray]
        | list[dict[tuple[IntegralType, int], npt.NDArray]]
    ):
        """Get the coefficients for the Jacobian form."""
        if self._A_coeffs is None:
            self._A_coeffs = _pack_coefficients(self._a)
        else:
            self._A_coeffs = _update_coefficients(self._a, self._A_coeffs)
        return self._A_coeffs

    def F(self, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """Assemble the residual F into the vector b.

        Args:
            x: The vector containing the latest solution
            b: Vector to assemble the residual into
        """
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0)

        b_coeffs = self._get_b_coeffs()
        dolfinx.fem.petsc.assemble_vector(b, self._L, coeffs=b_coeffs)

        # Apply boundary condition
        A_coeffs = self._get_A_coeffs()
        A_coeffs_lift = A_coeffs if isinstance(A_coeffs, list) else [A_coeffs]
        dolfinx.fem.petsc.apply_lifting(
            b, [self._a], bcs=[self.bcs], x0=[x], alpha=-1.0, coeffs=A_coeffs_lift
        )
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Args:
            x: The vector containing the latest solution
        """

        A.zeroEntries()
        A_coeffs = self._get_A_coeffs()
        dolfinx.fem.petsc.assemble_matrix(A, self._a, bcs=self.bcs, coeffs=A_coeffs)
        A.assemble()
