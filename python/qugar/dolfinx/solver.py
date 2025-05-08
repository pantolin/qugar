# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"Helper functions for simplifying the solution of linear systems using PETSc."

import importlib.util

if importlib.util.find_spec("petsc4py") is None:
    raise ValueError("petsc4py installation not found is required.")

from petsc4py import PETSc
from petsc4py.PETSc import Mat, Vec


def _create_Jacobi_preconditioner(A: Mat, symmetric: bool = True) -> Vec:
    """Compute the Jacobi preconditioner for a given matrix `A`.

    The Jacobi preconditioner is a diagonal matrix with the
    diagonal entries being the inverse of the square root of the
    absolute values of the diagonal entries of the matrix `A`.

    Args:
        A (Mat): Matrix from which to compute the Jacobi
            preconditioner.
        symmetric (bool): If `True`, the preconditioner is
            computed symmetrically (i.e., `Pinv = diag(1/sqrt(abs(diag(A))))`).
            Otherwise, it is computed only for the left side
            (i.e., `Pinv = diag(1/abs(diag(A)))`).

    Raises:
        e:  Error if the diagonal of the matrix is zero or negative.
            The Jacobi preconditioner is not defined for
            matrices with zero or negative diagonal entries.

    Returns:
        Vec: Vector containing the Jacobi preconditioner (only the
            matrix diagonal).
    """

    # 1. Create Pinv from the diagonal entries of A
    D_vec = A.getDiagonal()
    Pinv = D_vec.copy()

    # 2. Compute Pinv = sqrt(abs(D_vec)) if symmetric,
    #    Pinv = abs(D_vec) otherwise.
    Pinv.sqrtabs() if symmetric else Pinv.abs()

    # 3. Compute Pinv = 1 / Pinv (entry wise)
    try:
        Pinv.reciprocal()
    except PETSc.Error as e:
        if e.ierr == 73:  # PETSC_ERR_FP_DIVIDEBYZERO
            PETSc.Sys.Print(
                "Error: Division by zero encountered. "
                "Ensure A has non-zero diagonal entries for this scaling.",
                comm=A.comm,
            )
            exit(1)
        else:
            raise e

    return Pinv


def _precondition_matrix_Jacobi(A: Mat, Pinv: Vec, symmetric: bool) -> None:
    """Preconditions the matrix `A` in place using the diagonal Jacobi
    preconditioner.

    The preconditioner is applied symmetrically if `symmetric` is
    `True`, i.e., `A_prime = Pinv * A * Pinv`. Otherwise, it is applied
    only to the left, i.e., `A_prime = Pinv * A`.

    Args:
        A (Mat): Matrix to be preconditioned.
        Pinv (Vec): Diagonal Jacobi preconditioner (only the diagonal
            vector). It contains the inverse of the square root of the
            diagonal entries of the matrix `A`.
        symmetric (bool): If `True`, the preconditioner is applied
            symmetrically (left and right) to the matrix. Otherwise,
            only the left preconditioner is applied.
    """
    A.diagonalScale(L=Pinv, R=Pinv) if symmetric else A.diagonalScale(L=Pinv)
    return A


def _precondition_vector_Jacobi(b: Vec, Pinv: Vec) -> None:
    """Preconditions the right-hand-side vector `b` in place using the
    diagonal Jacobi preconditioner as `b[i] = Pinv[i] * b[i]`.

    Args:
        b (Vec): Right-hand-side vector to be preconditioned.
        Pinv (Vec): Diagonal Jacobi preconditioner (only the diagonal
            vector). It contains the inverse of the square root of the
            diagonal entries of the matrix `A`.
    """
    b.pointwiseMult(b, Pinv)  # b_prime[i] = Pinv[i] * b[i]


def _create_direct_solver(A: Mat, use_Cholesky: bool) -> PETSc.KSP:
    """Creates a direct solver for the linear system Ax = b.

    The solver is a Krylov subspace method that that only applies
    a preconditioner to the system. The preconditioner is
    either LU or Cholesky factorization, depending on the
    `use_Cholesky` argument.

    Args:
        A (Mat): Matrix representing the linear system.
        use_Cholesky (bool): Whether to use Cholesky factorization or
            LU factorization for the solver.

    Returns:
        PETSc.KSP: Generated solver.
    """
    solver = PETSc.KSP().create(comm=A.comm)
    solver.setOperators(A)

    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc_type = PETSc.PC.Type.CHOLESKY if use_Cholesky else PETSc.PC.Type.LU
    pc.setType(pc_type)

    return solver


def direct_solve_with_Jacobi(
    A: Mat,
    b: Vec,
    x: Vec,
    in_place_pc: bool = True,
    use_Cholesky: bool = False,
    symmetric_pc: bool = True,
    set_diagonal_zeros_to_one: bool = True,
) -> None:
    """
    Solve the linear system `Ax = b` using a direct solver with Jacobi preconditioner.

    Note:
        This function does not call the `scatter_forward()` method on
        the solution vector `x`. It is the caller's responsibility to do it.

    Args:
        A (Mat): PETSc's matrix representing the linear system.
        b (Vec): PETSc's right-hand side vector.
        x (Vec): PETSc's vector to store the solution (already created)
        in_place_pc (bool): If `True`, the preconditioner is applied in
            place, modifying the passed matrix and vector. Otherwise,
            copies of the matrix and vector are created for
            preconditioning.
        use_Cholesky (bool): If `True`, use Cholesky factorization for
            the solver. Otherwise, uses LU factorization. The Cholesky
            solver is only applicable for symmetric positive definite
            matrices. It is the caller's responsibility to ensure that.
        symmetric_pc (bool): If `True`, the preconditioner is applied
            symmetrically (left and right) to the matrix and vector.
            Otherwise, only the left preconditioner is applied.
        set_diagonal_zeros_to_one (bool): If `True`, set the diagonal
            entries of the preconditioned matrix to one if they are zero.
            This is useful in cases in which empty cells not contributing
            to the matrix are not removed, and therefore the contribution
            of some basis functions to the matrix is zero.
    """

    # 1. Creating copies (if needed).
    A = A if in_place_pc else A.copy()
    b = b if in_place_pc else b.copy()

    # 2. Set zeros to one if set_diagonal_zeros_to_one is True
    if set_diagonal_zeros_to_one:
        Adiag = A.getDiagonal()
        A.setDiagonal(Adiag * (Adiag != 0) + (Adiag == 0))

    # 3. Create Jacobi preconditioner
    Pinv = _create_Jacobi_preconditioner(A, symmetric_pc)

    # 3. Create A = Pinv * A * Pinv
    _precondition_matrix_Jacobi(A, Pinv, symmetric_pc)

    # 4 Create b = Pinv * b (if Jacobi preconditioner is symmetric)
    if symmetric_pc:
        _precondition_vector_Jacobi(b, Pinv)

    # 5. Solve A * x = b
    solver = _create_direct_solver(A, use_Cholesky)
    solver.solve(b, x)

    # 6. Transform back the solution if symmetric preconditioner
    # was applied: Pinv * x
    if symmetric_pc:
        _precondition_vector_Jacobi(x, Pinv)
