# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Test for the evaluation and gradient of implicit functions."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


import numpy as np
import pytest
from utils import (
    dtypes,  # type: ignore
)

import qugar.cpp
import qugar.impl


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n_test_pts", [1000])
@pytest.mark.parametrize("dtype", dtypes)
def test_evaluation(dim: int, n_test_pts: int, dtype: type[np.float32 | np.float64]):
    """
    Test the evaluation of the Schoen function using the qugar implementation.
    This test checks if the evaluation of the Schoen function using the qugar
    implementation matches the expected values computed using numpy.

    Args:
        dim (int): Dimension of the space (2 or 3).
        n_test_pts (int): Number of test points to evaluate.
        dtype (type): Data type for the test points (float32 or float64).

    Raises:
        AssertionError: If the evaluation of the Schoen function does not match
            the expected values within the specified tolerance.
    """

    func = qugar.impl.create_Schoen(np.ones(dim, dtype=dtype))

    def schoen(pts):
        if dim == 3:
            pts_ = pts
        else:
            assert dim == 2
            pts_ = np.zeros((pts.shape[0], 3), dtype=dtype)
            pts_[:, :2] = pts

        return (
            np.sin(2.0 * np.pi * pts_[:, 0]) * np.cos(2.0 * np.pi * pts_[:, 1])
            + np.sin(2.0 * np.pi * pts_[:, 1]) * np.cos(2.0 * np.pi * pts_[:, 2])
            + np.sin(2.0 * np.pi * pts_[:, 2]) * np.cos(2.0 * np.pi * pts_[:, 0])
        )

    test_pts = np.random.rand(n_test_pts, dim).astype(dtype)

    atol = np.sqrt(np.finfo(dtype).eps)  # type: ignore
    assert np.isclose(func.eval(test_pts), schoen(test_pts), atol=atol).all()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n_test_pts", [1000])
@pytest.mark.parametrize("dtype", dtypes)
def test_gradient(dim: int, n_test_pts: int, dtype: type[np.float32 | np.float64]):
    """Test the gradient of the Schoen function using the qugar implementation.
    This test checks if the gradient of the Schoen function using the qugar
    implementation matches the expected values computed using numpy.

    Args:
        dim (int): Dimension of the space (2 or 3).
        n_test_pts (int): Number of test points to evaluate.
        dtype (type): Data type for the test points (float32 or float64).

    Raises:
        AssertionError: If the evaluation of the Schoen function gradient
            does not match the expected values within the specified tolerance.
    """
    func = qugar.impl.create_Schoen(np.ones(dim, dtype=dtype))

    def schoen_grad(pts):
        if dim == 3:
            pts_ = pts
        else:
            assert dim == 2
            pts_ = np.zeros((pts.shape[0], 3), dtype=dtype)
            pts_[:, :2] = pts

        grads = np.zeros((pts.shape[0], dim), dtype=dtype)
        grads[:, 0] = np.cos(2.0 * np.pi * pts_[:, 0]) * np.cos(2.0 * np.pi * pts_[:, 1]) - np.sin(
            2.0 * np.pi * pts_[:, 0]
        ) * np.sin(2.0 * np.pi * pts_[:, 2])
        grads[:, 1] = np.cos(2.0 * np.pi * pts_[:, 1]) * np.cos(2.0 * np.pi * pts_[:, 2]) - np.sin(
            2.0 * np.pi * pts_[:, 1]
        ) * np.sin(2.0 * np.pi * pts_[:, 0])

        if dim == 3:
            grads[:, 2] = np.cos(2.0 * np.pi * pts_[:, 2]) * np.cos(
                2.0 * np.pi * pts_[:, 0]
            ) - np.sin(2.0 * np.pi * pts_[:, 2]) * np.sin(2.0 * np.pi * pts_[:, 1])

        grads *= 2.0 * np.pi

        return grads

    test_pts = np.random.rand(n_test_pts, dim).astype(dtype)

    atol = np.sqrt(np.finfo(dtype).eps)  # type: ignore
    assert np.isclose(func.grad(test_pts), schoen_grad(test_pts), atol=atol).all()


if __name__ == "__main__":
    test_evaluation(2, 1000, np.float32)
    test_gradient(2, 1000, np.float32)
    test_evaluation(3, 1000, np.float32)
    test_gradient(3, 1000, np.float32)
    test_evaluation(2, 1000, np.float64)
    test_gradient(2, 1000, np.float64)
    test_evaluation(3, 1000, np.float64)
    test_gradient(3, 1000, np.float64)
