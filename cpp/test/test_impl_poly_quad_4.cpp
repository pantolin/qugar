// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_poly_quad_4.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 4 for Implicit general Bezier polynomial unfitted implicit domain.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <catch2/catch_test_macros.hpp>

#include "quadrature_test_utils.hpp"

#include <qugar/cart_grid_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/primitive_funcs_lib.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>

#include <array>
#include <cstddef>
#include <memory>

// // NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Quadrature for Bezier function for cylinder 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto grid = std::make_shared<CartGridTP<3>>(std::array<std::size_t, 3>({ { 4, 5, 6 } }));
  const int n_quad_pts_dir{ 7 };

  const real radius{ 0.45 };
  const Point<3> origin{ 0.5, 0.5, 0.5 };

  for (int dir = 0; dir < 3; ++dir) {
    Point<3> axis{ 0.0, 0.0, 0.0 };
    axis(dir) = numbers::one;
    const auto cylinder = std::make_shared<funcs::CylinderBzr>(radius, origin, axis);

    const real target_volume{ numbers::pi * radius * radius };
    const auto target_centroid = origin;
    const real target_int_bound_volume = numbers::two * numbers::pi * radius;

    const Tolerance tol{ 1.0e-6 };
    test_volume_and_centroid<3>(
      cylinder, grid, n_quad_pts_dir, target_volume, target_centroid, target_int_bound_volume, tol);
  }
}
