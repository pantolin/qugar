// --------------------------------------------------------------------------
//
// Co (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_poly_quad_5.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 5 for Implicit general Bezier polynomial unfitted implicit domain.
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

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Quadrature for Bezier function for annulus 2D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ 16, 17 }));
  const int n_quad_pts_dir{ 7 };

  const real central_radius{ 0.25 };
  const real thickness{ 0.15 };
  const real R0 = central_radius - thickness;
  const real R1 = central_radius + thickness;

  const Point<2> origin{ 0.48, 0.55 };

  const auto annulus = std::make_shared<funcs::AnnulusBzr>(R0, R1, origin);


  const real target_volume{ numbers::pi * (R1 * R1 - R0 * R0) };
  const auto target_centroid = origin;
  const real target_int_bound_volume = numbers::two * numbers::pi * (R0 + R1);

  const Tolerance tol{ 1.0e-6 };
  test_volume_and_centroid<2>(
    annulus, grid, n_quad_pts_dir, target_volume, target_centroid, target_int_bound_volume, tol);
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Quadrature for Bezier function for torus 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto grid = std::make_shared<CartGridTP<3>>(std::array<std::size_t, 3>({ { 10, 12, 14 } }));
  // const CartGridTP<3> grid({ 16, 16, 16 });
  const int n_quad_pts_dir{ 7 };

  const real major_radius{ 0.35 };
  const real minor_radius{ 0.08 };
  const Point<3> origin{ 0.45, 0.51, 0.49 };
  const Point<3> axis{ 1.0, 1.2, 0.9 };


  const auto torus = std::make_shared<funcs::TorusBzr>(major_radius, minor_radius, origin, axis);

  const real target_volume{ numbers::two * numbers::pi * numbers::pi * minor_radius * minor_radius * major_radius };
  const auto target_centroid = origin;
  const real target_int_bound_volume = numbers::four * numbers::pi * numbers::pi * minor_radius * major_radius;

  const Tolerance tol{ 1.0e-3 };
  test_volume_and_centroid<3>(
    torus, grid, n_quad_pts_dir, target_volume, target_centroid, target_int_bound_volume, tol);
}
