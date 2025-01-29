// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_poly_quad_0.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 3 for Implicit general Bezier polynomial unfitted implicit domain.
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
#include <qugar/ref_system.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>

#include <array>
#include <cstddef>
#include <memory>

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Quadrature for Bezier function for ellipse 2D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<2> semi_axes{ 0.25, 0.15 };

  const Point<2> origin{ 0.45, 0.55 };
  const Point<2> x_axis{ 1.0, 0.5 };
  const RefSystem<2> system(origin, x_axis);

  const auto ellipse = std::make_shared<funcs::EllipsoidBzr<2>>(semi_axes, system);

  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ { 9, 10 } }));
  const int n_quad_pts_dir{ 7 };

  const real target_volume{ numbers::pi * semi_axes(0) * semi_axes(1) };
  const auto target_centroid = origin;
  const real target_int_bound_volume = 1.2763499392751274;// Computed numerically.

  const Tolerance tol{ 1.0e-6 };
  test_volume_and_centroid<2>(
    ellipse, grid, n_quad_pts_dir, target_volume, target_centroid, target_int_bound_volume, tol);
}

// // NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Quadrature for Bezier function for ellipsoid 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<3> semi_axes{ 0.3, 0.4, 0.35 };

  const Point<3> origin{ 0.45, 0.55, 0.49 };
  const Point<3> x_axis{ 1.0, 0.5, 1.0 };
  const Point<3> y_axis{ 0.0, 1.0, 0.0 };
  const RefSystem<3> system(origin, x_axis, y_axis);

  const auto ellipsoid = std::make_shared<funcs::EllipsoidBzr<3>>(semi_axes, system);

  const auto grid = std::make_shared<CartGridTP<3>>(std::array<std::size_t, 3>({ { 9, 10, 11 } }));
  const int n_quad_pts_dir{ 7 };

  const real target_volume{ numbers::four_thirds * numbers::pi * semi_axes(0) * semi_axes(1) * semi_axes(2) };
  const auto target_centroid = origin;
  const real target_int_bound_volume = 1.535246912774376;// Computed numerically.

  const Tolerance tol{ 1.0e-6 };
  test_volume_and_centroid<3>(
    ellipsoid, grid, n_quad_pts_dir, target_volume, target_centroid, target_int_bound_volume, tol);
}
