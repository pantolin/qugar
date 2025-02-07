// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_general_quad_1.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 1 for Implicit general quadrature.
//! @version 0.0.2
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
TEST_CASE("Quadrature for general function for sphere 2D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<2> origin{ 0.5, 0.5 };
  const real radius{ 0.4 };
  const auto sphere = std::make_shared<funcs::Sphere<2>>(radius, origin);

  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ 4, 5 }));
  const int n_quad_pts_dir{ 7 };

  const qugar::real target_volume{ qugar::numbers::pi * radius * radius };
  const auto target_centroid = origin;
  const qugar::real target_unf_bound_volume = qugar::numbers::two * qugar::numbers::pi * radius;

  const qugar::Tolerance tol{ 1.0e-8 };
  test_volume_and_centroid<2>(
    sphere, grid, n_quad_pts_dir, target_volume, target_centroid, target_unf_bound_volume, tol);
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Quadrature for general function for sphere 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<3> origin{ 0.5, 0.5, 0.5 };
  const real radius{ 0.4 };
  const auto sphere = std::make_shared<funcs::Sphere<3>>(radius, origin);

  const auto grid = std::make_shared<CartGridTP<3>>(std::array<std::size_t, 3>({ { 4, 5, 6 } }));
  const int n_quad_pts_dir{ 7 };

  const qugar::real target_volume{ numbers::four_thirds * numbers::pi * radius * radius * radius };
  const auto target_centroid = origin;
  const qugar::real target_unf_bound_volume = numbers::four * numbers::pi * radius * radius;

  const qugar::Tolerance tol{ 1.0e-6 };
  test_volume_and_centroid<3>(
    sphere, grid, n_quad_pts_dir, target_volume, target_centroid, target_unf_bound_volume, tol);
}
