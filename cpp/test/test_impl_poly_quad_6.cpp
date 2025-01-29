// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_poly_quad_6.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 6 for Implicit general Bezier polynomial unfitted implicit domain.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <catch2/catch_test_macros.hpp>

#include "quadrature_test_utils.hpp"
#include "qugar/numbers.hpp"

#include <qugar/primitive_funcs_lib.hpp>
#include <qugar/tolerance.hpp>

#include <array>
#include <cstddef>
#include <memory>

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Testing polynomial quadrature for case with aggregate quadratures in 2D", "[impl]")
{
  using namespace qugar;

  const Point<2> origin{ 0.5, 0.5 };
  const real radius{ 0.47 };
  const auto sphere = std::make_shared<impl::funcs::SphereBzr<2>>(radius, origin);

  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ { 1, 1 } }));
  const int n_quad_pts_dir = { 5 };

  const real target_volume{ numbers::pi * radius * radius };
  const auto target_centroid = origin;
  const real target_int_bound_volume = numbers::two * numbers::pi * radius;


  const Tolerance tol{ 1.0e-2 };
  test_volume_and_centroid<2>(
    sphere, grid, n_quad_pts_dir, target_volume, target_centroid, target_int_bound_volume, tol);
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Testing polynomial quadrature for case with aggregate quadratures in 3D", "[impl]")
{
  using namespace qugar;

  const Point<3> origin{ 0.5, 0.5, 0.5 };
  const real radius{ 0.47 };
  const auto sphere = std::make_shared<impl::funcs::SphereBzr<3>>(radius, origin);

  const auto grid = std::make_shared<CartGridTP<3>>(std::array<std::size_t, 3>({ { 1, 1, 1 } }));
  const int n_quad_pts_dir = { 7 };

  const real target_volume = numbers::four_thirds * numbers::pi * radius * radius * radius;
  const auto target_centroid = origin;
  const real target_int_bound_volume = numbers::four * numbers::pi * radius * radius;


  const Tolerance tol{ 1.0e-2 };
  test_volume_and_centroid<3>(
    sphere, grid, n_quad_pts_dir, target_volume, target_centroid, target_int_bound_volume, tol);
}
