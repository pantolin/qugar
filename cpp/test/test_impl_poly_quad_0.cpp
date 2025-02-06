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
//! @brief Test 0 for Implicit general Bezier polynomial unfitted implicit domain.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <catch2/catch_test_macros.hpp>

#include "quadrature_test_utils.hpp"


#include <qugar/bezier_tp.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tensor_index_tp.hpp>

#include <array>
#include <memory>
#include <vector>

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier polynomial unfitted implicit domain", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const TensorSizeTP<2> order(2);
  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ { 4, 5 } }));
  const int n_quad_pts_dir{ 3 };

  for (const auto &offset : { numbers::zero, 0.1, -0.1, 0.25, -0.35 }) {
    const std::vector<real> coefs({ 1.0 + offset, 1.0 + offset, -1.0, -1.0 });
    const auto bezier = std::make_shared<BezierTP<2, 1>>(order, coefs);

    const qugar::real target_volume{ 1.0 / (2.0 + offset) };
    const qugar::Point<2> target_centroid{ 0.5, 1.0 - (0.5 / (2.0 + offset)) };
    const qugar::real target_int_bound_area{ 1.0 };
    test_volume_and_centroid<2>(bezier, grid, n_quad_pts_dir, target_volume, target_centroid, target_int_bound_area);
  }
}
