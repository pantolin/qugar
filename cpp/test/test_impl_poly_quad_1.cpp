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
//! @brief Test 1 for Implicit general Bezier polynomial unfitted implicit domain.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <catch2/catch_test_macros.hpp>

#include "quadrature_test_utils.hpp"

#include <qugar/point.hpp>
#include <qugar/primitive_funcs_lib.hpp>
#include <qugar/types.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace {}// namespace

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier quadrature for plane 2D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<2> origin{ 0.5, 0.5 };
  const auto plane = std::make_shared<qugar::impl::funcs::PlaneBzr<2>>(origin, Point<2>{ 2.0, 1.0 });

  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ { 4, 5 } }));
  const int n_quad_pts_dir{ 2 };

  const qugar::real target_volume{ 0.5 };
  const qugar::Point<2> target_centroid{ 13.0 / 48.0, 5.0 / 12.0 };
  const qugar::real target_unf_bound_volume{ std::sqrt(1.25) };

  test_volume_and_centroid<2>(plane, grid, n_quad_pts_dir, target_volume, target_centroid, target_unf_bound_volume);
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier quadrature for plane 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto grid = std::make_shared<CartGridTP<3>>(std::array<std::size_t, 3>({ { 4, 5, 6 } }));

  const Point<3> origin{ 0.5, 0.5, 0.5 };
  const std::vector<Point<3>> normals{
    Point<3>{ 2.0, 1.0, 0.0 }, Point<3>{ 0.0, 2.0, 1.0 }, Point<3>{ 1.0, 0.0, 2.0 }
  };

  const std::vector<Point<3>> target_centroids{ Point<3>{ 13.0 / 48.0, 5.0 / 12.0, 0.5 },
    Point<3>{ 0.5, 13.0 / 48.0, 5.0 / 12.0 },
    Point<3>{ 5.0 / 12.0, 0.5, 13.0 / 48.0 } };

  const qugar::real target_volume{ 0.5 };
  const qugar::real target_unf_bound_volume{ std::sqrt(1.25) };

  auto target_centroid = target_centroids.cbegin();
  for (const auto normal : normals) {

    const auto plane = std::make_shared<qugar::impl::funcs::PlaneBzr<3>>(origin, normal);

    const int n_quad_pts_dir{ 2 };

    test_volume_and_centroid<3>(
      plane, grid, n_quad_pts_dir, target_volume, *target_centroid++, target_unf_bound_volume);
  }
}
