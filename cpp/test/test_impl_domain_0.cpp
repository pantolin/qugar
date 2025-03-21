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
//! @date 2025-03-07
//!
//! @copyright Copyright (c) 2025-present

#include <catch2/catch_test_macros.hpp>


#include <qugar/bezier_tp.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/point.hpp>
#include <qugar/primitive_funcs_lib.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace {
void horizontal_Bezier_test()
{
  using namespace qugar;
  using namespace qugar::impl;

  const TensorSizeTP<2> order(2);
  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ { 5, 4 } }));

  const std::vector<real> coefs({ -1.0, -1.0, 1.0, 1.0 });
  const auto bezier = std::make_shared<BezierTP<2, 1>>(order, coefs);

  const qugar::impl::UnfittedImplDomain<2> unf_domain(bezier, grid);

  for (const auto &cell_id : unf_domain.get_full_cells()) {
    REQUIRE((0 <= cell_id && cell_id < 10));
  }
  for (const auto &cell_id : unf_domain.get_empty_cells()) {
    REQUIRE((10 <= cell_id));
  }
  REQUIRE((unf_domain.get_cut_cells().empty()));
}


void vertical_Bezier_test()
{
  using namespace qugar;
  using namespace qugar::impl;

  const TensorSizeTP<2> order(2);
  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ { 4, 5 } }));

  const std::vector<real> coefs({ -1.0, 1.0, -1.0, 1.0 });
  const auto bezier = std::make_shared<BezierTP<2, 1>>(order, coefs);

  const qugar::impl::UnfittedImplDomain<2> unf_domain(bezier, grid);

  for (const auto &cell_id : unf_domain.get_full_cells()) {
    REQUIRE((cell_id % 4 < 2));
  }
  for (const auto &cell_id : unf_domain.get_empty_cells()) {
    REQUIRE(((cell_id % 4) >= 2));
  }
  REQUIRE((unf_domain.get_cut_cells().empty()));
}

void horizontal_general_test()
{
  using namespace qugar;
  using namespace qugar::impl;

  const TensorSizeTP<2> order(2);
  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ { 5, 4 } }));

  const Point<2> origin{ 0.5, 0.5 };
  const auto plane = std::make_shared<funcs::Plane<2>>(origin, Point<2>{ 0.0, 1.0 });

  const qugar::impl::UnfittedImplDomain<2> unf_domain(plane, grid);

  for (const auto &cell_id : unf_domain.get_full_cells()) {
    REQUIRE((0 <= cell_id && cell_id < 10));
  }
  for (const auto &cell_id : unf_domain.get_empty_cells()) {
    REQUIRE((10 <= cell_id));
  }
  REQUIRE((unf_domain.get_cut_cells().empty()));
}


void vertical_general_test()
{
  using namespace qugar;
  using namespace qugar::impl;

  const TensorSizeTP<2> order(2);
  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ { 4, 5 } }));

  const Point<2> origin{ 0.5, 0.5 };
  const auto plane = std::make_shared<funcs::Plane<2>>(origin, Point<2>{ 1.0, 0.0 });

  const qugar::impl::UnfittedImplDomain<2> unf_domain(plane, grid);

  for (const auto &cell_id : unf_domain.get_full_cells()) {
    REQUIRE((cell_id % 4 < 2));
  }
  for (const auto &cell_id : unf_domain.get_empty_cells()) {
    REQUIRE(((cell_id % 4) >= 2));
  }
  REQUIRE((unf_domain.get_cut_cells().empty()));
}

}// namespace

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier polynomial unfitted implicit domain decomposition", "[impl]")
{
  horizontal_Bezier_test();
  vertical_Bezier_test();
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("General polynomial unfitted implicit domain decomposition", "[impl]")
{
  horizontal_general_test();
  vertical_general_test();
}
