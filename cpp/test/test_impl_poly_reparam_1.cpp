// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_poly_reparam_1.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 0 for Implicit Bezier polynomial reparameterization for grids.
//! @date 2025-01-15
//!
//! @copyright Copyright (c) 2025-present

#include "reparam_test_utils.hpp"

#include <qugar/bbox.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/point.hpp>
#include <qugar/primitive_funcs_lib.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <memory>


// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier reparameterization for plane 2D in a grid", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<2> origin{ 0.75, 0.65 };
  const Point<2> normal{ 1.0, 2.0 };
  const auto bzr = std::make_shared<qugar::impl::funcs::PlaneBzr<2>>(origin, normal);

  const int n_elems_dir = 4;
  const int order{ 4 };

  const BoundBox<2> domain_01;

  const auto grid =
    std::make_shared<CartGridTP<2>>(domain_01, std::array<std::size_t, 2>({ { n_elems_dir, n_elems_dir } }));


  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam<2, false>(
    bzr, grid, order, 185, 272, 172, 94, 93, 15951, 9357, Point<2>{ 0.443783783783783858, 0.486756756756756781 });
  test_reparam<2, true>(bzr, grid, order, 16, 20, 0, 7, 0, 89, 0, Point<2>{ 0.515625, 0.767187500000000133 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier reparameterization for sphere 2D in a grid", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<2> center{ 0.5, 0.5 };
  const auto bzr = std::make_shared<qugar::impl::funcs::SphereBzr<2>>(0.45, center);

  const int n_elems_dir = 4;
  const int order{ 8 };

  const BoundBox<2> domain_01;

  const auto grid =
    std::make_shared<CartGridTP<2>>(domain_01, std::array<std::size_t, 2>({ { n_elems_dir, n_elems_dir } }));

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam<2, false>(
    bzr, grid, order, 813, 1024, 288, 408, 396, 270782, 70728, Point<2>{ 0.501162887659512335, 0.501162887659512335 });
  test_reparam<2, true>(
    bzr, grid, order, 84, 96, 0, 41, 0, 2630, 0, Point<2>{ 0.499487904818710138, 0.499487904818710138 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier reparameterization for plane 3D in a grid", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<3> origin{ 0.75, 0.75, 0.75 };
  const Point<3> normal{ 1.0, 2.0, 3.0 };
  const auto bzr = std::make_shared<qugar::impl::funcs::PlaneBzr<3>>(origin, normal);

  const int n_elems_dir = 4;
  const int order{ 4 };

  const BoundBox<3> domain_01;

  const auto grid = std::make_shared<CartGridTP<3>>(
    domain_01, std::array<std::size_t, 3>({ { n_elems_dir, n_elems_dir, n_elems_dir } }));

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam<3, false>(bzr,
    grid,
    order,
    2365,
    4480,
    1224,
    1183,
    1198,
    3314776,
    887964,
    Point<3>{ 0.498872445384073604, 0.499982381959126587, 0.504087385482734018 });
  test_reparam<3, true>(bzr,
    grid,
    order,
    130,
    208,
    120,
    64,
    63,
    7999,
    4550,
    Point<3>{ 0.580769230769230926, 0.724038461538461631, 0.823717948717948345 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}

// // NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier reparameterization for sphere 3D in a grid", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<3> center{ 0.5, 0.5, 0.5 };
  const auto bzr = std::make_shared<qugar::impl::funcs::SphereBzr<3>>(0.45, center);

  const int n_elems_dir = 4;
  const int order{ 8 };

  const BoundBox<3> domain_01;

  const auto grid = std::make_shared<CartGridTP<3>>(
    domain_01, std::array<std::size_t, 3>({ { n_elems_dir, n_elems_dir, n_elems_dir } }));

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam<3, false>(bzr,
    grid,
    order,
    23073,
    32768,
    1856,
    11674,
    11189,
    247673237,
    13213934,
    Point<3>{ 0.500880925000022903, 0.499260123893885133, 0.496336171563302953 });
  test_reparam<3, true>(bzr,
    grid,
    order,
    2814,
    3584,
    992,
    1418,
    1380,
    3341288,
    894590,
    Point<3>{ 0.502028314621738359, 0.499423318703030472, 0.496941331370618744 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}
