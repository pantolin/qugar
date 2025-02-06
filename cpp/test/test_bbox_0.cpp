// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_bbox_0.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Tests for bounding box class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <catch2/catch_test_macros.hpp>


#include <qugar/bbox.hpp>
#include <qugar/types.hpp>


// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Lengths are computed", "[bbox]")
{
  using namespace qugar;
  const BoundBox<2> box_2D_0(0.0, 1.0);
  const BoundBox<2> box_2D_1(Point<2>{ 1, 2 }, Point<2>{ 2, 4 });
  for (int dir = 0; dir < 2; ++dir) {
    // NOLINTBEGIN(cppcoreguidelines-avoid-do-while)
    REQUIRE((box_2D_0.length(dir) == 1.0));
    REQUIRE((box_2D_1.length(dir) == static_cast<real>(dir + 1)));
    // NOLINTEND(cppcoreguidelines-avoid-do-while)
  }


  BoundBox<3> box_3D_0(0.0, 1.0);
  const BoundBox<3> box_3D_1(Point<3>{ 1, 2, 3 }, Point<3>{ 2, 4, 6 });
  for (int dir = 0; dir < 3; ++dir) {
    // NOLINTBEGIN(cppcoreguidelines-avoid-do-while)
    REQUIRE((box_3D_0.length(dir) == 1.0));
    REQUIRE((box_3D_1.length(dir) == static_cast<real>(dir + 1)));
    // NOLINTEND(cppcoreguidelines-avoid-do-while)
  }
}
