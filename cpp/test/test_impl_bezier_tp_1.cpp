// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_bezier_tp_1.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 1 for BezierTP class (test Bezier products).
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"

#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <type_traits>

namespace {

template<int dim, int range> void test_Bezier_product(const int n_eval_pts)
{
  const auto bzr_0 = create_random_bezier<dim, range>(1, 5);
  const auto bzr_1 = create_random_bezier<dim, range>(1, 5);
  const auto prod_bzr = *bzr_0 * *bzr_1;

  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));

  const qugar::Tolerance tol(1000 * qugar::numbers::eps);

  using T = std::conditional_t<range == 1, qugar::real, qugar::Point<range>>;


  for (const auto &eval_pt : eval_pts) {
    const auto val_0 = bzr_0->operator()(eval_pt);
    const auto val_1 = bzr_1->operator()(eval_pt);
    const auto prod_val = prod_bzr->operator()(eval_pt);

    const T prod_val_0_val_1 = val_0 * val_1;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
    REQUIRE((compare_values(prod_val, prod_val_0_val_1, tol)));
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Beziers product", "[bezier_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_Bezier_product<1, 1>(n_eval_pts);
  test_Bezier_product<1, 2>(n_eval_pts);
  test_Bezier_product<1, 3>(n_eval_pts);

  test_Bezier_product<2, 1>(n_eval_pts);
  test_Bezier_product<2, 2>(n_eval_pts);
  test_Bezier_product<2, 3>(n_eval_pts);

  test_Bezier_product<3, 1>(n_eval_pts);
  test_Bezier_product<3, 2>(n_eval_pts);
  test_Bezier_product<3, 3>(n_eval_pts);
}