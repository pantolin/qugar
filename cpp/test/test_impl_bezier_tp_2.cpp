// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_bezier_tp_2.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 2 for BezierTP class (test Bezier composition).
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"

#include <qugar/bezier_tp.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

namespace {


template<int dim, int int_dim, int range> void test_Bezier_composition(const int n_eval_pts)
{
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto bzr_0 = create_random_bezier<dim, int_dim>(3, 5);
  const auto bzr_1 = create_random_bezier<int_dim, range>(2, 4);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  const auto comp_bzr = bzr_1->compose(*bzr_0);
  using Value = typename qugar::impl::BezierTP<dim, range>::template Value<qugar::real>;

  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));
  const qugar::Tolerance tol(1.0e-12);

  Value val;
  for (const auto &eval_pt : eval_pts) {
    const auto val_0 = bzr_0->operator()(eval_pt);
    if constexpr (int_dim == 1) {
      val = bzr_1->operator()(qugar::Point<1>(val_0));
    } else {
      val = bzr_1->operator()(val_0);
    }
    const Value comp_val = comp_bzr->operator()(eval_pt);
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
    REQUIRE((compare_values(comp_val, val, tol)));
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Beziers composition", "[bezier_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_Bezier_composition<1, 1, 1>(n_eval_pts);
  test_Bezier_composition<1, 2, 1>(n_eval_pts);
  test_Bezier_composition<1, 3, 1>(n_eval_pts);
  test_Bezier_composition<1, 1, 2>(n_eval_pts);
  test_Bezier_composition<1, 2, 2>(n_eval_pts);
  test_Bezier_composition<1, 3, 2>(n_eval_pts);
  test_Bezier_composition<1, 1, 3>(n_eval_pts);
  test_Bezier_composition<1, 2, 3>(n_eval_pts);
  test_Bezier_composition<1, 3, 3>(n_eval_pts);

  test_Bezier_composition<2, 1, 1>(n_eval_pts);
  test_Bezier_composition<2, 2, 1>(n_eval_pts);
  test_Bezier_composition<2, 3, 1>(n_eval_pts);
  test_Bezier_composition<2, 1, 2>(n_eval_pts);
  test_Bezier_composition<2, 2, 2>(n_eval_pts);
  test_Bezier_composition<2, 3, 2>(n_eval_pts);
  test_Bezier_composition<2, 1, 3>(n_eval_pts);
  test_Bezier_composition<2, 2, 3>(n_eval_pts);
  test_Bezier_composition<2, 3, 3>(n_eval_pts);

  test_Bezier_composition<3, 1, 1>(n_eval_pts);
  test_Bezier_composition<3, 2, 1>(n_eval_pts);
  test_Bezier_composition<3, 3, 1>(n_eval_pts);
  test_Bezier_composition<3, 1, 2>(n_eval_pts);
  test_Bezier_composition<3, 2, 2>(n_eval_pts);
  test_Bezier_composition<3, 3, 2>(n_eval_pts);
  test_Bezier_composition<3, 1, 3>(n_eval_pts);
  test_Bezier_composition<3, 2, 3>(n_eval_pts);
  test_Bezier_composition<3, 3, 3>(n_eval_pts);
}
