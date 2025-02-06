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
//! @brief Test 4 for BezierTP class (test Bezier degree elevation).
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"
#include "random_generator.hpp"

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tolerance.hpp>

#include <cstddef>

#include <catch2/catch_test_macros.hpp>

namespace {


template<int dim, int range> void test_Bezier_degree_elevation(const int n_eval_pts)
{
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto bezier = create_random_bezier<dim, range>(3, 5);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  const auto order_increase = qugar::rand::create_random_tensor_size<dim>(0, 3, true);
  const auto new_order = bezier->get_order() + order_increase;

  const auto bzr_elevated = bezier->raise_order(new_order);

  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE((bzr_elevated->get_order() == new_order));

  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));
  const qugar::Tolerance tol(1.0e-12);

  for (const auto &eval_pt : eval_pts) {

    const auto bzr_ele_val = bzr_elevated->operator()(eval_pt);
    const auto bzr_val = bezier->operator()(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_values(bzr_ele_val, bzr_val, tol)));

    const auto bzr_ele_grad = bzr_elevated->grad(eval_pt);
    const auto bzr_grad = bezier->grad(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors(bzr_ele_grad, bzr_grad, tol)));

    const auto bzr_ele_hess = bzr_elevated->hessian(eval_pt);
    const auto bzr_hess = bezier->hessian(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors(bzr_ele_hess, bzr_hess, tol)));
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Beziers order elevation", "[bezier_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_Bezier_degree_elevation<1, 1>(n_eval_pts);
  test_Bezier_degree_elevation<2, 1>(n_eval_pts);
  test_Bezier_degree_elevation<3, 1>(n_eval_pts);

  test_Bezier_degree_elevation<1, 2>(n_eval_pts);
  test_Bezier_degree_elevation<2, 2>(n_eval_pts);
  test_Bezier_degree_elevation<3, 2>(n_eval_pts);

  test_Bezier_degree_elevation<1, 3>(n_eval_pts);
  test_Bezier_degree_elevation<2, 3>(n_eval_pts);
  test_Bezier_degree_elevation<3, 3>(n_eval_pts);
}