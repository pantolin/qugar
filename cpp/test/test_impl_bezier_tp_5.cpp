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
//! @brief Test 5 for BezierTP class (test Bezier addition and subtraction).
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"
#include "random_generator.hpp"

#include <qugar/bezier_tp.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>

#include <cstddef>

#include <catch2/catch_test_macros.hpp>

namespace {


// NOLINTNEXTLINE (readability-function-cognitive-complexity)
template<int dim, int range = 1> void test_Bezier_addition_and_subtraction(const int n_eval_pts)
{
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto bezier_0 = create_random_bezier<dim, 1>(3, 5);
  const auto bezier_1 = create_random_bezier<dim, 1>(3, 5);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  const auto bezier_sum = *bezier_0 + *bezier_1;
  const auto bezier_sub = *bezier_0 - *bezier_1;

  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));
  const qugar::Tolerance tol(1.0e-12);

  for (const auto &eval_pt : eval_pts) {

    const auto bzr_0_val = bezier_0->operator()(eval_pt);
    const auto bzr_1_val = bezier_1->operator()(eval_pt);
    const auto bzr_sum_val = bezier_sum->operator()(eval_pt);
    const auto bzr_sub_val = bezier_sub->operator()(eval_pt);
    // NOLINTBEGIN(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_values(bzr_0_val + bzr_1_val, bzr_sum_val, tol)));
    REQUIRE((compare_values(bzr_0_val - bzr_1_val, bzr_sub_val, tol)));
    // NOLINTEND(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)

    const auto bzr_0_grad = bezier_0->grad(eval_pt);
    const auto bzr_1_grad = bezier_1->grad(eval_pt);
    const auto bzr_sum_grad = bezier_sum->grad(eval_pt);
    const auto bzr_sub_grad = bezier_sub->grad(eval_pt);

    using Grad = qugar::impl::BezierTP<dim, range>::template Gradient<qugar::real>;
    using Value = qugar::impl::BezierTP<dim, range>::template Value<qugar::real>;
    // NOLINTBEGIN(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors<Value, dim>(Grad(bzr_0_grad + bzr_1_grad), bzr_sum_grad, tol)));
    REQUIRE((compare_vectors<Value, dim>(Grad(bzr_0_grad - bzr_1_grad), bzr_sub_grad, tol)));
    // NOLINTEND(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)

    const auto bzr_0_hess = bezier_0->hessian(eval_pt);
    const auto bzr_1_hess = bezier_1->hessian(eval_pt);
    const auto bzr_sum_hess = bezier_sum->hessian(eval_pt);
    const auto bzr_sub_hess = bezier_sub->hessian(eval_pt);

    using Hess = qugar::impl::BezierTP<dim, range>::template Hessian<qugar::real>;
    static const int num_hessian = qugar::impl::BezierTP<dim, range>::num_hessian;
    // NOLINTBEGIN(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors<Value, num_hessian>(Hess(bzr_0_hess + bzr_1_hess), bzr_sum_hess, tol)));
    REQUIRE((compare_vectors<Value, num_hessian>(Hess(bzr_0_hess - bzr_1_hess), bzr_sub_hess, tol)));
    // NOLINTEND(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Beziers addition and subtraction", "[bezier_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_Bezier_addition_and_subtraction<1, 1>(n_eval_pts);
  test_Bezier_addition_and_subtraction<2, 1>(n_eval_pts);
  test_Bezier_addition_and_subtraction<3, 1>(n_eval_pts);

  test_Bezier_addition_and_subtraction<1, 2>(n_eval_pts);
  test_Bezier_addition_and_subtraction<2, 2>(n_eval_pts);
  test_Bezier_addition_and_subtraction<3, 2>(n_eval_pts);

  test_Bezier_addition_and_subtraction<1, 3>(n_eval_pts);
  test_Bezier_addition_and_subtraction<2, 3>(n_eval_pts);
  test_Bezier_addition_and_subtraction<3, 3>(n_eval_pts);
}