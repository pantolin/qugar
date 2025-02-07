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
//! @brief Test 6 for BezierTP class (test Bezier negation).
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


template<int dim, int range = 1> void test_Bezier_negation(const int n_eval_pts)
{
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  const auto bezier = create_random_bezier<dim, 1>(3, 5);

  const auto bezier_neg = bezier->negate();

  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));
  const qugar::Tolerance tol(1.0e-12);

  for (const auto &eval_pt : eval_pts) {

    const auto bzr_val = bezier->operator()(eval_pt);
    const auto bzr_neg_val = bezier_neg->operator()(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_values(bzr_val, -bzr_neg_val, tol)));

    const auto bzr_grad = bezier->grad(eval_pt);
    const auto bzr_neg_grad = bezier_neg->grad(eval_pt);

    using Grad = qugar::impl::BezierTP<dim, range>::template Gradient<qugar::real>;
    using Value = qugar::impl::BezierTP<dim, range>::template Value<qugar::real>;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors<Value, dim>(bzr_grad, Grad(-bzr_neg_grad), tol)));

    const auto bzr_hess = bezier->hessian(eval_pt);
    const auto bzr_neg_hess = bezier_neg->hessian(eval_pt);

    using Hess = qugar::impl::BezierTP<dim, range>::template Hessian<qugar::real>;
    static const int num_hessian = qugar::impl::BezierTP<dim, range>::num_hessian;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors<Value, num_hessian>(bzr_hess, Hess(-bzr_neg_hess), tol)));
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Beziers negation", "[bezier_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_Bezier_negation<1, 1>(n_eval_pts);
  test_Bezier_negation<2, 1>(n_eval_pts);
  test_Bezier_negation<3, 1>(n_eval_pts);

  test_Bezier_negation<1, 2>(n_eval_pts);
  test_Bezier_negation<2, 2>(n_eval_pts);
  test_Bezier_negation<3, 2>(n_eval_pts);

  test_Bezier_negation<1, 3>(n_eval_pts);
  test_Bezier_negation<2, 3>(n_eval_pts);
  test_Bezier_negation<3, 3>(n_eval_pts);
}
