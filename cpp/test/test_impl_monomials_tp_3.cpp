// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_monomials_tp_0.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 3 for MonomialsTP class (test transformation to Beziers).
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"
#include "random_generator.hpp"

#include <qugar/bezier_tp.hpp>
#include <qugar/tolerance.hpp>

#include <cstddef>
#include <memory>

#include <catch2/catch_test_macros.hpp>

namespace {


//! @brief Tests the transformation of monomials to Bezier form and compares their evaluations, gradients, and Hessians.
//!
//! This function generates random monomials and transforms them into Bezier form. It then evaluates the monomials and
//! their Bezier counterparts at random points, comparing their values, gradients, and Hessians to ensure the
//! transformation is accurate within a specified tolerance.
//!
//! @tparam dim The dimension of the space in which the monomials are defined.
//! @tparam range The range of the monomials.
//! @param n_eval_pts The number of evaluation points to generate for testing.
template<int dim, int range> void test_Bezier_transformation(const int n_eval_pts)
{
  using namespace qugar;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto monomials = create_random_monomials<dim, range>(3, 5);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  const auto bezier = std::make_shared<qugar::impl::BezierTP<dim, range>>(*monomials);

  const auto eval_pts = rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));
  const Tolerance tol(1.0e-12);

  for (const auto &eval_pt : eval_pts) {

    const auto mon_val = monomials->operator()(eval_pt);
    const auto bzr_val = bezier->operator()(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_values(mon_val, bzr_val, tol)));

    const auto mon_grad = monomials->grad(eval_pt);
    const auto bzr_grad = bezier->grad(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors(mon_grad, bzr_grad, tol)));

    const auto mon_hess = monomials->hessian(eval_pt);
    const auto bzr_hess = bezier->hessian(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors(mon_hess, bzr_hess, tol)));
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing transformation to Beziers", "[monomials_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_Bezier_transformation<1, 1>(n_eval_pts);
  test_Bezier_transformation<2, 1>(n_eval_pts);
  test_Bezier_transformation<3, 1>(n_eval_pts);

  test_Bezier_transformation<1, 2>(n_eval_pts);
  test_Bezier_transformation<2, 2>(n_eval_pts);
  test_Bezier_transformation<3, 2>(n_eval_pts);

  test_Bezier_transformation<1, 3>(n_eval_pts);
  test_Bezier_transformation<2, 3>(n_eval_pts);
  test_Bezier_transformation<3, 3>(n_eval_pts);
}
