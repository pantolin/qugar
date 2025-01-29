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
//! @brief Test 2 for MonomialsTP class (test hessians).
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"
#include "random_generator.hpp"

#include <qugar/monomials_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <array>
#include <cstddef>
#include <memory>

#include <catch2/catch_test_macros.hpp>

namespace {


template<int dim, int range = 1> void test_monomials_hessians(const int n_eval_pts)
{
  using namespace qugar;

  static const int num_hessian = dim * (dim + 1) / 2;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto monomials = create_random_monomials<dim, range>(3, 5);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  const auto eval_pts = rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));
  const Tolerance tol(1.0e-12);

  std::array<std::shared_ptr<qugar::impl::MonomialsTP<dim, range>>, num_hessian> hessians;
  for (int i = 0, ij = 0; i < dim; ++i) {
    const auto grad_i = monomials->create_derivative(i);
    for (int j = i; j < dim; ++j) {
      at(hessians, ij++) = grad_i->create_derivative(j);
    }
  }

  using Hessian = qugar::impl::MonomialsTP<dim, range>::template Hessian<real>;
  for (const auto &eval_pt : eval_pts) {

    Hessian expected_hess{ numbers::zero };
    for (int i = 0; i < num_hessian; ++i) {
      expected_hess(i) = at(hessians, i)->operator()(eval_pt);
    }

    const auto hess = monomials->hessian(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors(hess, expected_hess, tol)));
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Monomials hessians", "[monomials_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_monomials_hessians<1, 1>(n_eval_pts);
  test_monomials_hessians<2, 1>(n_eval_pts);
  test_monomials_hessians<3, 1>(n_eval_pts);

  test_monomials_hessians<1, 2>(n_eval_pts);
  test_monomials_hessians<2, 2>(n_eval_pts);
  test_monomials_hessians<3, 2>(n_eval_pts);

  test_monomials_hessians<1, 3>(n_eval_pts);
  test_monomials_hessians<2, 3>(n_eval_pts);
  test_monomials_hessians<3, 3>(n_eval_pts);
}
