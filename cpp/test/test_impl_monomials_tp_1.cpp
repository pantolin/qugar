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
//! @brief Test 1 for MonomialsTP class (test gradients).
//! @version 0.0.2
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


template<int dim, int range = 1> void test_monomials_gradients(const int n_eval_pts)
{
  using namespace qugar;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto monomials = create_random_monomials<dim, range>(3, 5);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  const auto eval_pts = rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));
  const Tolerance tol(1.0e-12);

  std::array<std::shared_ptr<qugar::impl::MonomialsTP<dim, range>>, dim> gradients;
  for (int dir = 0; dir < dim; ++dir) {
    at(gradients, dir) = monomials->create_derivative(dir);
  }

  using Gradient = qugar::impl::MonomialsTP<dim, range>::template Gradient<real>;
  for (const auto &eval_pt : eval_pts) {

    Gradient expected_grad{ numbers::zero };
    for (int dir = 0; dir < dim; ++dir) {
      expected_grad(dir) = at(gradients, dir)->operator()(eval_pt);
    }

    const auto grad = monomials->grad(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_vectors(grad, expected_grad, tol)));
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Monomials gradients", "[monomials_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_monomials_gradients<1, 1>(n_eval_pts);
  test_monomials_gradients<2, 1>(n_eval_pts);
  test_monomials_gradients<3, 1>(n_eval_pts);

  test_monomials_gradients<1, 2>(n_eval_pts);
  test_monomials_gradients<2, 2>(n_eval_pts);
  test_monomials_gradients<3, 2>(n_eval_pts);

  test_monomials_gradients<1, 3>(n_eval_pts);
  test_monomials_gradients<2, 3>(n_eval_pts);
  test_monomials_gradients<3, 3>(n_eval_pts);
}
