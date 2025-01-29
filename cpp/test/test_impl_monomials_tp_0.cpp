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
//! @brief Test 0 for MonomialsTP class (test evaluation).
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"
#include "random_generator.hpp"

#include <qugar/monomials_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>

#include <cstddef>

#include <catch2/catch_test_macros.hpp>

namespace {


template<int dim, int range = 1> void test_monomials_evaluation(const int n_eval_pts)
{
  using namespace qugar;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto monomials = create_random_monomials<dim, range>(3, 5);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  const auto eval_pts = rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));
  const Tolerance tol(1.0e-12);

  using Value = qugar::impl::MonomialsTP<dim, range>::template Value<real>;
  for (const auto &eval_pt : eval_pts) {


    Value expected_val{ numbers::zero };
    auto coef = monomials->get_coefs().cbegin();
    for (const auto tid : TensorIndexRangeTP<dim>(monomials->get_order())) {
      Value local_val{ *coef++ };
      for (int dir = 0; dir < dim; ++dir) {
        local_val *= std::pow(eval_pt(dir), tid(dir));
      }

      expected_val += local_val;
    }

    const auto val = monomials->operator()(eval_pt);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while, readability-suspicious-call-argument)
    REQUIRE((compare_values(val, expected_val, tol)));
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Monomials evaluation", "[monomials_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_monomials_evaluation<1, 1>(n_eval_pts);
  test_monomials_evaluation<2, 1>(n_eval_pts);
  test_monomials_evaluation<3, 1>(n_eval_pts);

  test_monomials_evaluation<1, 2>(n_eval_pts);
  test_monomials_evaluation<2, 2>(n_eval_pts);
  test_monomials_evaluation<3, 2>(n_eval_pts);

  test_monomials_evaluation<1, 3>(n_eval_pts);
  test_monomials_evaluation<2, 3>(n_eval_pts);
  test_monomials_evaluation<3, 3>(n_eval_pts);
}
