// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_bezier_tp_0.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 0 for BezierTP class.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"

#include <qugar/bezier_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tolerance.hpp>

#include <catch2/catch_test_macros.hpp>

namespace {

template<int dim, int range>
void test_Bezier_constant_evaluation(const qugar::impl::BezierTP<dim, range> &bzr, const int n_eval_pts)
{
  const auto ref_val = bzr.get_coefs().front();

  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));

  const qugar::Tolerance tol(1000 * qugar::numbers::eps);

  for (const auto &eval_pt : eval_pts) {
    const auto val = bzr(eval_pt);
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
    REQUIRE((compare_values(val, ref_val, tol)));
  }
}

template<int dim, int range>
void test_Bezier_constant_gradient(const qugar::impl::BezierTP<dim, range> &bzr, const int n_eval_pts)
{
  using Bzr = qugar::impl::BezierTP<dim, range>;
  using Gradient = Bzr::template Gradient<qugar::real>;

  const Gradient ref_grad{};

  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));

  const qugar::Tolerance tol(1000 * qugar::numbers::eps);

  for (const auto &eval_pt : eval_pts) {
    const auto grad = bzr.grad(eval_pt);
    for (int dir = 0; dir < dim; ++dir) {
      // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
      REQUIRE((compare_values(grad(dir), ref_grad(dir), tol)));
    }
  }
}

template<int dim, int range>
void test_Bezier_constant_hessian(const qugar::impl::BezierTP<dim, range> &bzr, const int n_eval_pts)
{
  using Bzr = qugar::impl::BezierTP<dim, range>;
  using Hessian = Bzr::template Hessian<qugar::real>;

  const Hessian ref_hess{};

  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts));

  const qugar::Tolerance tol(1000 * qugar::numbers::eps);

  for (const auto &eval_pt : eval_pts) {
    const auto hess = bzr.hessian(eval_pt);
    for (int dir = 0; dir < Bzr::num_hessian; ++dir) {
      // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
      REQUIRE((compare_values(hess(dir), ref_hess(dir), tol)));
    }
  }
}


template<int dim, int range> void test_Bezier_constant(const int n_eval_pts)
{
  const auto bzr_order_1 = create_random_bezier<dim, range>(1, 1);
  test_Bezier_constant_evaluation(*bzr_order_1, n_eval_pts);
  test_Bezier_constant_gradient(*bzr_order_1, n_eval_pts);
  test_Bezier_constant_hessian(*bzr_order_1, n_eval_pts);

  const auto bzr_general_order = create_constant_random_bezier<dim, range>(1, 5);
  test_Bezier_constant_evaluation(*bzr_general_order, n_eval_pts);
  test_Bezier_constant_gradient(*bzr_general_order, n_eval_pts);
  test_Bezier_constant_hessian(*bzr_general_order, n_eval_pts);
}

}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Beziers evaluation", "[bezier_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_Bezier_constant<1, 1>(n_eval_pts);
  test_Bezier_constant<1, 2>(n_eval_pts);
  test_Bezier_constant<1, 3>(n_eval_pts);

  test_Bezier_constant<2, 1>(n_eval_pts);
  test_Bezier_constant<2, 2>(n_eval_pts);
  test_Bezier_constant<2, 3>(n_eval_pts);

  test_Bezier_constant<3, 1>(n_eval_pts);
  test_Bezier_constant<3, 2>(n_eval_pts);
  test_Bezier_constant<3, 3>(n_eval_pts);
}
