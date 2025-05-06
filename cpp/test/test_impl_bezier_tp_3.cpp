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
//! @brief Test 3 for BezierTP class (test Bezier composition).
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"
#include "random_generator.hpp"

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/vector.hpp>

#include <cstddef>

#include <catch2/catch_test_macros.hpp>

namespace {


/**
 * @brief Tests the rescaling of a Bezier domain.
 *
 * This function creates a random Bezier curve or surface of the specified dimension,
 * generates random points to define a new bounding box domain, rescales the Bezier
 * object to this new domain, and then evaluates the Bezier object at random points
 * to ensure the rescaling was performed correctly.
 *
 * @tparam dim The dimension of the Bezier object.
 * @param n_eval_pts The number of evaluation points to test the rescaled Bezier object.
 */
template<int dim> void test_Bezier_domain_rescale(const int n_eval_pts)
{
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto bzr = create_random_bezier<dim, 1>(3, 5);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  auto pt_0 = qugar::rand::create_random_point<dim>(-qugar::numbers::two, qugar::numbers::two);
  auto pt_1 = qugar::rand::create_random_point<dim>(-qugar::numbers::two, qugar::numbers::two);

  for (int dir = 0; dir < dim; ++dir) {
    if (pt_0(dir) > pt_1(dir)) {
      std::swap(pt_0(dir), pt_1(dir));
    }
  }

  const qugar::BoundBox<dim> new_domain(pt_0, pt_1);

  const auto bzr_rescale = std::make_shared<qugar::impl::BezierTP<dim>>(*bzr);
  bzr_rescale->rescale_domain(new_domain);

  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts), pt_0, pt_1);
  const qugar::Tolerance tol(1.0e-10);

  for (const auto &eval_pt : eval_pts) {
    const auto scaled_pt = new_domain.scale_to_0_1(eval_pt);
    const auto val = bzr->operator()(eval_pt);
    const auto val_scaled = bzr_rescale->operator()(scaled_pt);
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
    REQUIRE((tol.equal(val, val_scaled)));
  }
}

// NOLINTNEXTLINE (readability-function-cognitive-complexity)
template<int dim> void test_Bezier_domain_rescale_face(const int n_eval_pts)
{
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto bzr = create_random_bezier<dim, 1>(3, 5);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  auto pt_0 = qugar::rand::create_random_point<dim>(-qugar::numbers::two, qugar::numbers::two);
  auto pt_1 = qugar::rand::create_random_point<dim>(-qugar::numbers::two, qugar::numbers::two);

  for (int dir = 0; dir < dim; ++dir) {
    if (pt_0(dir) > pt_1(dir)) {
      std::swap(pt_0(dir), pt_1(dir));
    }
  }


  const qugar::BoundBox<dim> new_domain(pt_0, pt_1);

  const auto bzr_rescale = std::make_shared<qugar::impl::BezierTP<dim>>(*bzr);
  bzr_rescale->rescale_domain(new_domain);


  const auto eval_pts = qugar::rand::create_random_real_vectors<dim>(static_cast<std::size_t>(n_eval_pts), pt_0, pt_1);
  const qugar::Tolerance tol(1.0e-10);

  for (int local_facet_id = 0; local_facet_id < dim * 2; ++local_facet_id) {

    const int const_dir = qugar::impl::get_facet_constant_dir<dim>(local_facet_id);
    const int side = qugar::impl::get_facet_side<dim>(local_facet_id);

    const auto bzr_rescale_facet = bzr_rescale->extract_facet(local_facet_id);

    for (auto eval_pt : eval_pts) {
      auto scaled_pt = new_domain.scale_to_0_1(eval_pt);
      scaled_pt(const_dir) = side == 0 ? qugar::numbers::zero : qugar::numbers::one;
      eval_pt(const_dir) = side == 0 ? new_domain.min(const_dir) : new_domain.max(const_dir);

      const qugar::Point<dim - 1> facet_scaled_pt = qugar::remove_component(scaled_pt, const_dir);
      // const qugar::Point<dim - 1> facet_eval_pt = qugar::remove_component(eval_pt, const_dir);

      const auto val = bzr->operator()(eval_pt);
      const auto val_scaled = bzr_rescale->operator()(scaled_pt);
      const auto val_scaled_facet = bzr_rescale_facet->operator()(facet_scaled_pt);

      // NOLINTBEGIN (cppcoreguidelines-avoid-do-while)
      REQUIRE((tol.equal(val, val_scaled)));
      REQUIRE((tol.equal(val, val_scaled_facet)));
      // NOLINTEND (cppcoreguidelines-avoid-do-while)
    }
  }
}
}// namespace

// NOLINTNEXTLINE (misc-use-anonymous-namespace)
TEST_CASE("Testing Beziers domain rescaling", "[bezier_tp]")
{
  using namespace qugar;
  const int n_eval_pts = 5;

  test_Bezier_domain_rescale<1>(n_eval_pts);
  test_Bezier_domain_rescale<2>(n_eval_pts);
  test_Bezier_domain_rescale<3>(n_eval_pts);

  test_Bezier_domain_rescale_face<2>(n_eval_pts);
  test_Bezier_domain_rescale_face<3>(n_eval_pts);
}
