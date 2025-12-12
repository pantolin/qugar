// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_bspline_tp.cpp
//! @author Dusan Cvijetic (dusan.cvijetic@epfl.ch)
//! @brief Tests for BSplineTP class.
//! @date 2025-12-08
//!
//! @copyright Copyright (c) 2025-present

#include "bezier_test_utils.hpp"

#include<algoim/bspline.hpp>
#include<algoim/interval.hpp>
#include<qugar/bspline_bezier_tp.hpp>
#include<qugar/bspline_tp.hpp>
#include<qugar/point.hpp>
#include<qugar/tolerance.hpp>

#include<catch2/catch_test_macros.hpp>

// TODO: More meaningful tests, this just checks whether the wrapper works.
TEST_CASE("B-spline tensor product basis functions sanity check", "[bspline_tp]") {
  constexpr int num_spans = 3;
  constexpr int degree = 3;
  constexpr int num_ctrl_pts_1D = (num_spans + degree - 1);
  constexpr int num_ctrl_pts = num_ctrl_pts_1D * num_ctrl_pts_1D * num_ctrl_pts_1D;
  auto open_knot    = algoim::bspline::Knots(0.0, 1.0, num_spans, degree);

  // Create control points in 1D
  auto ctrlpts= std::vector<double>();
  ctrlpts.resize(num_ctrl_pts);
  for (int i = 0; i < num_ctrl_pts; ++i) {
    ctrlpts[i] = static_cast<double>(i) / static_cast<double>(num_ctrl_pts - 1);
  }
  
  const qugar::Tolerance tol(1000 * qugar::numbers::eps);

  auto bspline_x    = std::make_shared<algoim::bspline::BSpline>(open_knot, degree);
  auto bspline_y    = std::make_shared<algoim::bspline::BSpline>(open_knot, degree);
  auto bspline_z    = std::make_shared<algoim::bspline::BSpline>(open_knot, degree);

  std::array<std::shared_ptr<const algoim::bspline::BSpline>, 3> bsplines_1D = {
    bspline_x,
    bspline_y,
    bspline_z
  };

  auto bspline_tp_algoim = std::make_shared<algoim::bspline::BSplineTP<3, 1, double>>(ctrlpts, bsplines_1D);
  auto bspline           = qugar::impl::BSplineTP<3, 1>(bspline_tp_algoim);
  auto bspline_bezier    = qugar::impl::BSplineBezierTP<3, 1>(bspline_tp_algoim);

  qugar::Point<3> point{0.5, 0.5, 0.5};
  qugar::Point<3> test_point_000{0.1, 0.1, 0.1};
  qugar::Point<3> test_point_111{0.5, 0.5, 0.5};
  qugar::Point<3> test_point_012{0.1, 0.5, 0.9};

  using Interval = algoim::Interval<3>;
  qugar::Point<3, Interval> interval_point(
    Interval(0.4, 0.6),
    Interval(0.4, 0.6),
    Interval(0.4, 0.6)
  );

  SECTION("Check bezier Indexing") {
    int index = 0;
    std::array<int, 3> multi_index{};
    multi_index = bspline_bezier.get_knot_multi_index(test_point_000);
    REQUIRE(multi_index[0] == 0);
    REQUIRE(multi_index[1] == 0);
    REQUIRE(multi_index[2] == 0);
    index = bspline_bezier.get_bezier_index(multi_index);
    REQUIRE(index == 0);
    multi_index = bspline_bezier.get_knot_multi_index(test_point_111);
    REQUIRE(multi_index[0] == 1);
    REQUIRE(multi_index[1] == 1);
    REQUIRE(multi_index[2] == 1);
    index = bspline_bezier.get_bezier_index(multi_index);
    REQUIRE(index == 1 * (num_spans) * (num_spans) + 1 * (num_spans) + 1);
    multi_index = bspline_bezier.get_knot_multi_index(test_point_012);
    REQUIRE(multi_index[0] == 0);
    REQUIRE(multi_index[1] == 1);
    REQUIRE(multi_index[2] == 2);
    index = bspline_bezier.get_bezier_index(multi_index);
    REQUIRE(index == 0 * (num_spans) * (num_spans) + 1 * (num_spans) + 2);
  }

  SECTION("Evaluate at a point") {
    auto value_bspl = (*bspline_tp_algoim)(point);
    auto value_bsbz = bspline_bezier(point);
    REQUIRE(compare_values(value_bspl, value_bsbz, tol));

    auto value_bspl_000 = (*bspline_tp_algoim)(test_point_000);
    auto value_bsbz_000 = bspline_bezier(test_point_000);
    REQUIRE(compare_values(value_bspl_000, value_bsbz_000, tol));

    auto value_bspl_111 = (*bspline_tp_algoim)(test_point_111);
    auto value_bsbz_111 = bspline_bezier(test_point_111);
    REQUIRE(compare_values(value_bspl_111, value_bsbz_111, tol));

    auto value_bspl_012 = (*bspline_tp_algoim)(test_point_012);
    auto value_bsbz_012 = bspline_bezier(test_point_012);
    REQUIRE(compare_values(value_bspl_012, value_bsbz_012, tol));
  }

  SECTION("Evaluate at an interval") {
    // TODO
  }

  SECTION("Gradient at a point") {
    auto grad_bspl = bspline_tp_algoim->grad(point);
    auto grad_bsbz = bspline_bezier.grad(point);
    std::cout << "B-spline gradient at point: " << grad_bspl << std::endl;
    std::cout << "B-spline Bezier gradient at point: " << grad_bsbz << std::endl;
    REQUIRE(compare_vectors(grad_bspl, grad_bsbz, tol));
  }

  SECTION("Gradient at an interval") {
    // TODO
  }

  SECTION("Hessian at a point") {
    auto hess_bspl = bspline_tp_algoim->hessian(point);
    auto hess_bsbz = bspline_bezier.hessian(point);
    REQUIRE(compare_vectors(hess_bspl, hess_bsbz, tol));
  }
  
}


