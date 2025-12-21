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

#include<qugar/cart_grid_tp.hpp>
#include<qugar/bspline_bezier_tp.hpp>
#include<qugar/bspline_tp.hpp>
#include<qugar/impl_unfitted_domain.hpp>
#include<qugar/point.hpp>
#include<qugar/tolerance.hpp>

#include<catch2/catch_test_macros.hpp>


/* Helper function to generate control points for a 2D linear B-spline
 f(u, v) = u + v + shift on the tensor-product basis:
 ctrlpts[i,j] = gu[i] + gv[j] + shift
 where gu and gv are the Gervill abscissae in u and v directions, respectively.
*/
void generate_linear_ctrlpts_2D(std::vector<double> &ctrlpts,
                                const std::vector<double> &gervill_abscissae,
                                int num_ctrl_pts_1D,
                                double shift = 0) {
  ctrlpts.erase(ctrlpts.begin(), ctrlpts.end());
  ctrlpts.resize(num_ctrl_pts_1D * num_ctrl_pts_1D);
  for (int j = 0; j < num_ctrl_pts_1D; ++j) {
    for (int i = 0; i < num_ctrl_pts_1D; ++i) {
      ctrlpts[i + j * num_ctrl_pts_1D] = gervill_abscissae[i] + gervill_abscissae[j] + shift;
    }
  }
}



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
    std::cout << "B-spline gradient at point: " << grad_bspl << "\n";
    std::cout << "B-spline Bezier gradient at point: " << grad_bsbz << "\n";
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

TEST_CASE("Quadrature cells for B-Splines", "[bspline_tp_quadrature]") {
  constexpr int num_spans = 3;
  constexpr int order = 2;
  constexpr int num_ctrl_pts_1D = (num_spans + order - 1);
  auto open_knot    = algoim::bspline::Knots(0.0, 1.0, num_spans, order);

  // Create Gervill abscissae on our knot vector
  std::vector<double> gervill_abscissae{};
  gervill_abscissae.resize(num_ctrl_pts_1D);
  for (size_t i = 0; i < num_ctrl_pts_1D; ++i) {
    gervill_abscissae[i] = open_knot.data[i + order - 1];
  }

  auto bspline_x    = std::make_shared<algoim::bspline::BSpline>(open_knot, order);
  auto bspline_y    = std::make_shared<algoim::bspline::BSpline>(open_knot, order);
  std::array<std::shared_ptr<const algoim::bspline::BSpline>, 2> bsplines_1D = {
    bspline_x,
    bspline_y
  };

  SECTION("Linear B-spline f(x,y) = x + y + 1, all positive cells") {
    // Create control points for f(x,y) = x + y + 1
    auto ctrlpts = std::vector<double>();
    generate_linear_ctrlpts_2D(ctrlpts, 
                               gervill_abscissae,
                               num_ctrl_pts_1D,
                               1.0);
    auto bspline_tp_algoim = std::make_shared<algoim::bspline::BSplineTP<2, 1, double>>(ctrlpts, bsplines_1D);
    auto bspline           = std::make_shared<qugar::impl::BSplineBezierTP<2, 1>>(bspline_tp_algoim);

    std::cout << "B-spline evaluation:" << "\n";
    for (int i = 0; i < num_spans; ++i) {
      for (int j = 0; j < num_spans; ++j) {
        double x = (i + 0.5) / static_cast<double>(num_spans);
        double y = (j + 0.5) / static_cast<double>(num_spans);
        std::cout << std::setw(6)
                  << std::fixed
                  << std::setprecision(2)
                  << bspline->operator()(qugar::Point<2>{x, y}) << " ";
      }
      std::cout << "\n";
    }

    // Craete cartesian grid for quadrature cells
    std::array<std::vector<double>, 2> breaks{};
    for (int d = 0; d < 2; ++d) {
      breaks[d] = open_knot.getUnique();
    }
    auto grid = std::make_shared<qugar::CartGridTP<2>>(breaks);
    auto impl_domain = qugar::impl::UnfittedImplDomain<2>(bspline, grid);

    auto num_full_cells = impl_domain.get_num_full_cells();
    auto num_cut_cells  = impl_domain.get_num_cut_cells();
    auto num_empty_cells= impl_domain.get_num_empty_cells();

    std::cout << "ALL POSITIVE CELLS TEST" << "\n";
    std::cout << "Quadrature cells classification:" << "\n";
    std::cout << "f c e" << "\n";
    std::cout << num_full_cells << " "
              << num_cut_cells  << " "
              << num_empty_cells<< "\n";
    
    REQUIRE(num_empty_cells == 9);
  };

  SECTION("Linear B-spline f(x,y) = x + y") {
    // Create control points for f(x,y) = x + y
    auto ctrlpts = std::vector<double>();
    generate_linear_ctrlpts_2D(ctrlpts, 
                               gervill_abscissae,
                               num_ctrl_pts_1D,
                               -1.0);
    auto bspline_tp_algoim = std::make_shared<algoim::bspline::BSplineTP<2, 1, double>>(ctrlpts, bsplines_1D);
    auto bspline           = std::make_shared<qugar::impl::BSplineBezierTP<2, 1>>(bspline_tp_algoim);

    std::cout << "B-spline evaluation:" << "\n";
    for (int i = 0; i < num_spans; ++i) {
      for (int j = 0; j < num_spans; ++j) {
        double x = (i + 0.5) / static_cast<double>(num_spans);
        double y = (j + 0.5) / static_cast<double>(num_spans);
        std::cout << std::setw(6)
                  << std::fixed
                  << std::setprecision(2)
                  << bspline->operator()(qugar::Point<2>{x, y}) << " ";
      }
      std::cout << "\n";
    }

    // Craete cartesian grid for quadrature cells
    std::array<std::vector<double>, 2> breaks{};
    for (int d = 0; d < 2; ++d) {
      breaks[d] = open_knot.getUnique();
    }
    auto grid = std::make_shared<qugar::CartGridTP<2>>(breaks);
    auto impl_domain = qugar::impl::UnfittedImplDomain<2>(bspline, grid);

    auto num_full_cells = impl_domain.get_num_full_cells();
    auto num_cut_cells  = impl_domain.get_num_cut_cells();
    auto num_empty_cells= impl_domain.get_num_empty_cells();

    std::cout << "MIXED CELLS TEST" << "\n";
    std::cout << "Quadrature cells classification:" << "\n";
    std::cout << "f c e" << "\n";
    std::cout << num_full_cells << " "
              << num_cut_cells  << " "
              << num_empty_cells<< "\n";
    
    REQUIRE(num_full_cells == 3);
    REQUIRE(num_cut_cells  == 3);
    REQUIRE(num_empty_cells== 3);
  };

  SECTION("Linear B-spline f(x,y) = x + y - 3, all negative") {
    // Create control points for f(x,y) = x + y
    auto ctrlpts = std::vector<double>();
    generate_linear_ctrlpts_2D(ctrlpts, 
                               gervill_abscissae,
                               num_ctrl_pts_1D,
                               -3.0);
    auto bspline_tp_algoim = std::make_shared<algoim::bspline::BSplineTP<2, 1, double>>(ctrlpts, bsplines_1D);
    auto bspline           = std::make_shared<qugar::impl::BSplineBezierTP<2, 1>>(bspline_tp_algoim);

    std::cout << "B-spline evaluation:" << "\n";
    for (int i = 0; i < num_spans; ++i) {
      for (int j = 0; j < num_spans; ++j) {
        double x = (i + 0.5) / static_cast<double>(num_spans);
        double y = (j + 0.5) / static_cast<double>(num_spans);
        std::cout << std::setw(6)
                  << std::fixed
                  << std::setprecision(2)
                  << bspline->operator()(qugar::Point<2>{x, y}) << " ";
      }
      std::cout << "\n";
    }

    // Craete cartesian grid for quadrature cells
    std::array<std::vector<double>, 2> breaks{};
    for (int d = 0; d < 2; ++d) {
      breaks[d] = open_knot.getUnique();
    }
    auto grid = std::make_shared<qugar::CartGridTP<2>>(breaks);
    auto impl_domain = qugar::impl::UnfittedImplDomain<2>(bspline, grid);

    auto num_full_cells = impl_domain.get_num_full_cells();
    auto num_cut_cells  = impl_domain.get_num_cut_cells();
    auto num_empty_cells= impl_domain.get_num_empty_cells();

    std::cout << "ALL NEGATIVE CELLS TEST" << "\n";
    std::cout << "Quadrature cells classification:" << "\n";
    std::cout << "f c e" << "\n";
    std::cout << num_full_cells << " "
              << num_cut_cells  << " "
              << num_empty_cells<< "\n";
    
    REQUIRE(num_full_cells == 9);
  }
}

