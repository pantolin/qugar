// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_tpms_derivatives_0.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 0 for the derivatives of the TPMS functions: gradient against central finite differences
//! of the value, hessian against central finite differences of the gradient.
//! @date 2026-10-19
//!
//! @copyright Copyright (c) 2025-present

#include <catch2/catch_test_macros.hpp>

#include <qugar/point.hpp>
#include <qugar/tpms_lib.hpp>
#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace {

using qugar::Point;
using qugar::real;
using qugar::Vector;

//! Finite-difference step: balances the truncation error of the central difference,
//! about step^2 * k^3 / 6 for a gradient whose third derivative scales as k^3 (k = 2 pi times the
//! largest period), against its rounding error, about eps * k / step.
constexpr real fd_step = 1.0e-5;

//! Tolerance on |H_fd - H| relative to k^2, the scale of the hessian. With k at most 4 pi below,
//! both errors of `fd_step` are below 1e-8 relative; a wrong entry differs by O(1) relative.
constexpr real rel_tol = 1.0e-6;

//! Index of the (i, j) entry, i <= j, in the packed hessian: rows of the upper triangle.
template<int dim> int packed_index(const int i, const int j)
{
  return i * dim - (i * (i - 1)) / 2 + (j - i);
}

//! Points where the derivatives are checked: generic, off every symmetry plane of the TPMS.
constexpr std::array<std::array<real, 3>, 4> test_points{
  { { 0.13, 0.71, 0.37 }, { 0.58, 0.29, 0.83 }, { 0.91, 0.47, 0.06 }, { 0.24, 0.66, 0.52 } }
};

template<int dim> Point<dim> make_point(const std::array<real, 3> &coords)
{
  Point<dim> point;
  for (int i = 0; i < dim; ++i) {
    point(i) = coords[static_cast<std::size_t>(i)];
  }
  return point;
}

//! Gradient against central differences of the value; the same error analysis as the hessian's,
//! one derivative lower, so the tolerance is relative to k.
template<int dim, typename Func> void check_gradient(const Func &func, const real k)
{
  for (const auto &coords : test_points) {
    const Point<dim> point = make_point<dim>(coords);
    const auto grad = func.grad(point);

    for (int j = 0; j < dim; ++j) {
      Point<dim> plus = point;
      Point<dim> minus = point;
      plus(j) += fd_step;
      minus(j) -= fd_step;
      const real fd = (func(plus) - func(minus)) / (2.0 * fd_step);
      CAPTURE(j, coords[0], coords[1], coords[2], fd, grad(j));
      CHECK(std::abs(fd - grad(j)) <= rel_tol * k);
    }
  }
}

template<int dim, typename Func> void check_hessian(const Func &func, const real k)
{
  for (const auto &coords : test_points) {
    const Point<dim> point = make_point<dim>(coords);
    const auto hess = func.hessian(point);

    for (int j = 0; j < dim; ++j) {
      Point<dim> plus = point;
      Point<dim> minus = point;
      plus(j) += fd_step;
      minus(j) -= fd_step;
      const auto grad_plus = func.grad(plus);
      const auto grad_minus = func.grad(minus);
      for (int i = 0; i <= j; ++i) {
        const real fd = (grad_plus(i) - grad_minus(i)) / (2.0 * fd_step);
        const real exact = hess(packed_index<dim>(i, j));
        CAPTURE(i, j, coords[0], coords[1], coords[2], fd, exact);
        CHECK(std::abs(fd - exact) <= rel_tol * k * k);
      }
    }
  }
}

template<template<int> class TPMS> void check_tpms()
{
  // Distinct periods per direction, so a component scaled by another direction's period shows.
  const Vector<real, 3> periods3(1.0, 2.0, 1.5);
  const real k3 = 2.0 * std::numbers::pi * 2.0;
  check_gradient<3>(TPMS<3>(periods3), k3);
  check_hessian<3>(TPMS<3>(periods3), k3);

  const Vector<real, 2> periods2(1.5, 1.0);
  const real k2 = 2.0 * std::numbers::pi * 1.5;
  check_gradient<2>(TPMS<2>(periods2, 0.3), k2);
  check_hessian<2>(TPMS<2>(periods2, 0.3), k2);
}

}// namespace

TEST_CASE("Schoen derivatives against finite differences", "[tpms]")
{
  check_tpms<qugar::impl::tpms::Schoen>();
}

TEST_CASE("Schoen IWP derivatives against finite differences", "[tpms]")
{
  check_tpms<qugar::impl::tpms::SchoenIWP>();
}

TEST_CASE("Schoen FRD derivatives against finite differences", "[tpms]")
{
  check_tpms<qugar::impl::tpms::SchoenFRD>();
}

TEST_CASE("Fischer-Koch S derivatives against finite differences", "[tpms]")
{
  check_tpms<qugar::impl::tpms::FischerKochS>();
}

TEST_CASE("Schwarz Diamond derivatives against finite differences", "[tpms]")
{
  check_tpms<qugar::impl::tpms::SchwarzDiamond>();
}

TEST_CASE("Schwarz Primitive derivatives against finite differences", "[tpms]")
{
  check_tpms<qugar::impl::tpms::SchwarzPrimitive>();
}
