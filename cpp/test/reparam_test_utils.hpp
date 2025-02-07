// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file reparam_test_utils.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Utils for reparameterization testing.
//! @date 2025-01-08
//!
//! @copyright Copyright (c) 2025-present


#ifndef QUGAR_TEST_REPARAM_TEST_UTILS_HPP
#define QUGAR_TEST_REPARAM_TEST_UTILS_HPP

#include "quadrature_test_utils.hpp"

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_reparam.hpp>
#include <qugar/impl_reparam_bezier.hpp>
#include <qugar/impl_reparam_general.hpp>
#include <qugar/impl_reparam_mesh.hpp>
#include <qugar/tolerance.hpp>

#include <cstddef>
#include <iomanip>
#include <vector>

#include <catch2/catch_test_macros.hpp>

inline std::size_t compute_average(const std::vector<std::size_t> &values)
{
  if (values.empty()) {
    return 0;
  } else {
    // Not in C++20 ranges. Maybe in C++23.
    // NOLINTNEXTLINE (boost-use-ranges)
    return std::accumulate(values.cbegin(), values.cend(), std::size_t{ 0 }) / values.size();
  }
}

inline std::size_t compute_moment(const std::vector<std::size_t> &values)
{
  std::size_t moment{ 0 };
  for (std::size_t i = 0; i < values.size(); ++i) {
    moment += i * values[i];
  }
  if (!values.empty()) {
    moment /= values.size();
  }

  return moment;
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim, int range>
void test_reparam_generic(const qugar::impl::ImplReparamMesh<dim, range> &reparam,
  const std::size_t n_points,
  const std::size_t n_connectivity,
  const std::size_t n_wires_connectivity,
  const std::size_t connecitivity_avg,
  const std::size_t connecitivity_wires_avg,
  const std::size_t connecitivity_moment,
  const std::size_t connecitivity_wires_moment,
  const qugar::Point<range> &expected_centroid)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  using namespace qugar;
  using namespace qugar::impl;

#if 0
  static_cast<void>(n_points);
  static_cast<void>(n_connectivity);
  static_cast<void>(n_wires_connectivity);
  static_cast<void>(connecitivity_avg);
  static_cast<void>(connecitivity_wires_avg);
  static_cast<void>(connecitivity_moment);
  static_cast<void>(connecitivity_wires_moment);
  static_cast<void>(expected_centroid);

  const auto centroid = compute_points_centroid(reparam.get_points());
  std::cerr << reparam.get_points().size() << ", ";
  std::cerr << reparam.get_connectivity().size() << ", ";
  std::cerr << reparam.get_wires_connectivity().size() << ", ";
  std::cerr << compute_average(reparam.get_connectivity()) << ", ";
  std::cerr << compute_average(reparam.get_wires_connectivity()) << ", ";
  std::cerr << compute_moment(reparam.get_connectivity()) << ", ";
  std::cerr << compute_moment(reparam.get_wires_connectivity()) << ", ";
  std::cerr << std::setprecision(18);
  std::cerr << "Point<" << range << ">{" << centroid(0);
  for (int dir = 1; dir < range; ++dir) {
    std::cerr << ", " << centroid(dir);
  }
  std::cerr << "}" << std::endl;
#else
  // NOLINTBEGIN (bugprone-chained-comparison)
  REQUIRE((reparam.get_points().size() == n_points));
  REQUIRE((reparam.get_connectivity().size() == n_connectivity));
  REQUIRE((reparam.get_wires_connectivity().size() == n_wires_connectivity));

  REQUIRE((compute_average(reparam.get_connectivity()) == connecitivity_avg));
  REQUIRE((compute_average(reparam.get_wires_connectivity()) == connecitivity_wires_avg));

  REQUIRE((compute_moment(reparam.get_connectivity()) == connecitivity_moment));
  REQUIRE((compute_moment(reparam.get_wires_connectivity()) == connecitivity_wires_moment));

  const Tolerance tol(101.0 * numbers::eps);
  const auto centroid = compute_points_centroid(reparam.get_points());
  REQUIRE((tol.coincident(centroid, expected_centroid)));
// NOLINTEND (bugprone-chained-comparison)
#endif
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim, bool levelset>
void test_reparam_Bezier(const std::shared_ptr<const qugar::impl::BezierTP<dim, 1>> bzr,
  const qugar::BoundBox<dim> &domain,
  const int order,
  const std::size_t n_points,
  const std::size_t n_connectivity,
  const std::size_t n_wires_connectivity,
  const std::size_t connecitivity_avg,
  const std::size_t connecitivity_wires_avg,
  const std::size_t connecitivity_moment,
  const std::size_t connecitivity_wires_moment,
  const qugar::Point<dim> &expected_centroid)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto reparam = reparam_Bezier<dim, levelset>(*bzr, domain, order);

  const Tolerance tol(2.e4 * numbers::eps);
  reparam->merge_coincident_points(tol);


  test_reparam_generic(*reparam,
    n_points,
    n_connectivity,
    n_wires_connectivity,
    connecitivity_avg,
    connecitivity_wires_avg,
    connecitivity_moment,
    connecitivity_wires_moment,
    expected_centroid);
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim, bool levelset>
void test_reparam_general(const std::shared_ptr<const qugar::impl::ImplicitFunc<dim>> func,
  const qugar::BoundBox<dim> &domain,
  const int order,
  const std::size_t n_points,
  const std::size_t n_connectivity,
  const std::size_t n_wires_connectivity,
  const std::size_t connecitivity_avg,
  const std::size_t connecitivity_wires_avg,
  const std::size_t connecitivity_moment,
  const std::size_t connecitivity_wires_moment,
  const qugar::Point<dim> &expected_centroid)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto reparam = reparam_general<dim, levelset>(*func, domain, order);
  const Tolerance tol(2.0e4 * numbers::eps);
  reparam->merge_coincident_points(tol);

  test_reparam_generic(*reparam,
    n_points,
    n_connectivity,
    n_wires_connectivity,
    connecitivity_avg,
    connecitivity_wires_avg,
    connecitivity_moment,
    connecitivity_wires_moment,
    expected_centroid);
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
void test_reparam_general_facet(const std::shared_ptr<const qugar::impl::ImplicitFunc<dim>> func,
  const qugar::BoundBox<dim> &domain,
  const int local_facet_id,
  const int order,
  const std::size_t n_points,
  const std::size_t n_connectivity,
  const std::size_t n_wires_connectivity,
  const std::size_t connecitivity_avg,
  const std::size_t connecitivity_wires_avg,
  const std::size_t connecitivity_moment,
  const std::size_t connecitivity_wires_moment,
  const qugar::Point<dim> &expected_centroid)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto reparam = reparam_general_facet<dim>(*func, domain, local_facet_id, order);

  const Tolerance tol(2.e4 * numbers::eps);
  reparam->merge_coincident_points(tol);

  test_reparam_generic(*reparam,
    n_points,
    n_connectivity,
    n_wires_connectivity,
    connecitivity_avg,
    connecitivity_wires_avg,
    connecitivity_moment,
    connecitivity_wires_moment,
    expected_centroid);
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim, bool levelset>
void test_reparam(const std::shared_ptr<const qugar::impl::ImplicitFunc<dim>> func,
  const std::shared_ptr<const qugar::CartGridTP<dim>> grid,
  const int order,
  const std::size_t n_points,
  const std::size_t n_connectivity,
  const std::size_t n_wires_connectivity,
  const std::size_t connecitivity_avg,
  const std::size_t connecitivity_wires_avg,
  const std::size_t connecitivity_moment,
  const std::size_t connecitivity_wires_moment,
  const qugar::Point<dim> &expected_centroid)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  using namespace qugar;
  using namespace qugar::impl;

  assert(func != nullptr);
  assert(grid != nullptr);

  const qugar::impl::UnfittedImplDomain<dim> unf_domain(func, grid);

  const auto reparam = create_reparameterization<dim, levelset>(unf_domain, order);

  test_reparam_generic(*reparam,
    n_points,
    n_connectivity,
    n_wires_connectivity,
    connecitivity_avg,
    connecitivity_wires_avg,
    connecitivity_moment,
    connecitivity_wires_moment,
    expected_centroid);
}

#endif// QUGAR_TEST_REPARAM_TEST_UTILS_HPP