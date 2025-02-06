// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_general_quad_0.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 0 for Implicit general implict functions quadrature.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <catch2/catch_test_macros.hpp>

#include "quadrature_test_utils.hpp"

#include <qugar/cart_grid_tp.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_quadrature.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/tpms_lib.hpp>
#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <vector>


// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("General function quadrature", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto func = std::make_shared<tpms::Schoen<3>>(qugar::Vector<real, 3>(2., 2., 2.));
  const auto grid = std::make_shared<CartGridTP<3>>(std::array<std::size_t, 3>({ 16, 16, 16 }));

  const UnfittedImplDomain<3> unf_domain(func, grid);

  // NOLINTBEGIN (bugprone-chained-comparison)
  REQUIRE(unf_domain.get_cut_cells().size() == 2304);
  REQUIRE(unf_domain.get_empty_cells().size() == 896);
  REQUIRE(unf_domain.get_full_cells().size() == 896);
  // NOLINTEND (bugprone-chained-comparison)

  const int n_pts_dir{ 3 };

  std::vector<int> cut_facets_cells;
  std::vector<int> cut_facets_local_facets_ids;
  std::vector<int> full_facets_cells;
  std::vector<int> full_facets_local_facets_ids;
  std::vector<int> empty_facets_cells;
  std::vector<int> empty_facets_local_facets_ids;
  unf_domain.get_empty_facets(empty_facets_cells, empty_facets_local_facets_ids);
  unf_domain.get_full_facets(full_facets_cells, full_facets_local_facets_ids);
  unf_domain.get_cut_facets(cut_facets_cells, cut_facets_local_facets_ids);

  // NOLINTBEGIN (bugprone-chained-comparison)
  REQUIRE(cut_facets_cells.size() == 11136);
  REQUIRE(cut_facets_local_facets_ids.size() == 11136);
  REQUIRE(full_facets_cells.size() == 5376);
  REQUIRE(full_facets_local_facets_ids.size() == 5376);
  REQUIRE(empty_facets_cells.size() == 8064);
  REQUIRE(empty_facets_local_facets_ids.size() == 8064);
  // NOLINTEND (bugprone-chained-comparison)

  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  const Tolerance tol(1.0e-6);

  const auto quad = create_quadrature<3>(unf_domain, unf_domain.get_cut_cells(), n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(quad->points.size() == 795837);

  const Point<3> target_centroid(0.5000000310737452, 0.4999998788332324, 0.4999999483123341);
  const auto centroid = compute_points_centroid(quad->points, quad->weights);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(centroid, target_centroid));

  const auto int_bound_quad = create_interior_bound_quadrature<3>(unf_domain, unf_domain.get_cut_cells(), n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(int_bound_quad->points.size() == 177421);

  const auto int_bound_centroid = compute_points_centroid<3>(int_bound_quad->points, int_bound_quad->weights);
  const Point<3> target_int_bound_centroid(0.4999957334273454, 0.4999960466067558, 0.4999963189195731);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(int_bound_centroid, target_int_bound_centroid));

  const auto facet_quad =
    create_facets_quadrature<3>(unf_domain, cut_facets_cells, cut_facets_local_facets_ids, n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(facet_quad->points.size() == 281988);

  const auto facet_centroid = compute_points_centroid<2>(facet_quad->points, facet_quad->weights);
  const Point<2> target_facet_centroid(0.499999990594248, 0.4999999928073127);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(facet_centroid, target_facet_centroid));
}