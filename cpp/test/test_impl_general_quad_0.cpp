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
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <catch2/catch_test_macros.hpp>

#include "quadrature_test_utils.hpp"
#include "qugar/unfitted_domain.hpp"

#include <qugar/cart_grid_tp.hpp>
#include <qugar/cut_quadrature.hpp>
#include <qugar/impl_reparam.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/tpms_lib.hpp>
#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

template<int dim>
void check_num_cells(const qugar::UnfittedDomain<dim> &domain,
  const std::size_t n_full_cells,
  const std::size_t n_empty_cells,
  const std::size_t n_cut_cells)
{
  // NOLINTBEGIN (bugprone-chained-comparison)
  std::vector<std::int64_t> full_cells;
  domain.get_full_cells(full_cells);
  REQUIRE(full_cells.size() == n_full_cells);

  std::vector<std::int64_t> empty_cells;
  domain.get_empty_cells(empty_cells);
  REQUIRE(empty_cells.size() == n_empty_cells);

  std::vector<std::int64_t> cut_cells;
  domain.get_cut_cells(cut_cells);
  REQUIRE(cut_cells.size() == n_cut_cells);
  // NOLINTEND (bugprone-chained-comparison)
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Schwarz-Diamond 2D function quadrature", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto func = std::make_shared<tpms::SchwarzDiamond<2>>(qugar::Vector<real, 2>(1., 1.), numbers::zero);
  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ 12, 12 }));

  const UnfittedImplDomain<2> unf_domain(func, grid);

  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  check_num_cells(unf_domain, 72, 72, 0);

  const int n_pts_dir{ 3 };

  std::vector<std::int64_t> cut_facets_cells;
  std::vector<int> cut_facets_local_facets_ids;
  std::vector<std::int64_t> full_facets_cells;
  std::vector<int> full_facets_local_facets_ids;
  std::vector<std::int64_t> empty_facets_cells;
  std::vector<int> empty_facets_local_facets_ids;
  std::vector<std::int64_t> full_unfitted_facets_cells;
  std::vector<int> full_unfitted_facets_local_facets_ids;
  std::vector<std::int64_t> unfitted_facets_cells;
  std::vector<int> unfitted_facets_local_facets_ids;
  unf_domain.get_empty_facets(empty_facets_cells, empty_facets_local_facets_ids);
  unf_domain.get_full_facets(full_facets_cells, full_facets_local_facets_ids);
  unf_domain.get_cut_facets(cut_facets_cells, cut_facets_local_facets_ids);
  unf_domain.get_full_unfitted_facets(full_unfitted_facets_cells, full_unfitted_facets_local_facets_ids);
  unf_domain.get_unfitted_facets(unfitted_facets_cells, unfitted_facets_local_facets_ids);

  // NOLINTBEGIN (bugprone-chained-comparison)
  REQUIRE(cut_facets_cells.size() == 0);
  REQUIRE(cut_facets_local_facets_ids.size() == 0);
  REQUIRE(full_facets_cells.size() == 240);
  REQUIRE(full_facets_local_facets_ids.size() == 240);
  REQUIRE(empty_facets_cells.size() == 288);
  REQUIRE(empty_facets_local_facets_ids.size() == 288);
  REQUIRE(full_unfitted_facets_cells.size() == 48);
  REQUIRE(full_unfitted_facets_local_facets_ids.size() == 48);
  REQUIRE(unfitted_facets_cells.size() == 48);
  REQUIRE(unfitted_facets_local_facets_ids.size() == 48);
  // NOLINTEND (bugprone-chained-comparison)

  const Tolerance tol(1.0e-6);

  std::vector<std::int64_t> cut_cells;
  unf_domain.get_cut_cells(cut_cells);

  const auto quad = create_quadrature<2>(unf_domain, cut_cells, n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(quad->points.size() == 0);

  const Point<2> target_centroid(0.0, 0.0);
  const auto centroid = compute_points_centroid(quad->points, quad->weights);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(centroid, target_centroid));

  constexpr bool include_facet_unf_bdry{ true };
  constexpr bool exclude_ext_bdry{ true };
  const auto unf_bound_quad =
    create_unfitted_bound_quadrature<2>(unf_domain, cut_cells, n_pts_dir, include_facet_unf_bdry, exclude_ext_bdry);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(unf_bound_quad->points.size() == 0);

  auto facet_quad =
    create_facets_quadrature_exterior_integral<2>(unf_domain, cut_facets_cells, cut_facets_local_facets_ids, n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(facet_quad->points.size() == 0);

  facet_quad = create_facets_quadrature_exterior_integral<2>(
    unf_domain, full_unfitted_facets_cells, full_unfitted_facets_local_facets_ids, n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(facet_quad->points.size() == 144);

  auto facet_centroid = compute_points_centroid<1>(facet_quad->points, facet_quad->weights);
  const Point<1> target_facet_centroid(0.5);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(facet_centroid, target_facet_centroid));

  facet_quad = create_facets_quadrature_exterior_integral<2>(
    unf_domain, unfitted_facets_cells, unfitted_facets_local_facets_ids, n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(facet_quad->points.size() == 144);

  facet_centroid = compute_points_centroid<1>(facet_quad->points, facet_quad->weights);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(facet_centroid, target_facet_centroid));
}


// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Schoen Gyroid 2D function quadrature", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto func = std::make_shared<tpms::Schoen<2>>(qugar::Vector<real, 2>(1., 1.), numbers::zero);
  const auto grid = std::make_shared<CartGridTP<2>>(std::array<std::size_t, 2>({ 4, 4 }));

  const UnfittedImplDomain<2> unf_domain(func, grid);

  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  check_num_cells(unf_domain, 4, 4, 8);

  const int n_pts_dir{ 3 };

  std::vector<std::int64_t> cut_facets_cells;
  std::vector<int> cut_facets_local_facets_ids;
  std::vector<std::int64_t> full_facets_cells;
  std::vector<int> full_facets_local_facets_ids;
  std::vector<std::int64_t> empty_facets_cells;
  std::vector<int> empty_facets_local_facets_ids;
  std::vector<std::int64_t> full_unfitted_facets_cells;
  std::vector<int> full_unfitted_facets_local_facets_ids;
  std::vector<std::int64_t> unfitted_facets_cells;
  std::vector<int> unfitted_facets_local_facets_ids;
  unf_domain.get_empty_facets(empty_facets_cells, empty_facets_local_facets_ids);
  unf_domain.get_full_facets(full_facets_cells, full_facets_local_facets_ids);
  unf_domain.get_cut_facets(cut_facets_cells, cut_facets_local_facets_ids);
  unf_domain.get_full_unfitted_facets(full_unfitted_facets_cells, full_unfitted_facets_local_facets_ids);
  unf_domain.get_unfitted_facets(unfitted_facets_cells, unfitted_facets_local_facets_ids);

  // NOLINTBEGIN (bugprone-chained-comparison)
  REQUIRE(cut_facets_cells.size() == 8);
  REQUIRE(cut_facets_local_facets_ids.size() == 8);
  REQUIRE(full_facets_cells.size() == 28);
  REQUIRE(full_facets_local_facets_ids.size() == 28);
  REQUIRE(empty_facets_cells.size() == 28);
  REQUIRE(empty_facets_local_facets_ids.size() == 28);
  REQUIRE(full_unfitted_facets_cells.size() == 0);
  REQUIRE(full_unfitted_facets_local_facets_ids.size() == 0);
  REQUIRE(unfitted_facets_cells.size() == 0);
  REQUIRE(unfitted_facets_local_facets_ids.size() == 0);
  // NOLINTEND (bugprone-chained-comparison)

  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  const Tolerance tol(1.0e-6);

  std::vector<std::int64_t> cut_cells;
  unf_domain.get_cut_cells(cut_cells);

  const auto quad = create_quadrature<2>(unf_domain, cut_cells, n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(quad->points.size() == 213);

  const Point<2> target_centroid(0.5, 0.5);
  const auto centroid = compute_points_centroid(quad->points, quad->weights);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(centroid, target_centroid));

  constexpr bool include_facet_unf_bdry{ true };
  constexpr bool exclude_ext_bdry{ true };
  const auto unf_bound_quad =
    create_unfitted_bound_quadrature<2>(unf_domain, cut_cells, n_pts_dir, include_facet_unf_bdry, exclude_ext_bdry);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(unf_bound_quad->points.size() == 60);

  const auto unf_bound_centroid = compute_points_centroid<2>(unf_bound_quad->points, unf_bound_quad->weights);
  const Point<2> target_unf_bound_centroid(0.5, 0.5);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(unf_bound_centroid, target_unf_bound_centroid));

  const auto facet_quad =
    create_facets_quadrature_exterior_integral<2>(unf_domain, cut_facets_cells, cut_facets_local_facets_ids, n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(facet_quad->points.size() == 0);

  const auto facet_centroid = compute_points_centroid<1>(facet_quad->points, facet_quad->weights);
  const Point<1> target_facet_centroid(0.0);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(facet_centroid, target_facet_centroid));
}

// // NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("General function quadrature", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const auto func = std::make_shared<tpms::Schoen<3>>(qugar::Vector<real, 3>(2., 2., 2.));
  const auto grid = std::make_shared<CartGridTP<3>>(std::array<std::size_t, 3>({ 16, 16, 16 }));

  const UnfittedImplDomain<3> unf_domain(func, grid);

  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  check_num_cells(unf_domain, 896, 896, 2304);

  const int n_pts_dir{ 3 };

  std::vector<std::int64_t> cut_facets_cells;
  std::vector<int> cut_facets_local_facets_ids;
  std::vector<std::int64_t> full_facets_cells;
  std::vector<int> full_facets_local_facets_ids;
  std::vector<std::int64_t> empty_facets_cells;
  std::vector<int> empty_facets_local_facets_ids;
  unf_domain.get_empty_facets(empty_facets_cells, empty_facets_local_facets_ids);
  unf_domain.get_full_facets(full_facets_cells, full_facets_local_facets_ids);
  unf_domain.get_cut_facets(cut_facets_cells, cut_facets_local_facets_ids);

  // NOLINTBEGIN (bugprone-chained-comparison)
  REQUIRE(cut_facets_cells.size() == 8448);
  REQUIRE(cut_facets_local_facets_ids.size() == 8448);
  REQUIRE(full_facets_cells.size() == 8064);
  REQUIRE(full_facets_local_facets_ids.size() == 8064);
  REQUIRE(empty_facets_cells.size() == 8064);
  REQUIRE(empty_facets_local_facets_ids.size() == 8064);
  // NOLINTEND (bugprone-chained-comparison)

  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  const Tolerance tol(1.0e-6);

  std::vector<std::int64_t> cut_cells;
  unf_domain.get_cut_cells(cut_cells);

  const auto quad = create_quadrature<3>(unf_domain, cut_cells, n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(quad->points.size() == 795837);

  const Point<3> target_centroid(0.5000000310737452, 0.4999998788332324, 0.4999999483123341);
  const auto centroid = compute_points_centroid(quad->points, quad->weights);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(centroid, target_centroid));

  constexpr bool include_facet_unf_bdry{ true };
  constexpr bool exclude_ext_bdry{ true };
  const auto unf_bound_quad =
    create_unfitted_bound_quadrature<3>(unf_domain, cut_cells, n_pts_dir, include_facet_unf_bdry, exclude_ext_bdry);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(unf_bound_quad->points.size() == 177421);

  const auto unf_bound_centroid = compute_points_centroid<3>(unf_bound_quad->points, unf_bound_quad->weights);
  const Point<3> target_unf_bound_centroid(0.4999957334273454, 0.4999960466067558, 0.4999963189195731);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(unf_bound_centroid, target_unf_bound_centroid));

  const auto facet_quad =
    create_facets_quadrature_exterior_integral<3>(unf_domain, cut_facets_cells, cut_facets_local_facets_ids, n_pts_dir);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE(facet_quad->points.size() == 4626);

  const auto facet_centroid = compute_points_centroid<2>(facet_quad->points, facet_quad->weights);
  const Point<2> target_facet_centroid(0.50000000000001399, 0.50000000000003864);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE(tol.coincident(facet_centroid, target_facet_centroid));
}
