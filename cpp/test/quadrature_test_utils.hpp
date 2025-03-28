// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file quadrature_test_utils.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Utils for quadrature testing.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#ifndef QUGAR_TEST_QUADRATURE_TEST_UTILS_HPP
#define QUGAR_TEST_QUADRATURE_TEST_UTILS_HPP

#include <qugar/bezier_tp.hpp>
#include <qugar/cut_quadrature.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_quadrature.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <cstdint>
#include <numeric>
#include <vector>

template<int dim>
qugar::Point<dim> compute_points_centroid(const std::vector<qugar::Point<dim>> &points,
  const std::vector<qugar::real> &weights)
{
  qugar::Point<dim> centroid;
  auto w_it = weights.cbegin();
  qugar::real vol{ qugar::numbers::zero };
  for (const auto &point : points) {
    vol += *w_it;
    centroid = centroid + point * *w_it++;
  }
  return (qugar::numbers::one / vol) * centroid;
}


template<int dim> qugar::Point<dim> compute_points_centroid(const std::vector<qugar::Point<dim>> &points)
{
  if (points.empty()) {
    return qugar::Point<dim>{};
  } else {
    std::vector<qugar::real> weights(points.size(), qugar::numbers::one);
    return compute_points_centroid(points, weights);
  }
}

template<int dim>
qugar::Point<dim> compute_centroid(const qugar::impl::UnfittedImplDomain<dim> &unf_domain,
  const qugar::CutCellsQuad<dim> &quad)
{
  using qugar::at;
  qugar::real vol{ qugar::numbers::zero };
  qugar::Point<dim> centroid{ qugar::numbers::zero };

  std::vector<std::int64_t> full_cells;
  unf_domain.get_full_cells(full_cells);

  const auto grid = unf_domain.get_grid();
  for (const auto &cell_id : full_cells) {
    const auto domain = grid->get_cell_domain(cell_id);
    const auto cell_volume = domain.volume();
    vol += cell_volume;
    centroid = centroid + cell_volume * domain.mid_point();
  }

  auto weight = quad.weights.cbegin();
  auto point = quad.points.cbegin();

  const auto n_cut_cells = static_cast<int>(quad.cells.size());
  for (int i = 0; i < n_cut_cells; ++i) {
    const auto cell_id = at(quad.cells, i);
    const auto domain = grid->get_cell_domain(cell_id);
    const auto cell_vol = domain.volume();

    const auto n_pts = at(quad.n_pts_per_cell, i);
    for (int pt_id = 0; pt_id < n_pts; ++pt_id, ++weight, ++point) {
      vol += cell_vol * *weight;
      centroid = centroid + cell_vol * *weight * domain.scale_from_0_1(*point);
    }
  }


  return (qugar::numbers::one / vol) * centroid;
}

//! @brief Computes the volume of a given unfitted implicit domain and quadrature.
//!
//! This function calculates the total volume by summing up the volumes of full cells
//! and cut cells within the provided unfitted implicit domain and quadrature.
//!
//! @tparam dim The dimension of the unfitted implicit domain and quadrature.
//! @param unf_domain The unfitted implicit domain object containing the grid and full cells.
//! @param quad The quadrature object containing the cut cells and their weights.
//! @return The computed volume as a `qugar::real` value.
template<int dim>
qugar::real compute_volume(const qugar::impl::UnfittedImplDomain<dim> &unf_domain, const qugar::CutCellsQuad<dim> &quad)
{
  using qugar::at;

  qugar::real vol{ qugar::numbers::zero };

  std::vector<std::int64_t> full_cells;
  unf_domain.get_full_cells(full_cells);

  const auto grid = unf_domain.get_grid();
  for (const auto &cell_id : full_cells) {
    const auto domain = grid->get_cell_domain(cell_id);
    vol += domain.volume();
  }

  auto weight = quad.weights.cbegin();

  const auto n_cut_cells = static_cast<int>(quad.cells.size());
  for (int i = 0; i < n_cut_cells; ++i) {
    const auto cell_id = at(quad.cells, i);
    const auto cell_vol = grid->get_cell_domain(cell_id).volume();

    const auto n_pts = at(quad.n_pts_per_cell, i);
    vol += cell_vol * std::accumulate(weight, weight + n_pts, qugar::numbers::zero);
    weight += n_pts;
  }


  return vol;
}

//! @brief Computes the volume of the unfitted boundary for a given unfitted implicit domain and quadrature.
//!
//! This function calculates the volume of the unfitted boundary by iterating through the cut cells
//! in the provided quadrature and summing up the weighted volumes of these cells.
//!
//! @tparam dim The dimension of the unfitted implicit domain.
//! @param unf_domain The unfitted implicit domain object containing the grid information.
//! @param quad The quadrature object containing the cut cells and their corresponding weights.
//! @return The computed volume of the unfitted boundary.
template<int dim>
qugar::real compute_unfitted_boundary_volume(const qugar::impl::UnfittedImplDomain<dim> &unf_domain,
  const qugar::CutUnfBoundsQuad<dim> &quad)
{
  using qugar::at;
  qugar::real vol{ qugar::numbers::zero };

  const auto grid = unf_domain.get_grid();

  auto weight_it = quad.weights.cbegin();
  auto normal_it = quad.normals.cbegin();

  const auto n_cut_cells = static_cast<int>(quad.cells.size());
  for (int i = 0; i < n_cut_cells; ++i) {
    const auto cell_id = at(quad.cells, i);
    const auto domain = grid->get_cell_domain(cell_id);
    const auto cell_vol = domain.volume();
    const auto cell_lengths = domain.get_lengths();

    const auto n_pts = at(quad.n_pts_per_cell, i);
    for (int pt_id = 0; pt_id < n_pts; ++pt_id) {
      auto mapped_normal = *normal_it++;
      for (int dir = 0; dir < dim; ++dir) {
        mapped_normal(dir) /= cell_lengths(dir);
      }
      vol += *weight_it++ * norm(mapped_normal) * cell_vol;
    }
  }

  return vol;
}

//! @brief Tests the volume and centroid calculations for a given Bezier tensor product.
//!
//! This function computes the volume, unfitted boundary area, and centroid of a given
//! Bezier tensor product using a specified quadrature rule. It then compares the computed
//! values with the exact values provided as input parameters.
//!
//! @tparam dim The dimension of the Bezier tensor product.
//! @param bezier The Bezier tensor product to be tested.
//! @param grid The Cartesian grid associated with the Bezier tensor product.
//! @param n_quad_pts_dir The number of quadrature points per direction.
//! @param exact_volume The exact volume to compare against.
//! @param exact_centroid The exact centroid to compare against.
//! @param exact_unf_bound_area The exact unfitted boundary area to compare against.
template<int dim>
void test_volume_and_centroid(const std::shared_ptr<const qugar::impl::ImplicitFunc<dim>> func,
  const std::shared_ptr<const qugar::CartGridTP<dim>> grid,
  // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
  const int n_quad_pts_dir,
  const qugar::real exact_volume,
  const qugar::Point<dim> &exact_centroid,
  const qugar::real exact_unf_bound_area,
  const qugar::Tolerance tol = qugar::Tolerance())
{
  assert(grid != nullptr);
  assert(grid != nullptr);

  const qugar::impl::UnfittedImplDomain<dim> unf_domain(func, grid);

  std::vector<std::int64_t> cut_cells;
  unf_domain.get_cut_cells(cut_cells);

  const auto quad = qugar::create_quadrature(unf_domain, cut_cells, n_quad_pts_dir, true);

  const auto unf_bound_quad = qugar::create_unfitted_bound_quadrature(unf_domain, cut_cells, n_quad_pts_dir);

  const auto volume = compute_volume(unf_domain, *quad);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE((tol.equal(volume, exact_volume)));

  const auto unf_bound_area = compute_unfitted_boundary_volume(unf_domain, *unf_bound_quad);
  // NOLINTNEXTLINE (bugprone-chained-comparison)
  REQUIRE((tol.equal(unf_bound_area, exact_unf_bound_area)));

  const auto centroid = compute_centroid(unf_domain, *quad);
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-do-while)
  REQUIRE((tol.coincident(centroid, exact_centroid)));
}

#endif// QUGAR_TEST_QUADRATURE_TEST_UTILS_HPP