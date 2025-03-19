// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#include <qugar/bbox.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/cut_quadrature.hpp>
#include <qugar/impl_quadrature.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/primitive_funcs_lib.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>

namespace {


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

  const auto grid = unf_domain.get_grid();
  for (const auto &cell_id : unf_domain.get_full_cells()) {
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

//! @brief Computes the area of the unfitted boundary for a given unfitted implicit domain and quadrature.
//!
//! This function calculates the area of the unfitted boundary by iterating through the cut cells
//! in the provided quadrature and summing up the weighted areas of these cells.
//!
//! @tparam dim The dimension of the unfitted implicit domain.
//! @param unf_domain The unfitted implicit domain object containing the grid information.
//! @param quad The quadrature object containing the cut cells and their corresponding weights.
//! @return The computed area of the unfitted boundary.
template<int dim>
qugar::real compute_boundary_area(const qugar::impl::UnfittedImplDomain<dim> &unf_domain,
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

}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(/* int argc, const char **argv */)
{

  using namespace qugar;
  using namespace qugar::impl;

  const real radius = 0.45;
  const Point<3> center{ 0.5, 0.5, 0.5 };

  const int n_elems_dir = 8;
  const int n_quad_pts_dir{ 8 };

  const real exact_volume = 4.0 / 3.0 * numbers::pi * radius * radius * radius;
  const real exact_area = 4.0 * numbers::pi * radius * radius;

  const auto bzr = std::make_shared<qugar::impl::funcs::SphereBzr<3>>(radius, center);

  const BoundBox<3> domain_01;

  const auto grid = std::make_shared<CartGridTP<3>>(
    domain_01, std::array<std::size_t, 3>({ { n_elems_dir, n_elems_dir, n_elems_dir } }));
  const qugar::impl::UnfittedImplDomain<3> unf_domain(bzr, grid);

  const auto quad = qugar::impl::create_quadrature<3>(unf_domain, unf_domain.get_cut_cells(), n_quad_pts_dir, true);

  const auto unf_bound_quad =
    qugar::impl::create_unfitted_bound_quadrature<3>(unf_domain, unf_domain.get_cut_cells(), n_quad_pts_dir);

  std::cerr << "Volume: " << compute_volume<3>(unf_domain, *quad) << "\n";
  std::cerr << "Exact volume: " << exact_volume << "\n\n";

  std::cerr << "Boundary area: " << compute_boundary_area<3>(unf_domain, *unf_bound_quad) << "\n";
  std::cerr << "Exact boundary area: " << exact_area << "\n\n";

  return 0;
}
