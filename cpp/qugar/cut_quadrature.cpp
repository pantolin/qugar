// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file cut_quadrature.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of cut quadratures functionalities.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/cut_quadrature.hpp>

#include <qugar/impl_quadrature.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/unfitted_domain.hpp>

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>


namespace qugar {

template<int dim, typename Domain> using FacetQuadCreator = std::function<void(int, int, CutIsoBoundsQuad<dim - 1> &)>;

namespace {

  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  template<int dim, typename QuadCreator>
  std::shared_ptr<const CutIsoBoundsQuad<dim - 1>> create_facets_quadrature_generic(const std::vector<int> &cells,
    const std::vector<int> &facets,
    const int n_pts_dir,
    const QuadCreator &facet_quad_creator)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    static_assert(dim == 2 || dim == 3, "Invalid dimension.");

    assert(cells.size() == facets.size());
    assert(0 < n_pts_dir);

    // Estimation of number of points.
    const int n_cells = static_cast<int>(cells.size());
    const int n_quad_set_per_facet_estimate{ 2 };// This is (almost surely) an overestimation.
    const int n_pts_per_quad_set = n_pts_dir * (dim == 3 ? n_pts_dir : 1);
    const int n_pts_per_facet_estimate = n_quad_set_per_facet_estimate * n_pts_per_quad_set;
    const int n_pts_estimate = n_cells * n_pts_per_facet_estimate;

    const auto quad = std::make_shared<CutIsoBoundsQuad<dim - 1>>();
    quad->reserve(n_cells, n_pts_estimate);

    quad->cells = cells;
    quad->local_facet_ids = facets;

    auto facet_it = facets.cbegin();

    for (const int cell_id : cells) {
      const auto local_facet_id = *facet_it++;
      facet_quad_creator(cell_id, local_facet_id, *quad);
    }

    return quad;
  }

}// namespace

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
template<int dim> void CutCellsQuad<dim>::reserve(const int n_cells, const int n_tot_pts)
{
  cells.reserve(cells.size() + static_cast<std::size_t>(n_cells));
  n_pts_per_cell.reserve(n_pts_per_cell.size() + static_cast<std::size_t>(n_cells));
  points.reserve(points.size() + static_cast<std::size_t>(n_tot_pts));
  weights.reserve(weights.size() + static_cast<std::size_t>(n_tot_pts));
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
template<int dim> void CutIsoBoundsQuad<dim>::reserve(const int n_cells, const int n_tot_pts)
{
  cells.reserve(cells.size() + static_cast<std::size_t>(n_cells));
  n_pts_per_facet.reserve(n_pts_per_facet.size() + static_cast<std::size_t>(n_cells));
  points.reserve(points.size() + static_cast<std::size_t>(n_tot_pts));
  weights.reserve(weights.size() + static_cast<std::size_t>(n_tot_pts));
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
template<int dim> void CutUnfBoundsQuad<dim>::reserve(const int n_cells, const int n_tot_pts)
{
  cells.reserve(cells.size() + static_cast<std::size_t>(n_cells));
  n_pts_per_cell.reserve(n_pts_per_cell.size() + static_cast<std::size_t>(n_cells));
  points.reserve(points.size() + static_cast<std::size_t>(n_tot_pts));
  weights.reserve(weights.size() + static_cast<std::size_t>(n_tot_pts));
  normals.reserve(normals.size() + static_cast<std::size_t>(n_tot_pts));
}

template<int dim>
std::shared_ptr<const CutCellsQuad<dim>> create_quadrature(const UnfittedDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const int n_pts_dir,
  const bool full_cells)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(0 < n_pts_dir);

  // Estimation of number of points.
  const int n_cells = static_cast<int>(cells.size());
  const int n_quad_set_per_cell_estimate{ 2 };// This is an estimation.
  const int n_pts_per_quad_set = n_pts_dir * n_pts_dir * (dim == 3 ? n_pts_dir : 1);
  const int n_pts_per_cell_estimate = n_quad_set_per_cell_estimate * n_pts_per_quad_set;
  const int n_pts_estimate = n_cells * n_pts_per_cell_estimate;

  const auto quad = std::make_shared<CutCellsQuad<dim>>();
  quad->reserve(n_cells, n_pts_estimate);

  quad->cells = cells;

  const auto *unf_impl_domain = dynamic_cast<const impl::UnfittedImplDomain<dim> *>(&unf_domain);
  if (unf_impl_domain != nullptr) {
    for (const int cell_id : cells) {
      impl::create_cell_quadrature<dim>(*unf_impl_domain, cell_id, n_pts_dir, full_cells, *quad);
    }
  } else {
    // Not implemented for non-implicit domains.
    assert(false);
  }

  return quad;
}

template<int dim>
std::shared_ptr<const CutUnfBoundsQuad<dim>> create_unfitted_bound_quadrature(const UnfittedDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const int n_pts_dir)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(0 < n_pts_dir);

  // Estimation of number of points.
  const int n_cells = static_cast<int>(cells.size());
  const int n_quad_set_per_cell_estimate{ 2 };// This is an estimation.
  const int n_pts_per_quad_set = n_pts_dir * (dim == 3 ? n_pts_dir : 1);
  const int n_pts_per_cell_estimate = n_quad_set_per_cell_estimate * n_pts_per_quad_set;
  const int n_pts_estimate = n_cells * n_pts_per_cell_estimate;

  const auto quad = std::make_shared<CutUnfBoundsQuad<dim>>();
  quad->reserve(n_cells, n_pts_estimate);

  quad->cells = cells;

  const auto *unf_impl_domain = dynamic_cast<const impl::UnfittedImplDomain<dim> *>(&unf_domain);
  if (unf_impl_domain != nullptr) {
    for (const int cell_id : cells) {
      impl::create_cell_unfitted_bound_quadrature(*unf_impl_domain, cell_id, n_pts_dir, *quad);
    }
  } else {
    // Not implemented for non-implicit domains.
    assert(false);
  }

  return quad;
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
std::shared_ptr<const CutIsoBoundsQuad<dim - 1>> create_interior_facets_quadrature(
  const UnfittedDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const std::vector<int> &facets,
  const int n_pts_dir,
  const bool full_facets)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(0 < n_pts_dir);

  const auto *unf_impl_domain = dynamic_cast<const impl::UnfittedImplDomain<dim> *>(&unf_domain);

  if (unf_impl_domain != nullptr) {
    using QuadCreator = FacetQuadCreator<dim, impl::UnfittedImplDomain<dim>>;
    const QuadCreator facet_quad_creator = [unf_impl_domain, n_pts_dir, full_facets](
                                             int cell_id, int local_facet_id, CutIsoBoundsQuad<dim - 1> &quad) {
      constexpr bool remove_unf = { false };
      constexpr bool remove_cut = { true };

      impl::create_facet_quadrature<dim>(
        *unf_impl_domain, cell_id, local_facet_id, n_pts_dir, full_facets, remove_unf, remove_cut, quad);
    };

    return create_facets_quadrature_generic<dim, QuadCreator>(cells, facets, n_pts_dir, facet_quad_creator);
  } else {
    // Not implemented for non-implicit domains.
    assert(false);
    return nullptr;// To avoid warning.
  }
}


// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
std::shared_ptr<const CutIsoBoundsQuad<dim - 1>> create_exterior_facets_quadrature(
  const UnfittedDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const std::vector<int> &facets,
  const int n_pts_dir,
  const bool full_facets)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(0 < n_pts_dir);

  const auto *unf_impl_domain = dynamic_cast<const impl::UnfittedImplDomain<dim> *>(&unf_domain);

  if (unf_impl_domain != nullptr) {
    using QuadCreator = FacetQuadCreator<dim, impl::UnfittedImplDomain<dim>>;
    const QuadCreator facet_quad_creator = [unf_impl_domain, n_pts_dir, full_facets](
                                             int cell_id, int local_facet_id, CutIsoBoundsQuad<dim - 1> &quad) {
      constexpr bool remove_unf = { false };

      // On facets that are not on the boundary of the grid, we only consider the unfitted part.
      const bool on_bound = unf_impl_domain->get_grid()->on_boundary(cell_id, local_facet_id);
      const bool remove_cut = !on_bound;

      impl::create_facet_quadrature<dim>(
        *unf_impl_domain, cell_id, local_facet_id, n_pts_dir, full_facets, remove_unf, remove_cut, quad);
    };

    return create_facets_quadrature_generic<dim, QuadCreator>(cells, facets, n_pts_dir, facet_quad_creator);
  } else {
    // Not implemented for non-implicit domains.
    assert(false);
    return nullptr;// To avoid warning.
  }
}


// Instantiations

template struct CutCellsQuad<1>;
template struct CutCellsQuad<2>;
template struct CutCellsQuad<3>;

template struct CutIsoBoundsQuad<1>;
template struct CutIsoBoundsQuad<2>;
template struct CutIsoBoundsQuad<3>;

template struct CutUnfBoundsQuad<1>;
template struct CutUnfBoundsQuad<2>;
template struct CutUnfBoundsQuad<3>;


template std::shared_ptr<const CutCellsQuad<2>>
  create_quadrature<2>(const UnfittedDomain<2> &, const std::vector<int> &, const int, const bool full_cells);
template std::shared_ptr<const CutCellsQuad<3>>
  create_quadrature<3>(const UnfittedDomain<3> &, const std::vector<int> &, const int, const bool full_cells);

template std::shared_ptr<const CutUnfBoundsQuad<2>>
  create_unfitted_bound_quadrature<2>(const UnfittedDomain<2> &, const std::vector<int> &, const int);
template std::shared_ptr<const CutUnfBoundsQuad<3>>
  create_unfitted_bound_quadrature<3>(const UnfittedDomain<3> &, const std::vector<int> &, const int);

template std::shared_ptr<const CutIsoBoundsQuad<1>> create_interior_facets_quadrature<2>(const UnfittedDomain<2> &,
  const std::vector<int> &,
  const std::vector<int> &,
  const int,
  const bool);
template std::shared_ptr<const CutIsoBoundsQuad<2>> create_interior_facets_quadrature<3>(const UnfittedDomain<3> &,
  const std::vector<int> &,
  const std::vector<int> &,
  const int,
  const bool);

template std::shared_ptr<const CutIsoBoundsQuad<1>> create_exterior_facets_quadrature<2>(const UnfittedDomain<2> &,
  const std::vector<int> &,
  const std::vector<int> &,
  const int,
  const bool);
template std::shared_ptr<const CutIsoBoundsQuad<2>> create_exterior_facets_quadrature<3>(const UnfittedDomain<3> &,
  const std::vector<int> &,
  const std::vector<int> &,
  const int,
  const bool);

}// namespace qugar