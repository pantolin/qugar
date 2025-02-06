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
//! @version 0.0.2
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
template<int dim> void CutIntBoundsQuad<dim>::reserve(const int n_cells, const int n_tot_pts)
{
  cells.reserve(cells.size() + static_cast<std::size_t>(n_cells));
  n_pts_per_cell.reserve(n_pts_per_cell.size() + static_cast<std::size_t>(n_cells));
  points.reserve(points.size() + static_cast<std::size_t>(n_tot_pts));
  weights.reserve(weights.size() + static_cast<std::size_t>(n_tot_pts));
  normals.reserve(normals.size() + static_cast<std::size_t>(n_tot_pts));
}

template<int dim>
std::shared_ptr<const CutCellsQuad<dim>>
  create_quadrature(const UnfittedDomain<dim> &unf_domain, const std::vector<int> &cells, const int n_pts_dir)
{
  const auto *unf_impl_domain = dynamic_cast<const impl::UnfittedImplDomain<dim> *>(&unf_domain);
  assert(unf_impl_domain != nullptr);
  return impl::create_quadrature<dim>(*unf_impl_domain, cells, n_pts_dir);
}

template<int dim>
std::shared_ptr<const CutIntBoundsQuad<dim>> create_interior_bound_quadrature(const UnfittedDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const int n_pts_dir)
{
  const auto *unf_impl_domain = dynamic_cast<const impl::UnfittedImplDomain<dim> *>(&unf_domain);
  assert(unf_impl_domain != nullptr);
  return impl::create_interior_bound_quadrature<dim>(*unf_impl_domain, cells, n_pts_dir);
}

template<int dim>
std::shared_ptr<const CutIsoBoundsQuad<dim - 1>> create_facets_quadrature(const UnfittedDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const std::vector<int> &facets,
  const int n_pts_dir)
{
  const auto *unf_impl_domain = dynamic_cast<const impl::UnfittedImplDomain<dim> *>(&unf_domain);
  assert(unf_impl_domain != nullptr);
  return impl::create_facets_quadrature<dim>(*unf_impl_domain, cells, facets, n_pts_dir);
}


// Instantiations

template struct CutCellsQuad<1>;
template struct CutCellsQuad<2>;
template struct CutCellsQuad<3>;

template struct CutIsoBoundsQuad<1>;
template struct CutIsoBoundsQuad<2>;
template struct CutIsoBoundsQuad<3>;

template struct CutIntBoundsQuad<1>;
template struct CutIntBoundsQuad<2>;
template struct CutIntBoundsQuad<3>;


template std::shared_ptr<const CutCellsQuad<2>>
  create_quadrature<2>(const UnfittedDomain<2> &, const std::vector<int> &, const int);
template std::shared_ptr<const CutCellsQuad<3>>
  create_quadrature<3>(const UnfittedDomain<3> &, const std::vector<int> &, const int);

template std::shared_ptr<const CutIntBoundsQuad<2>>
  create_interior_bound_quadrature<2>(const UnfittedDomain<2> &, const std::vector<int> &, const int);
template std::shared_ptr<const CutIntBoundsQuad<3>>
  create_interior_bound_quadrature<3>(const UnfittedDomain<3> &, const std::vector<int> &, const int);


template std::shared_ptr<const CutIsoBoundsQuad<1>>
  create_facets_quadrature<2>(const UnfittedDomain<2> &, const std::vector<int> &, const std::vector<int> &, const int);
template std::shared_ptr<const CutIsoBoundsQuad<2>>
  create_facets_quadrature<3>(const UnfittedDomain<3> &, const std::vector<int> &, const std::vector<int> &, const int);

}// namespace qugar