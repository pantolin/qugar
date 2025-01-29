// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_reparam.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of reparameterization generization tools implicit functions on grids.
//! @version 0.0.1
//! @date 2025-01-13
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_reparam.hpp>

#include <qugar/bezier_tp.hpp>
#include <qugar/impl_reparam_bezier.hpp>
#include <qugar/impl_reparam_general.hpp>
#include <qugar/impl_reparam_mesh.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tolerance.hpp>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <ranges>
#include <vector>


namespace qugar::impl {

namespace {

  void
    intersection(const std::vector<int> &sorted_lhs, const std::vector<int> &sorted_rhs, std::vector<int> &intersection)
  {
    intersection.clear();
    intersection.reserve(std::max(sorted_lhs.size(), sorted_rhs.size()));

    std::ranges::set_intersection(sorted_lhs, sorted_rhs, std::back_inserter(intersection));
  }

}// namespace

template<int dim, bool levelset>
std::shared_ptr<const ImplReparamMesh<levelset ? dim - 1 : dim, dim>>
  create_reparameterization(const UnfittedImplDomain<dim> &unf_domain, const int n_pts_dir)
{
  const auto rng = std::ranges::iota_view<int, int>{ 0, unf_domain.get_grid()->get_num_cells() };
  const std::vector<int> cells(rng.begin(), rng.end());
  return create_reparameterization<dim, levelset>(unf_domain, cells, n_pts_dir);
}

template<int dim, bool levelset>
std::shared_ptr<const ImplReparamMesh<levelset ? dim - 1 : dim, dim>> create_reparameterization(
  const UnfittedImplDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const int n_pts_dir)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(1 < n_pts_dir);

  std::vector<int> sorted_cells = cells;
  std::ranges::sort(sorted_cells);

  std::vector<int> cut_cells;
  intersection(unf_domain.get_cut_cells(), sorted_cells, cut_cells);

  const auto grid = unf_domain.get_grid();

  static const int param_dim = levelset ? dim - 1 : dim;
  const auto reparam = std::make_shared<ImplReparamMesh<param_dim, dim>>(n_pts_dir);

  const auto n_reparam_tiles_per_cell = param_dim == 2 ? 3 : 13;// Estimation
  const auto n_cells_est = cut_cells.size() * n_reparam_tiles_per_cell + unf_domain.get_full_cells().size();
  reparam->reserve_cells(n_cells_est);

  const auto phi = unf_domain.get_impl_func();
  const auto *bzr = dynamic_cast<const BezierTP<dim, 1> *>(phi.get());
  const bool is_bzr = bzr != nullptr;


  for (const auto &cell_id : cut_cells) {
    const auto domain = grid->get_cell_domain(cell_id);

    if (is_bzr) {
      BezierTP<dim> bzr_domain(*bzr);
      bzr_domain.rescale_domain(domain);

      reparam_Bezier<dim, levelset>(bzr_domain, domain, *reparam);
    } else {
      reparam_general<dim, levelset>(*phi, domain, *reparam);
    }
  }

  if constexpr (!levelset) {
    std::vector<int> full_cells;
    intersection(unf_domain.get_full_cells(), sorted_cells, full_cells);

    constexpr bool wirebasket = true;
    reparam->add_full_cells(*grid, full_cells, wirebasket);
  }

  const Tolerance tol(1.0e4 * numbers::eps);
  reparam->merge_coincident_points(tol);


  return reparam;
}

// Instantiations.

template std::shared_ptr<const ImplReparamMesh<2, 2>>
  create_reparameterization<2, false>(const UnfittedImplDomain<2> &, const std::vector<int> &, const int);
template std::shared_ptr<const ImplReparamMesh<1, 2>>
  create_reparameterization<2, true>(const UnfittedImplDomain<2> &, const std::vector<int> &, const int);

template std::shared_ptr<const ImplReparamMesh<3, 3>>
  create_reparameterization<3, false>(const UnfittedImplDomain<3> &, const std::vector<int> &, const int);
template std::shared_ptr<const ImplReparamMesh<2, 3>>
  create_reparameterization<3, true>(const UnfittedImplDomain<3> &, const std::vector<int> &, const int);

template std::shared_ptr<const ImplReparamMesh<2, 2>> create_reparameterization<2, false>(const UnfittedImplDomain<2> &,
  const int);
template std::shared_ptr<const ImplReparamMesh<1, 2>> create_reparameterization<2, true>(const UnfittedImplDomain<2> &,
  const int);

template std::shared_ptr<const ImplReparamMesh<3, 3>> create_reparameterization<3, false>(const UnfittedImplDomain<3> &,
  const int);
template std::shared_ptr<const ImplReparamMesh<2, 3>> create_reparameterization<3, true>(const UnfittedImplDomain<3> &,
  const int);

}// namespace qugar::impl