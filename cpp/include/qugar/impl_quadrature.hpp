// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_QUADRATURE_HPP
#define QUGAR_IMPL_QUADRATURE_HPP

//! @file impl_quadrature.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of quadratures for general implicit functions on grids.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/cart_grid_tp.hpp>
#include <qugar/cut_quadrature.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_unfitted_domain.hpp>

#include <memory>
#include <vector>


namespace qugar::impl {

template<int dim>
void create_cell_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  int cell_id,
  int n_pts_dir,
  bool full_cells,
  CutCellsQuad<dim> &quad);

template<int dim>
void create_cell_unfitted_bound_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  int cell_id,
  int n_pts_dir,
  CutUnfBoundsQuad<dim> &quad);

template<int dim>
void create_facet_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  int cell_id,
  int local_facet_id,
  int n_pts_dir,
  bool full_facets,
  bool remove_unf_bdry,
  bool remove_cut,
  CutIsoBoundsQuad<dim - 1> &quad);


}// namespace qugar::impl

#endif// QUGAR_IMPL_QUADRATURE_HPP
