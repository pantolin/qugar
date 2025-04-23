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

#include <cstdint>
#include <memory>
#include <vector>


namespace qugar::impl {

/**
 * @brief Creates a quadrature for cut cells.
 * @tparam dim Dimension of the domain.
 * @param unf_domain  Unfitted domain.
 * @param cell_id Cell for which the quadrature is created.
 * @param n_pts_dir Number of points in each direction for generated custom quadratures.
 * @param quad Quadrature object to be filled with the generated quadrature.
 */
template<int dim>
void create_cell_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  std::int64_t cell_id,
  int n_pts_dir,
  CutCellsQuad<dim> &quad);

/**
 * @brief Creates a quadrature for the unfitted boundary.
 * @tparam dim Dimension of the domain.
 * @param unf_domain  Unitted domain.
 * @param cell_id Cell for which the quadrature is created.
 * @param n_pts_dir Number of points in each direction for generated custom quadratures.
 * @param include_facet_unf_bdry  If true, the quadrature includes the parts of the unfitted boundary
 *                                that belong to the cells' facets.
 * @param exclude_ext_bdry If the previous parameter is true, and this is one is false, the parts
 *                         of the unfitted boundary that belong to the external facets are not included.
 * @param quad Quadrature object to be filled with the generated quadrature.
 */
template<int dim>
void create_cell_unfitted_bound_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  std::int64_t cell_id,
  int n_pts_dir,
  bool include_facet_unf_bdry,
  bool exclude_ext_bdry,
  CutUnfBoundsQuad<dim> &quad);

template<int dim>
void create_facet_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  std::int64_t cell_id,
  int local_facet_id,
  int n_pts_dir,
  bool remove_unf_bdry,
  bool remove_cut,
  CutIsoBoundsQuad<dim - 1> &quad);


}// namespace qugar::impl

#endif// QUGAR_IMPL_QUADRATURE_HPP
