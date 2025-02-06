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
//! @version 0.0.2
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
std::shared_ptr<const CutCellsQuad<dim>>
  create_quadrature(const UnfittedImplDomain<dim> &unf_domain, const std::vector<int> &cells, int n_pts_dir);

template<int dim>
std::shared_ptr<const CutIntBoundsQuad<dim>> create_interior_bound_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  int n_pts_dir);

template<int dim>
std::shared_ptr<const CutIsoBoundsQuad<dim - 1>> create_facets_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const std::vector<int> &facets,
  int n_pts_dir);

}// namespace qugar::impl

#endif// QUGAR_IMPL_QUADRATURE_HPP
