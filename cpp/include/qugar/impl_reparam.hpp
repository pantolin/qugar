// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_REPARAM_HPP
#define QUGAR_IMPL_REPARAM_HPP

//! @file impl_reparam.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of reparameterization generization tools implicit functions on grids.
//! @date 2025-01-13
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/domain_function.hpp>
#include <qugar/impl_reparam_mesh.hpp>
#include <qugar/impl_unfitted_domain.hpp>

#include <cstddef>
#include <memory>
#include <vector>


namespace qugar::impl {

template<int dim, bool levelset>
std::shared_ptr<const ImplReparamMesh<levelset ? dim - 1 : dim, dim>>
  create_reparameterization(const UnfittedImplDomain<dim> &unf_domain, int n_pts_dir, bool merge_points);

template<int dim, bool levelset>
std::shared_ptr<const ImplReparamMesh<levelset ? dim - 1 : dim, dim>> create_reparameterization(
  const UnfittedImplDomain<dim> &unf_domain,
  const std::vector<std::int64_t> &cells,
  int n_pts_dir,
  bool merge_points);


}// namespace qugar::impl

#endif// QUGAR_IMPL_REPARAM_HPP
