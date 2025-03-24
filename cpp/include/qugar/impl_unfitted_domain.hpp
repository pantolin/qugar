// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_UNFITTED_DOMAIN_HPP
#define QUGAR_IMPL_UNFITTED_DOMAIN_HPP


//! @file impl_unfitted_domain.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of of UnfittedImplDomain class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/unfitted_domain.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace qugar::impl {


template<int dim> class UnfittedImplDomain : public UnfittedDomain<dim>
{
public:
  using FuncSign = qugar::impl::FuncSign;
  using FacetsStatus = std::array<ImmersedFacetStatus, static_cast<std::size_t>(dim) * 2>;
  using GridPtr = std::shared_ptr<const CartGridTP<dim>>;
  using FuncPtr = std::shared_ptr<const ImplicitFunc<dim>>;

  explicit UnfittedImplDomain(const FuncPtr phi, GridPtr grid);

  explicit UnfittedImplDomain(const FuncPtr phi, GridPtr grid, const std::vector<std::int64_t> &cells);

  [[nodiscard]] FuncPtr get_impl_func() const;

private:
  FuncPtr phi_;

  // NOLINTNEXTLINE (misc-no-recursion)
  void create_decomposition(const SubCartGridTP<dim> &subgrid,
    const std::function<FuncSign(const BoundBox<dim> &)> &func_sign,
    const std::optional<std::vector<std::int64_t>> &target_cells);

  void classify_undetermined_sign_cell(const SubCartGridTP<dim> &subgrid);
};


}// namespace qugar::impl


#endif// QUGAR_IMPL_UNFITTED_DOMAIN_HPP
