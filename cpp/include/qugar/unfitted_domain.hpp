// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_UNFITTED_DOMAIN_HPP
#define QUGAR_UNFITTED_DOMAIN_HPP


//! @file unfitted_domain.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of of UnfittedDomain class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/cart_grid_tp.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace qugar {

enum class ImmersedFacetStatus : std::uint8_t {
  cut,
  full,
  empty,
  cut_unf_bdry,
  cut_ext_bdry,
  cut_unf_bdry_ext_bdry,
  full_unf_bdry,
  full_ext_bdry,
  unf_bdry,
  ext_bdry,
  unf_bdry_ext_bdry
};

template<int dim> class UnfittedDomain
{
public:
  using FacetsStatus = std::array<ImmersedFacetStatus, static_cast<std::size_t>(dim) * 2>;
  using GridPtr = std::shared_ptr<const CartGridTP<dim>>;

protected:
  explicit UnfittedDomain(const GridPtr &grid);

public:
  virtual ~UnfittedDomain() = default;

  [[nodiscard]] GridPtr get_grid() const;
  [[nodiscard]] const std::vector<int> &get_full_cells() const;
  [[nodiscard]] const std::vector<int> &get_empty_cells() const;
  [[nodiscard]] const std::vector<int> &get_cut_cells() const;
  [[nodiscard]] FacetsStatus get_cell_facets_status(int cell_id) const;

  void get_empty_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_full_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_cut_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const;

  [[nodiscard]] bool is_full_cell(int cell_id) const;
  [[nodiscard]] bool is_empty_cell(int cell_id) const;
  [[nodiscard]] bool is_cut_cell(int cell_id) const;

  [[nodiscard]] bool is_full_facet(int cell_id, int local_facet_id) const;
  [[nodiscard]] bool is_empty_facet(int cell_id, int local_facet_id) const;
  [[nodiscard]] bool is_cut_facet(int cell_id, int local_facet_id) const;

  [[nodiscard]] bool has_unfitted_boundary(int cell_id, int local_facet_id) const;
  [[nodiscard]] bool has_unfitted_boundary_on_domain_boundary(int cell_id, int local_facet_id) const;
  [[nodiscard]] bool has_external_boundary(int cell_id, int local_facet_id) const;

protected:
  GridPtr grid_;
  std::vector<int> full_cells_;
  std::vector<int> empty_cells_;
  std::vector<int> cut_cells_;
  std::unordered_map<int, FacetsStatus> facets_status_;

  void sort();
};


}// namespace qugar

#endif// QUGAR_UNFITTED_DOMAIN_HPP
