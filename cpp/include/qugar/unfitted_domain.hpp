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
  static const std::size_t n_facets_per_cell = static_cast<std::size_t>(dim) * 2;
  using FacetsStatus = std::array<ImmersedFacetStatus, n_facets_per_cell>;
  using GridPtr = std::shared_ptr<const CartGridTP<dim>>;

protected:
  explicit UnfittedDomain(const GridPtr &grid);

public:
  virtual ~UnfittedDomain() = default;

  /// Copy constructor
  UnfittedDomain(const UnfittedDomain &other) = default;

  /// Copy assignment operator
  UnfittedDomain &operator=(const UnfittedDomain &other) = default;

  /// Move constructor
  UnfittedDomain(UnfittedDomain &&other) noexcept = default;

  /// Move assignment operator
  UnfittedDomain &operator=(UnfittedDomain &&other) noexcept = default;

  [[nodiscard]] GridPtr get_grid() const;
  [[nodiscard]] const std::vector<int> &get_full_cells() const;
  [[nodiscard]] const std::vector<int> &get_empty_cells() const;
  [[nodiscard]] const std::vector<int> &get_cut_cells() const;
  [[nodiscard]] FacetsStatus get_cell_facets_status(int cell_id) const;

  void get_empty_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_full_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_unfitted_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_full_unfitted_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_cut_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const;

  void get_empty_facets(const std::vector<int> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<int> &cell_ids,
    std::vector<int> &local_facets_ids) const;
  void get_full_facets(const std::vector<int> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<int> &cell_ids,
    std::vector<int> &local_facets_ids) const;
  void get_unfitted_facets(const std::vector<int> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<int> &cell_ids,
    std::vector<int> &local_facets_ids) const;
  void get_full_unfitted_facets(const std::vector<int> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<int> &cell_ids,
    std::vector<int> &local_facets_ids) const;
  void get_cut_facets(const std::vector<int> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<int> &cell_ids,
    std::vector<int> &local_facets_ids) const;

  [[nodiscard]] bool is_full_cell(int cell_id) const;
  [[nodiscard]] bool is_empty_cell(int cell_id) const;
  [[nodiscard]] bool is_cut_cell(int cell_id) const;

  // Full unfitted facets are not considered full.
  [[nodiscard]] bool is_full_facet(int cell_id, int local_facet_id) const;
  [[nodiscard]] bool is_empty_facet(int cell_id, int local_facet_id) const;
  [[nodiscard]] bool is_cut_facet(int cell_id, int local_facet_id) const;
  [[nodiscard]] bool is_full_unfitted_facet(int cell_id, int local_facet_id) const;

  [[nodiscard]] bool has_unfitted_boundary(int cell_id, int local_facet_id) const;
  [[nodiscard]] bool has_external_boundary(int cell_id, int local_facet_id) const;

  // Full unfitted facets are not considered full.
  [[nodiscard]] static bool is_full_facet(ImmersedFacetStatus status);

  [[nodiscard]] static bool is_empty_facet(ImmersedFacetStatus status);
  [[nodiscard]] static bool is_cut_facet(ImmersedFacetStatus status);
  [[nodiscard]] static bool is_full_unfitted_facet(ImmersedFacetStatus status);

  [[nodiscard]] static bool has_unfitted_boundary(ImmersedFacetStatus status);
  [[nodiscard]] static bool has_external_boundary(ImmersedFacetStatus status);

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
