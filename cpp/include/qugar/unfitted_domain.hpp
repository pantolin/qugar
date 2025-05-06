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
#include <qugar/unfitted_domain_kd_tree.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
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
  static const int n_facets_per_cell = dim * 2;
  using FacetsStatus = std::array<ImmersedFacetStatus, n_facets_per_cell>;
  using GridPtr = std::shared_ptr<const CartGridTP<dim>>;
  using KDTreePtr = std::shared_ptr<UnfittedKDTree<dim>>;

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

  [[nodiscard]] bool is_exterior_facet(std::int64_t cell_id, int local_facet_id) const;

  [[nodiscard]] std::size_t get_num_total_cells() const;
  [[nodiscard]] std::size_t get_num_full_cells() const;
  [[nodiscard]] std::size_t get_num_empty_cells() const;
  [[nodiscard]] std::size_t get_num_cut_cells() const;
  [[nodiscard]] bool has_facets_with_unf_bdry() const;

  void get_full_cells(std::vector<std::int64_t> &cell_ids) const;
  void get_empty_cells(std::vector<std::int64_t> &cell_ids) const;
  void get_cut_cells(std::vector<std::int64_t> &cell_ids) const;

  void get_full_cells(const std::vector<std::int64_t> &target_cell_ids, std::vector<std::int64_t> &cell_ids) const;
  void get_empty_cells(const std::vector<std::int64_t> &target_cell_ids, std::vector<std::int64_t> &cell_ids) const;
  void get_cut_cells(const std::vector<std::int64_t> &target_cell_ids, std::vector<std::int64_t> &cell_ids) const;

  void get_empty_facets(std::vector<std::int64_t> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_full_facets(std::vector<std::int64_t> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_unfitted_facets(std::vector<std::int64_t> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_full_unfitted_facets(std::vector<std::int64_t> &cell_ids, std::vector<int> &local_facets_ids) const;
  void get_cut_facets(std::vector<std::int64_t> &cell_ids, std::vector<int> &local_facets_ids) const;

  void get_empty_facets(const std::vector<std::int64_t> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<std::int64_t> &cell_ids,
    std::vector<int> &local_facets_ids) const;
  void get_full_facets(const std::vector<std::int64_t> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<std::int64_t> &cell_ids,
    std::vector<int> &local_facets_ids) const;
  void get_unfitted_facets(const std::vector<std::int64_t> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<std::int64_t> &cell_ids,
    std::vector<int> &local_facets_ids) const;
  void get_full_unfitted_facets(const std::vector<std::int64_t> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<std::int64_t> &cell_ids,
    std::vector<int> &local_facets_ids) const;
  void get_cut_facets(const std::vector<std::int64_t> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<std::int64_t> &cell_ids,
    std::vector<int> &local_facets_ids) const;

  [[nodiscard]] bool is_full_cell(std::int64_t cell_id) const;
  [[nodiscard]] bool is_empty_cell(std::int64_t cell_id) const;
  [[nodiscard]] bool is_cut_cell(std::int64_t cell_id) const;
  [[nodiscard]] bool is_full_with_unf_bdry_cell(std::int64_t cell_id) const;

  // Full unfitted facets are not considered full.
  [[nodiscard]] bool is_full_facet(std::int64_t cell_id, int local_facet_id) const;
  [[nodiscard]] bool is_empty_facet(std::int64_t cell_id, int local_facet_id) const;
  [[nodiscard]] bool is_cut_facet(std::int64_t cell_id, int local_facet_id) const;
  [[nodiscard]] bool is_full_unfitted_facet(std::int64_t cell_id, int local_facet_id) const;
  [[nodiscard]] bool is_cell_with_unf_bdry(std::int64_t cell_id) const;

  [[nodiscard]] bool has_unfitted_boundary(std::int64_t cell_id, int local_facet_id) const;
  [[nodiscard]] bool has_external_boundary(std::int64_t cell_id, int local_facet_id) const;

  // Full unfitted facets are not considered full.
  [[nodiscard]] static bool is_full_facet(ImmersedFacetStatus status);

  [[nodiscard]] static bool is_empty_facet(ImmersedFacetStatus status);
  [[nodiscard]] static bool is_cut_facet(ImmersedFacetStatus status);
  [[nodiscard]] static bool is_full_unfitted_facet(ImmersedFacetStatus status);

  [[nodiscard]] static bool has_unfitted_boundary(ImmersedFacetStatus status);
  [[nodiscard]] static bool has_external_boundary(ImmersedFacetStatus status);

protected:
  GridPtr grid_;
  KDTreePtr kd_tree_;

  std::unordered_map<std::int64_t, FacetsStatus> facets_status_;
  std::vector<std::int64_t> full_cells_with_unf_bdry_;

private:
  static void get_facets_target(const std::vector<std::int64_t> &target_cell_ids,
    const std::vector<int> &target_local_facets_ids,
    std::vector<std::int64_t> &cell_ids,
    std::vector<int> &local_facets_ids,
    const std::function<bool(std::int64_t, int)> &func);

protected:
  void init_full_cells_with_unf_bdry();
};


}// namespace qugar

#endif// QUGAR_UNFITTED_DOMAIN_HPP
