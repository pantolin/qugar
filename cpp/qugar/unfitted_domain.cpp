// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file unfitted_domain.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of UnfittedDomain class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/unfitted_domain.hpp>

#include <qugar/unfitted_domain_kd_tree.hpp>
#include <qugar/utils.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <vector>

namespace qugar {


template<int dim>
UnfittedDomain<dim>::UnfittedDomain(const GridPtr &grid)
  : grid_(grid), kd_tree_(std::make_shared<UnfittedKDTree<dim>>(grid))
{
  assert(grid_ != nullptr);
  assert(kd_tree_ != nullptr);
}


template<int dim> auto UnfittedDomain<dim>::get_grid() const -> GridPtr
{
  return this->grid_;
}

template<int dim>
bool UnfittedDomain<dim>::is_exterior_facet(const std::int64_t cell_id, const int local_facet_id) const
{
  assert(0 <= local_facet_id && local_facet_id < n_facets_per_cell);
  return grid_->on_boundary(cell_id, local_facet_id);
}

template<int dim> std::size_t UnfittedDomain<dim>::get_num_total_cells() const
{
  return this->grid_->get_num_cells();
}

template<int dim> std::size_t UnfittedDomain<dim>::get_num_full_cells() const
{
  return this->kd_tree_->get_num_cells(ImmersedCellStatus::full);
}

template<int dim> std::size_t UnfittedDomain<dim>::get_num_empty_cells() const
{
  return this->kd_tree_->get_num_cells(ImmersedCellStatus::empty);
}

template<int dim> std::size_t UnfittedDomain<dim>::get_num_cut_cells() const
{
  return this->kd_tree_->get_num_cells(ImmersedCellStatus::cut);
}

template<int dim> bool UnfittedDomain<dim>::has_facets_with_unf_bdry() const
{
  return std::ranges::any_of(this->facets_status_, [](const auto &pair) {
    return std::ranges::any_of(pair.second, [](const auto &status) { return has_unfitted_boundary(status); });
  });
}

template<int dim> void UnfittedDomain<dim>::get_full_cells(std::vector<std::int64_t> &cell_ids) const
{
  this->kd_tree_->get_cell_ids(ImmersedCellStatus::full, cell_ids);
}

template<int dim> void UnfittedDomain<dim>::get_empty_cells(std::vector<std::int64_t> &cell_ids) const
{
  this->kd_tree_->get_cell_ids(ImmersedCellStatus::empty, cell_ids);
}

template<int dim> void UnfittedDomain<dim>::get_cut_cells(std::vector<std::int64_t> &cell_ids) const
{
  this->kd_tree_->get_cell_ids(ImmersedCellStatus::cut, cell_ids);
}


template<int dim>
void UnfittedDomain<dim>::get_full_cells(const std::vector<std::int64_t> &target_cell_ids,
  std::vector<std::int64_t> &cell_ids) const
{
  this->kd_tree_->get_cell_ids(ImmersedCellStatus::full, target_cell_ids, cell_ids);
}

template<int dim>
void UnfittedDomain<dim>::get_empty_cells(const std::vector<std::int64_t> &target_cell_ids,
  std::vector<std::int64_t> &cell_ids) const
{
  this->kd_tree_->get_cell_ids(ImmersedCellStatus::empty, target_cell_ids, cell_ids);
}

template<int dim>
void UnfittedDomain<dim>::get_cut_cells(const std::vector<std::int64_t> &target_cell_ids,
  std::vector<std::int64_t> &cell_ids) const
{
  this->kd_tree_->get_cell_ids(ImmersedCellStatus::cut, target_cell_ids, cell_ids);
}

template<int dim>
void UnfittedDomain<dim>::get_empty_facets(std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
{
  const auto n_facets_estimate = (this->get_num_empty_cells() + this->get_num_cut_cells()) * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  std::vector<std::int64_t> empty_cells;
  this->get_empty_cells(empty_cells);

  for (const auto &cell_id : empty_cells) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }

  for (const auto &[cell_id, facets] : this->facets_status_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {
      if (is_empty_facet(at(facets, local_facet_id))) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}

template<int dim>
void UnfittedDomain<dim>::get_full_facets(std::vector<std::int64_t> &cell_ids, std::vector<int> &local_facets_ids) const
{
  const auto n_facets_estimate = (this->get_num_full_cells() + this->facets_status_.size()) * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &[cell_id, facets] : this->facets_status_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {
      if (is_full_facet(at(facets, local_facet_id))) {
        // This do not include facets that correspond to full unfitted boundaries.
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }

  std::vector<std::int64_t> full_cells;
  this->get_full_cells(full_cells);

  for (const auto &cell_id : full_cells) {
    if (this->facets_status_.contains(cell_id)) {
      continue;
    }

    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }
}

template<int dim>
void UnfittedDomain<dim>::get_cut_facets(std::vector<std::int64_t> &cell_ids, std::vector<int> &local_facets_ids) const
{
  const auto n_facets_estimate = this->get_num_cut_cells() * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &[cell_id, facets] : this->facets_status_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {
      if (is_cut_facet(at(facets, local_facet_id))) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}

template<int dim>
void UnfittedDomain<dim>::get_full_unfitted_facets(std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
{
  const auto n_facets_estimate = this->facets_status_.size() * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &[cell_id, facets] : this->facets_status_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {
      if (is_full_unfitted_facet(at(facets, local_facet_id))) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}


template<int dim>
void UnfittedDomain<dim>::get_unfitted_facets(std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
{
  const auto n_facets_estimate = this->facets_status_.size() * n_facets_per_cell;// An overestimation.

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &[cell_id, facets] : this->facets_status_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {
      if (has_unfitted_boundary(at(facets, local_facet_id))) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}

template<int dim>
// NOLINTBEGIN (bugprone-easily-swappable-parameters)
void UnfittedDomain<dim>::get_facets_target(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids,
  const std::function<bool(std::int64_t, int)> &func)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  assert(target_cell_ids.size() == target_local_facets_ids.size());

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(target_cell_ids.size());
  local_facets_ids.reserve(target_cell_ids.size());

  auto it_cell = target_cell_ids.cbegin();
  auto it_facet = target_local_facets_ids.cbegin();

  for (; it_cell != target_cell_ids.cend(); ++it_cell, ++it_facet) {
    const auto cell_id = *it_cell;
    const auto local_facet_id = *it_facet;
    if (func(cell_id, local_facet_id)) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }
}

template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
void UnfittedDomain<dim>::get_empty_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
{
  const auto func = [this](std::int64_t cell_id, int local_facet_id) -> bool {
    return this->is_empty_facet(cell_id, local_facet_id);
  };

  this->get_facets_target(target_cell_ids, target_local_facets_ids, cell_ids, local_facets_ids, func);
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
void UnfittedDomain<dim>::get_full_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  const auto func = [this](std::int64_t cell_id, int local_facet_id) -> bool {
    return this->is_full_facet(cell_id, local_facet_id);
  };

  this->get_facets_target(target_cell_ids, target_local_facets_ids, cell_ids, local_facets_ids, func);
}


// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
void UnfittedDomain<dim>::get_cut_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  const auto func = [this](std::int64_t cell_id, int local_facet_id) -> bool {
    return this->is_cut_facet(cell_id, local_facet_id);
  };

  this->get_facets_target(target_cell_ids, target_local_facets_ids, cell_ids, local_facets_ids, func);
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
void UnfittedDomain<dim>::get_full_unfitted_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  const auto func = [this](std::int64_t cell_id, int local_facet_id) -> bool {
    return this->is_full_unfitted_facet(cell_id, local_facet_id);
  };

  this->get_facets_target(target_cell_ids, target_local_facets_ids, cell_ids, local_facets_ids, func);
}


// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
void UnfittedDomain<dim>::get_unfitted_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  const auto func = [this](std::int64_t cell_id, int local_facet_id) -> bool {
    return this->has_unfitted_boundary(cell_id, local_facet_id);
  };

  this->get_facets_target(target_cell_ids, target_local_facets_ids, cell_ids, local_facets_ids, func);
}


template<int dim> bool UnfittedDomain<dim>::is_full_cell(const std::int64_t cell_id) const
{
  return this->kd_tree_->is_cell(ImmersedCellStatus::full, cell_id);
}

template<int dim> bool UnfittedDomain<dim>::is_full_with_unf_bdry_cell(const std::int64_t cell_id) const
{
  return this->facets_status_.contains(cell_id) && std::ranges::binary_search(this->full_cells_with_unf_bdry_, cell_id);
}

template<int dim> bool UnfittedDomain<dim>::is_empty_cell(const std::int64_t cell_id) const
{
  return this->kd_tree_->is_cell(ImmersedCellStatus::empty, cell_id);
}

template<int dim> bool UnfittedDomain<dim>::is_cut_cell(const std::int64_t cell_id) const
{
  return this->facets_status_.contains(cell_id) && this->kd_tree_->is_cell(ImmersedCellStatus::cut, cell_id);
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
template<int dim> bool UnfittedDomain<dim>::is_full_facet(const std::int64_t cell_id, const int local_facet_id) const
{
  assert(0 <= local_facet_id && local_facet_id < n_facets_per_cell);
  const auto it = this->facets_status_.find(cell_id);
  if (it != this->facets_status_.end()) {
    return is_full_facet(at(it->second, local_facet_id));
  } else {
    return this->is_full_cell(cell_id);
  }
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
template<int dim> bool UnfittedDomain<dim>::is_empty_facet(const std::int64_t cell_id, const int local_facet_id) const
{
  assert(0 <= local_facet_id && local_facet_id < n_facets_per_cell);
  const auto it = this->facets_status_.find(cell_id);
  if (it != this->facets_status_.end()) {
    return is_empty_facet(at(it->second, local_facet_id));
  } else {
    return this->is_empty_cell(cell_id);
  }
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
template<int dim> bool UnfittedDomain<dim>::is_cut_facet(const std::int64_t cell_id, const int local_facet_id) const
{
  assert(0 <= local_facet_id && local_facet_id < n_facets_per_cell);
  const auto it = this->facets_status_.find(cell_id);
  if (it != this->facets_status_.end()) {
    return is_cut_facet(at(it->second, local_facet_id));
  } else {
    return false;
  }
}

template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
bool UnfittedDomain<dim>::is_full_unfitted_facet(const std::int64_t cell_id, const int local_facet_id) const
{
  assert(0 <= local_facet_id && local_facet_id < n_facets_per_cell);
  const auto it = this->facets_status_.find(cell_id);
  if (it != this->facets_status_.end()) {
    return is_full_unfitted_facet(at(it->second, local_facet_id));
  } else {
    return false;
  }
}

template<int dim> bool UnfittedDomain<dim>::is_cell_with_unf_bdry(const std::int64_t cell_id) const
{
  return this->is_cut_cell(cell_id) || this->is_full_with_unf_bdry_cell(cell_id);
}

template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
bool UnfittedDomain<dim>::has_unfitted_boundary(const std::int64_t cell_id, const int local_facet_id) const
{
  assert(0 <= local_facet_id && local_facet_id < n_facets_per_cell);
  const auto it = this->facets_status_.find(cell_id);
  if (it != this->facets_status_.end()) {
    return has_unfitted_boundary(at(it->second, local_facet_id));
  } else {
    return false;
  }
}

template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
bool UnfittedDomain<dim>::has_external_boundary(const std::int64_t cell_id, const int local_facet_id) const
{
  // Note here that empty facets with external boundary are not considered
  assert(0 <= local_facet_id && local_facet_id < n_facets_per_cell);
  const auto it = this->facets_status_.find(cell_id);
  if (it != this->facets_status_.end()) {
    return has_external_boundary(at(it->second, local_facet_id));
  } else {
    return false;
  }
}

template<int dim> bool UnfittedDomain<dim>::is_full_facet(const ImmersedFacetStatus status)
{
  return status == ImmersedFacetStatus::full;
}

template<int dim> bool UnfittedDomain<dim>::is_empty_facet(const ImmersedFacetStatus status)
{
  switch (status) {
  case ImmersedFacetStatus::empty:
  case ImmersedFacetStatus::full_ext_bdry:
  case ImmersedFacetStatus::ext_bdry:
    return true;
  default:
    return false;
  }
}

template<int dim> bool UnfittedDomain<dim>::is_cut_facet(const ImmersedFacetStatus status)
{
  switch (status) {
  case ImmersedFacetStatus::cut:
  case ImmersedFacetStatus::cut_unf_bdry:
  case ImmersedFacetStatus::cut_ext_bdry:
  case ImmersedFacetStatus::cut_unf_bdry_ext_bdry:
    return true;
  default:
    return false;
  }
}

template<int dim> bool UnfittedDomain<dim>::is_full_unfitted_facet(const ImmersedFacetStatus status)
{
  return status == ImmersedFacetStatus::full_unf_bdry;
}

template<int dim> bool UnfittedDomain<dim>::has_unfitted_boundary(const ImmersedFacetStatus status)
{
  switch (status) {
  case ImmersedFacetStatus::cut_unf_bdry:
  case ImmersedFacetStatus::cut_unf_bdry_ext_bdry:
  case ImmersedFacetStatus::full_unf_bdry:
  case ImmersedFacetStatus::unf_bdry:
  case ImmersedFacetStatus::unf_bdry_ext_bdry:
    return true;
  default:
    return false;
  }
}

template<int dim> bool UnfittedDomain<dim>::has_external_boundary(const ImmersedFacetStatus status)
{
  switch (status) {
  case ImmersedFacetStatus::cut_ext_bdry:
  case ImmersedFacetStatus::cut_unf_bdry_ext_bdry:
  case ImmersedFacetStatus::full_ext_bdry:
  case ImmersedFacetStatus::ext_bdry:
  case ImmersedFacetStatus::unf_bdry_ext_bdry:
    return true;
  default:
    return false;
  }
}

template<int dim> void UnfittedDomain<dim>::init_full_cells_with_unf_bdry()
{
  const auto n_cut_cells = this->get_num_cut_cells();
  const auto n_full_cells_with_unf_bdry = this->facets_status_.size() - n_cut_cells;

  if (n_full_cells_with_unf_bdry == 0) {
    return;
  }

  full_cells_with_unf_bdry_.clear();
  full_cells_with_unf_bdry_.reserve(n_full_cells_with_unf_bdry);

  std::vector<std::int64_t> cut_cells;
  this->get_cut_cells(cut_cells);
  const auto is_cut = [&cut_cells](
                        const std::int64_t cell_id) { return std::ranges::binary_search(cut_cells, cell_id); };

  for (const auto &cell_id_facets : this->facets_status_) {
    const auto cell_id = cell_id_facets.first;
    if (!is_cut(cell_id)) {
      this->full_cells_with_unf_bdry_.push_back(cell_id);
    }
  }

  std::ranges::sort(this->full_cells_with_unf_bdry_);
}


// Instantiations
template class UnfittedDomain<2>;
template class UnfittedDomain<3>;

}// namespace qugar
