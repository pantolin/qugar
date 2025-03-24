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

#include <qugar/utils.hpp>

#include <cassert>
#include <cstddef>
#include <ranges>
#include <vector>

namespace qugar {


template<int dim> UnfittedDomain<dim>::UnfittedDomain(const GridPtr &grid) : grid_(grid)
{
  assert(grid_ != nullptr);

  // Overpesimistic estimate.
  const auto n_cells = static_cast<std::size_t>(this->grid_->get_num_cells());
  this->full_cells_.reserve(n_cells);
  this->empty_cells_.reserve(n_cells);
  this->cut_cells_.reserve(n_cells);
}


template<int dim> void UnfittedDomain<dim>::sort()
{
  std::ranges::sort(this->full_cells_);
  std::ranges::sort(this->empty_cells_);
  std::ranges::sort(this->cut_cells_);
}

template<int dim> auto UnfittedDomain<dim>::get_grid() const -> GridPtr
{
  return this->grid_;
}

template<int dim> const std::vector<std::int64_t> &UnfittedDomain<dim>::get_full_cells() const
{
  return this->full_cells_;
}

template<int dim> const std::vector<std::int64_t> &UnfittedDomain<dim>::get_empty_cells() const
{
  return this->empty_cells_;
}

template<int dim> const std::vector<std::int64_t> &UnfittedDomain<dim>::get_cut_cells() const
{
  return this->cut_cells_;
}

template<int dim> auto UnfittedDomain<dim>::get_cell_facets_status(const std::int64_t cell_id) const -> FacetsStatus
{
  return this->facets_status_.at(cell_id);
}


template<int dim>
void UnfittedDomain<dim>::get_empty_facets(std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
{
  const auto n_facets_estimate = (this->empty_cells_.size() + this->cut_cells_.size()) * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &cell_id : this->empty_cells_) {
    for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }

  for (const auto cell_id : this->cut_cells_) {
    for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
      if (this->is_empty_facet(cell_id, local_facet_id)) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}

template<int dim>
void UnfittedDomain<dim>::get_full_facets(std::vector<std::int64_t> &cell_ids, std::vector<int> &local_facets_ids) const
{
  const auto n_facets_estimate = (this->full_cells_.size() + this->cut_cells_.size()) * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &cell_id : this->full_cells_) {
    const auto it = this->facets_status_.find(cell_id);
    if (it != this->facets_status_.end()) {
      // Full cells may contain full unfitted facets (that are not considered full).
      for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
        if (is_full_facet(at(it->second, local_facet_id))) {
          cell_ids.push_back(cell_id);
          local_facets_ids.push_back(local_facet_id);
        }
      }
    } else {
      for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }

  for (const auto &cell_id : this->cut_cells_) {
    for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
      if (this->is_full_facet(cell_id, local_facet_id)) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}


template<int dim>
void UnfittedDomain<dim>::get_cut_facets(std::vector<std::int64_t> &cell_ids, std::vector<int> &local_facets_ids) const
{
  const auto n_facets_estimate = this->cut_cells_.size() * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto cell_id : this->cut_cells_) {
    for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
      if (this->is_cut_facet(cell_id, local_facet_id)) {
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
  const auto n_facets_estimate = this->cut_cells_.size() * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &cell_id : this->full_cells_) {
    const auto it = this->facets_status_.find(cell_id);
    if (it != this->facets_status_.end()) {
      // Full cells may contain full unfitted facets (that are not considered full).
      for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
        if (is_full_unfitted_facet(at(it->second, local_facet_id))) {
          cell_ids.push_back(cell_id);
          local_facets_ids.push_back(local_facet_id);
        }
      }
    }
  }

  for (const auto cell_id : this->cut_cells_) {
    for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
      if (this->is_full_unfitted_facet(cell_id, local_facet_id)) {
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
  const auto n_facets_estimate = this->cut_cells_.size() * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &cell_id : this->full_cells_) {

    const auto it = this->facets_status_.find(cell_id);
    if (it != this->facets_status_.end()) {
      // Full cells may contain full unfitted facets (that are not considered full).
      for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
        if (has_unfitted_boundary(at(it->second, local_facet_id))) {
          cell_ids.push_back(cell_id);
          local_facets_ids.push_back(local_facet_id);
        }
      }
    }
  }

  for (const auto cell_id : this->cut_cells_) {
    for (int local_facet_id = 0; local_facet_id < static_cast<int>(n_facets_per_cell); ++local_facet_id) {
      if (this->has_unfitted_boundary(cell_id, local_facet_id)) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}

template<int dim>
void UnfittedDomain<dim>::get_empty_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
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
    if (this->is_empty_facet(cell_id, local_facet_id)) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }
}

template<int dim>
void UnfittedDomain<dim>::get_full_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
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
    if (this->is_full_facet(cell_id, local_facet_id)) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }
}


template<int dim>
void UnfittedDomain<dim>::get_cut_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
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
    if (this->is_cut_facet(cell_id, local_facet_id)) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }
}

template<int dim>
void UnfittedDomain<dim>::get_full_unfitted_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
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
    if (this->is_full_unfitted_facet(cell_id, local_facet_id)) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }
}


template<int dim>
void UnfittedDomain<dim>::get_unfitted_facets(const std::vector<std::int64_t> &target_cell_ids,
  const std::vector<int> &target_local_facets_ids,
  std::vector<std::int64_t> &cell_ids,
  std::vector<int> &local_facets_ids) const
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
    if (this->has_unfitted_boundary(cell_id, local_facet_id)) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }
}


template<int dim> bool UnfittedDomain<dim>::is_full_cell(const std::int64_t cell_id) const
{
  // NOLINTNEXTLINE (misc-include-cleaner) // <algorithm> should be enough.
  return std::ranges::binary_search(this->full_cells_, cell_id);
}

template<int dim> bool UnfittedDomain<dim>::is_empty_cell(const std::int64_t cell_id) const
{
  // NOLINTNEXTLINE (misc-include-cleaner) // <algorithm> should be enough.
  return std::ranges::binary_search(this->empty_cells_, cell_id);
}

template<int dim> bool UnfittedDomain<dim>::is_cut_cell(const std::int64_t cell_id) const
{
  // NOLINTNEXTLINE (misc-include-cleaner) // <algorithm> should be enough.
  return std::ranges::binary_search(this->cut_cells_, cell_id);
}

template<int dim> bool UnfittedDomain<dim>::is_full_facet(const std::int64_t cell_id, const int local_facet_id) const
{
  assert(0 <= local_facet_id && local_facet_id < static_cast<int>(n_facets_per_cell));
  const auto it = this->facets_status_.find(cell_id);
  if (it != this->facets_status_.end()) {
    return is_full_facet(at(it->second, local_facet_id));
  } else {
    return this->is_full_cell(cell_id);
  }
}

template<int dim> bool UnfittedDomain<dim>::is_empty_facet(const std::int64_t cell_id, const int local_facet_id) const
{
  assert(0 <= local_facet_id && local_facet_id < static_cast<int>(n_facets_per_cell));
  const auto it = this->facets_status_.find(cell_id);
  if (it != this->facets_status_.end()) {
    return is_empty_facet(at(it->second, local_facet_id));
  } else {
    return this->is_empty_cell(cell_id);
  }
}

template<int dim> bool UnfittedDomain<dim>::is_cut_facet(const std::int64_t cell_id, const int local_facet_id) const
{
  return is_cut_facet(at(this->facets_status_.at(cell_id), local_facet_id));
}

template<int dim>
bool UnfittedDomain<dim>::is_full_unfitted_facet(const std::int64_t cell_id, const int local_facet_id) const
{
  return is_full_unfitted_facet(at(this->facets_status_.at(cell_id), local_facet_id));
}

template<int dim>
bool UnfittedDomain<dim>::has_unfitted_boundary(const std::int64_t cell_id, const int local_facet_id) const
{
  return has_unfitted_boundary(at(this->facets_status_.at(cell_id), local_facet_id));
}

template<int dim>
bool UnfittedDomain<dim>::has_external_boundary(const std::int64_t cell_id, const int local_facet_id) const
{
  return has_external_boundary(at(this->facets_status_.at(cell_id), local_facet_id));
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

// Instantiations
template class UnfittedDomain<2>;
template class UnfittedDomain<3>;

}// namespace qugar
