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
//! @version 0.0.2
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

template<int dim> const std::vector<int> &UnfittedDomain<dim>::get_full_cells() const
{
  return this->full_cells_;
}

template<int dim> const std::vector<int> &UnfittedDomain<dim>::get_empty_cells() const
{
  return this->empty_cells_;
}

template<int dim> const std::vector<int> &UnfittedDomain<dim>::get_cut_cells() const
{
  return this->cut_cells_;
}

template<int dim> auto UnfittedDomain<dim>::get_cell_facets_status(const int cell_id) const -> FacetsStatus
{
  return this->facets_status_.at(cell_id);
}


template<int dim>
void UnfittedDomain<dim>::get_empty_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const
{
  const auto n_facets_per_cell = dim * 2;
  const auto n_facets_estimate = (this->empty_cells_.size() + this->cut_cells_.size()) * n_facets_per_cell;

  cell_ids.clear();
  local_facets_ids.clear();
  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &cell_id : this->empty_cells_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }

  for (const auto cell_id : this->cut_cells_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {

      if (this->is_empty_facet(cell_id, local_facet_id)) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}

template<int dim>
void UnfittedDomain<dim>::get_full_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const
{
  cell_ids.clear();
  local_facets_ids.clear();

  const auto n_facets_per_cell = dim * 2;
  const auto n_facets_estimate = (this->full_cells_.size() + this->cut_cells_.size()) * n_facets_per_cell;

  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto &cell_id : this->full_cells_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {
      cell_ids.push_back(cell_id);
      local_facets_ids.push_back(local_facet_id);
    }
  }


  for (const auto cell_id : this->cut_cells_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {

      if (this->is_full_facet(cell_id, local_facet_id)) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}


template<int dim>
void UnfittedDomain<dim>::get_cut_facets(std::vector<int> &cell_ids, std::vector<int> &local_facets_ids) const
{
  cell_ids.clear();
  local_facets_ids.clear();

  const auto n_facets_per_cell = dim * 2;
  const auto n_facets_estimate = this->cut_cells_.size() * n_facets_per_cell;

  cell_ids.reserve(n_facets_estimate);
  local_facets_ids.reserve(n_facets_estimate);

  for (const auto cell_id : this->cut_cells_) {
    for (int local_facet_id = 0; local_facet_id < n_facets_per_cell; ++local_facet_id) {

      if (this->is_cut_facet(cell_id, local_facet_id)) {
        cell_ids.push_back(cell_id);
        local_facets_ids.push_back(local_facet_id);
      }
    }
  }
}


template<int dim> bool UnfittedDomain<dim>::is_full_cell(const int cell_id) const
{
  // NOLINTNEXTLINE (misc-include-cleaner) // <algorithm> should be enough.
  return std::ranges::binary_search(this->full_cells_, cell_id);
}

template<int dim> bool UnfittedDomain<dim>::is_empty_cell(const int cell_id) const
{
  // NOLINTNEXTLINE (misc-include-cleaner) // <algorithm> should be enough.
  return std::ranges::binary_search(this->empty_cells_, cell_id);
}

template<int dim> bool UnfittedDomain<dim>::is_cut_cell(const int cell_id) const
{
  // NOLINTNEXTLINE (misc-include-cleaner) // <algorithm> should be enough.
  return std::ranges::binary_search(this->cut_cells_, cell_id);
}

template<int dim> bool UnfittedDomain<dim>::is_full_facet(const int cell_id, const int local_facet_id) const
{
  const auto facet = at(this->facets_status_.at(cell_id), local_facet_id);

  if (facet == ImmersedFacetStatus::full) {
    return true;
  }

  else if (facet == ImmersedFacetStatus::full_unf_bdry) {
    if (this->grid_->on_boundary(cell_id, local_facet_id)) {
      return true;
    }
  }

  return false;
}

template<int dim> bool UnfittedDomain<dim>::is_empty_facet(const int cell_id, const int local_facet_id) const
{
  const auto facet = at(this->facets_status_.at(cell_id), local_facet_id);

  if (facet == ImmersedFacetStatus::empty || facet == ImmersedFacetStatus::ext_bdry
      || facet == ImmersedFacetStatus::full_ext_bdry) {
    return true;
  }

  else if (facet == ImmersedFacetStatus::unf_bdry || facet == ImmersedFacetStatus::full_unf_bdry
           || facet == ImmersedFacetStatus::unf_bdry_ext_bdry) {
    if (!this->grid_->on_boundary(cell_id, local_facet_id)) {
      return true;
    }
  }

  return false;
}
template<int dim> bool UnfittedDomain<dim>::is_cut_facet(const int cell_id, const int local_facet_id) const
{
  const auto facet = at(this->facets_status_.at(cell_id), local_facet_id);


  if (facet == ImmersedFacetStatus::cut || facet == ImmersedFacetStatus::cut_unf_bdry
      || facet == ImmersedFacetStatus::cut_ext_bdry || facet == ImmersedFacetStatus::cut_unf_bdry_ext_bdry) {
    return true;
  }

  else if (facet == ImmersedFacetStatus::unf_bdry || facet == ImmersedFacetStatus::unf_bdry_ext_bdry) {
    if (this->grid_->on_boundary(cell_id, local_facet_id)) {
      return true;
    }
  }

  return false;
}

template<int dim> bool UnfittedDomain<dim>::has_unfitted_boundary(const int cell_id, const int local_facet_id) const
{
  if (this->grid_->on_boundary(cell_id, local_facet_id)) {
    // All the internal interfaces that are on the boundary are
    // considered as cut.
    return false;
  }

  const auto facet = at(this->facets_status_.at(cell_id), local_facet_id);

  return facet == ImmersedFacetStatus::cut_unf_bdry || facet == ImmersedFacetStatus::cut_unf_bdry_ext_bdry
         || facet == ImmersedFacetStatus::full_unf_bdry || facet == ImmersedFacetStatus::unf_bdry
         || facet == ImmersedFacetStatus::unf_bdry_ext_bdry;
}

template<int dim>
bool UnfittedDomain<dim>::has_unfitted_boundary_on_domain_boundary(const int cell_id, const int local_facet_id) const
{
  if (this->grid_->on_boundary(cell_id, local_facet_id)) {
    const auto facet = at(this->facets_status_.at(cell_id), local_facet_id);

    return facet == ImmersedFacetStatus::cut_unf_bdry || facet == ImmersedFacetStatus::cut_unf_bdry_ext_bdry
           || facet == ImmersedFacetStatus::full_unf_bdry || facet == ImmersedFacetStatus::unf_bdry
           || facet == ImmersedFacetStatus::unf_bdry_ext_bdry;
  } else {
    return false;
  }
}

template<int dim> bool UnfittedDomain<dim>::has_external_boundary(const int cell_id, const int local_facet_id) const
{
  const auto facet = at(this->facets_status_.at(cell_id), local_facet_id);

  return facet == ImmersedFacetStatus::cut_ext_bdry || facet == ImmersedFacetStatus::cut_unf_bdry_ext_bdry
         || facet == ImmersedFacetStatus::full_ext_bdry || facet == ImmersedFacetStatus::ext_bdry
         || facet == ImmersedFacetStatus::unf_bdry_ext_bdry;
}

// Instantiations
template class UnfittedDomain<2>;
template class UnfittedDomain<3>;

}// namespace qugar
