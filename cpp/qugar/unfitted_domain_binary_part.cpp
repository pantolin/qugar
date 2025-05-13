// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file unfitted_domain_binary_part.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of UnfittedDomain class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/unfitted_domain_binary_part.hpp>

#include <qugar/cart_grid_tp.hpp>
#include <qugar/utils.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <vector>

namespace qugar {

template<int dim>
UnfittedBinarySpacePart<dim>::UnfittedBinarySpacePart(const GridPtr grid)
  : UnfittedBinarySpacePart<dim>(std::make_shared<SubCartGridTP<dim>>(grid))
{}

template<int dim>
UnfittedBinarySpacePart<dim>::UnfittedBinarySpacePart(const SubGridPtr subgrid)
  : subgrid_(subgrid), status_(ImmersedCellStatus::unknown), children_()
{
  assert(subgrid_ != nullptr);
}

template<int dim> bool UnfittedBinarySpacePart<dim>::is_leaf() const
{
  return this->get_child(0) == nullptr && this->get_child(1) == nullptr;
}

template<int dim> void UnfittedBinarySpacePart<dim>::set_status(const ImmersedCellStatus &status)
{
#ifndef NDEBUG
  assert(this->is_leaf());
  assert(this->status_ == ImmersedCellStatus::unknown);
  if (status == ImmersedCellStatus::cut) {
    assert(this->subgrid_->is_unique_cell());
  }
#endif
  this->status_ = status;
}


template<int dim> ImmersedCellStatus UnfittedBinarySpacePart<dim>::get_status() const
{
  assert(this->get_child(0) != nullptr && this->get_child(1) != nullptr);
  return this->status_;
}

template<int dim> auto UnfittedBinarySpacePart<dim>::get_subgrid() const -> SubGridPtr
{
  return this->subgrid_;
}

template<int dim> auto UnfittedBinarySpacePart<dim>::get_grid() const -> GridPtr
{
  return this->subgrid_->get_grid();
}

template<int dim> auto UnfittedBinarySpacePart<dim>::get_child(const int index) -> SelfPtr
{
  const auto const_child = const_cast<const UnfittedBinarySpacePart<dim> &>(*this).get_child(index);
  return std::const_pointer_cast<UnfittedBinarySpacePart<dim>>(const_child);
}

template<int dim> auto UnfittedBinarySpacePart<dim>::get_child(const int index) const -> SelfConstPtr
{
  assert(index == 0 || index == 1);
  return at(this->children_, index);
}

template<int dim> void UnfittedBinarySpacePart<dim>::branch()
{
  assert(this->is_leaf());
  assert(this->status_ == ImmersedCellStatus::unknown);
  assert(!this->subgrid_->is_unique_cell());

  const auto [left, right] = this->subgrid_->split();

  this->children_[0] = std::make_shared<UnfittedBinarySpacePart<dim>>(left);
  this->children_[1] = std::make_shared<UnfittedBinarySpacePart<dim>>(right);
}

template<int dim> bool UnfittedBinarySpacePart<dim>::is_in_tree(const std::int64_t cell_id) const
{
  const auto size = this->get_grid()->get_num_cells_dir();
  return this->subgrid_->get_range().is_in_range(cell_id, size);
}

template<int dim>
// NOLINTNEXTLINE (misc-no-recursion)
std::shared_ptr<const UnfittedBinarySpacePart<dim>> UnfittedBinarySpacePart<dim>::find_leaf(
  const std::int64_t cell_id) const
{
  assert(this->is_in_tree(cell_id));
  if (this->is_leaf()) {
    return this->shared_from_this();
  } else if (this->get_child(0)->is_in_tree(cell_id)) {
    return this->get_child(0)->find_leaf(cell_id);
  } else {
    return this->get_child(1)->find_leaf(cell_id);
  }
}


template<int dim>
template<typename Func_0, typename Func_1>
// NOLINTNEXTLINE (misc-no-recursion)
void UnfittedBinarySpacePart<dim>::transverse_tree(const Func_0 &func_0, const Func_1 &func_1) const
{
  if (this->is_leaf()) {
    if (func_0(*this)) {
      func_1(*this);
    }
  } else {
    this->get_child(0)->transverse_tree(func_0, func_1);
    this->get_child(1)->transverse_tree(func_0, func_1);
  }
}

template<int dim>
template<typename Func_0, typename Func_1>
// NOLINTNEXTLINE (misc-no-recursion)
std::size_t UnfittedBinarySpacePart<dim>::reduce(const Func_0 &func_0, const Func_1 &func_1) const
{
  if (this->is_leaf()) {
    if (func_0(*this)) {
      return func_1(*this);
    } else {
      return 0;
    }
  } else {
    return this->get_child(0)->reduce(func_0, func_1) + this->get_child(1)->reduce(func_0, func_1);
  }
}

template<int dim>
void UnfittedBinarySpacePart<dim>::get_leaves(const ImmersedCellStatus status, std::vector<SelfConstPtr> &leaves) const
{
  leaves.clear();
  leaves.reserve(this->get_num_leaves(status));

  const auto func_0 = create_leaf_checker(status);
  const auto func_1 = [&leaves](const auto &leaf) { leaves.push_back(leaf.shared_from_this()); };

  this->transverse_tree(func_0, func_1);
}

template<int dim>
void UnfittedBinarySpacePart<dim>::get_cell_ids(const ImmersedCellStatus status,
  std::vector<std::int64_t> &cell_ids) const
{
  cell_ids.clear();
  cell_ids.reserve(this->get_num_cells(status));

  const auto n_cells = this->get_grid()->get_num_cells_dir();

  const auto func_0 = create_leaf_checker(status);
  const auto func_1 = [&cell_ids, &n_cells](const Self &leaf) {
    for (const auto it : leaf.subgrid_->get_range()) {
      cell_ids.push_back(it.flat(n_cells));
    }
  };

  this->transverse_tree(func_0, func_1);

  std::ranges::sort(cell_ids);
}

template<int dim>
void UnfittedBinarySpacePart<dim>::get_cell_ids(const ImmersedCellStatus status,
  const std::vector<std::int64_t> &target_cell_ids,
  std::vector<std::int64_t> &cell_ids) const
{
  cell_ids.clear();
  cell_ids.reserve(std::min(this->get_num_cells(status), target_cell_ids.size()));

  const auto func = create_leaf_checker(status);

  // TODO: this could be likely improved.

  for (const auto &cell_id : target_cell_ids) {
    assert(this->is_in_tree(cell_id));

    const auto leaf = this->find_leaf(cell_id);
    if (func(*leaf)) {
      cell_ids.push_back(cell_id);
    }
  }

  std::ranges::sort(cell_ids);
}


template<int dim> std::size_t UnfittedBinarySpacePart<dim>::get_num_cells(const ImmersedCellStatus status) const
{
  const auto func_0 = create_leaf_checker(status);
  const auto func_1 = [](const auto &leaf) -> std::size_t {
    return static_cast<std::size_t>(leaf.subgrid_->get_num_cells());
  };
  return reduce(func_0, func_1);
}

template<int dim> std::size_t UnfittedBinarySpacePart<dim>::get_num_leaves(const ImmersedCellStatus status) const
{
  const auto func_0 = create_leaf_checker(status);
  const auto func_1 = [](const auto & /*leaf*/) -> std::size_t { return 1; };
  return reduce(func_0, func_1);
}

template<int dim>
bool UnfittedBinarySpacePart<dim>::is_cell(const ImmersedCellStatus status, const std::int64_t cell_id) const
{
  const auto leaf = this->find_leaf(cell_id);

  return leaf->status_ == status;
}

template<int dim>
auto UnfittedBinarySpacePart<dim>::create_leaf_checker(
  const ImmersedCellStatus status) -> std::function<bool(const Self &)>
{
  return [status](const Self &leaf) {
    assert(leaf.is_leaf());
    return leaf.status_ == status;
  };
}

// Instantiations
template class UnfittedBinarySpacePart<2>;
template class UnfittedBinarySpacePart<3>;

}// namespace qugar
