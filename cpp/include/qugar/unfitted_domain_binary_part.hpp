// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_UNFITTED_DOMAIN_BINARY_PART_HPP
#define QUGAR_UNFITTED_DOMAIN_BINARY_PART_HPP


//! @file unfitted_domain_binary_part.hpp
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
#include <functional>
#include <memory>
#include <vector>

namespace qugar {

enum class ImmersedCellStatus : std::uint8_t {
  cut,
  full,
  empty,
  unknown,
};

template<int dim> class UnfittedBinarySpacePart : public std::enable_shared_from_this<UnfittedBinarySpacePart<dim>>
{
  using GridPtr = std::shared_ptr<const CartGridTP<dim>>;
  using SubGridPtr = std::shared_ptr<const SubCartGridTP<dim>>;
  using Self = UnfittedBinarySpacePart<dim>;
  using SelfConstPtr = std::shared_ptr<const Self>;
  using SelfPtr = std::shared_ptr<Self>;

public:
  explicit UnfittedBinarySpacePart(const GridPtr grid);
  explicit UnfittedBinarySpacePart(const SubGridPtr subgrid);

  [[nodiscard]] bool is_leaf() const;

  void set_status(const ImmersedCellStatus &status);

  void branch();

  [[nodiscard]] ImmersedCellStatus get_status() const;

  [[nodiscard]] SubGridPtr get_subgrid() const;
  [[nodiscard]] GridPtr get_grid() const;

  [[nodiscard]] SelfPtr get_child(int index);
  [[nodiscard]] SelfConstPtr get_child(int index) const;

  [[nodiscard]] bool is_in_tree(std::int64_t cell_id) const;

  [[nodiscard]] SelfConstPtr find_leaf(std::int64_t cell_id) const;

  void get_leaves(ImmersedCellStatus status, std::vector<SelfConstPtr> &leaves) const;

  void get_cell_ids(ImmersedCellStatus status, std::vector<std::int64_t> &cell_ids) const;


  void get_cell_ids(ImmersedCellStatus status,
    const std::vector<std::int64_t> &target_cell_ids,
    std::vector<std::int64_t> &cell_ids) const;

  [[nodiscard]] std::size_t get_num_cells(ImmersedCellStatus status) const;

  [[nodiscard]] std::size_t get_num_leaves(ImmersedCellStatus status) const;

  [[nodiscard]] bool is_cell(ImmersedCellStatus status, std::int64_t cell_id) const;


private:
  SubGridPtr subgrid_;
  ImmersedCellStatus status_;
  std::array<std::shared_ptr<UnfittedBinarySpacePart<dim>>, 2> children_;

  template<typename Func_0, typename Func_1> void transverse_tree(const Func_0 &func_0, const Func_1 &func_1) const;
  template<typename Func_0, typename Func_1> std::size_t reduce(const Func_0 &func_0, const Func_1 &func_1) const;

  static std::function<bool(const Self &)> create_leaf_checker(ImmersedCellStatus status);
};


}// namespace qugar

#endif// QUGAR_UNFITTED_DOMAIN_BINARY_PART_HPP
