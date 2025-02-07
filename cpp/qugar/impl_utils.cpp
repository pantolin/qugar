// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_utils.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of utility functions for implicit geometries.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_utils.hpp>

#include <qugar/domain_function.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <cassert>
#include <iterator>


namespace qugar::impl {

template<int dim> bool on_levelset(const ImplicitFunc<dim> &phi, const Point<dim> &point, const Tolerance tol)
{
  return tol.is_zero(phi(point));
}

template<int dim> int get_facet_constant_dir(const int local_facet_id)
{
  // NOLINTNEXTLINE (readability-simplify-boolean-expr)
  assert(0 <= local_facet_id && local_facet_id < dim * 2);
  return local_facet_id / 2;
}

template<int dim> [[nodiscard]] int get_facet_side(const int local_facet_id)
{
  // NOLINTNEXTLINE (readability-simplify-boolean-expr)
  assert(0 <= local_facet_id && local_facet_id < dim * 2);
  return local_facet_id % 2;
}

template<int dim> [[nodiscard]] int get_local_facet_id(const int const_dir, const int side)
{
  assert(0 <= const_dir && const_dir < dim);
  assert(0 <= side && side <= 1);
  return (const_dir * 2) + side;
}

template<int dim> Vector<int, dim - 1> get_edge_constant_dirs(const int local_edge_id)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  if constexpr (dim == 2) {
    return get_facet_constant_dir<dim>(local_edge_id);
  } else {
    assert(0 <= local_edge_id && local_edge_id < 12);
    if (local_edge_id < 4) {
      return Vector<int, 2>(0, 1);
    } else if (local_edge_id < 8) {// NOLINT (cppcoreguidelines-avoid-magic-numbers)
      return Vector<int, 2>(0, 2);
    } else {// if (edge_id < 12) {
      return Vector<int, 2>(1, 2);
    }
  }
}

template<int dim> Vector<int, dim - 1> get_edge_sides(const int local_edge_id)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  if constexpr (dim == 2) {
    return get_facet_side<dim>(local_edge_id);
  } else {
    assert(0 <= local_edge_id && local_edge_id < 12);
    return Vector<int, 2>(local_edge_id % 2, (local_edge_id / 2) % 2);
  }
}

template<int dim> RootsIntervals<dim>::RootsIntervals()
{
  constexpr int estimate{ 6 };
  this->roots.reserve(estimate);
  this->func_ids.reserve(estimate);
  this->roots_ids.reserve(estimate);

  // This "funny" number is related to https://stackoverflow.com/a/50519283
  constexpr int estimate_bool{ 65 };
  this->active_intervals.reserve(estimate_bool);
}

template<int dim> void RootsIntervals<dim>::clear()
{
  this->roots.clear();
  this->func_ids.clear();
  this->active_intervals.clear();
  this->roots_ids.clear();
  this->point = Point<dim>();
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
template<int dim> void RootsIntervals<dim>::add_root(const real root, const int func_id)
{
  assert(-1 <= func_id);
  this->roots.push_back(root);
  this->func_ids.push_back(func_id);
  this->roots_ids.emplace_back(root, func_id);
}

template<int dim> bool RootsIntervals<dim>::empty() const
{
  return this->roots.empty();
}

template<int dim> int RootsIntervals<dim>::get_num_roots() const
{
  return static_cast<int>(this->roots.size());
}

template<int dim> void RootsIntervals<dim>::sort_roots()
{
  // Sort.
  std::sort(this->roots_ids.begin(), this->roots_ids.end(), [](const auto &lhs, const auto &rhs) -> bool {
    return lhs.first < rhs.first;
  });

  // Unzip.
  for (int i = 0; i < this->get_num_roots(); ++i) {
    at(this->roots, i) = at(roots_ids, i).first;
    at(this->func_ids, i) = at(roots_ids, i).second;
  }
}

template<int dim> void RootsIntervals<dim>::adjust_roots(const Tolerance &tol, const real x0, const real x1)
{
  if (roots.empty()) {
    return;
  }

  this->sort_roots();

  // Adjusting near roots and enforcing root=x0, func_id=-1, and root=x1, func_id=-1,
  // if present, to be the first and last, respectively.

  auto roots_old = this->roots;
  auto funcs_old = this->func_ids;

  this->roots.clear();
  this->func_ids.clear();
  this->roots_ids.clear();

  this->roots.push_back(x0);
  this->func_ids.push_back(-1);
  this->roots_ids.emplace_back(x0, -1);

  bool has_x0{ false };
  bool has_x1{ false };
  for (int i = 0; i < static_cast<int>(roots_old.size()); ++i) {
    auto root = at(roots_old, i);
    const auto func_id = at(funcs_old, i);

    if (tol.equal(root, x0)) {
      if (func_id == -1) {
        has_x0 = true;
        continue;
      } else {
        root = x0;
      }
    } else if (tol.equal(root, x1)) {
      if (func_id == -1) {
        has_x1 = true;
        continue;
      } else {
        root = x1;
      }
    } else if (0 < i && tol.equal(root, at(roots_old, i - 1))) {
      root = at(roots_old, i - 1);
    }

    this->roots.push_back(root);
    this->func_ids.push_back(func_id);
    this->roots_ids.emplace_back(root, func_id);
  }// i

  if (!has_x0) {
    this->roots.erase(this->roots.begin());
    this->func_ids.erase(this->func_ids.begin());
    this->roots_ids.erase(this->roots_ids.begin());
  }

  if (has_x1) {
    this->roots.push_back(x1);
    this->func_ids.push_back(-1);
    this->roots_ids.emplace_back(x1, -1);
  }
}


// Instantiations.

template bool on_levelset<2>(const ImplicitFunc<2> &, const Point<2> &, const Tolerance);
template bool on_levelset<3>(const ImplicitFunc<3> &, const Point<3> &, const Tolerance);

template int get_facet_constant_dir<2>(const int);
template int get_facet_constant_dir<3>(const int);

template int get_facet_side<2>(const int);
template int get_facet_side<3>(const int);

template Vector<int, 1> get_edge_constant_dirs<2>(const int);
template Vector<int, 2> get_edge_constant_dirs<3>(const int);

template Vector<int, 1> get_edge_sides<2>(const int);
template Vector<int, 2> get_edge_sides<3>(const int);

template int get_local_facet_id<2>(const int, const int);
template int get_local_facet_id<3>(const int, const int);

template struct RootsIntervals<0>;
template struct RootsIntervals<1>;
template struct RootsIntervals<2>;
template struct RootsIntervals<3>;

}// namespace qugar::impl