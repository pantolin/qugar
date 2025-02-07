// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file cart_grid_tp.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of Cartesian grid class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/cart_grid_tp.hpp>

#include <qugar/bbox.hpp>
#include <qugar/point.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace qugar {

namespace _impl {
  template<int dim> TensorSizeTP<dim> get_num_cells_dir(const std::array<std::vector<real>, dim> &breaks)
  {
    TensorSizeTP<dim> n_cells;

    for (int dir = 0; dir < dim; ++dir) {
      const auto &brks = at(breaks, dir);
      assert(1 < brks.size());
      n_cells(dir) = static_cast<int>(brks.size()) - 1;
    }

    return n_cells;
  }

  template<int dim>
  std::array<std::vector<real>, dim> create_equispaced_breaks(const BoundBox<dim> &domain,
    const std::array<std::size_t, dim> &n_intrvs_dir)
  {
    std::array<std::vector<real>, dim> breaks{};
    for (int dir = 0; dir < dim; ++dir) {
      const auto n_intervs = at(n_intrvs_dir, dir);
#ifndef NDEBUG
      assert(0 < n_intervs);
#endif

      const auto dx = domain.length(dir) / static_cast<real>(n_intervs);
      const auto x0 = domain.min(dir);

      auto &breaks_dir = at(breaks, dir);
      breaks_dir.reserve(n_intervs + 1);

      for (std::size_t i = 0; i <= n_intervs; ++i) {
        breaks_dir.push_back(x0 + dx * static_cast<real>(i));
      }
    }

    return breaks;
  }
}// namespace _impl


template<int dim>
CartGridTP<dim>::CartGridTP(const std::array<std::vector<real>, dim> &breaks) : breaks_(breaks), domain_(), range_(1)
{
#ifndef NDEBUG
  for (int dir = 0; dir < dim; ++dir) {
    const auto &breaks_dir = at(breaks, dir);
    const auto n_brks = breaks_dir.size();
    assert(1 < n_brks);

    auto it0 = breaks_dir.cbegin();
    auto it1 = std::next(breaks_dir.cbegin());
    const auto end = breaks_dir.cend();

    for (; it1 < end; ++it0, ++it1) {
      // FIXME: to use tolerance here?
      assert(*it0 < *it1);
    }
  }
#endif

  // NOLINTBEGIN (misc-const-correctness)
  Point<dim> min;
  Point<dim> max;
  // NOLINTEND (misc-const-correctness)

  for (int dir = 0; dir < dim; ++dir) {
    min(dir) = at(breaks, dir).front();
    max(dir) = at(breaks, dir).back();
  }

  domain_.set(min, max);

  range_ = TensorIndexRangeTP<dim>(TensorIndexTP<dim>(_impl::get_num_cells_dir<dim>(breaks)));
}


template<int dim>
CartGridTP<dim>::CartGridTP(const BoundBox<dim> &domain, const std::array<std::size_t, dim> &n_intrvs_dir)
  : CartGridTP(_impl::create_equispaced_breaks<dim>(domain, n_intrvs_dir))
{}

template<int dim>
CartGridTP<dim>::CartGridTP(const std::array<std::size_t, dim> &n_intrvs_dir)
  : CartGridTP(BoundBox<dim>(0.0, 1.0), n_intrvs_dir)
{}

template<int dim> const std::vector<real> &CartGridTP<dim>::get_breaks(int dir) const
{
  return at(breaks_, dir);
}

template<int dim> const BoundBox<dim> &CartGridTP<dim>::get_domain() const
{
  return domain_;
}

template<int dim> int CartGridTP<dim>::get_cell_id(const PointType &point, const Tolerance &tolerance) const
{
  TensorIndexTP<dim> tid;
  for (int dir = 0; dir < dim; ++dir) {
    const auto &breaks = this->get_breaks(dir);
    const auto n_spans = static_cast<int>(breaks.size()) - 1;
    assert(0 < n_spans);

    const auto coord = point(dir);

    const auto begin = breaks.cbegin();
    const auto it = std::prev(std::upper_bound(begin, breaks.cend(), coord + tolerance.value()));

    auto dist = static_cast<int>(std::distance(begin, it));
    if (tolerance.equal(coord, *it)) {
      if (dist == n_spans) {
        // Point at last knot.
        --dist;
      }
    }
    // NOLINTNEXTLINE (readability-simplify-boolean-expr)
    assert(0 <= dist && dist < n_spans);

    tid(dir) = dist;
  }

  return tid.flat(this->get_num_cells_dir());
}

template<int dim>
std::optional<int> CartGridTP<dim>::at_cells_boundary(const PointType &point, const Tolerance &tolerance) const
{
  for (int dir = 0; dir < dim; ++dir) {
    const auto &breaks = this->get_breaks(dir);
    const auto n_spans = static_cast<int>(breaks.size()) - 1;
    assert(0 < n_spans);

    const auto coord = point(dir);

    const auto begin = breaks.cbegin();
    const auto it = std::prev(std::upper_bound(begin, breaks.cend(), coord + tolerance.value()));
    const auto break_coord = *it;

    const auto dist = static_cast<int>(std::distance(begin, it));
    assert(0 <= dist);
    if (tolerance.equal(coord, break_coord) && 0 < dist && dist < n_spans) {
      return dir;
    }
  }

  return std::optional<int>{};
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
template<int dim> bool CartGridTP<dim>::on_boundary(const int cell_id, const int local_facet_id) const
{
  // NOLINTNEXLINE (readability-simplify-boolean-expr)
  assert(0 <= local_facet_id && local_facet_id < dim * 2);

  const int const_dir = local_facet_id / 2;
  const int side = local_facet_id % 2;

  const auto n_cells_dir = this->get_num_cells_dir();
  const TensorIndexTP<dim> cell_tid(cell_id, n_cells_dir);
  return cell_tid(const_dir) == (side == 0 ? 0 : n_cells_dir(const_dir) - 1);
}

template<int dim> std::vector<int> CartGridTP<dim>::get_boundary_cells(int facet_id) const
{
  assert(0 <= facet_id && facet_id < dim * 2);

  // TODO: to substitute with functions
  const int const_dir = facet_id / 2;
  const int side = facet_id % 2;

  const auto n_cells_dir = this->get_num_cells_dir();

  TensorIndexTP<dim> tid_0;
  TensorIndexTP<dim> tid_1(n_cells_dir);
  tid_0(const_dir) = side == 0 ? 0 : n_cells_dir(const_dir) - 1;
  tid_1(const_dir) = side == 0 ? 1 : n_cells_dir(const_dir);

  std::vector<int> cells;
  const auto n_cells_facet = remove_component(n_cells_dir.as_Vector(), const_dir);
  cells.reserve(static_cast<std::size_t>(prod(n_cells_facet)));

  for (const auto cell_tid : TensorIndexRangeTP<dim>(tid_0, tid_1)) {
    cells.push_back(cell_tid.flat(n_cells_dir));
  }

  return cells;
}

template<int dim> int CartGridTP<dim>::to_flat(const TensorIndexTP<dim> &tid) const
{
  return tid.flat(this->get_num_cells_dir());
}

template<int dim> TensorIndexTP<dim> CartGridTP<dim>::to_tensor(const int fid) const
{
  return TensorIndexTP<dim>(fid, this->get_num_cells_dir());
}

template<int dim> TensorSizeTP<dim> CartGridTP<dim>::get_num_cells_dir() const
{
  return this->range_.get_sizes();
}

template<int dim> int CartGridTP<dim>::get_num_cells() const
{
  return this->get_num_cells_dir().size();
}

template<int dim> BoundBox<dim> CartGridTP<dim>::get_cell_domain(const int cell_fid) const
{
  const auto tid = TensorIndexTP<dim>(cell_fid, this->get_num_cells_dir());
  BoundBox<dim> domain;

  // NOLINTBEGIN (misc-const-correctness)
  PointType min_pt;
  PointType max_pt;
  // NOLINTEND (misc-const-correctness)

  for (int dir = 0; dir < dim; ++dir) {
    const auto &brks = this->get_breaks(dir);
    min_pt(dir) = at(brks, tid(dir));
    max_pt(dir) = at(brks, tid(dir) + 1);
  }

  domain.set(min_pt, max_pt);

  return domain;
}

template<int dim>
// NOLINTBEGIN (bugprone-easily-swappable-parameters)
SubCartGridTP<dim>::SubCartGridTP(const CartGridTP<dim> &grid,
  const TensorIndexTP<dim> &indices_start,
  const TensorIndexTP<dim> &indices_end)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  : SubCartGridTP(grid, TensorIndexRangeTP<dim>{ indices_start, indices_end })
{}

template<int dim>
SubCartGridTP<dim>::SubCartGridTP(const CartGridTP<dim> &grid, const TensorIndexRangeTP<dim> &indices_range)
  : grid_(grid), range_(indices_range)
{
  assert(0 < this->get_num_cells());
}

template<int dim>

SubCartGridTP<dim>::SubCartGridTP(const CartGridTP<dim> &grid)
  : SubCartGridTP(grid, TensorIndexTP<dim>{}, TensorIndexTP<dim>(grid.get_num_cells_dir()))
{}

template<int dim> TensorSizeTP<dim> SubCartGridTP<dim>::get_num_cells_dir() const
{
  return this->range_.get_sizes();
}

template<int dim> int SubCartGridTP<dim>::get_num_cells() const
{
  return this->range_.size();
}

template<int dim> bool SubCartGridTP<dim>::is_unique_cell() const
{
  return this->get_num_cells() == 1;
}

template<int dim> const TensorIndexRangeTP<dim> &SubCartGridTP<dim>::get_range() const
{
  return range_;
}

template<int dim> std::array<std::shared_ptr<const SubCartGridTP<dim>>, 2> SubCartGridTP<dim>::split() const
{
  assert(!this->is_unique_cell());

  const auto new_ranges = this->range_.split();

  const auto sub_grid_0 = std::make_shared<SubCartGridTP<dim>>(this->grid_, at(new_ranges, 0));
  const auto sub_grid_1 = std::make_shared<SubCartGridTP<dim>>(this->grid_, at(new_ranges, 1));

  return { sub_grid_0, sub_grid_1 };
}

template<int dim> BoundBox<dim> SubCartGridTP<dim>::get_domain() const
{
  Point<dim> min;
  Point<dim> max;

  const auto &lower = this->range_.get_lower_bound();
  const auto &upper = this->range_.get_upper_bound();

  for (int dir = 0; dir < dim; ++dir) {
    const auto &breaks = this->grid_.get_breaks(dir);
    min(dir) = at(breaks, lower(dir));
    max(dir) = at(breaks, upper(dir));
  }

  return BoundBox<dim>(min, max);
}
template<int dim> const CartGridTP<dim> &SubCartGridTP<dim>::get_grid() const
{
  return this->grid_;
}

template<int dim> int SubCartGridTP<dim>::to_flat(const TensorIndexTP<dim> &tid) const
{
  assert(range_.is_in_range(tid));
  return this->grid_.to_flat(tid);
}

template<int dim> int SubCartGridTP<dim>::get_single_cell() const
{
  assert(this->is_unique_cell());
  return this->to_flat(this->range_.get_lower_bound());
}


// Instantiations
template class CartGridTP<2>;
template class CartGridTP<3>;

template class SubCartGridTP<2>;
template class SubCartGridTP<3>;
}// namespace qugar
