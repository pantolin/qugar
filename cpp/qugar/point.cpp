// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file point.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of point related functionalities.
//! @date 2025-01-14
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/point.hpp>

#include <qugar/tolerance.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <span>
#include <vector>


namespace qugar {

namespace {

  template<int dim, int sub_dim>
  void find_coincident_points(const std::vector<Point<dim>> &points,
    const Tolerance &tol,
    const std::span<std::size_t> &points_indices,
    std::vector<std::size_t> &points_map)
  {
    static_assert(0 <= sub_dim && sub_dim < dim, "Invalid sub-dimension.");

    const auto n_pts = points_indices.size();
    if (n_pts < 2) {
      return;
    }

    std::ranges::sort(points_indices, [&points](const std::size_t &lhs, const std::size_t &rhs) -> bool {
      return points[lhs](sub_dim) < points[rhs](sub_dim);
    });

    for (std::size_t i = 0; i < n_pts - 1;) {
      const auto it_0 = points_indices.begin() + static_cast<std::ptrdiff_t>(i);
      const auto i_0 = i++;

      auto it_1 = it_0;
      auto it_2 = std::next(it_0);
      for (; i < n_pts; ++i, ++it_1, ++it_2) {
        if (!tol.equal(points[*it_1](sub_dim), points[*it_2](sub_dim))) {
          break;
        }
      }

      const auto n_coincident = i - i_0;
      if (n_coincident >= 2) {
        if constexpr (sub_dim == 0) {
          auto it = std::next(it_0);
          for (std::size_t j = i_0 + 1; j < i; ++j, ++it) {
            points_map[*it] = *it_0;
          }
        } else {
          const std::span<std::size_t> points_indices_sub(it_0, n_coincident);
          find_coincident_points<dim, sub_dim - 1>(points, tol, points_indices_sub, points_map);
        }
      }
    }
  }


}// namespace

template<int dim>
void find_coincident_points(const std::vector<Point<dim>> &points,
  const Tolerance &tol,
  std::vector<std::size_t> &points_map)
{
  // Note: do not use n_pts as the size of points_map,
  // for a reason misterious to me, GCC complains about it.
  // Do not either declare n_pts before this line
  points_map.resize(points.size());

  const auto n_pts = points.size();

  for (std::size_t i = 0; i < n_pts; ++i) {
    points_map[i] = i;
  }

  if (n_pts > 1) {
    auto points_indices = points_map;
    find_coincident_points<dim, dim - 1>(points, tol, std::span<std::size_t>(points_indices), points_map);
  }
}

template<int dim>
void make_points_unique(std::vector<Point<dim>> &points, const Tolerance &tol, std::vector<std::size_t> &old_to_new)
{
  const auto points_copy = points;
  std::vector<std::size_t> old_to_unique;
  find_coincident_points(points_copy, tol, old_to_unique);

  old_to_new.resize(points_copy.size());
  auto point_copy = points_copy.cbegin();
  auto point = points.begin();
  auto otn = old_to_new.begin();
  auto otu = old_to_unique.cbegin();
  std::size_t n_pts = 0;
  for (std::size_t i = 0; i < points_copy.size(); ++i, ++point_copy, ++otn) {
    if (*otu++ == i) {
      *otn = n_pts++;
      *point++ = *point_copy;
    }
  }
  points.resize(n_pts);

  otn = old_to_new.begin();
  for (const auto &otu_ : old_to_unique) {
    *otn++ = old_to_new[otu_];
  }
}


// Instantiations.

template void
  find_coincident_points<2>(const std::vector<Point<2>> &points, const Tolerance &, std::vector<std::size_t> &);
template void
  find_coincident_points<3>(const std::vector<Point<3>> &points, const Tolerance &, std::vector<std::size_t> &);

template void make_points_unique<2>(std::vector<Point<2>> &, const Tolerance &, std::vector<std::size_t> &);
template void make_points_unique<3>(std::vector<Point<3>> &, const Tolerance &, std::vector<std::size_t> &);

}// namespace qugar