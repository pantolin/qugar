// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_POINT_HPP
#define QUGAR_POINT_HPP

//! @file point.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Definition and implementation of Point class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <cstddef>
#include <vector>

namespace qugar {

class Tolerance;

//! @brief Class representing a <tt>dim</tt>-dimensional Point.
//!
//! @tparam dim Dimension of point.
template<int dim, typename T = real> using Point = Vector<T, dim>;


//! @brief Finds coincident points in a given set of points.
//!
//! This function identifies pairs of points in the provided vector that are
//! coincident within a specified tolerance. It returns a vector that maps
//! the (indices of the) points in the input vector, to the unique ones.
//! If more than two points are coincident, all those points will be associated
//! to a single one.
//!
//! @tparam dim The dimension of the points.
//! @param points A vector of points to be checked for coincidences.
//! @param tol The tolerance within which points are considered coincident.
//! @param points_map A vector mapping original point indices to unique ones.
template<int dim>
void find_coincident_points(const std::vector<Point<dim>> &points,
  const Tolerance &tol,
  std::vector<std::size_t> &points_map);

//! @brief Makes a given vector of points unique by merging coincident points.
//!
//! This function identifies pairs of points in the provided vector that are
//! coincident within a specified tolerance, and merges them into a single one.
//! It returns a vector that map the (indices of the) original points, to the
//! new (unique) ones. If more than two points are coincident, all those points
//! will be associated to a single one.
//!
//! @tparam dim The dimension of the points.
//! @param points Vector of points to make unique.
//! @param tol The tolerance within which points are considered coincident.
//! @param old_to_new Result vector mapping original point indices to the new (unique) ones.
template<int dim>
void make_points_unique(std::vector<Point<dim>> &points, const Tolerance &tol, std::vector<std::size_t> &old_to_new);

}// namespace qugar

#endif// QUGAR_POINT_HPP