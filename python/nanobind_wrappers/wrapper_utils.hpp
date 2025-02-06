// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file wrapper_utils.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Helper tools for creating nanobind wrappers.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/point.hpp>
#include <qugar/types.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <vector>

namespace nb = nanobind;

namespace qugar::wrappers {

template<int dim> using npPoint = nb::ndarray<real, nb::numpy, nb::shape<dim>, nb::c_contig>;
template<int dim> using npPointConst = nb::ndarray<const real, nb::numpy, nb::shape<dim>, nb::c_contig>;
template<int dim> using npPointArray = nb::ndarray<real, nb::numpy, nb::shape<-1, dim>, nb::c_contig>;
template<int dim> using npPointConstArray = nb::ndarray<const real, nb::numpy, nb::shape<-1, dim>, nb::c_contig>;

template<int dim> auto get_points_array(const std::vector<Point<dim>> &points)
{
  if (points.empty()) {
    return npPointConstArray<dim>(nullptr, { 0, dim }, nb::handle());
  } else {
    return npPointConstArray<dim>(points.data()->data(), { points.size(), dim }, nb::handle());
  }
}


template<int dim> Point<dim> transform_point(const npPointConst<dim> &point_py)
{
  Point<dim> point;
  for (int dir = 0; dir < dim; ++dir) {
    point(dir) = point_py(dir);
  }
  return point;
}

}// namespace qugar::wrappers