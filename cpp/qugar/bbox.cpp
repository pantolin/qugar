// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file bbox.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of Cartesian bounding box class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>

#include <qugar/concepts.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <algoim/hyperrectangle.hpp>

#include <array>
#include <cassert>
#include <cstddef>

namespace qugar {

template<int dim> BoundBox<dim>::BoundBox() : min_(numbers::zero), max_(numbers::one) {}

template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
BoundBox<dim>::BoundBox(const std::array<real, static_cast<std::size_t>(dim)> &min,
  const std::array<real, static_cast<std::size_t>(dim)> &max)
  : BoundBox()
{
  Point<dim> min_pt;
  Point<dim> max_pt;

  for (int dir = 0; dir < dim; ++dir) {
    min_pt(dir) = at(min, dir);
    max_pt(dir) = at(max, dir);
  }

  this->set(min_pt, max_pt);
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
template<int dim> BoundBox<dim>::BoundBox(const Point<dim> &min, const Point<dim> &max) : BoundBox()
{
  this->set(min, max);
}

template<int dim>
BoundBox<dim>::BoundBox(const real min, const real max) : BoundBox(Point<dim>{ min }, Point<dim>{ max })
{}

template<int dim>
BoundBox<dim>::BoundBox(const ::algoim::HyperRectangle<real, dim> &rectangle)
  : BoundBox<dim>(rectangle.min(), rectangle.max())
{}


// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
template<int dim> void BoundBox<dim>::set(const Point<dim> &min, const Point<dim> &max)
{
  for (int dir = 0; dir < dim; ++dir) {
    assert(min(dir) <= max(dir));
  }
  this->min_ = min;
  this->max_ = max;
}

template<int dim> void BoundBox<dim>::extend(const Point<dim> &point)
{
  for (int dir = 0; dir < dim; ++dir) {
    min_(dir) = std::min(min_(dir), point(dir));
    max_(dir) = std::max(max_(dir), point(dir));
  }
}


template<int dim> real BoundBox<dim>::min(int dir) const
{
  return min_(dir);
}

template<int dim> real BoundBox<dim>::max(int dir) const
{
  return max_(dir);
}

template<int dim>
template<int dim_aux>
  requires Is1D<dim_aux, dim>
real BoundBox<dim>::min() const
{
  return this->min(0);
}

template<int dim>
template<int dim_aux>
  requires Is1D<dim_aux, dim>
real BoundBox<dim>::max() const
{
  return this->max(0);
}


template<int dim> const Point<dim> &BoundBox<dim>::min_corner() const
{
  return min_;
}

template<int dim> const Point<dim> &BoundBox<dim>::max_corner() const
{
  return max_;
}

template<int dim>::algoim::HyperRectangle<real, dim> BoundBox<dim>::to_hyperrectangle() const
{
  const Point<dim> dummy_vec;
  ::algoim::HyperRectangle<real, dim> rectangle{ dummy_vec, dummy_vec };
  for (int dir = 0; dir < dim; ++dir) {
    rectangle.min(dir) = this->min(dir);
    rectangle.max(dir) = this->max(dir);
  }
  return rectangle;
}

template<int dim> BoundBox<dim> BoundBox<dim>::extend(const real delta) const
{
  Point<dim> min_pt = this->min_corner();
  Point<dim> max_pt = this->max_corner();

  for (int dir = 0; dir < dim; ++dir) {
    min_pt(dir) -= delta;
    max_pt(dir) += delta;
  }

  return BoundBox<dim>(min_pt, max_pt);
}


template<int dim> real BoundBox<dim>::length(const int dir) const
{
  return this->max(dir) - this->min(dir);
}

template<int dim> Point<dim> BoundBox<dim>::get_lengths() const
{
  Point<dim> lengths;
  for (int dir = 0; dir < dim; ++dir) {
    lengths(dir) = this->length(dir);
  }
  return lengths;
}

template<int dim>
template<int dim_aux>
  requires Is1D<dim_aux, dim>
real BoundBox<dim>::length() const
{
  return this->length(0);
}

template<int dim> real BoundBox<dim>::volume() const
{
  real volume{ 1.0 };
  for (int dir = 0; dir < dim; ++dir) {
    volume *= this->length(dir);
  }
  return volume;
}

template<int dim> Point<dim> BoundBox<dim>::mid_point() const
{
  return Point<dim>{ numbers::half * (this->min_ + this->max_) };
}

template<int dim>
Point<dim> BoundBox<dim>::scale_to_new_domain(const BoundBox<dim> &new_domain, const Point<dim> &point) const
{
  Point<dim> scaled_point;
  for (int dir = 0; dir < dim; ++dir) {
    scaled_point(dir) =
      (point(dir) - this->min(dir)) * new_domain.length(dir) / this->length(dir) + new_domain.min(dir);
  }
  return scaled_point;
}

template<int dim> Point<dim> BoundBox<dim>::scale_to_0_1(const Point<dim> &point) const
{
  Point<dim> scaled_point;
  for (int dir = 0; dir < dim; ++dir) {
    scaled_point(dir) = (point(dir) - this->min(dir)) / this->length(dir);
  }
  return scaled_point;
}

template<int dim> Point<dim> BoundBox<dim>::scale_from_0_1(const Point<dim> &point_01) const
{
  Point<dim> scaled_point;
  for (int dir = 0; dir < dim; ++dir) {
    scaled_point(dir) = point_01(dir) * this->length(dir) + this->min(dir);
  }
  return scaled_point;
}

// Instantiations
template class BoundBox<1>;
template class BoundBox<2>;
template class BoundBox<3>;

template real BoundBox<1>::min<1>() const;
template real BoundBox<1>::max<1>() const;
template real BoundBox<1>::length<1>() const;

}// namespace qugar
