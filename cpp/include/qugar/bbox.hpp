// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_LIBRARY_BBOX_HPP

#define QUGAR_LIBRARY_BBOX_HPP

//! @file bbox.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Definition of Cartesian bounding box class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <qugar/concepts.hpp>
#include <qugar/point.hpp>
#include <qugar/types.hpp>

#include <algoim/hyperrectangle.hpp>

#include <array>
#include <cstddef>

namespace qugar {

//!  @brief Class representing a <tt>dim</tt>-dimensional Cartesian product
//!  bounding box.
//!  @tparam dim Dimension of the domain.
template<int dim> class BoundBox
{

public:
  //! @name Constructors
  //@{
  //! @brief Construct a new default @ref BoundBox.
  //! Initializes the box to the unit_ domain [0, 1].
  BoundBox();

  //! @brief Construct a new @ref BoundBox object from the max and min coordinates.
  //! @param min Minimum coordinates of the box.
  //! @param max Maximum coordinates of the box.
  BoundBox(const std::array<real, static_cast<std::size_t>(dim)> &min,
    const std::array<real, static_cast<std::size_t>(dim)> &max);

  //! @brief Construct a new @ref BoundBox object from the max and min coordinates.
  //! @param min Minimum coordinates of the box in all directions.
  //! @param max Maximum coordinates of the box in all directions.
  BoundBox(const real min, const real max);

  //! @brief Construct a new @ref BoundBox object from the max and min coordinates.
  //! @param min Minimum coordinates of the box.
  //! @param max Maximum coordinates of the box.
  BoundBox(const Point<dim> &min, const Point<dim> &max);

  //! @brief Constructs a BoundBox from a given Algoim's hyperrectangle.
  //!
  //! This constructor initializes a BoundBox object using the provided
  //! hyperrectangle from the Algoim library. The hyperrectangle is templated
  //! on the type `real` and the dimension `dim`.
  //!
  //! @param rectangle A constant reference to an algoim::hyperrectangle object
  //!                  representing the bounding box dimensions and coordinates.
  explicit BoundBox(const ::algoim::HyperRectangle<real, dim> &rectangle);

  //@}

private:
  //! @brief Lower box bound.
  Point<dim> min_;

  //! @brief Upper box bound.
  Point<dim> max_;

public:
  //! @brief Sets the bounds of the domain.
  //! @param min Minimum coordinates of the box.
  //! @param max Maximum coordinates of the box.
  void set(const Point<dim> &min, const Point<dim> &max);

  //! @brief Expands the current bounding box such that it contains the
  //! given @p point.
  //! @param point Point that the expanded box must contain.
  void extend(const Point<dim> &point);

  //! @name Query methods
  //@{
  //! @brief Gets the minimum value of the box along direction @p dir.
  //! @param dir Direction along which the minimum value is retrieved.
  //! @return Minimum value along @p dir.
  [[nodiscard]] real min(int dir) const;

  //! @brief Gets the maximum value of the box along direction @p dir.
  //! @param dir Direction along which the maximum value is retrieved.
  //! @return Maximum value along @p dir.
  [[nodiscard]] real max(int dir) const;

  //! @brief Gets the minimum value of the box for 1D boxes.
  //! @return Minimum value.
  template<int dim_aux = dim>
    requires Is1D<dim_aux, dim>
  [[nodiscard]] real min() const;

  //! @brief Gets the maxnimum value of the box for 1D boxes.
  //! @return Maximum value.
  template<int dim_aux = dim>
    requires Is1D<dim_aux, dim>
  [[nodiscard]] real max() const;

  //! @brief Gets the minimum bounds along all the directions.
  //! @return Minimum bounds along all <tt>dim</tt> directions.
  [[nodiscard]] const Point<dim> &min_corner() const;

  //! @brief Gets the maximum bounds along all the directions.
  //! @return Maximum bounds along all <tt>dim</tt> directions.
  [[nodiscard]] const Point<dim> &max_corner() const;

  //! @brief Extends the current bounding box by a given +/- delta on each side.
  //! @param delta Amount by which the box is extended.
  //! @return Extended bounding box.
  [[nodiscard]] BoundBox<dim> extend(real delta) const;

  //! @brief Converts the current object to an Algoim's hyperrectangle.
  //!
  //! This function transforms the current object into an instance of
  //! algoim::hyperrectangle with the specified real type and dimension.
  //!
  //! @return An instance of algoim::hyperrectangle<real, dim> representing
  //!         the current object as a hyperrectangle.
  ::algoim::HyperRectangle<real, dim> to_hyperrectangle() const;

  //! @brief Retrieves the lengths of the bounding box along each dimension.
  //!
  //! This function returns a Point object representing the lengths of the bounding box
  //! along each dimension.
  //!
  //! @return A Point object containing the lengths of the bounding box along each dimension.
  [[nodiscard]] Point<dim> get_lengths() const;

  //! @brief Gets the box length along the given direction.
  //! @param dir Direction along which the length is computed.
  //! @return Box's length along @p dir.
  [[nodiscard]] real length(int dir) const;

  //! @brief Gets the box length for 1D boxes.
  //! @return Box's length.
  template<int dim_aux = dim>
    requires Is1D<dim_aux, dim>
  [[nodiscard]] real length() const;

  //! @brief Computes the box volume.
  //! @return Box's volume.
  [[nodiscard]] real volume() const;

  //! @brief Gets the mid point of the box.
  //! @return Box's mid point.
  [[nodiscard]] Point<dim> mid_point() const;

  //! @brief Scales the given point from the current domain to the @p new_domain.
  //! @param new_domain New domain to which the point is scaled to.
  //! @param point Point to be scaled.
  //! @return Scaledd point.
  [[nodiscard]] Point<dim> scale_to_new_domain(const BoundBox<dim> &new_domain, const Point<dim> &point) const;

  //! @brief Scales the given point from the current domain to the [0,1]^dim domain.
  //! @param point Point to be scaled.
  //! @return Scaled point.
  [[nodiscard]] Point<dim> scale_to_0_1(const Point<dim> &point) const;

  //! @brief Scales the given point from the [0,1]^dim domain to the current domain.
  //! @param point_01 Point to be scaled.
  //! @return Scaled point.
  [[nodiscard]] Point<dim> scale_from_0_1(const Point<dim> &point_01) const;

  //! @brief Performs a slice of the domain reduding it by one dimension.
  //! @return Dim-1 dimensional domain.
  template<int dim_aux = dim>
    requires(dim_aux == dim && dim > 1)
  [[nodiscard]] BoundBox<dim - 1> slice(const int const_dir) const
  {
    Point<dim - 1> _min_corner;
    Point<dim - 1> _max_corner;
    for (int dir = 0, dir2 = 0; dir < dim; ++dir) {
      if (dir != const_dir) {
        _min_corner(dir2) = this->min(dir);
        _max_corner(dir2) = this->max(dir);
        ++dir2;
      }
    }
    return BoundBox<dim - 1>(_min_corner, _max_corner);
  }


  //@}
};

//! @brief Alias representing an interval as a <tt>1</tt>-dimensional bounding box.
using Interval = BoundBox<1>;

}// namespace qugar

#endif// QUGAR_LIBRARY_BBOX_HPP