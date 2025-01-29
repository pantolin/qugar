// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file ref_system.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of reference system class.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/ref_system.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <array>
#include <cassert>

namespace qugar::impl {

namespace {
  //! @brief Computes the cross product of two 3-dimensional vectors.
  //!
  //! @param lhs The left-hand side vector.
  //! @param rhs The right-hand side vector.
  //! @return A Point<3> object representing the cross product of the two vectors.
  Point<3> cross_product(const Point<3> &lhs, const Point<3> &rhs)
  {
    return Point<3>{ (lhs(1) * rhs(2)) - (lhs(2) * rhs(1)),
      (lhs(2) * rhs(0)) - (lhs(0) * rhs(2)),
      (lhs(0) * rhs(1)) - (lhs(1) * rhs(0)) };
  };


  //! @brief Creates a 3D orthonormal system from two given vectors.
  //!
  //! The given @p axis_x and @p axis_y define the xy-plane, but they may not be orthonormal
  //! vectors. Thus, the way in which the new system directions are computed is as follows:
  //! First, the @p axis_x is normalized. Then, the z-axis is computed as the normalized
  //! cross-product of the @p axis_x and @p axis_y. Then, the y-axis is recomputed as the
  //! cross-product between normalized vectors along the z- and the x-axes.
  //!
  //! @param axis_x Direction of first vector.
  //! @param axis_y Direction defining the xy-plane together with @p axis_x
  //! @return Orthonormal basis.
  std::array<Point<3>, 3> create_orthonormal_system(Point<3> axis_x, Point<3> axis_y)
  {
    axis_x = (numbers::one / norm(axis_x)) * axis_x;

    auto axis_z = cross_product(axis_x, axis_y);
    assert(norm(axis_z) > numbers::zero);

    axis_z = (numbers::one / norm(axis_z)) * axis_z;

    axis_y = cross_product(axis_z, axis_x);

    return { axis_x, axis_y, axis_z };
  }

  //! @brief Creates an orthonormal system with the given axis.
  //!
  //! This function takes a single axis (@p axis_x) and normalizes it to create an orthonormal system.
  //!
  //! @param axis_x The input axis to be normalized.
  //! @return Generated basis.
  std::array<Point<1>, 1> create_orthonormal_system(Point<1> axis_x)
  {
    axis_x = (numbers::one / norm(axis_x)) * axis_x;
    return { axis_x };
  }

  //! @brief Creates an orthonormal coordinate system based on a given axis.
  //!
  //! This function takes a 2D point representing an axis (@p axis_x) and generates
  //! an orthonormal coordinate system. The input axis is normalized to unit length,
  //! and a perpendicular axis (axis_y) is computed to form the orthonormal system.
  //! The y-axis is just a counter-clockwise rotation of the x-axis by 90 degrees.
  //!
  //! @param axis_x Initial x-axis.
  //! @return Generated basis.
  std::array<Point<2>, 2> create_orthonormal_system(Point<2> axis_x)
  {
    axis_x = (numbers::one / norm(axis_x)) * axis_x;

    const Point<2> axis_y(-axis_x(1), axis_x(0));

    return { axis_x, axis_y };
  }

  //! @brief Creates an orthonormal basis from a z-axis.
  //!
  //! This function takes a primary axis (axis_z) and generates two additional
  //! orthogonal axes (axis_x and axis_y) to form an orthonormal coordinate system.
  //! The resulting axes are returned as an array of three vectors.
  //!
  //! @param axis_z The primary axis of the coordinate system. It must be a non-zero vector.
  //! @return Generated basis.
  //!
  //! @note The function ensures that the input axis_z is normalized and then computes
  //!       the other two orthogonal axes. If the cross product of axis_z and the initial
  //!       axis_x (which is set to (1, 0, 0)) is too small, it uses a different initial
  //!       axis_y (set to (0, 1, 0)) to avoid numerical instability.
  std::array<Point<3>, 3> create_orthonormal_system(Point<3> axis_z)
  {
#ifndef NDEBUG
    const bool is_zero = Tolerance().is_zero(norm(axis_z));
    assert(!is_zero);
#endif// NDEBUG

    axis_z = (numbers::one / norm(axis_z)) * axis_z;


    Point<3> axis_x(numbers::one, numbers::zero, numbers::zero);

    auto axis_y = cross_product(axis_z, axis_x);
    const auto norm_axis_y = norm(axis_y);

    constexpr real tol{ 1.0e-1 };
    if (norm_axis_y > tol) {
      axis_y = (numbers::one / norm_axis_y) * axis_y;
      axis_x = cross_product(axis_y, axis_z);
    } else {
      axis_y = Point<3>(numbers::zero, numbers::one, numbers::zero);

      axis_x = cross_product(axis_y, axis_z);
      axis_x = (numbers::one / norm(axis_x)) * axis_x;
      axis_y = cross_product(axis_z, axis_x);
    }

    return { axis_x, axis_y, axis_z };
  }


}// namespace


template<int dim> RefSystem<dim>::RefSystem() : RefSystem(Point<dim>()) {}

template<int dim>
RefSystem<dim>::RefSystem(const Point<dim> &origin, const std::array<Point<dim>, dim> &basis)
  : origin_(origin), basis_(basis)
{}

template<> RefSystem<1>::RefSystem(const Point<1> &origin) : RefSystem(origin, { Point<1>(numbers::one) }) {}

template<>
RefSystem<2>::RefSystem(const Point<2> &origin)
  : RefSystem(origin, { Point<2>(numbers::one, numbers::zero), Point<2>(numbers::zero, numbers::one) })
{}


template<>
RefSystem<3>::RefSystem(const Point<3> &origin)
  : RefSystem(origin,
      { Point<3>(numbers::one, numbers::zero, numbers::zero),
        Point<3>(numbers::zero, numbers::one, numbers::zero),
        Point<3>(numbers::zero, numbers::zero, numbers::one) })
{}

template<int dim>
RefSystem<dim>::RefSystem(const Point<dim> &origin, const Point<dim> &axis)
  : RefSystem(origin, create_orthonormal_system(axis))
{}

template<int dim>
template<int dim_aux>
  requires(dim_aux == dim && dim == 3)
RefSystem<dim>::RefSystem(const Point<dim> &origin, const Point<dim> &axis_x, const Point<dim> &axis_y)
  : RefSystem(origin, create_orthonormal_system(axis_x, axis_y))
{}

template<int dim> const Point<dim> &RefSystem<dim>::get_origin() const
{
  return origin_;
}

template<int dim> const std::array<Point<dim>, dim> &RefSystem<dim>::get_basis() const
{
  return basis_;
}

template<int dim> bool RefSystem<dim>::is_Cartesian_oriented() const
{
  const Tolerance tol;
  for (int i = 0; i < dim; ++i) {
    if (!tol.equal(at(basis_, i)(i), numbers::one)) {
      return false;
    }
  }
  return true;
}

// Instantiations

template class RefSystem<1>;
template class RefSystem<2>;
template class RefSystem<3>;

template RefSystem<3>::RefSystem(const Point<3> &, const Point<3> &, const Point<3> &);

}// namespace qugar::impl
