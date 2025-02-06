// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_REF_SYSTEM_HPP
#define QUGAR_IMPL_REF_SYSTEM_HPP

//! @file ref_system.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of reference system class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/point.hpp>

namespace qugar::impl {

//! @class RefSystem
//! @brief A class representing a reference system in a given dimension.
//!
//! This class provides functionality to create and manage a reference system
//! in a specified dimension. It supports creating Cartesian reference systems
//! centered at the origin with canonical directions, as well as custom reference
//! systems with specified origins and axes.
//!
//! @tparam dim The dimension of the reference system.
template<int dim> class RefSystem
{
public:
  //! @brief Constructs a new RefSystem object.
  //!
  //! This is the default constructor for the RefSystem class.
  //!
  //! It creates a reference system with the origin at the origin point and the basis
  //! with along the Cartesian axes.
  RefSystem();

  //! @brief Constructs a RefSystem object with the specified origin point.
  //!
  //! It creates a reference system with the origin at the given @p origin point and the basis
  //! with along the Cartesian axes.
  //!
  //! @param origin The origin point of the reference system.
  explicit RefSystem(const Point<dim> &origin);

  //! @brief Constructs a reference system with a specified origin.
  //!
  //! The given @p axis has different meanings depending on the dimension of the reference system.
  //! For 1D, the axis is the x-axis; for 2D, the axis is the x-axis; and for 3D, the axis is the z-axis.
  //! Thus, in 2D, the y-axis is computed by rotating the given @p axis 90 degrees.
  //! In 3D, given @p axis is considered to be the z-axis, then, x- and y-axes are computed
  //! to be orthonormal to @p axis.
  //!
  //! @param origin The origin point of the reference system.
  //! @param axis The axis point of the reference system.
  RefSystem(const Point<dim> &origin, const Point<dim> &axis);

  //! @brief Constructs a reference system with a specified origin and two axes.
  //!
  //! This constructor is only available for 3-dimensional reference systems.
  //!
  //! The given @p axis_x and @p axis_y define the xy-plane, but they may not be orthonormal
  //! vectors. Thus, the way in which the new system directions are computed is as follows:
  //! First, the @p axis_x is normalized. Then, the z-axis is computed as the normalized
  //! cross-product of the @p axis_x and @p axis_y. Then, the y-axis is recomputed as the
  //! cross-product between normalized vectors along the z- and the x-axes.
  //!
  //! @tparam dim_aux Auxiliary template parameter to enforce the dimension check.
  //! @param origin The origin point of the reference system.
  //! @param axis_x The x-axis point of the reference system.
  //! @param axis_y The y-axis point of the reference system.
  template<int dim_aux = dim>
    requires(dim_aux == dim && dim == 3)
  RefSystem(const Point<dim> &origin, const Point<dim> &axis_x, const Point<dim> &axis_y);

private:
  //! @brief Constructs a reference system with a specified origin and basis.
  //!
  //! This constructor is private and is used internally.
  //!
  //! @param origin The origin point of the reference system.
  //! @param basis The orthonormal basis of the reference system.
  RefSystem(const Point<dim> &origin, const std::array<Point<dim>, dim> &basis);

public:
  //! @brief Gets the origin point of the reference system.
  //!
  //! @return Constant reference to the origin point of the reference system.
  const Point<dim> &get_origin() const;

  //! @brief Gets the orthonormal basis of the reference system.
  //!
  //! @return Constant reference to the orthonormal basis of the reference system.
  const std::array<Point<dim>, dim> &get_basis() const;

  //! @brief Checks if the reference system is Cartesian oriented.
  //!
  //! This function determines whether the current reference system
  //! is oriented in a Cartesian manner. I.e., if the basis vectors are oriented
  //! along the Cartesian directions x, y, z.
  //!
  //! @return true if the reference system is Cartesian oriented, false otherwise.
  [[nodiscard]] bool is_Cartesian_oriented() const;

private:
  //! @brief Origin.
  Point<dim> origin_;

  //! @brief Orthonormal basis.
  std::array<Point<dim>, dim> basis_;
};


}// namespace qugar::impl


#endif// QUGAR_IMPL_REF_SYSTEM_HPP