// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_AFFINE_TRANSF_HPP
#define QUGAR_IMPL_AFFINE_TRANSF_HPP

//! @file affine_transf.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of affine transformation class.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <utility>

namespace qugar::impl {

std::pair<Point<3>, Point<3>> create_reference_system_around_axis(Point<3> axis_z);


//! @brief Class for representing affine transformations.
//!
//! It transforms points as y = A * x + b, where x is the original
//! point, y is the new point, b is the translation vector, and
//! A is the transformation matrix that may be defined by rotations
//! and scaling.
//!
//! The member coefs_ stores the involved coefficients. Thus, first
//! dim*dim values of coefs_ contain the values of the matrix A
//! (row-wise), and the last dim values, the ones of the vector b.
//!
//! @tparam dim Parametric direction.
template<int dim> class AffineTransf
{
public:
  static constexpr int n_coefs = dim * (dim + 1);
  template<typename T> using Tensor = Vector<T, dim *(dim + 1) / 2>;

  //! @brief Default constructor.
  //! Creates the identity transformation.
  explicit AffineTransf();

  //! @brief Constructs a new affine tranformation that simply translates the origin.
  //!
  //! @param origin New origin of the reference system.
  explicit AffineTransf(const Point<dim> &origin);

  //! @brief Constructs a new affine transformation that first applies an isotropic
  //! scaling and then translates the origin.
  //!
  //! @param origin New origin of the reference system.
  //! @param scale Scaling factor to be applied along all the directions.
  AffineTransf(const Point<dim> &origin, real scale);

  //! @brief Constructs a new 2D affine transformation defined by an @p origin and the @p axis_x.
  //!
  //! It first rotates the axes and then performs the translation.
  //!
  //! The y-axis is computed by rotating @p axis_x 90 degrees counter-clockwise.
  //!
  //! @param origin New origin of the reference system.
  //! @param axis_x New x-axis of the system (it will be normalized).
  template<int dim_aux = dim>
    requires(dim_aux == dim && dim == 2)
  AffineTransf(const Point<dim> &origin, const Point<dim> &axis_x);

  //! @brief Constructs a new 3D affine transformation defined by an @p origin and a couple of
  //! directions that define the xy-plane.
  //!
  //! It first rotates the axes and then performs the translation.
  //!
  //! The given @p axis_x and @p axis_y define the xy-plane, but they may not be orthonormal
  //! vectors. Thus, the way in which the new system directions are computed is as follows:
  //! First, the @p axis_x is normalized. Then, the z-axis is computed as the normalized
  //! cross-product of the normalized vector @p axis_x and @p axis_y.
  //! Then, the y-axis is recomputed as the cross-product between normalized vectors along
  //! the z- and the x-axes.
  //!
  //! @param origin New origin of the reference system.
  //! @param axis_x New x-axis of the system (it will be rescaled).
  //! @param axis_y Axis defining the xy-plane together with @p axis_x. It must be not parallel
  //! to @p axis_x.
  template<int dim_aux = dim>
    requires(dim_aux == dim && dim == 3)
  AffineTransf(const Point<dim> &origin, const Point<dim> &axis_x, const Point<dim> &axis_y);

  //! @brief Constructs a new 2D affine transformation defined by an @p origin, the @p axis_x, and an
  //! orthotropic scaling along the x- and y-axes.
  //!
  //! It first applies the scaling, then rotates the axes, and finally performs the translation.
  //!
  //! The y-axis is computed by rotating @p axis_x 90 degrees counter-clockwise.
  //!
  //! @param origin New origin of the reference system.
  //! @param axis_x New x-axis of the system (it will be rescaled).
  //! @param scale_x Scaling along the new x-axis.
  //! @param scale_y Scaling along the new y-axis.
  template<int dim_aux = dim>
    requires(dim_aux == dim && dim == 2)
  AffineTransf(const Point<dim> &origin, const Point<dim> &axis_x, real scale_x, real scale_y);

  //! @brief Constructs a new 3D affine transformation defined by an @p origin, a couple of directions
  //! that define the xy-plane, and an orthotropic scaling along the x-, y, and z-axes.
  //!
  //! It first applies the scaling, then rotates the axes, and finally performs the translation.
  //!
  //! The given @p axis_x and @p axis_y define the xy-plane, but they may not be orthonormal
  //! vectors. Thus, the way in which the new system directions are computed is as follows:
  //! First, the @p axis_x is normalized. Then, the z-axis is computed as the normalized
  //! cross-product of the normalized vector @p axis_x and @p axis_y.
  //! Then, the y-axis is recomputed as the cross-product between normalized vectors along
  //! the z- and the x-axes.
  //!
  //! @param origin New origin of the reference system.
  //! @param axis_x New x-axis of the system (it will be rescaled).
  //! @param axis_y Axis defining the xy-plane together with @p axis_x. It must be not parallel
  //! to @p axis_x.
  //! @param scale_x Scaling along the new x-axis.
  //! @param scale_y Scaling along the new y-axis.
  //! @param scale_z Scaling along the new z-axis.
  template<int dim_aux = dim>
    requires(dim_aux == dim && dim == 3)
  AffineTransf(const Point<dim> &origin,
    const Point<dim> &axis_x,
    const Point<dim> &axis_y,
    real scale_x,
    real scale_y,
    real scale_z);

  //! @brief Constructs a new object through its coefficients. a new Affine Transf object
  //!
  //! See the class documentation for information about the coefficients.
  //!
  //! @param coefs Transformation coefficients.
  explicit AffineTransf(const Vector<real, n_coefs> &coefs);

  //! @brief Concatenates two affine transformation to generate a new one.
  //!
  //! The new generated transformation is equivalent to, first, applying
  //! the given @p rhs transformation to a point (gradient, hessian), and then,
  //! the current one.
  //!
  //! @param rhs First transformation to apply.
  //! @return Concatenated transformations.
  AffineTransf<dim> operator*(const AffineTransf<dim> &rhs) const;

  //! @brief Inverts the current transformation.
  //!
  //! @return Inverted transformation.
  AffineTransf<dim> inverse() const;

  //! @brief Transforms a point from the original to the new reference system.
  //!
  //! @tparam T Type of the point.
  //! @param point Point to be transformed.
  //! @return Transformed point.
  template<typename T> [[nodiscard]] Vector<T, dim> transform_point(const Vector<T, dim> &point) const;

  //! @brief Transforms a vector from the original to the new reference system
  //! (without translation).
  //!
  //! @tparam T Type of the vector.
  //! @param vector Vector to be transformed.
  //! @return Transformed vector.
  template<typename T> [[nodiscard]] Vector<T, dim> transform_vector(const Vector<T, dim> &vector) const;

  //! @brief Transforms a (second-order symmetric) tensor from the original to the new reference system
  //! (without translation).
  //!
  //! @tparam T Type of the tensor.
  //! @param tensor Second-order symmetric tensor to be transformed.
  //! @return Transformed tnesor.
  template<typename T> [[nodiscard]] Tensor<T> transform_tensor(const Tensor<T> &tensor) const;

private:
  //! @brief Computes the coefficients of a new 2D affine transformation that translates the origin and apply an
  //! isotropic scaling.
  //!
  //! It first applies the scaling and then performs the translation.
  //!
  //! @param origin New origin of the reference system.
  //! @param scale Scaling factor to be applied along all the directions.
  //! @return Computed coefficients.
  [[nodiscard]] static Vector<real, n_coefs> compute_coefs(const Point<dim> &origin, real scale);

  //! @brief Computes the coefficients of a new 3D affine transformation defined by an @p origin, a couple of directions
  //! that define the xy-plane, and an orthotropic scaling along the x-, y, and z-axes.
  //!
  //! It first applies the scaling, then rotates the axes, and finally performs the translation.
  //!
  //! The given @p axis_x and @p axis_y define the xy-plane, but they may not be orthonormal
  //! vectors. Thus, the way in which the new system directions are computed is as follows:
  //! First, the @p axis_x is normalized. Then, the z-axis is computed as the normalized
  //! cross-product of the normalized vector @p axis_x and @p axis_y.
  //! Then, the y-axis is recomputed as the cross-product between normalized vectors along
  //! the z- and the x-axes.
  //!
  //! The scaling is performed respect to the new (rotated/translated) reference system.
  //!
  //! @param origin New origin of the reference system.
  //! @param axis_x New x-axis of the system (it will be rescaled).
  //! @param axis_y Axis defining the xy-plane together with @p axis_x. It must be not parallel
  //! to @p axis_x.
  //! @param scale_x Scaling along the new x-axis.
  //! @param scale_y Scaling along the new y-axis.
  //! @param scale_z Scaling along the new z-axis.
  //! @return Computed coefficients.
  template<int dim_aux = dim>
    requires(dim_aux == dim && dim == 3)
  [[nodiscard]] static Vector<real, n_coefs> compute_coefs(const Point<dim> &origin,
    const Point<dim> &axis_x = Point<dim>(numbers::one, numbers::zero, numbers::zero),
    const Point<dim> &axis_y = Point<dim>(numbers::zero, numbers::one, numbers::zero),
    real scale_x = numbers::one,
    real scale_y = numbers::one,
    real scale_z = numbers::one);

  //! @brief Computes the coefficients of a new 2D affine transformation defined by an @p origin,
  //! the @p axis_x, and an orthotropic scaling along the x- and y-axes.
  //!
  //! It first applies the scaling, then rotates the axes, and finally performs the translation.
  //!
  //! The y-axis is computed by rotating @p axis_x 90 degrees counter-clockwise.
  //!
  //! The scaling is performed respect to the new (rotated/translated) reference system.
  //!
  //! @param origin New origin of the reference system.
  //! @param axis_x New x-axis of the system (it will be rescaled).
  //! @param scale_x Scaling along the new x-axis.
  //! @param scale_y Scaling along the new y-axis.
  template<int dim_aux = dim>
    requires(dim_aux == dim && dim == 2)
  [[nodiscard]] static Vector<real, n_coefs> compute_coefs(const Point<dim> &origin,
    const Point<dim> &axis_x = Point<dim>(numbers::one, numbers::zero),
    real scale_x = numbers::one,
    real scale_y = numbers::one);

  //! @brief  Transformation coefficients.
  Vector<real, n_coefs> coefs_;
};


}// namespace qugar::impl


#endif// QUGAR_IMPL_AFFINE_TRANSF_HPP