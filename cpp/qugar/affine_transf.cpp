// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file affine_transf.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of affine transformation class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/affine_transf.hpp>

#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/interval.hpp>

#include <cassert>
#include <utility>

namespace qugar::impl {

template<int dim> using Interval = ::algoim::Interval<dim>;

namespace {
  const auto cross_product = [](const auto &lhs, const auto &rhs) -> Vector<real, 3> {
    return Vector<real, 3>{ (lhs(1) * rhs(2)) - (lhs(2) * rhs(1)),
      (lhs(2) * rhs(0)) - (lhs(0) * rhs(2)),
      (lhs(0) * rhs(1)) - (lhs(1) * rhs(0)) };
  };

  Vector<real, 4> invert_matrix(const Vector<real, 4> &matrix)
  {
    const auto &mat = matrix;

    const real det = (mat(0) * mat(3)) - (mat(1) * mat(2));
#ifndef NDEBUG
    const bool is_zero = Tolerance().is_zero(det);
    assert(!is_zero);
#endif// NDEBUG
    const real inv_det = numbers::one / det;

    Vector<real, 4> inv_matrix;
    inv_matrix(0) = inv_det * mat(3);
    inv_matrix(1) = -inv_det * mat(1);
    inv_matrix(2) = -inv_det * mat(2);
    inv_matrix(3) = inv_det * mat(0);
    return inv_matrix;
  }

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  Vector<real, 9> invert_matrix(const Vector<real, 9> &matrix)
  {
    const auto &mat = matrix;

    // computes the inverse of a matrix m
    const real det = mat(0) * (mat(4) * mat(8) - mat(7) * mat(5)) - mat(1) * (mat(3) * mat(8) - mat(5) * mat(6))
                     + mat(2) * (mat(3) * mat(7) - mat(4) * mat(6));
#ifndef NDEBUG
    const bool is_zero = Tolerance().is_zero(det);
    assert(!is_zero);
#endif// NDEBUG
    const real inv_det = numbers::one / det;

    return Vector<real, 9>{
      (mat(4) * mat(8) - mat(7) * mat(5)) * inv_det,
      (mat(2) * mat(7) - mat(1) * mat(8)) * inv_det,
      (mat(1) * mat(5) - mat(2) * mat(4)) * inv_det,
      (mat(5) * mat(6) - mat(3) * mat(8)) * inv_det,
      (mat(0) * mat(8) - mat(2) * mat(6)) * inv_det,
      (mat(3) * mat(2) - mat(0) * mat(5)) * inv_det,
      (mat(3) * mat(7) - mat(6) * mat(4)) * inv_det,
      (mat(6) * mat(1) - mat(0) * mat(7)) * inv_det,
      (mat(0) * mat(4) - mat(3) * mat(1)) * inv_det,
    };
  }


  //! @brief Creates a orthonormal system from two given vectors.
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
  std::array<Vector<real, 3>, 3> create_orthonormal_system(const Point<3> &axis_x, const Point<3> &axis_y)
  {

    Point<3> e_0 = axis_x;
    assert(norm(e_0) > numbers::zero);
    e_0 /= norm(e_0);


    auto e_2 = cross_product(e_0, axis_y);
    assert(norm(e_2) > numbers::zero);
    e_2 /= norm(e_2);

    const auto e_1 = cross_product(e_2, e_0);

    return { e_0, e_1, e_2 };
  }
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)


}// namespace

std::pair<Vector<real, 3>, Vector<real, 3>> create_reference_system_around_axis(Vector<real, 3> axis_z)
{
#ifndef NDEBUG
  const bool is_zero = Tolerance().is_zero(norm(axis_z));
  assert(!is_zero);
#endif// NDEBUG
  axis_z /= norm(axis_z);

  const Point<3> guess_axis_x(numbers::one, numbers::zero, numbers::zero);

  Vector<real, 3> axis_x;

  auto axis_y = cross_product(axis_z, guess_axis_x);
  const auto norm_axis_y = norm(axis_y);

  constexpr real tol{ 1.0e-4 };
  if (norm_axis_y > tol) {
    axis_y /= norm_axis_y;
    axis_x = cross_product(axis_y, axis_z);
  } else {
    const Point<3> guess_axis_y(numbers::zero, numbers::one, numbers::zero);

    axis_x = cross_product(guess_axis_y, axis_z);
    axis_x /= norm(axis_x);
    axis_y = cross_product(axis_z, axis_y);
  }

  return std::make_pair(axis_x, axis_y);
}

template<int dim> AffineTransf<dim>::AffineTransf() : AffineTransf(Point<dim>()) {}


template<int dim> AffineTransf<dim>::AffineTransf(const Point<dim> &origin) : AffineTransf(origin, numbers::one) {}

template<int dim>
AffineTransf<dim>::AffineTransf(const Point<dim> &origin, const real scale) : AffineTransf(compute_coefs(origin, scale))
{}

template<int dim>
template<int dim_aux>
  requires(dim_aux == dim && dim == 2)
AffineTransf<dim>::AffineTransf(const Point<dim> &origin, const Point<dim> &axis_x)
  : AffineTransf(origin, axis_x, numbers::one, numbers::one)
{}

template<int dim>
template<int dim_aux>
  requires(dim_aux == dim && dim == 3)
AffineTransf<dim>::AffineTransf(const Point<dim> &origin, const Point<dim> &axis_x, const Point<dim> &axis_y)
  : AffineTransf(origin, axis_x, axis_y, numbers::one, numbers::one, numbers::one)
{}

template<int dim>
template<int dim_aux>
  requires(dim_aux == dim && dim == 2)
AffineTransf<dim>::AffineTransf(const Point<dim> &origin,
  const Point<dim> &axis_x,
  const real scale_x,
  const real scale_y)
  : AffineTransf(compute_coefs(origin, axis_x, scale_x, scale_y))
{}

template<int dim>
template<int dim_aux>
  requires(dim_aux == dim && dim == 3)
AffineTransf<dim>::AffineTransf(const Point<dim> &origin,
  const Point<dim> &axis_x,
  const Point<dim> &axis_y,
  const real scale_x,
  const real scale_y,
  const real scale_z)
  : AffineTransf(compute_coefs(origin, axis_x, axis_y, scale_x, scale_y, scale_z))
{}

template<int dim> AffineTransf<dim>::AffineTransf(const Vector<real, n_coefs> &coefs) : coefs_(coefs) {}

template<int dim>
auto AffineTransf<dim>::compute_coefs(const Point<dim> &origin, const real scale) -> Vector<real, n_coefs>
{
  if constexpr (dim == 2) {
    return Vector<real, n_coefs>{ scale, numbers::zero, numbers::zero, scale, origin(0), origin(1) };
  } else {// if { constexpr (dim == 3)
    return Vector<real, n_coefs>{ scale,
      numbers::zero,
      numbers::zero,
      numbers::zero,
      scale,
      numbers::zero,
      numbers::zero,
      numbers::zero,
      scale,
      origin(0),
      origin(1),
      origin(2) };
  }
}

template<int dim>
template<int dim_aux>
  requires(dim_aux == dim && dim == 3)
// NOLINTBEGIN (bugprone-easily-swappable-parameters)
auto AffineTransf<dim>::compute_coefs(const Point<dim> &origin,
  const Point<dim> &axis_x,
  const Point<dim> &axis_y,
  const real scale_x,
  const real scale_y,
  const real scale_z) -> Vector<real, n_coefs>
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  assert(scale_x > numbers::zero);
  assert(scale_y > numbers::zero);
  assert(scale_z > numbers::zero);

  const auto system = create_orthonormal_system(axis_x, axis_y);
  const auto &e_0 = at(system, 0);
  const auto &e_1 = at(system, 1);
  const auto &e_2 = at(system, 2);

  Vector<real, n_coefs> coefs;
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  coefs(0) = scale_x * e_0(0);
  coefs(1) = scale_y * e_1(0);
  coefs(2) = scale_z * e_2(0);
  coefs(3) = scale_x * e_0(1);
  coefs(4) = scale_y * e_1(1);
  coefs(5) = scale_z * e_2(1);
  coefs(6) = scale_x * e_0(2);
  coefs(7) = scale_y * e_1(2);
  coefs(8) = scale_z * e_2(2);
  coefs(9) = origin(0);
  coefs(10) = origin(1);
  coefs(11) = origin(2);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  return coefs;
}

template<int dim>
template<int dim_aux>
  requires(dim_aux == dim && dim == 2)
// NOLINTBEGIN (bugprone-easily-swappable-parameters)
auto AffineTransf<dim>::compute_coefs(const Point<dim> &origin,
  const Point<dim> &axis_x,
  const real scale_x,
  const real scale_y) -> Vector<real, n_coefs>
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  assert(scale_x > numbers::zero);
  assert(scale_y > numbers::zero);

  Point<2> e_0 = axis_x;
  assert(norm(e_0) > numbers::zero);
  e_0 /= norm(e_0);

  const Vector<real, 2> e_1{ -e_0(1), e_0(0) };

  Vector<real, n_coefs> coefs;
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  coefs(0) = scale_x * e_0(0);
  coefs(1) = scale_y * e_1(0);
  coefs(2) = scale_x * e_0(1);
  coefs(3) = scale_y * e_1(1);
  coefs(4) = origin(0);
  coefs(5) = origin(1);
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
  return coefs;
}

template<int dim> AffineTransf<dim> AffineTransf<dim>::operator*(const AffineTransf<dim> &rhs) const
{
  const auto &lhs = *this;

  Vector<real, n_coefs> new_coefs;

  const auto ind = [](const int row, const int col) { return (row * dim) + col; };

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      auto &coef = new_coefs(ind(i, j));
      coef = numbers::zero;
      for (int k = 0; k < dim; ++k) {
        coef += lhs.coefs_(ind(i, k)) * rhs.coefs_(ind(k, j));
      }
    }

    auto &coef = new_coefs((dim * dim) + i);
    coef = lhs.coefs_((dim * dim) + i);
    for (int k = 0; k < dim; ++k) {
      coef += lhs.coefs_(ind(i, k)) * rhs.coefs_((dim * dim) + k);
    }
  }

  return AffineTransf<dim>(new_coefs);
}

template<int dim> AffineTransf<dim> AffineTransf<dim>::inverse() const
{
  Vector<real, dim * dim> matrix;
  for (int i = 0; i < dim * dim; ++i) {
    matrix(i) = this->coefs_(i);
  }

  const auto inv_matrix = invert_matrix(matrix);
  Vector<real, n_coefs> new_coefs;
  for (int i = 0; i < dim * dim; ++i) {
    new_coefs(i) = inv_matrix(i);
  }

  for (int i = 0; i < dim; ++i) {
    auto &coef = new_coefs((dim * dim) + i);
    coef = numbers::zero;
    for (int k = 0; k < dim; ++k) {
      coef -= new_coefs((dim * i) + k) * this->coefs_((dim * dim) + k);
    }
  }

  return AffineTransf<dim>(new_coefs);
}


template<int dim>
template<typename T>
Vector<T, dim> AffineTransf<dim>::transform_point(const Vector<T, dim> &point) const
{
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  if constexpr (dim == 2) {
    return Vector<T, dim>{ coefs_(0) * point(0) + coefs_(1) * point(1) + coefs_(4),
      coefs_(2) * point(0) + coefs_(3) * point(1) + coefs_(5) };
  } else {// if constexpr (dim == 3)

    return Vector<T, dim>{ coefs_(0) * point(0) + coefs_(1) * point(1) + coefs_(2) * point(2) + coefs_(9),
      coefs_(3) * point(0) + coefs_(4) * point(1) + coefs_(5) * point(2) + coefs_(10),
      coefs_(6) * point(0) + coefs_(7) * point(1) + coefs_(8) * point(2) + coefs_(11) };
  }
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}


template<int dim>
template<typename T>
Vector<T, dim> AffineTransf<dim>::transform_vector(const Vector<T, dim> &vector) const
{
  const auto &cfs = this->coefs_;
  const auto &vec = vector;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  if constexpr (dim == 2) {
    return Vector<T, dim>{ cfs(0) * vec(0) + cfs(2) * vec(1), cfs(1) * vec(0) + cfs(3) * vec(1) };
  } else {// if constexpr (dim == 3)
    return Vector<T, dim>{ cfs(0) * vec(0) + cfs(3) * vec(1) + cfs(6) * vec(2),
      cfs(1) * vec(0) + cfs(4) * vec(1) + cfs(7) * vec(2),
      cfs(2) * vec(0) + cfs(5) * vec(1) + cfs(8) * vec(2) };
  }
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}

template<int dim>
template<typename T>
auto AffineTransf<dim>::transform_tensor(const Tensor<T> &tensor) const -> Tensor<T>
{
  const auto &tns = tensor;
  const auto &cfs = this->coefs_;
  constexpr real two = numbers::two;

  Tensor<T> new_tensor{};
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  if constexpr (dim == 2) {
    new_tensor(0) = tns(0) * cfs(0) * cfs(0) + two * tns(1) * cfs(0) * cfs(2) + tns(2) * cfs(2) * cfs(2);

    new_tensor(1) = tns(0) * cfs(0) * cfs(1) + tns(1) * (cfs(0) * cfs(3) + cfs(1) * cfs(2)) + tns(2) * cfs(2) * cfs(3);

    new_tensor(2) = tns(0) * cfs(1) * cfs(1) + two * tns(1) * cfs(1) * cfs(3) + tns(2) * cfs(3) * cfs(3);
  } else {// if constexpr (dim == 3)
    new_tensor(0) = tns(0) * cfs(0) * cfs(0) + two * tns(1) * cfs(0) * cfs(3) + two * tns(2) * cfs(0) * cfs(6)
                    + tns(3) * cfs(3) * cfs(3) + two * tns(4) * cfs(3) * cfs(6) + tns(5) * cfs(6) * cfs(6);

    new_tensor(1) = tns(0) * cfs(0) * cfs(1) + tns(1) * (cfs(0) * cfs(4) + cfs(1) * cfs(3))
                    + tns(2) * (cfs(0) * cfs(7) + cfs(1) * cfs(6)) + tns(3) * cfs(3) * cfs(4)
                    + tns(4) * (cfs(3) * cfs(7) + cfs(4) * cfs(6)) + tns(5) * cfs(6) * cfs(7);

    new_tensor(2) = tns(0) * cfs(0) * cfs(2) + tns(1) * (cfs(0) * cfs(5) + cfs(2) * cfs(3))
                    + tns(2) * (cfs(0) * cfs(8) + cfs(2) * cfs(6)) + tns(3) * cfs(3) * cfs(5)
                    + tns(4) * (cfs(3) * cfs(8) + cfs(5) * cfs(6)) + tns(5) * cfs(6) * cfs(8);

    new_tensor(3) = tns(0) * cfs(1) * cfs(1) + two * tns(1) * cfs(1) * cfs(4) + two * tns(2) * cfs(1) * cfs(7)
                    + tns(3) * cfs(4) * cfs(4) + two * tns(4) * cfs(4) * cfs(7) + tns(5) * cfs(7) * cfs(7);

    new_tensor(4) = tns(0) * cfs(1) * cfs(2) + tns(1) * (cfs(1) * cfs(5) + cfs(2) * cfs(4))
                    + tns(2) * (cfs(1) * cfs(8) + cfs(2) * cfs(7)) + tns(3) * cfs(4) * cfs(5)
                    + tns(4) * (cfs(4) * cfs(8) + cfs(5) * cfs(7)) + tns(5) * cfs(7) * cfs(8);

    new_tensor(5) = tns(0) * cfs(2) * cfs(2) + two * tns(1) * cfs(2) * cfs(5) + two * tns(2) * cfs(2) * cfs(8)
                    + tns(3) * cfs(5) * cfs(5) + two * tns(4) * cfs(5) * cfs(8) + tns(5) * cfs(8) * cfs(8);
  }
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

  return new_tensor;
}


// Instantiations

template class AffineTransf<2>;
template class AffineTransf<3>;

template AffineTransf<2>::AffineTransf(const Point<2> &, const Point<2> &);
template AffineTransf<2>::AffineTransf(const Point<2> &, const Point<2> &, const real, const real);

template AffineTransf<3>::AffineTransf(const Point<3> &, const Point<3> &, const Point<3> &);
template AffineTransf<3>::AffineTransf(const Point<3> &,
  const Point<3> &,
  const Point<3> &,
  const real,
  const real,
  const real);


template Vector<real, 2> AffineTransf<2>::transform_point(const Vector<real, 2> &) const;
template Vector<real, 3> AffineTransf<3>::transform_point(const Vector<real, 3> &) const;

template Vector<Interval<2>, 2> AffineTransf<2>::transform_point(const Vector<Interval<2>, 2> &) const;
template Vector<Interval<3>, 3> AffineTransf<3>::transform_point(const Vector<Interval<3>, 3> &) const;


template Vector<real, 2> AffineTransf<2>::transform_vector(const Vector<real, 2> &) const;
template Vector<real, 3> AffineTransf<3>::transform_vector(const Vector<real, 3> &) const;

template Vector<Interval<2>, 2> AffineTransf<2>::transform_vector(const Vector<Interval<2>, 2> &) const;
template Vector<Interval<3>, 3> AffineTransf<3>::transform_vector(const Vector<Interval<3>, 3> &) const;

template Vector<real, 3> AffineTransf<2>::transform_tensor(const Vector<real, 3> &) const;
template Vector<real, 6> AffineTransf<3>::transform_tensor(const Vector<real, 6> &) const;

template Vector<Interval<3>, 3> AffineTransf<2>::transform_tensor(const Vector<Interval<3>, 3> &) const;
template Vector<Interval<6>, 6> AffineTransf<3>::transform_tensor(const Vector<Interval<6>, 6> &) const;

}// namespace qugar::impl
