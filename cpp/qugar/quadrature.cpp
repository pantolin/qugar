// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file quadrature.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of quadrature related functionalities.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <qugar/quadrature.hpp>

#include <qugar/point.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>


namespace qugar {

namespace {

  //! @brief Gets quadrature abscissa from an array of quadrature data.
  //!
  //! @tparam N Number of dimensions.
  //! @param n_points Number of points of the 1D quadrature.
  //! It must be smaller or equal than @p N.
  //! @param pt_id Id of the point whose abscissa is returned.
  //! The point must be in the range [0, @p n_points [.
  //! @param quad_data Array containing the quadrature abscissae and weights.
  //! The data is sorted in the following way: for a given number
  //! of points <tt>n</tt>, from 1 to @p N, first the abcissae
  //! in ascending order (in [0,1]) are stored and then associated weights.
  //! Then, the abscissae for <tt>n+1</tt> points and weights, etc.
  //! @return Abscissa of the point.
  template<std::size_t N>
  // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
  real get_abscissa_(const int n_points, const int pt_id, const std::array<real, N *(N + 1)> &quad_data)
  {
    // NOLINTBEGIN (readability-simplify-boolean-expr)
    assert(0 < n_points && n_points <= static_cast<int>(N));
    assert(0 <= pt_id && pt_id < n_points);
    // NOLINTEND (readability-simplify-boolean-expr)
    const int offset = (n_points - 1) * n_points;
    return at(quad_data, offset + (pt_id * 2));
  }


  //! @brief Gets quadrature weight from an array of quadrature data.
  //!
  //! @tparam N Number of dimensions.
  //! @param n_points Number of points of the 1D quadrature.
  //! It must be smaller or equal than @p N.
  //! @param pt_id Id of the point whose weight is returned.
  //! The point must be in the range [0, @p n_points [.
  //! @param quad_data Array containing the quadrature abscissae and weights.
  //! The data is sorted in the following way: for a given number
  //! of points <tt>n</tt>, from 1 to @p N first the abcissae
  //! in ascending order (in [0,1]) are stored and then associated weights.
  //! Then, the abscissae for <tt>n+1</tt> points and weights, etc.
  //! @return Abscissa of the point.
  template<std::size_t N>
  // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
  real get_weight_(const int n_points, const int pt_id, const std::array<real, N *(N + 1)> &quad_data)
  {
    // NOLINTBEGIN (readability-simplify-boolean-expr)
    assert(0 < n_points && n_points <= static_cast<int>(N));
    assert(0 <= pt_id && pt_id < n_points);
    // NOLINTEND (readability-simplify-boolean-expr)
    const int offset = ((n_points - 1) * n_points) + 1;
    return at(quad_data, offset + (pt_id * 2));
  }

}// namespace


real Gauss::get_abscissa(const int n_points, const int pt_id)
{
  return get_abscissa_<n_max>(n_points, pt_id, get_quad_data());
}

real Gauss::get_weight(const int n_points, const int pt_id)
{
  return get_weight_<n_max>(n_points, pt_id, get_quad_data());
}

real TanhSinh::get_abscissa(const int n_points, const int pt_id)
{
  return get_abscissa_<n_max>(n_points, pt_id, get_quad_data());
}

real TanhSinh::get_weight(const int n_points, const int pt_id)
{
  return get_weight_<n_max>(n_points, pt_id, get_quad_data());
}

template<int dim>
Quadrature<dim>::Quadrature(const std::vector<Pt> &points, const std::vector<real> &weights)
  : points_(points), weights_(weights)
{
  assert(!this->points_.empty());
  assert(this->points_.size() == this->weights_.size());
}

template<int dim> std::shared_ptr<Quadrature<dim>> Quadrature<dim>::create_Gauss_01(const int n_points)
{
  std::array<int, dim> n_points_dir{};
  n_points_dir.fill(n_points);
  return create_Gauss_01(n_points_dir);
}

template<int dim> std::shared_ptr<Quadrature<dim>> Quadrature<dim>::create_Gauss_01(const std::array<int, dim> n_points)
{
  for (int dir = 0; dir < dim; ++dir) {
    // NOLINTNEXTLINE (readability-simplify-boolean-expr)
    assert(0 <= at(n_points, dir) && at(n_points, dir) <= Gauss::n_max);
  }

  return create_Gauss(n_points, Domain());
}

template<int dim>
std::shared_ptr<Quadrature<dim>> Quadrature<dim>::create_Gauss(const std::array<int, dim> n_points,
  const Domain &domain)
{
  return create_tp_quadrature<Gauss>(n_points, domain);
}

template<int dim> std::shared_ptr<Quadrature<dim>> Quadrature<dim>::create_tanh_sinh_01(const int n_points)
{
  std::array<int, dim> n_points_dir{};
  n_points_dir.fill(n_points);
  return create_tanh_sinh_01(n_points_dir);
}

template<int dim>
std::shared_ptr<Quadrature<dim>> Quadrature<dim>::create_tanh_sinh_01(const std::array<int, dim> n_points)
{
  for (int dir = 0; dir < dim; ++dir) {
    // NOLINTNEXTLINE (readability-simplify-boolean-expr)
    assert(0 <= at(n_points, dir) && at(n_points, dir) <= TanhSinh::n_max);
  }

  return create_tanh_sinh(n_points, Domain());
}

template<int dim>
std::shared_ptr<Quadrature<dim>> Quadrature<dim>::create_tanh_sinh(const std::array<int, dim> n_points,
  const Domain &domain)
{
  return create_tp_quadrature<TanhSinh>(n_points, domain);
}

template<int dim> auto Quadrature<dim>::points() const -> const std::vector<Pt> &
{
  return points_;
}

template<int dim> const std::vector<real> &Quadrature<dim>::weights() const
{
  return weights_;
}

template<int dim> std::vector<real> &Quadrature<dim>::weights()
{
  // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
  return const_cast<std::vector<real> &>(const_cast<const Quadrature<dim> &>(*this).weights());
}

template<int dim> auto Quadrature<dim>::points() -> std::vector<Pt> &
{
  // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
  return const_cast<std::vector<Pt> &>(const_cast<const Quadrature<dim> &>(*this).points());
}

template<int dim> std::size_t Quadrature<dim>::get_num_points() const
{
  return points_.size();
}

template<int dim> void Quadrature<dim>::scale_weights(const real ratio)
{
  std::for_each(this->weights_.begin(), this->weights_.end(), [ratio](auto &weight) { weight *= ratio; });
}

template<int dim>
template<typename QuadType>
std::shared_ptr<Quadrature<dim>> Quadrature<dim>::create_tp_quadrature(const std::array<int, dim> n_points,
  const Domain &domain)
{
  int n_points_flat{ 1 };
  for (int dir = 0; dir < dim; ++dir) {
    // NOLINTNEXTLINE (readability-simplify-boolean-expr)
    assert(0 <= at(n_points, dir));
    n_points_flat *= at(n_points, dir);
  }

  std::vector<Pt> pts(static_cast<std::size_t>(n_points_flat));
  std::vector<real> wgh(static_cast<std::size_t>(n_points_flat));

  auto it_p = pts.begin();
  auto it_w = wgh.begin();

  if constexpr (dim == 1) {

    const real len = domain.length();
    const real min = domain.min();
    for (int i = 0; i < n_points[0]; ++i) {
      *it_p++ = Point<1>(min + (QuadType::get_abscissa(n_points[0], i) * len));
      at(wgh, i) = QuadType::get_weight(n_points[0], i) * len;
    }

  } else if constexpr (dim == 2) {

    const auto n_1 = at(n_points, 1);
    const auto n_0 = at(n_points, 0);

    const real len_0 = domain.length(0);
    const real min_0 = domain.min(0);
    const real len_1 = domain.length(1);
    const real min_1 = domain.min(1);

    for (int j = 0; j < n_1; ++j) {
      const auto p_j = min_1 + (QuadType::get_abscissa(n_1, j) * len_1);
      const auto w_j = QuadType::get_weight(n_1, j) * len_1;
      for (int i = 0; i < n_0; ++i) {
        const auto p_i = min_0 + (QuadType::get_abscissa(n_0, i) * len_0);
        const auto w_i = QuadType::get_weight(n_0, i) * len_0;
        *it_p++ = Point<2>(p_i, p_j);
        *it_w++ = w_i * w_j;
      }
    }

  } else {// if constexpr (dim == 3)

    const auto n_2 = at(n_points, 2);
    const auto n_1 = at(n_points, 1);
    const auto n_0 = at(n_points, 0);

    const real len_0 = domain.length(0);
    const real min_0 = domain.min(0);
    const real len_1 = domain.length(1);
    const real min_1 = domain.min(1);
    const real len_2 = domain.length(2);
    const real min_2 = domain.min(2);

    for (int k = 0; k < n_2; ++k) {
      const auto p_k = QuadType::get_abscissa(n_2, k) * len_2;
      const auto w_k = min_2 + (QuadType::get_weight(n_2, k) * len_2);
      for (int j = 0; j < n_1; ++j) {
        const auto p_j = QuadType::get_abscissa(n_1, j) * len_1;
        const auto w_j = min_1 + (QuadType::get_weight(n_1, j) * len_1);
        for (int i = 0; i < n_0; ++i) {
          const auto p_i = QuadType::get_abscissa(n_0, i) * len_0;
          const auto w_i = min_0 + (QuadType::get_weight(n_0, i) * len_0);
          *it_p++ = Point<3>(p_i, p_j, p_k);
          *it_w++ = w_i * w_j * w_k;
        }
      }
    }
  }

  return std::make_shared<Quadrature<dim>>(pts, wgh);
}

// Instantiations.
template class Quadrature<1>;
template class Quadrature<2>;
template class Quadrature<3>;

}// namespace qugar
