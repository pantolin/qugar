// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_LIBRARY_QUADRATURE_HPP

#define QUGAR_LIBRARY_QUADRATURE_HPP

//! @file quadrature.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Definition of quadrature related functionalities.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <qugar/bbox.hpp>
#include <qugar/point.hpp>
#include <qugar/types.hpp>

#include <array>
#include <memory>
#include <type_traits>


namespace qugar {


//! @brief Helper class for computing the abscissae and weights
//! of the
//! <a href="https://en.wikipedia.org/wiki/Gauss–Legendre_quadrature">Gauss-Legendre quadrature rule</a>
//! referred to the [0,1].
//! The maximum number of abscissae/weights is prescribed by @ref n_max.
//! The abscissae / weights are precomputed with enough precision
//! (34 significant digits) for double or quadruple precision applications.
//!
class Gauss
{
public:
  //! @brief Maximum precomputed degree of Gauss-Legendre points.
  static constexpr int n_max = 100;

  //! @brief Gets a Gauss-Legendre abscissa in the [0,1] domain.
  //! @param n_points Number of points of the 1D quadrature.
  //! It must be smaller or equal than @ref n_max.
  //! @param pt_id Id of the point whose abscissa is returned.
  //! The point must be in the range [0, @p n_points [.
  //! @return Abscissa of the point.
  static real get_abscissa(const int n_points, const int pt_id);

  //! @brief Gets a Gauss-Legendre weight in the [0,1] domain.
  //! @param n_points Number of points of the 1D quadrature.
  //! It must be smaller or equal than @ref n_max.
  //! @param pt_id Id of the point whose weight is returned.
  //! The point must be in the range [0, @p n_points [.
  //! @return Weight of the point.
  static real get_weight(const int n_points, const int pt_id);


private:
  //! @brief Number of entries in data vectors.
  static constexpr int n_entries = n_max * (n_max + 1);

  //! @brief Gets a reference to a static array containing all
  //! the information of the quadrature.
  //!
  //! The array contains the abscissae and weights from 1 up to
  //! @ref n_max points.
  //! The data is sorted in the following way: for a given number
  //! of points <tt>n</tt>, from 1 to @ref n_max, first the abcissae
  //! in ascending order (in [0,1]) are stored and then associated weights.
  //! Then, the abscissae for <tt>n+1</tt> points and weights, etc.
  //!
  //! @return Constant reference to static array containing all the data.
  static const std::array<real, n_entries> &get_quad_data();
};

//! @brief Helper class for computing the abscissae and weights
//! of the
//! <a href="https://en.wikipedia.org/wiki/Tanh-sinh_quadrature">tanh-sinh quadrature rule</a>
//! referred to the [0,1].
//! The maximum number of abscissae/weights is prescribed by @ref n_max.
//! The abscissae / weights are precomputed with enough precision
//! (34 significant digits) for double or quadruple precision applications.
//!
class TanhSinh
{
public:
  //! @brief Maximum precomputed degree of tanh-sinh points.
  static constexpr int n_max = 100;

  //! @brief Gets a tanh-sinh abscissa in the [0,1] domain.
  //! @param n_points Number of points of the 1D quadrature.
  //! It must be smaller or equal than @ref n_max.
  //! @param pt_id Id of the point whose abscissa is returned.
  //! The point must be in the range [0, @p n_points [.
  //! @return Abscissa of the point.
  static real get_abscissa(const int n_points, const int pt_id);

  //! @brief Gets a tanh-sinh weight in the [0,1] domain.
  //! @param n_points Number of points of the 1D quadrature.
  //! It must be smaller or equal than @ref n_max.
  //! @param pt_id Id of the point whose weight is returned.
  //! The point must be in the range [0, @p n_points [.
  //! @return Weight of the point.
  static real get_weight(const int n_points, const int pt_id);


private:
  //! @brief Number of entries in data vectors.
  static constexpr int n_entries = n_max * (n_max + 1);

  //! @brief Gets a reference to a static array containing all
  //! the information of the quadrature.
  //!
  //! The array contains the abscissae and weights from 1 up to
  //! @ref n_max points.
  //! The data is sorted in the following way: for a given number
  //! of points <tt>n</tt>, from 1 to @ref n_max, first the abcissae
  //! in ascending order (in [0,1]) are stored and then associated weights.
  //! Then, the abscissae for <tt>n+1</tt> points and weights, etc.
  //!
  //! @return Constant reference to static array containing all the data.
  static const std::array<real, n_entries> &get_quad_data();
};

//! @brief Class for storing dim-dimensional quadratures (non-tensor product).
template<int dim> class Quadrature
{
public:
  /// @brief Point type.
  using Pt = Point<dim>;

  /// @brief Domain type.
  using Domain = std::conditional_t<dim == 1, Interval, BoundBox<dim>>;


  //! @brief Default constructor.
  //! @warning Not allowed to be used.
  Quadrature() = delete;

  //! @brief Constructor.
  //! @param points Vector of points referred to [0, 1].
  //! @param weights Vector of weights referred to [0, 1].
  //! @note @p points and @p weight must have the same length.
  Quadrature(const std::vector<Pt> &points, const std::vector<real> &weights);

  //! @brief Creates a new quadrature class instance with a Gauss-Legendre quadrature in [0, 1].
  //! @param n_points Number of points of the quadrature rule along all the parametric directions.
  //! @return Gauss-Legendre quadrature rule wrapped in a shared pointer.
  [[nodiscard]] static std::shared_ptr<Quadrature> create_Gauss_01(int n_points);

  //! @brief Creates a new quadrature class instance with a Gauss-Legendre quadrature in [0, 1].
  //! @param n_points Number of points of the quadrature rule in each parametric direction.
  //! @return Gauss-Legendre quadrature rule wrapped in a shared pointer.
  [[nodiscard]] static std::shared_ptr<Quadrature> create_Gauss_01(const std::array<int, dim> n_points);

  //! @brief Creates a new quadrature class instance with a Gauss-Legendre quadrature in the domain @p box.
  //! @param n_points Number of points of the quadrature rule in each parametric direction.
  //! @param domain Domain in which the quadrature is created.
  //! @return Gauss-Legendre quadrature rule wrapped in a shared pointer.
  [[nodiscard]] static std::shared_ptr<Quadrature> create_Gauss(const std::array<int, dim> n_points,
    const Domain &domain);

  //! @brief Creates a new quadrature class instance with a tanh-sinh quadrature in [0, 1].
  //! @param n_points Number of points of the quadrature rule along all the parametric directions.
  //! @return Tanh-sinh quadrature rule wrapped in a shared pointer.
  [[nodiscard]] static std::shared_ptr<Quadrature> create_tanh_sinh_01(int n_points);

  //! @brief Creates a new quadrature class instance with a tanh-sinh quadrature in [0, 1].
  //! @param n_points Number of points of the quadrature rule in each parametric direction.
  //! @return Tanh-sinh quadrature rule wrapped in a shared pointer.
  [[nodiscard]] static std::shared_ptr<Quadrature> create_tanh_sinh_01(const std::array<int, dim> n_points);

  //! @brief Creates a new quadrature class instance with a tanh-sinh quadrature in the domain @p box.
  //! @param n_points Number of points of the quadrature rule in each parametric direction.
  //! @param domain Domain in which the quadrature is created.
  //! @return Tanh-sinh quadrature rule wrapped in a shared pointer.
  [[nodiscard]] static std::shared_ptr<Quadrature> create_tanh_sinh(const std::array<int, dim> n_points,
    const Domain &domain);

  //! @brief Scales the weights multiplying them by the
  //! given @p ratio.
  //! @param ratio Ratio respect to which the weights are
  //! scaled.
  void scale_weights(const real ratio);

  //! @brief Returns the vector of points.
  //! @return Constant reference to the vector of points.
  [[nodiscard]] const std::vector<Pt> &points() const;

  //! @brief Returns the vector of weights.
  //! @return Constant reference to the vector of weights.
  [[nodiscard]] const std::vector<real> &weights() const;

  //! @brief Returns the vector of points.
  //! @return Non-constant reference to the vector of points.
  [[nodiscard]] std::vector<Pt> &points();

  //! @brief Returns the vector of weights.
  //! @return Non-constant reference to the vector of weights.
  [[nodiscard]] std::vector<real> &weights();

  //! @brief Returns the number of points (and weights).
  //! @return Number of points.
  [[nodiscard]] std::size_t get_num_points() const;

private:
  //! @brief Vector of points.
  std::vector<Pt> points_;
  //! @brief Vector of weights.
  std::vector<real> weights_;

  //! @brief Creates a tensor-product quadrature.
  //!
  //! @tparam QuadType Type of 1D quadrature to use.
  //! @param n_points Number of points of the quadrature rule in each parametric direction.
  //! @param domain Domain in which the quadrature is created.
  //! @return Computed quadrature wrapped in a shared pointer.
  template<typename QuadType>
  [[nodiscard]] static std::shared_ptr<Quadrature<dim>> create_tp_quadrature(const std::array<int, dim> n_points,
    const Domain &domain);
};


}// namespace qugar

#endif// QUGAR_LIBRARY_QUADRATURE_HPP