// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_DOMAIN_FUNCTION_HPP
#define QUGAR_IMPL_DOMAIN_FUNCTION_HPP

//! @file domain_function.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of a few implicit functions template class ready to be consumed by Algoim.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/types.hpp>

#include <qugar/point.hpp>
#include <qugar/vector.hpp>

#include <algoim/interval.hpp>

#include <cstdint>
#include <type_traits>

namespace qugar::impl {

enum FuncSign : std::int8_t { negative, positive, undetermined };

//! @brief Domain functions.
//!
//! @tparam dim Parametric dimension.
//! @tparam range Image dimension.
//! @tparam T Type.
template<int dim, int range> class DomainFunc
{
public:
  //! @brief Algoim's interval alias.
  template<int N> using Interval = ::algoim::Interval<N>;

  //! @brief Value type.
  //! @tparam T Type of the input coordinates.
  template<typename T> using Value = std::conditional_t<range == 1, T, Vector<T, range>>;

  //! @brief Gradient type.
  //! @tparam T Type of the input coordinates.
  template<typename T> using Gradient = Vector<Value<T>, dim>;

  //! @brief Number of Hessian (symmetric) components.
  static const int num_hessian = dim * (dim + 1) / 2;

  //! @brief Hessian (symmetric type).
  //! @tparam T Type of the input coordinates.
  template<typename T> using Hessian = Vector<Value<T>, num_hessian>;

  //! @brief Default constructor.
  DomainFunc() = default;
  //! @brief Default copy constructor.
  DomainFunc(const DomainFunc &) = default;
  //! @brief Default move constructor.
  DomainFunc(DomainFunc &&) = default;
  //! @brief Default copy assignment operator.
  DomainFunc &operator=(const DomainFunc &) = default;
  //! @brief Default move assignment operator.
  DomainFunc &operator=(DomainFunc &&) = default;
  //! @brief Default virtual destructor.
  virtual ~DomainFunc() = default;

  //! @brief Evaluator operator.
  //!
  //! @param point Point at which the function is evaluated.
  //! @return Function value at @p point.
  //! @note This is a purely virtual method and must be implemented in derived classes.
  [[nodiscard]] virtual Value<real> operator()(const Point<dim> &point) const = 0;

  //! @brief Evaluator operator.
  //!
  //! @param point Point at which the function is evaluated.
  //! @return Function value at @p point.
  //! @note This is a purely virtual method and must be implemented in derived classes.
  [[nodiscard]] virtual Value<Interval<dim>> operator()(const Point<dim, Interval<dim>> &point) const = 0;

  //! @brief Gradient evaluator operator.
  //!
  //! @param point Point at which the function's gradient is evaluated.
  //! @return Function gradient at @p point.
  //! @note This is a purely virtual method and must be implemented in derived classes.
  [[nodiscard]] virtual Gradient<real> grad(const Point<dim> &point) const = 0;

  //! @brief Gradient evaluator operator.
  //!
  //! @param point Point at which the function's gradient is evaluated.
  //! @return Function gradient at @p point.
  //! @note This is a purely virtual method and must be implemented in derived classes.
  [[nodiscard]] virtual Gradient<Interval<dim>> grad(const Point<dim, Interval<dim>> &point) const = 0;

  //! @brief Hessian evaluator operator.
  //!
  //! @param point Point at which the function's hessian is evaluated.
  //! @return Function hessian at @p point.
  //! @note This is a purely virtual method and must be implemented in derived classes.
  [[nodiscard]] virtual Hessian<real> hessian(const Point<dim> &point) const = 0;
};

//! @brief Alias for scalar functions.
template<int dim> using ScalarFunc = DomainFunc<dim, 1>;

//! @brief Alias for implicit functions.
template<int dim> using ImplicitFunc = ScalarFunc<dim>;

}// namespace qugar::impl


#endif// QUGAR_IMPL_DOMAIN_FUNCTION_HPP