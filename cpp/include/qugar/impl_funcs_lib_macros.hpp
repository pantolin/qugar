// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_FUNCS_LIB_MACROS_HPP
#define QUGAR_IMPL_FUNCS_LIB_MACROS_HPP

//! @file impl_funcs_lib_macros.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of macros to ease the definition/implementation of implicit functions.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


// NOLINTBEGIN (cppcoreguidelines-macro-usage, bugprone-macro-parentheses)
#define declare_impl_func_virtual_interface                                                                         \
  template<typename T> using Hessian = typename ImplicitFunc<dim>::template Hessian<T>;                             \
                                                                                                                    \
  [[nodiscard]] virtual real operator()(const Point<dim> &point) const final;                                       \
                                                                                                                    \
  [[nodiscard]] virtual ::algoim::Interval<dim> operator()(const Point<dim, ::algoim::Interval<dim>> &point)        \
    const final;                                                                                                    \
                                                                                                                    \
  [[nodiscard]] virtual Vector<real, dim> grad(const Point<dim> &point) const final;                                \
                                                                                                                    \
  [[nodiscard]] virtual Vector<::algoim::Interval<dim>, dim> grad(const Point<dim, ::algoim::Interval<dim>> &point) \
    const final;                                                                                                    \
                                                                                                                    \
  [[nodiscard]] Hessian<real> virtual hessian(const Point<dim> &point) const final;                                 \
                                                                                                                    \
private:                                                                                                            \
  template<typename T> [[nodiscard]] T eval_(const Point<dim, T> &point) const;                                     \
                                                                                                                    \
  template<typename T> [[nodiscard]] Vector<T, dim> grad_(const Point<dim, T> &point) const;                        \
                                                                                                                    \
  template<typename T> [[nodiscard]] Hessian<T> hessian_(const Point<dim, T> &point) const

#define declare_impl_func_virtual_interface_2D                                                                      \
  template<typename T> using Hessian = ImplicitFunc<2>::template Hessian<T>;                                        \
                                                                                                                    \
  [[nodiscard]] virtual real operator()(const Point<2> &point) const final;                                         \
                                                                                                                    \
  [[nodiscard]] virtual ::algoim::Interval<2> operator()(const Point<2, ::algoim::Interval<2>> &point) const final; \
                                                                                                                    \
  [[nodiscard]] virtual Vector<real, 2> grad(const Point<2> &point) const final;                                    \
                                                                                                                    \
  [[nodiscard]] virtual Vector<::algoim::Interval<2>, 2> grad(const Point<2, ::algoim::Interval<2>> &point)         \
    const final;                                                                                                    \
                                                                                                                    \
  [[nodiscard]] Hessian<real> virtual hessian(const Point<2> &point) const final;                                   \
                                                                                                                    \
private:                                                                                                            \
  template<typename T> [[nodiscard]] T eval_(const Point<2, T> &point) const;                                       \
                                                                                                                    \
  template<typename T> [[nodiscard]] Vector<T, 2> grad_(const Point<2, T> &point) const;                            \
                                                                                                                    \
  template<typename T> [[nodiscard]] Hessian<T> hessian_(const Point<2, T> &point) const


#define declare_impl_func_virtual_interface_3D                                                                      \
  template<typename T> using Hessian = ImplicitFunc<3>::template Hessian<T>;                                        \
                                                                                                                    \
  [[nodiscard]] virtual real operator()(const Point<3> &point) const final;                                         \
                                                                                                                    \
  [[nodiscard]] virtual ::algoim::Interval<3> operator()(const Point<3, ::algoim::Interval<3>> &point) const final; \
                                                                                                                    \
  [[nodiscard]] virtual Vector<real, 3> grad(const Point<3> &point) const final;                                    \
                                                                                                                    \
  [[nodiscard]] virtual Vector<::algoim::Interval<3>, 3> grad(const Point<3, ::algoim::Interval<3>> &point)         \
    const final;                                                                                                    \
                                                                                                                    \
  [[nodiscard]] Hessian<real> virtual hessian(const Point<3> &point) const final;                                   \
                                                                                                                    \
private:                                                                                                            \
  template<typename T> [[nodiscard]] T eval_(const Point<3, T> &point) const;                                       \
                                                                                                                    \
  template<typename T> [[nodiscard]] Vector<T, 3> grad_(const Point<3, T> &point) const;                            \
                                                                                                                    \
  template<typename T> [[nodiscard]] Hessian<T> hessian_(const Point<3, T> &point) const


#define implement_impl_func(FUNC_NAME)                                                                              \
                                                                                                                    \
  template<int dim> real FUNC_NAME<dim>::operator()(const Point<dim> &point) const                                  \
  {                                                                                                                 \
    return this->eval_(point);                                                                                      \
  }                                                                                                                 \
                                                                                                                    \
  template<int dim>                                                                                                 \
  ::algoim::Interval<dim> FUNC_NAME<dim>::operator()(const Point<dim, ::algoim::Interval<dim>> &point) const        \
  {                                                                                                                 \
    return this->eval_(point);                                                                                      \
  }                                                                                                                 \
                                                                                                                    \
  template<int dim> Vector<real, dim> FUNC_NAME<dim>::grad(const Point<dim> &point) const                           \
  {                                                                                                                 \
    return this->grad_(point);                                                                                      \
  }                                                                                                                 \
                                                                                                                    \
  template<int dim>                                                                                                 \
  Vector<::algoim::Interval<dim>, dim> FUNC_NAME<dim>::grad(const Point<dim, ::algoim::Interval<dim>> &point) const \
  {                                                                                                                 \
    return this->grad_(point);                                                                                      \
  }                                                                                                                 \
                                                                                                                    \
  template<int dim> auto FUNC_NAME<dim>::hessian(const Point<dim> &point) const -> Hessian<real>                    \
  {                                                                                                                 \
    return this->hessian_(point);                                                                                   \
  }

#define implement_impl_func_2D(FUNC_NAME)                                                              \
                                                                                                       \
  real FUNC_NAME::operator()(const Point<2> &point) const                                              \
  {                                                                                                    \
    return this->eval_(point);                                                                         \
  }                                                                                                    \
                                                                                                       \
  ::algoim::Interval<2> FUNC_NAME::operator()(const Point<2, ::algoim::Interval<2>> &point) const      \
  {                                                                                                    \
    return this->eval_(point);                                                                         \
  }                                                                                                    \
                                                                                                       \
  Vector<real, 2> FUNC_NAME::grad(const Point<2> &point) const                                         \
  {                                                                                                    \
    return this->grad_(point);                                                                         \
  }                                                                                                    \
                                                                                                       \
  Vector<::algoim::Interval<2>, 2> FUNC_NAME::grad(const Point<2, ::algoim::Interval<2>> &point) const \
  {                                                                                                    \
    return this->grad_(point);                                                                         \
  }                                                                                                    \
                                                                                                       \
  auto FUNC_NAME::hessian(const Point<2> &point) const -> Hessian<real>                                \
  {                                                                                                    \
    return this->hessian_(point);                                                                      \
  }

#define implement_impl_func_3D(FUNC_NAME)                                                              \
                                                                                                       \
  real FUNC_NAME::operator()(const Point<3> &point) const                                              \
  {                                                                                                    \
    return this->eval_(point);                                                                         \
  }                                                                                                    \
                                                                                                       \
  ::algoim::Interval<3> FUNC_NAME::operator()(const Point<3, ::algoim::Interval<3>> &point) const      \
  {                                                                                                    \
    return this->eval_(point);                                                                         \
  }                                                                                                    \
                                                                                                       \
  Vector<real, 3> FUNC_NAME::grad(const Point<3> &point) const                                         \
  {                                                                                                    \
    return this->grad_(point);                                                                         \
  }                                                                                                    \
                                                                                                       \
  Vector<::algoim::Interval<3>, 3> FUNC_NAME::grad(const Point<3, ::algoim::Interval<3>> &point) const \
  {                                                                                                    \
    return this->grad_(point);                                                                         \
  }                                                                                                    \
                                                                                                       \
  auto FUNC_NAME::hessian(const Point<3> &point) const -> Hessian<real>                                \
  {                                                                                                    \
    return this->hessian_(point);                                                                      \
  }

#define implement_impl_func_transf(FUNC_NAME)                                                                       \
                                                                                                                    \
  template<int dim> real FUNC_NAME<dim>::operator()(const Point<dim> &point) const                                  \
  {                                                                                                                 \
    return this->eval_(this->transf_.transform_point(point));                                                       \
  }                                                                                                                 \
                                                                                                                    \
  template<int dim>                                                                                                 \
  ::algoim::Interval<dim> FUNC_NAME<dim>::operator()(const Point<dim, ::algoim::Interval<dim>> &point) const        \
  {                                                                                                                 \
    return this->eval_(this->transf_.transform_point(point));                                                       \
  }                                                                                                                 \
                                                                                                                    \
  template<int dim> Vector<real, dim> FUNC_NAME<dim>::grad(const Point<dim> &point) const                           \
  {                                                                                                                 \
    return this->transf_.transform_vector(this->grad_(this->transf_.transform_point(point)));                       \
  }                                                                                                                 \
                                                                                                                    \
  template<int dim>                                                                                                 \
  Vector<::algoim::Interval<dim>, dim> FUNC_NAME<dim>::grad(const Point<dim, ::algoim::Interval<dim>> &point) const \
  {                                                                                                                 \
    return this->transf_.transform_vector(this->grad_(this->transf_.transform_point(point)));                       \
  }                                                                                                                 \
                                                                                                                    \
  template<int dim> auto FUNC_NAME<dim>::hessian(const Point<dim> &point) const -> Hessian<real>                    \
  {                                                                                                                 \
    return this->transf_.transform_tensor(this->hessian_(this->transf_.transform_point(point)));                    \
  }
// NOLINTEND (cppcoreguidelines-macro-usage, bugprone-macro-parentheses)


#endif// QUGAR_IMPL_FUNCS_LIB_MACROS_HPP