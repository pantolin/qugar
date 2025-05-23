// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_TPMS_LIB_HPP
#define QUGAR_IMPL_TPMS_LIB_HPP

//! @file tpms_lib.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of TPMS functions to be consumed by Algoim.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/domain_function.hpp>
#include <qugar/vector.hpp>

#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/types.hpp>

//! Namespace for Triple-Periodic Minimal Surfaces. Namely, Schoen gyroid, Schoen IWP, Scheon FRD, Fischer-Koch S,
//! Schwarz diamond, and Schwarz primitive. These function are ready to be consumed by Algoim.
namespace qugar::impl::tpms {

template<int dim> class TPMSBase : public qugar::impl::ImplicitFunc<dim>
{
protected:
  //! Constructor.
  //! @param mnq Function periods.
  //! @param z Constant z coordinate for 2D functions.
  template<int dim_aux = dim>
    requires(dim == dim_aux && dim_aux == 2)
  TPMSBase(const Vector<real, 2> &mnq, real z) : z_(z)
  {
    mnq_(0) = mnq(0);
    mnq_(1) = mnq(1);
  }

  //! Constructor.
  //! @param mnq Function periods.
  template<int dim_aux = dim>
    requires(dim == dim_aux && dim_aux == 3)
  explicit TPMSBase(const Vector<real, 3> &mnq) : mnq_(mnq)
  {}

  TPMSBase() = default;

  //! Function periods.
  Vector<real, 3> mnq_{ numbers::one };

  //! Constant z coordinate for 2D functions.
  real z_{ numbers::zero };
};

// NOLINTNEXTLINE (cppcoreguidelines-macro-usage)
#define declare_tpms(TPMS_NAME)                                                                 \
  template<int dim> class TPMS_NAME : public TPMSBase<dim>                                      \
  {                                                                                             \
    using Parent = TPMSBase<dim>;                                                               \
    template<typename T> using Gradient = qugar::impl::ImplicitFunc<dim>::template Gradient<T>; \
    template<typename T> using Hessian = qugar::impl::ImplicitFunc<dim>::template Hessian<T>;   \
                                                                                                \
  public:                                                                                       \
    TPMS_NAME() = default;                                                                      \
                                                                                                \
    template<int dim_aux = dim>                                                                 \
      requires(dim == dim_aux && dim_aux == 2)                                                  \
    explicit TPMS_NAME(const Vector<real, dim> &mnq, real z) : Parent(mnq, z)                   \
    {}                                                                                          \
                                                                                                \
    template<int dim_aux = dim>                                                                 \
      requires(dim == dim_aux && dim_aux == 3)                                                  \
    explicit TPMS_NAME(const Vector<real, dim> &mnq) : Parent(mnq)                              \
    {}                                                                                          \
                                                                                                \
    [[nodiscard]] virtual real operator()(const Point<dim> &point) const final;                 \
                                                                                                \
    [[nodiscard]] virtual ::algoim::Interval<dim> operator()(                                   \
      const Point<dim, ::algoim::Interval<dim>> &point) const final;                            \
                                                                                                \
    [[nodiscard]] virtual Gradient<real> grad(const Point<dim> &point) const final;             \
                                                                                                \
    [[nodiscard]] virtual Gradient<::algoim::Interval<dim>> grad(                               \
      const Point<dim, ::algoim::Interval<dim>> &point) const final;                            \
                                                                                                \
    [[nodiscard]] virtual Hessian<real> hessian(const Point<dim> &point) const final;           \
                                                                                                \
  private:                                                                                      \
    template<typename T> [[nodiscard]] T eval_(const Point<3, T> &point) const;                 \
                                                                                                \
    template<typename T> [[nodiscard]] Gradient<T> grad_(const Point<3, T> &point) const;       \
                                                                                                \
    template<typename T> [[nodiscard]] Hessian<T> hessian_(const Point<3, T> &point) const;     \
  };

//! @brief Schoen's gyroid function.
//! Defined as
//!   f(x,y,z,m,n,q) = sin(2 pi m x) * cos(2 pi n y) + sin(2 pi n y) * cos(2 pi q z) + sin(2 pi q z) * cos(2 pi m x)
//! this is a triply periodic function with period (m, n, q).
//!
//! See https:// en.wikipedia.org/wiki/Gyroid
declare_tpms(Schoen);

//! @brief Schoen IWP's gyroid function.
//! Defined as
//!   f(x,y,z,m,n,q) = 2 * (cos(2 pi m x) * cos(2 pi n y) + cos(2 pi n y) * cos(2 pi q z)
//!     + cos(2 pi q z) * cos(2 pi m x)) - cos(4 pi m x) - cos(4 pi m n y) - cos(4 pi q z)
//! this is a triply periodic function with period (m, n, q).
//!
//! See https://en.wikipedia.org/wiki/Gyroid
declare_tpms(SchoenIWP);

//! @brief Schoen FRD's gyroid function.
//! Defined as
//!   f(x,y,z,m,n,q) = 4 * cos(2 pi m x) * cos(2 pi n y) * cos(2 pi q z) - cos(4 pi m x) * cos(4 pi n y)
//!     - cos(4 pi n y) * cos(4 pi q z) - cos(4 pi m q z* cos(4 pi m x)
//! this is a triply periodic function with period (m, n, q).
//!
//! See https://en.wikipedia.org/wiki/Gyroid
declare_tpms(SchoenFRD);

//! @brief Fischer-Koch S' gyroid function.
//! Defined as
//!   f(x,y,z,m,n,q) = cos(4 pi m x) * sin(2 pi n y) * cos(2 pi q z)
//!     + cos(2 pi m x) * cos(4 pi n y) * sin(2 pi q z)
//!     + sin(2 pi m x) * cos(2 pi n y) * cos(4 pi q z)
//! this is a triply periodic function with period (m, n, q).
//!
//! See https://en.wikipedia.org/wiki/Gyroid
declare_tpms(FischerKochS);

//! @brief Schwarz diamond's gyroid function.
//! Defined as
//!   f(x,y,z,m,n,q) = cos(2 pi m x) * cos(2 pi n y) * cos(2 pi q z)
//!                  - sin(2 pi m x) * sin(2 pi n y) * sin(2 pi q z)
//! this is a triply periodic function with period (m, n, q).
//!
//! See https://en.wikipedia.org/wiki/Gyroid
declare_tpms(SchwarzDiamond);

//! @brief Schwarz primitive's gyroid function.
//! Defined as
//!   f(x,y,z,m,n,q) = cos(2 pi mpoint(0)) + cos(2 pi n y) + cos(2 pi q z)
//! this is a triply periodic function with period (m, n, q).
//!
//! See https://en.wikipedia.org/wiki/Gyroid
declare_tpms(SchwarzPrimitive);

}// namespace qugar::impl::tpms


#endif// QUGAR_IMPL_TPMS_LIB_HPP