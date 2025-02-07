// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file tpms_lib.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of TPMS functions to be consumed by Algoim.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <qugar/tpms_lib.hpp>

#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <algoim/interval.hpp>

namespace qugar::impl::tpms {

namespace alg = ::algoim;
template<int dim> using Interval = alg::Interval<dim>;

namespace {

  template<typename T, int dim> Point<3, T> extend_to_3D(const Point<dim, T> &vec)
  {
    if constexpr (dim == 2) {
      return Point<3, T>(vec(0), vec(1), T{ numbers::zero });
    } else {
      return vec;
    }
  }

  template<int dim> Point<3, real> extend_to_3D(const Point<dim> &point)
  {
    if constexpr (dim == 2) {
      return Point<3, real>(point(0), point(1), numbers::zero);
    } else {
      return Point<3, real>(point(0), point(1), point(2));
    }
  }

}// namespace

// NOLINTBEGIN (cppcoreguidelines-macro-usage, bugprone-macro-parentheses)
#define implement_tpms(TPMS_NAME)                                                                                      \
  template<int dim> TPMS_NAME<dim>::TPMS_NAME(const Vector<real, dim> &mnq) : TPMSBase<dim>(mnq) {}                    \
                                                                                                                       \
  template<int dim> TPMS_NAME<dim>::TPMS_NAME() : TPMS_NAME(numbers::one) {}                                           \
                                                                                                                       \
  template<int dim> real TPMS_NAME<dim>::operator()(const Point<dim> &point) const                                     \
  {                                                                                                                    \
    return this->eval_(point);                                                                                         \
  }                                                                                                                    \
                                                                                                                       \
  template<int dim> Interval<dim> TPMS_NAME<dim>::operator()(const Point<dim, Interval<dim>> &point) const             \
  {                                                                                                                    \
    return this->eval_(point);                                                                                         \
  }                                                                                                                    \
                                                                                                                       \
  template<int dim> auto TPMS_NAME<dim>::grad(const Point<dim> &point) const -> Gradient<real>                         \
  {                                                                                                                    \
    return this->grad_(point);                                                                                         \
  }                                                                                                                    \
                                                                                                                       \
  template<int dim> auto TPMS_NAME<dim>::grad(const Point<dim, Interval<dim>> &point) const -> Gradient<Interval<dim>> \
  {                                                                                                                    \
    return this->grad_(point);                                                                                         \
  }                                                                                                                    \
                                                                                                                       \
  template<int dim> auto TPMS_NAME<dim>::hessian(const Point<dim> &point) const -> Hessian<real>                       \
  {                                                                                                                    \
    return this->hessian_(point);                                                                                      \
  }
// NOLINTEND (cppcoreguidelines-macro-usage, bugprone-macro-parentheses)


// NOLINTBEGIN (readability-math-missing-parentheses)
template<int dim> TPMSBase<dim>::TPMSBase(const Vector<real, dim> &mnq) : mnq_(extend_to_3D(mnq)) {}

implement_tpms(Schoen);


template<int dim> template<typename T> T Schoen<dim>::eval_(const Point<dim, T> &point) const
{
  const Vector<T, 3> pi_2mnq_point = (numbers::two * numbers::pi) * (this->mnq_ * extend_to_3D(point));

  return sin(pi_2mnq_point(0)) * cos(pi_2mnq_point(1)) + sin(pi_2mnq_point(1)) * cos(pi_2mnq_point(2))
         + sin(pi_2mnq_point(2)) * cos(pi_2mnq_point(0));
}

template<int dim> template<typename T> auto Schoen<dim>::grad_(const Point<dim, T> &point) const -> Gradient<T>
{
  const Vector<real, 3> pi_2mnq = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2mnq_point = pi_2mnq * extend_to_3D(point);

  Gradient<T> grad_val;
  grad_val(0) =
    (cos(pi_2mnq_point(0)) * cos(pi_2mnq_point(1)) - sin(pi_2mnq_point(0)) * sin(pi_2mnq_point(2))) * pi_2mnq(0);
  grad_val(1) =
    (cos(pi_2mnq_point(1)) * cos(pi_2mnq_point(2)) - sin(pi_2mnq_point(1)) * sin(pi_2mnq_point(0))) * pi_2mnq(1);

  if constexpr (dim == 3) {
    grad_val(2) =
      (cos(pi_2mnq_point(2)) * cos(pi_2mnq_point(0)) - sin(pi_2mnq_point(2)) * sin(pi_2mnq_point(1))) * pi_2mnq(2);
  }

  return grad_val;
}

template<int dim> template<typename T> auto Schoen<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  const Vector<real, 3> pi_2mnq = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2mnq_point = pi_2mnq * extend_to_3D(point);

  Hessian<T> hess;

  hess(0) = pi_2mnq(0) * pi_2mnq(0)
            * (-sin(pi_2mnq_point(0)) * cos(pi_2mnq_point(1)) - sin(pi_2mnq_point(2)) * cos(pi_2mnq_point(0)));
  hess(1) = pi_2mnq(0) * pi_2mnq(1) * (-cos(pi_2mnq_point(0)) * sin(pi_2mnq_point(1)));

  constexpr int yy = dim == 2 ? 2 : 3;
  hess(yy) = pi_2mnq(1) * pi_2mnq(1)
             * (-sin(pi_2mnq_point(1)) * cos(pi_2mnq_point(2)) - sin(pi_2mnq_point(0)) * cos(pi_2mnq_point(1)));

  if constexpr (dim == 3) {
    hess(2) = pi_2mnq(0) * pi_2mnq(2) * (-cos(pi_2mnq_point(2)) * sin(pi_2mnq_point(2)));
    hess(4) = pi_2mnq(1) * pi_2mnq(2) * (-cos(pi_2mnq_point(1)) * sin(pi_2mnq_point(2)));
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
    hess(5) = pi_2mnq(2) * pi_2mnq(2)
              * (-sin(pi_2mnq_point(2)) * cos(pi_2mnq_point(0)) - sin(pi_2mnq_point(1)) * cos(pi_2mnq_point(2)));
  }

  return hess;
}


implement_tpms(SchoenIWP);

template<int dim> template<typename T> T SchoenIWP<dim>::eval_(const Point<dim, T> &point) const
{
  const Vector<T, 3> pi_2_m_point = (numbers::two * numbers::pi) * (this->mnq_ * extend_to_3D(point));
  const Vector<T, 3> pi_4_m_point = numbers::two * pi_2_m_point;

  return numbers::two
           * (cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) + cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
              + cos(pi_2_m_point(2)) * cos(pi_2_m_point(0)))
         - cos(pi_4_m_point(0)) - cos(pi_4_m_point(1)) - cos(pi_4_m_point(2));
}

template<int dim> template<typename T> auto SchoenIWP<dim>::grad_(const Point<dim, T> &point) const -> Gradient<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<real, 3> pi_4_m = numbers::two * pi_2_m;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);
  const Vector<T, 3> pi_4_m_point = numbers::two * pi_2_m_point;

  Gradient<T> grad_val;

  grad_val(0) =
    (sin(pi_4_m_point(0)) - sin(pi_2_m_point(0)) * (cos(pi_2_m_point(1)) + cos(pi_2_m_point(2)))) * pi_4_m(0);
  grad_val(1) =
    (sin(pi_4_m_point(1)) - sin(pi_2_m_point(1)) * (cos(pi_2_m_point(0)) + cos(pi_2_m_point(2)))) * pi_4_m(0);
  if constexpr (dim == 3) {
    grad_val(2) =
      (sin(pi_4_m_point(2)) - sin(pi_2_m_point(2)) * (cos(pi_2_m_point(0)) + cos(pi_2_m_point(1)))) * pi_4_m(2);
  }

  return grad_val;
}

template<int dim> template<typename T> auto SchoenIWP<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<real, 3> pi_4_m = (numbers::four * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);
  const Vector<T, 3> pi_4_m_point = numbers::two * pi_2_m_point;

  Hessian<T> hess;
  hess(0) = -pi_2_m(0) * pi_4_m(0)
            * (cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) + cos(pi_2_m_point(2)) * cos(pi_2_m_point(0))
               - numbers::two * cos(pi_4_m_point(0)));
  hess(1) = pi_2_m(0) * pi_4_m(1) * sin(pi_2_m_point(0)) * sin(pi_2_m_point(1));

  constexpr int yy = dim == 2 ? 2 : 3;
  hess(yy) = -pi_2_m(1) * pi_4_m(1)
             * (cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) + cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                - numbers::two * cos(pi_4_m_point(1)));

  if constexpr (dim == 3) {
    hess(2) = pi_2_m(0) * pi_4_m(2) * sin(pi_2_m_point(2)) * sin(pi_2_m_point(0));
    hess(4) = pi_2_m(1) * pi_4_m(2) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2));
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
    hess(5) = -pi_2_m(2) * pi_4_m(2)
              * (cos(pi_2_m_point(1)) * cos(pi_2_m_point(2)) + cos(pi_2_m_point(2)) * cos(pi_2_m_point(0))
                 - numbers::two * cos(pi_4_m_point(2)));
  }


  return hess;
}

implement_tpms(SchoenFRD);

template<int dim> template<typename T> T SchoenFRD<dim>::eval_(const Point<dim, T> &point) const
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);
  const Vector<T, 3> pi_4_m_point = numbers::two * pi_2_m_point;

  return 4 * cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
         - cos(pi_4_m_point(0)) * cos(pi_4_m_point(1)) - cos(pi_4_m_point(1)) * cos(pi_4_m_point(2))
         - cos(pi_4_m_point(2)) * cos(pi_4_m_point(0));
}

template<int dim> template<typename T> auto SchoenFRD<dim>::grad_(const Point<dim, T> &point) const -> Gradient<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<real, 3> pi_4_m = numbers::two * pi_2_m;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);
  const Vector<T, 3> pi_4_m_point = numbers::two * pi_2_m_point;

  Gradient<T> grad_val;

  grad_val(0) = (-numbers::two * sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                  + sin(pi_4_m_point(0)) * cos(pi_4_m_point(1)) + cos(pi_4_m_point(2)) * sin(pi_4_m_point(0)))
                * pi_4_m(0);

  grad_val(1) = (-numbers::two * cos(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                  + cos(pi_4_m_point(0)) * sin(pi_4_m_point(1)) + sin(pi_4_m_point(1)) * cos(pi_4_m_point(2)))
                * pi_4_m(1);

  if constexpr (dim == 3) {
    grad_val(2) = (-numbers::two * cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * sin(pi_2_m_point(2))
                    + cos(pi_4_m_point(1)) * sin(pi_4_m_point(2)) + sin(pi_4_m_point(2)) * cos(pi_4_m_point(0)))
                  * pi_4_m(2);
  }

  return grad_val;
}

template<int dim> template<typename T> auto SchoenFRD<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<real, 3> pi_4_m = (numbers::four * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);
  const Vector<T, 3> pi_4_m_point = numbers::two * pi_2_m_point;

  Hessian<T> hess;

  hess(0) = pi_2_m(0) * pi_4_m(0)
            * (-numbers::two * cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
               + numbers::two * cos(pi_4_m_point(0)) * cos(pi_4_m_point(1))
               + numbers::two * cos(pi_4_m_point(2)) * cos(pi_4_m_point(0)));
  hess(1) = pi_2_m(0) * pi_4_m(1)
            * (numbers::two * sin(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2))
               - numbers::two * sin(pi_4_m_point(0)) * sin(pi_4_m_point(1)));

  constexpr int yy = dim == 2 ? 2 : 3;
  hess(yy) = pi_2_m(1) * pi_4_m(1)
             * (-numbers::two * cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                + numbers::two * cos(pi_4_m_point(0)) * cos(pi_4_m_point(1))
                + numbers::two * cos(pi_4_m_point(1)) * cos(pi_4_m_point(2)));

  if constexpr (dim == 3) {
    hess(2) = pi_2_m(0) * pi_4_m(2)
              * (numbers::two * sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * sin(pi_2_m_point(2))
                 - numbers::two * sin(pi_4_m_point(2)) * sin(pi_4_m_point(0)));
    hess(4) = pi_2_m(1) * pi_4_m(2)
              * (numbers::two * cos(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2))
                 - numbers::two * sin(pi_4_m_point(1)) * sin(pi_4_m_point(2)));
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
    hess(5) = pi_2_m(2) * pi_4_m(2)
              * (-numbers::two * cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                 + numbers::two * cos(pi_4_m_point(1)) * cos(pi_4_m_point(2))
                 + numbers::two * cos(pi_4_m_point(2)) * cos(pi_4_m_point(0)));
  }

  return hess;
}

implement_tpms(FischerKochS);

template<int dim> template<typename T> T FischerKochS<dim>::eval_(const Point<dim, T> &point) const
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);
  const Vector<T, 3> pi_4_m_point = numbers::two * pi_2_m_point;

  return cos(pi_4_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2))
         + cos(pi_2_m_point(0)) * cos(pi_4_m_point(1)) * sin(pi_2_m_point(2))
         + sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_4_m_point(2));
}

template<int dim> template<typename T> auto FischerKochS<dim>::grad_(const Point<dim, T> &point) const -> Gradient<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);
  const Vector<T, 3> pi_4_m_point = numbers::two * pi_2_m_point;

  Gradient<T> grad_val;

  grad_val(0) = (-numbers::two * sin(pi_4_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                  - sin(pi_2_m_point(0)) * cos(pi_4_m_point(1)) * sin(pi_2_m_point(2))
                  + cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_4_m_point(2)))
                * pi_2_m(0);

  grad_val(1) = (cos(pi_4_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                  - numbers::two * cos(pi_2_m_point(0)) * sin(pi_4_m_point(1)) * sin(pi_2_m_point(2))
                  - sin(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_4_m_point(2)))
                * pi_2_m(1);

  if constexpr (dim == 3) {
    grad_val(2) = (-cos(pi_4_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2))
                    + cos(pi_2_m_point(0)) * cos(pi_4_m_point(1)) * cos(pi_2_m_point(2))
                    - numbers::two * sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * sin(pi_4_m_point(2)))
                  * pi_2_m(2);
  }

  return grad_val;
}

template<int dim> template<typename T> auto FischerKochS<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<real, 3> pi_4_m = (numbers::four * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);
  const Vector<T, 3> pi_4_m_point = numbers::two * pi_2_m_point;

  Hessian<T> hess;
  hess(0) = -pi_4_m(0) * pi_4_m(0) * cos(pi_4_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2))
            - pi_2_m(0) * pi_2_m(0) * cos(pi_2_m_point(0)) * cos(pi_4_m_point(1)) * sin(pi_2_m_point(2))
            - pi_2_m(0) * pi_2_m(0) * sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_4_m_point(2));
  hess(1) = -pi_4_m(0) * pi_2_m(1) * sin(pi_4_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
            + pi_2_m(0) * pi_4_m(1) * sin(pi_2_m_point(0)) * sin(pi_4_m_point(1)) * sin(pi_2_m_point(2))
            - pi_2_m(0) * pi_2_m(1) * cos(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_4_m_point(2));

  constexpr int yy = dim == 2 ? 2 : 3;
  hess(yy) = -pi_2_m(1) * pi_2_m(1) * cos(pi_4_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2))
             - pi_4_m(1) * pi_4_m(1) * cos(pi_2_m_point(0)) * cos(pi_4_m_point(1)) * sin(pi_2_m_point(2))
             - pi_2_m(1) * pi_2_m(1) * sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_4_m_point(2));

  if constexpr (dim == 3) {
    hess(2) = pi_4_m(0) * pi_2_m(2) * sin(pi_4_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2))
              - pi_2_m(0) * pi_2_m(2) * sin(pi_2_m_point(0)) * cos(pi_4_m_point(1)) * cos(pi_2_m_point(2))
              - pi_2_m(0) * pi_4_m(2) * cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * sin(pi_4_m_point(2));
    hess(4) = -pi_2_m(1) * pi_2_m(2) * cos(pi_4_m_point(0)) * cos(pi_2_m_point(1)) * sin(pi_2_m_point(2))
              - pi_4_m(1) * pi_2_m(2) * cos(pi_2_m_point(0)) * sin(pi_4_m_point(1)) * cos(pi_2_m_point(2))
              + pi_2_m(1) * pi_4_m(2) * sin(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_4_m_point(2));
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
    hess(5) = -pi_2_m(2) * pi_2_m(2) * cos(pi_4_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2))
              - pi_2_m(2) * pi_2_m(2) * cos(pi_2_m_point(0)) * cos(pi_4_m_point(1)) * sin(pi_2_m_point(2))
              - pi_4_m(2) * pi_4_m(2) * sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_4_m_point(2));
  }

  return hess;
}

implement_tpms(SchwarzDiamond);

template<int dim> template<typename T> T SchwarzDiamond<dim>::eval_(const Point<dim, T> &point) const
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);

  return cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
         - sin(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2));
}

template<int dim> template<typename T> auto SchwarzDiamond<dim>::grad_(const Point<dim, T> &point) const -> Gradient<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);

  Gradient<T> grad_val;

  grad_val(0) = -(sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                  + cos(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2)))
                * pi_2_m(0);

  grad_val(1) = -(cos(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                  + sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * sin(pi_2_m_point(2)))
                * pi_2_m(1);

  if constexpr (dim == 3) {
    grad_val(2) = -(cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * sin(pi_2_m_point(2))
                    + sin(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2)))
                  * pi_2_m(2);
  }

  return grad_val;
}

template<int dim>
template<typename T>
auto SchwarzDiamond<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);

  Hessian<T> hess;
  hess(0) = -pi_2_m(0) * pi_2_m(0)
            * (cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
               - sin(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2)));
  hess(1) = -pi_2_m(0) * pi_2_m(1)
            * (-sin(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2))
               + cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * sin(pi_2_m_point(2)));

  constexpr int yy = dim == 2 ? 2 : 3;
  hess(yy) = -pi_2_m(1) * pi_2_m(1)
             * (cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                - sin(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2)));

  if constexpr (dim == 3) {

    hess(2) = -pi_2_m(0) * pi_2_m(2)
              * (-sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * sin(pi_2_m_point(2))
                 + cos(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * cos(pi_2_m_point(2)));
    hess(4) = -pi_2_m(1) * pi_2_m(2)
              * (-cos(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2))
                 + sin(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2)));
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
    hess(5) = -pi_2_m(2) * pi_2_m(2)
              * (cos(pi_2_m_point(0)) * cos(pi_2_m_point(1)) * cos(pi_2_m_point(2))
                 - sin(pi_2_m_point(0)) * sin(pi_2_m_point(1)) * sin(pi_2_m_point(2)));
  }

  return hess;
}

implement_tpms(SchwarzPrimitive);

template<int dim> template<typename T> T SchwarzPrimitive<dim>::eval_(const Point<dim, T> &point) const
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);

  return cos(pi_2_m_point(0)) + cos(pi_2_m_point(1)) + cos(pi_2_m_point(2));
}

template<int dim>
template<typename T>
auto SchwarzPrimitive<dim>::grad_(const Point<dim, T> &point) const -> Gradient<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);

  Gradient<T> grad_val;

  grad_val(0) = -sin(pi_2_m_point(0)) * pi_2_m(0);
  grad_val(1) = -sin(pi_2_m_point(1)) * pi_2_m(1);
  if constexpr (dim == 3) {
    grad_val(2) = -sin(pi_2_m_point(2)) * pi_2_m(2);
  }

  return grad_val;
}

template<int dim>
template<typename T>
auto SchwarzPrimitive<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  const Vector<real, 3> pi_2_m = (numbers::two * numbers::pi) * this->mnq_;
  const Vector<real, 3> pi2_4_m2 = pi_2_m * pi_2_m;
  const Vector<T, 3> pi_2_m_point = pi_2_m * extend_to_3D(point);

  Hessian<T> hess;

  hess(0) = -pi2_4_m2(0) * cos(pi_2_m_point(0));
  hess(1) = 0.0;

  constexpr int yy = dim == 2 ? 2 : 3;
  hess(yy) = -pi2_4_m2(1) * cos(pi_2_m_point(1));

  if constexpr (dim == 3) {
    hess(2) = 0.0;
    hess(4) = 0.0;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
    hess(5) = -pi2_4_m2(2) * cos(pi_2_m_point(2));
  }

  return hess;
}
// NOLINTEND (readability-math-missing-parentheses)


// Instantiations

template class Schoen<2>;
template class Schoen<3>;

template class SchoenIWP<2>;
template class SchoenIWP<3>;

template class SchoenFRD<2>;
template class SchoenFRD<3>;

template class FischerKochS<2>;
template class FischerKochS<3>;

template class SchwarzDiamond<2>;
template class SchwarzDiamond<3>;

template class SchwarzPrimitive<2>;
template class SchwarzPrimitive<3>;


}// namespace qugar::impl::tpms