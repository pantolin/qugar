// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file monomials_tp.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tensor-product monomials class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/monomials_tp.hpp>

#include <qugar/bezier_tp.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/polynomial_tp.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/binomial.hpp>
#include <algoim/interval.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace qugar::impl {


namespace alg = ::algoim;

template<int dim> using Interval = alg::Interval<dim>;

namespace {
  // NOLINTNEXTLINE (misc-no-recursion)
  template<typename T> void evaluate_monomial_basis(const T &point, const int order, int der, std::vector<T> &values)
  {
    assert(der >= 0);
    assert(order > 0);

    values.resize(static_cast<std::size_t>(order), T{ qugar::numbers::zero });

    real x_n = qugar::numbers::one;
    for (int i = der; i < order; ++i) {
      auto &val = at(values, i);
      val = x_n;
      for (int j = 0; j < der; ++j) {
        val *= static_cast<real>(i - j);
      }
      x_n *= point;
    }
  }

}// namespace


template<int dim, int range>
MonomialsTP<dim, range>::MonomialsTP(const TensorSizeTP<dim> &order) : PolynomialTP<dim, range>(order)
{}

template<int dim, int range>
MonomialsTP<dim, range>::MonomialsTP(const TensorSizeTP<dim> &order, const std::vector<CoefsType> &coefs)
  : PolynomialTP<dim, range>(order, coefs)
{}

template<int dim, int range>
MonomialsTP<dim, range>::MonomialsTP(const MonomialsTP<dim, range> &monomials)
  : MonomialsTP(monomials.get_order(), monomials.get_coefs())
{}

template<int dim, int range>
std::shared_ptr<MonomialsTP<dim, range>> MonomialsTP<dim, range>::create_derivative(const int dir) const
{
  assert(0 <= dir && dir < dim);

  if (this->get_degree(dir) == 0) {
    // Just a clone
    return std::make_shared<MonomialsTP<dim, range>>(this->get_order(), this->get_coefs());
  }

  auto new_order = this->get_order();
  --new_order(dir);

  std::vector<CoefsType> new_coefs(static_cast<std::size_t>(new_order.size()));

  for (TensorIndexTP<dim> tid : TensorIndexRangeTP<dim>(new_order)) {
    const auto flat_id = tid.flat(new_order);

    ++tid(dir);
    const auto old_flat_id = tid.flat(this->get_order());

    at(new_coefs, flat_id) = static_cast<real>(tid(dir)) * this->get_coef(old_flat_id);
  }

  return std::make_shared<MonomialsTP<dim, range>>(new_order, new_coefs);
}


template<int dim, int range>
template<typename T>
auto MonomialsTP<dim, range>::eval_(const Point<dim, T> &point) const -> Value<T>
{
  auto coefs_it = this->coefs_.cbegin();
  return MonomialsTP<dim, range>::horner<T>(point, coefs_it, this->order_);
}

template<int dim, int range>
template<typename T>
auto MonomialsTP<dim, range>::grad_(const Point<dim, T> &point) const -> Gradient<T>
{
  auto coefs_it = this->coefs_.cbegin();
  const auto val_der = this->template horner_der<T>(point, coefs_it, this->order_);
  return remove_component(val_der, 0);
}

template<int dim, int range>
template<typename T>
auto MonomialsTP<dim, range>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  // TODO: in the future this may be improved by implementing the Casteljau algorithm for second derivatives.
  std::array<std::array<std::vector<T>, dim>, 3> basis_and_ders;
  for (int dir = 0; dir < dim; ++dir) {
    for (int der = 0; der <= 2; ++der) {
      auto &bases = at(at(basis_and_ders, der), dir);
      bases.resize(static_cast<std::size_t>(this->order_(dir)));
      evaluate_monomial_basis(point(dir), this->order_(dir), der, bases);
    }
  }

  std::array<int, dim> ders{};
  ders.fill(0);

  Hessian<T> hess{};
  for (int i = 0, k = 0; i < dim; ++i) {
    ++at(ders, i);
    for (int j = i; j < dim; ++j, ++k) {
      ++at(ders, j);

      auto &hess_k = hess(k);
      hess_k = T{ qugar::numbers::zero };

      auto coefs_it = this->coefs_.cbegin();
      for (const auto tid : TensorIndexRangeTP<dim>(this->order_)) {
        auto val = *coefs_it++;
        for (int dir = 0; dir < dim; ++dir) {
          val *= at(at(at(basis_and_ders, at(ders, dir)), dir), tid(dir));
        }
        hess_k += val;
      }

      --at(ders, j);
    }
    --at(ders, i);
  }

  return hess;
}

template<int dim, int range> auto MonomialsTP<dim, range>::operator()(const Point<dim> &point) const -> Value<real>
{
  return this->eval_(point);
}

template<int dim, int range>
auto MonomialsTP<dim, range>::operator()(const Point<dim, Interval<dim>> &point) const -> Value<Interval<dim>>
{
  return this->eval_(point);
}

template<int dim, int range> auto MonomialsTP<dim, range>::grad(const Point<dim> &point) const -> Gradient<real>
{
  return this->grad_(point);
}

template<int dim, int range>
auto MonomialsTP<dim, range>::grad(const Point<dim, Interval<dim>> &point) const -> Gradient<Interval<dim>>
{
  return this->grad_(point);
}

template<int dim, int range> auto MonomialsTP<dim, range>::hessian(const Point<dim> &point) const -> Hessian<real>
{
  return this->hessian_(point);
}

template<int dim, int range>
template<int dim_aux>
  requires(dim == dim_aux && dim > 1)
std::shared_ptr<MonomialsTP<dim - 1, range>> MonomialsTP<dim, range>::extract_facet(const int local_facet_id) const
{
  const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
  const int side = get_facet_side<dim>(local_facet_id);

  const auto new_order = TensorSizeTP<dim - 1>(remove_component(this->get_order().as_Vector(), const_dir));

  std::vector<CoefsType> new_coefs(static_cast<std::size_t>(new_order.size()));

  TensorIndexTP<dim> lower_bound(0);
  TensorIndexTP<dim> upper_bound(this->get_order());
  if (side == 0) {
    upper_bound(const_dir) = 1;
  } else {
    lower_bound(const_dir) = this->get_order(const_dir) - 1;
  }
  const TensorIndexRangeTP<dim> indices_range(lower_bound, upper_bound);

  auto new_coef = new_coefs.begin();
  for (const auto &tid : indices_range) {
    *new_coef++ = at(this->coefs_, tid.flat(this->get_order()));
  }

  return std::make_shared<MonomialsTP<dim - 1, range>>(new_order, new_coefs);
}

template<int dim, int range>
template<typename T>
auto MonomialsTP<dim, range>::horner(const Point<dim, T> &point,
  typename std::vector<CoefsType>::const_iterator &coefs,
  const Vector<int, dim> &order) -> Value<T>
{
  const int last_order = order(dim - 1);
  assert(last_order > 0);

  std::vector<Value<T>> beta(static_cast<std::size_t>(last_order));
  if constexpr (dim == 1) {
    beta.assign(coefs, coefs + last_order);
    coefs += last_order;
  } else {
    const Point<dim - 1, T> sub_point = remove_component(point, dim - 1);
    const Vector<int, dim - 1> sub_order = remove_component(order, dim - 1);
    for (auto &beta_i : beta) {
      beta_i = MonomialsTP<dim - 1, range>::template horner<T>(sub_point, coefs, sub_order);
    }
  }

  Value<T> val{ beta.back() };
  const T x = point(dim - 1);

  for (auto beta_it = std::next(beta.crbegin()); beta_it != beta.crend(); ++beta_it) {
    val = val * x + *beta_it;
  }
  return val;
}

template<int dim, int range>
template<typename T>
auto MonomialsTP<dim, range>::horner_der(const Point<dim, T> &point,
  typename std::vector<CoefsType>::const_iterator &coefs,
  const Vector<int, dim> &order) -> Vector<Value<T>, dim + 1>
{
  const int last_order = order(dim - 1);
  assert(last_order > 0);
  const int last_degree = last_order - 1;

  std::vector<Gradient<T>> beta(static_cast<std::size_t>(last_order));
  if constexpr (dim == 1) {
    for (auto &beta_i : beta) {
      beta_i(0) = *coefs++;
    }
  } else {
    const Point<dim - 1, T> sub_point = remove_component(point, dim - 1);
    const Vector<int, dim - 1> sub_order = remove_component(order, dim - 1);
    for (auto &beta_i : beta) {
      beta_i = MonomialsTP<dim - 1, range>::template horner_der<T>(sub_point, coefs, sub_order);
    }
  }

  Gradient<T> val{ beta.back() };
  Value<T> der{ static_cast<real>(last_degree) * val(0) };
  const T x = point(dim - 1);

  for (int i = last_degree - 1; i > 0; --i) {
    val = val * x + at(beta, i);
    der = der * x + static_cast<real>(i) * at(beta, i)(0);
  }
  val = val * x + beta.front();

  return add_component(val, dim, der);
}

template<int dim, int range>
void MonomialsTP<dim, range>::transform_coefs_to_Bezier(const MonomialsTP<dim, range> &monomials,
  std::vector<CoefsType> &bzr_coefs)
{
  const auto &monomials_coefs = monomials.get_coefs();
  const auto &order = monomials.get_order();

  assert(static_cast<std::size_t>(order.size()) == monomials_coefs.size());

  bzr_coefs.resize(monomials_coefs.size(), CoefsType{ qugar::numbers::zero });
  auto mon_coef = monomials_coefs.cbegin();

  const auto get_binomial = [](const int degree) {
    return std::span<const real>(alg::Binomial::row(degree), static_cast<std::size_t>(degree + 1));
  };

  std::array<std::span<const real>, dim> binomials;
  for (int dir = 0; dir < dim; ++dir) {
    at(binomials, dir) = get_binomial(order(dir) - 1);
  }

  for (const auto tid : TensorIndexRangeTP<dim>(order)) {

    auto coef = *mon_coef++;
    for (int dir = 0; dir < dim; ++dir) {
      coef /= at(at(binomials, dir), tid(dir));
    }

    for (const auto tid2 : TensorIndexRangeTP<dim>(tid, TensorIndexTP<dim>(order))) {
      auto bzr_coef = coef;
      for (int dir = 0; dir < dim; ++dir) {
        bzr_coef *= at(get_binomial(tid2(dir)), tid(dir));
      }

      at(bzr_coefs, tid2.flat(order)) += bzr_coef;
    }
  }
}


// Instantiations

template class MonomialsTP<1, 1>;
template class MonomialsTP<1, 2>;
template class MonomialsTP<1, 3>;

template class MonomialsTP<2, 1>;
template class MonomialsTP<2, 2>;
template class MonomialsTP<2, 3>;

template class MonomialsTP<3, 1>;
template class MonomialsTP<3, 2>;
template class MonomialsTP<3, 3>;


template std::shared_ptr<MonomialsTP<1, 1>> MonomialsTP<2, 1>::extract_facet<2>(const int) const;
template std::shared_ptr<MonomialsTP<1, 2>> MonomialsTP<2, 2>::extract_facet<2>(const int) const;
template std::shared_ptr<MonomialsTP<1, 3>> MonomialsTP<2, 3>::extract_facet<2>(const int) const;
template std::shared_ptr<MonomialsTP<2, 1>> MonomialsTP<3, 1>::extract_facet<3>(const int) const;
template std::shared_ptr<MonomialsTP<2, 2>> MonomialsTP<3, 2>::extract_facet<3>(const int) const;
template std::shared_ptr<MonomialsTP<2, 3>> MonomialsTP<3, 3>::extract_facet<3>(const int) const;


}// namespace qugar::impl