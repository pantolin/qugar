// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file bezier_tp.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tensor-product Bezier class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bezier_tp.hpp>

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp_utils.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/monomials_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/polynomial_tp.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/bernstein.hpp>
#include <algoim/interval.hpp>
#include <algoim/xarray.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace qugar::impl {


namespace alg = ::algoim;

template<int dim> using Interval = alg::Interval<dim>;

template<int dim, int range>
BezierTP<dim, range>::BezierTP(const TensorSizeTP<dim> &order) : BezierTP(order, CoefsType{ numbers::zero })
{}

template<int dim, int range>
BezierTP<dim, range>::BezierTP(const TensorSizeTP<dim> &order, const CoefsType &value)
  : PolynomialTP<dim, range>(order, value), coefs_xarray_(this->coefs_.data(), permute_vector_directions(this->order_))
{}

template<int dim, int range>
BezierTP<dim, range>::BezierTP(const TensorSizeTP<dim> &order, const std::vector<CoefsType> &coefs)
  : PolynomialTP<dim, range>(order, coefs), coefs_xarray_(this->coefs_.data(), permute_vector_directions(this->order_))
{}

template<int dim, int range>
BezierTP<dim, range>::BezierTP(const BezierTP<dim, range> &bezier) : BezierTP(bezier.get_order(), bezier.get_coefs())
{}

template<int dim, int range>
BezierTP<dim, range>::BezierTP(const MonomialsTP<dim, range> &monomials) : BezierTP(monomials.get_order())
{
  MonomialsTP<dim, range>::transform_coefs_to_Bezier(monomials, this->coefs_);
}

template<int dim, int range> auto BezierTP<dim, range>::get_xarray() const -> const ::algoim::xarray<CoefsType, dim> &
{
  return this->coefs_xarray_;
}

template<int dim, int range>
template<typename T>
auto BezierTP<dim, range>::eval_(const Vector<T, dim> &point) const -> Value<T>
{
  auto coefs_it = this->coefs_.cbegin();
  return BezierTP<dim, range>::casteljau<T>(point, coefs_it, this->order_);
}

template<int dim, int range>
template<typename T>
auto BezierTP<dim, range>::grad_(const Vector<T, dim> &point) const -> Gradient<T>
{
  auto coefs_it = this->coefs_.cbegin();
  const auto val_der = this->template casteljau_der<T>(point, coefs_it, this->order_);
  return remove_component(val_der, 0);
}

template<int dim, int range>
template<typename T>
auto BezierTP<dim, range>::hessian_(const Vector<T, dim> &point) const -> Hessian<T>
{
  // TODO: in the future this may be improved by implementing the Casteljau algorithm for second derivatives.
  std::array<std::array<std::vector<T>, dim>, 3> basis_and_ders;
  for (int dir = 0; dir < dim; ++dir) {
    for (int der = 0; der <= 2; ++der) {
      auto &bases = at(at(basis_and_ders, der), dir);
      bases.resize(static_cast<std::size_t>(this->order_(dir)));
      evaluate_Bernstein(point(dir), this->order_(dir), der, bases);
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
      hess_k = T{ numbers::zero };

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

template<int dim, int range> auto BezierTP<dim, range>::operator()(const Vector<real, dim> &point) const -> Value<real>
{
  return this->eval_(point);
}

template<int dim, int range>
auto BezierTP<dim, range>::operator()(const Vector<Interval<dim>, dim> &point) const -> Value<Interval<dim>>
{
  return this->eval_(point);
}

template<int dim, int range> auto BezierTP<dim, range>::grad(const Vector<real, dim> &point) const -> Gradient<real>
{
  return this->grad_(point);
}

template<int dim, int range>
auto BezierTP<dim, range>::grad(const Vector<Interval<dim>, dim> &point) const -> Gradient<Interval<dim>>
{
  return this->grad_(point);
}

template<int dim, int range> auto BezierTP<dim, range>::hessian(const Vector<real, dim> &point) const -> Hessian<real>
{
  return this->hessian_(point);
}

template<int dim, int range>
template<typename T>
auto BezierTP<dim, range>::casteljau(const Vector<T, dim> &point,
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
    const Vector<T, dim - 1> sub_point = remove_component(point, dim - 1);
    const Vector<int, dim - 1> sub_order = remove_component(order, dim - 1);
    for (auto &beta_i : beta) {
      beta_i = BezierTP<dim - 1, range>::template casteljau<T>(sub_point, coefs, sub_order);
    }
  }

  const T x = point(dim - 1);
  for (int j = 1; j < last_order; ++j) {
    auto beta_k = beta.begin();
    for (int k = 0; k < (last_order - j); ++k, ++beta_k) {
      *beta_k += (*std::next(beta_k) - *beta_k) * x;
    }
  }

  return beta.front();
}


template<int dim, int range>
template<typename T>
auto BezierTP<dim, range>::casteljau_der(const Vector<T, dim> &point,
  typename std::vector<CoefsType>::const_iterator &coefs,
  const Vector<int, dim> &order) -> Vector<Value<T>, dim + 1>
{
  const int last_order = order(dim - 1);
  assert(last_order > 0);

  const int degree = last_order - 1;

  std::vector<Gradient<T>> beta(static_cast<std::size_t>(last_order));
  std::vector<Value<T>> gamma(beta.size(), Value<T>{ numbers::zero });

  if constexpr (dim == 1) {
    for (auto &beta_i : beta) {
      beta_i(0) = *coefs++;
    }
  } else {
    const Vector<T, dim - 1> sub_point = remove_component(point, dim - 1);
    const Vector<int, dim - 1> sub_order = remove_component(order, dim - 1);
    for (auto &beta_i : beta) {
      beta_i = BezierTP<dim - 1, range>::casteljau_der(sub_point, coefs, sub_order);
    }
  }

  if (last_order > 1) {
    // Beta stores the Bezier value and gamma its derivative.
    auto gamma_i = gamma.begin();

    *gamma_i++ = -degree * beta.front()(0);
    *gamma_i-- = -beta.front()(0);

    auto beta_i = std::next(beta.cbegin());

    for (int i = 1; i < degree; ++i) {
      const auto beta_i_0 = beta_i++->operator()(0);
      *gamma_i++ += ((last_order - i) * beta_i_0);
      *gamma_i++ += ((2 * i) - degree) * beta_i_0;
      *gamma_i-- -= (i + 1) * beta_i_0;
    }
    *gamma_i++ += beta.back()(0);
    *gamma_i += degree * beta.back()(0);

    const T x = point(dim - 1);
    for (int j = 1; j < last_order; ++j) {
      auto beta_k = beta.begin();
      auto gamma_k = gamma.begin();
      for (int k = 0; k < (last_order - j); ++k, ++beta_k, ++gamma_k) {
        *beta_k += (*std::next(beta_k) - *beta_k) * x;
        *gamma_k += (*std::next(gamma_k) - *gamma_k) * x;
      }
    }
  }


  return add_component(beta.front(), dim, gamma.front());
}

template<int dim, int range>
std::shared_ptr<BezierTP<dim, range>> BezierTP<dim, range>::operator*(const BezierTP<dim, range> &rhs) const
{
  return Bezier_product(*this, rhs);
}

template<int dim, int range>
// NOLINTNEXTLINE (misc-no-recursion)
std::shared_ptr<BezierTP<dim, range>> BezierTP<dim, range>::operator+(const BezierTP<dim, range> &rhs) const
{
  if (rhs.get_order() == this->get_order()) {
    auto new_coefs = this->get_coefs();
    for (int i = 0; i < static_cast<int>(this->get_num_coefs()); ++i) {
      at(new_coefs, i) += rhs.get_coef(i);
    }
    return std::make_shared<BezierTP<dim, range>>(this->get_order(), new_coefs);
  }

  bool rhs_is_higher = true;
  bool lhs_is_higher = true;
  TensorSizeTP<dim> new_order;
  for (int dir = 0; dir < dim; ++dir) {
    const int diff_dir = rhs.get_order(dir) - this->get_order(dir);
    if (diff_dir > 0) {
      lhs_is_higher = false;
    } else if (diff_dir < 0) {
      rhs_is_higher = false;
    }
    new_order(dir) = std::max(rhs.get_order(dir), this->get_order(dir));
  }

  // NOLINTBEGIN (misc-no-recursion)
  if (rhs_is_higher) {
    const auto new_lhs = this->raise_order(rhs.get_order());
    return *new_lhs + rhs;
  } else if (lhs_is_higher) {
    const auto new_rhs = rhs.raise_order(this->get_order());
    return *this + *new_rhs;
  } else {
    const auto new_lhs = this->raise_order(new_order);
    const auto new_rhs = rhs.raise_order(new_order);
    return *new_lhs + *new_rhs;
  }
  // NOLINTEND (misc-no-recursion)
}

template<int dim, int range>
// NOLINTNEXTLINE (misc-no-recursion)
std::shared_ptr<BezierTP<dim, range>> BezierTP<dim, range>::operator-(const BezierTP<dim, range> &rhs) const
{
  if (rhs.get_order() == this->get_order()) {
    auto new_coefs = this->get_coefs();
    for (int i = 0; i < static_cast<int>(this->get_num_coefs()); ++i) {
      at(new_coefs, i) -= rhs.get_coef(i);
    }
    return std::make_shared<BezierTP<dim, range>>(this->get_order(), new_coefs);
  }

  bool rhs_is_higher = true;
  bool lhs_is_higher = true;
  TensorSizeTP<dim> new_order;
  for (int dir = 0; dir < dim; ++dir) {
    const int diff_dir = rhs.get_order(dir) - this->get_order(dir);
    if (diff_dir > 0) {
      lhs_is_higher = false;
    } else if (diff_dir < 0) {
      rhs_is_higher = false;
    }
    new_order(dir) = std::max(rhs.get_order(dir), this->get_order(dir));
  }

  // NOLINTBEGIN (misc-no-recursion)
  if (rhs_is_higher) {
    const auto new_lhs = this->raise_order(rhs.get_order());
    return *new_lhs - rhs;
  } else if (lhs_is_higher) {
    const auto new_rhs = rhs.raise_order(this->get_order());
    return *this - *new_rhs;
  } else {
    const auto new_lhs = this->raise_order(new_order);
    const auto new_rhs = rhs.raise_order(new_order);
    return *new_lhs - *new_rhs;
  }
  // NOLINTEND (misc-no-recursion)
}

template<int dim, int range>
template<int sub_dim>
std::shared_ptr<BezierTP<sub_dim, range>> BezierTP<dim, range>::compose(const BezierTP<sub_dim, dim> &rhs) const
{
  return Bezier_composition(*this, rhs);
}

template<int dim, int range>
template<int dim_aux>
  requires(dim == dim_aux && dim > 1)
std::shared_ptr<BezierTP<dim - 1, range>> BezierTP<dim, range>::extract_facet(const int local_facet_id) const
{
  const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
  const int side = get_facet_side<dim>(local_facet_id);

  const TensorSizeTP<dim - 1> new_order{ remove_component(this->get_order(), const_dir) };
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

  return std::make_shared<BezierTP<dim - 1, range>>(new_order, new_coefs);
}

template<int dim, int range>
std::shared_ptr<BezierTP<dim, range>> BezierTP<dim, range>::raise_order(const TensorSizeTP<dim> &new_order) const
{
  if (new_order == this->get_order()) {
    return std::make_shared<BezierTP<dim, range>>(*this);
  }

  TensorSizeTP<dim> diff;
  for (int dir = 0; dir < dim; ++dir) {
    assert(new_order(dir) >= this->get_order(dir));
    diff(dir) = new_order(dir) - this->get_order(dir) + 1;
  }

  const BezierTP<dim, range> unit{ diff, CoefsType{ numbers::one } };

  return *this * unit;
}

template<int dim, int range> std::shared_ptr<BezierTP<dim, range>> BezierTP<dim, range>::negate() const
{
  const auto negative = std::make_shared<BezierTP<dim, range>>(*this);
  negative->coefs_linear_transform(-numbers::one, CoefsType{ numbers::zero });
  return negative;
}

template<int dim, int range>
template<int range_aux>
  requires(range_aux == range && range == 1)
void BezierTP<dim, range>::rescale_domain(const BoundBox<dim> &new_domain)
{
  Vector<real, dim> domain_min;
  Vector<real, dim> domain_max;
  for (int dir = 0; dir < dim; ++dir) {
    domain_min(dir) = new_domain.min(dim - dir - 1);
    domain_max(dir) = new_domain.max(dim - dir - 1);
  }
  alg::bernstein::deCasteljau(this->coefs_xarray_, domain_min.data(), domain_max.data());
}

template<int dim, int range>
template<int range_aux>
  requires(range_aux == range && range == 1)
FuncSign BezierTP<dim, range>::sign() const
{
  switch (alg::bernstein::uniformSign<dim>(this->coefs_xarray_)) {
  case 1:
    return FuncSign::positive;
  case -1:
    return FuncSign::negative;
  case 0:
  default:
    return FuncSign::undetermined;
  }
}

// Instantiations

template class BezierTP<1, 1>;
template class BezierTP<1, 2>;
template class BezierTP<1, 3>;

template class BezierTP<2, 1>;
template class BezierTP<2, 2>;
template class BezierTP<2, 3>;

template class BezierTP<3, 1>;
template class BezierTP<3, 2>;
template class BezierTP<3, 3>;

template void BezierTP<1, 1>::rescale_domain<1>(const BoundBox<1> &new_domain);
template void BezierTP<2, 1>::rescale_domain<1>(const BoundBox<2> &new_domain);
template void BezierTP<3, 1>::rescale_domain<1>(const BoundBox<3> &new_domain);

template FuncSign BezierTP<1, 1>::sign<1>() const;
template FuncSign BezierTP<2, 1>::sign<1>() const;
template FuncSign BezierTP<3, 1>::sign<1>() const;

template std::shared_ptr<BezierTP<1, 1>> BezierTP<2, 1>::extract_facet<2>(const int) const;
template std::shared_ptr<BezierTP<1, 2>> BezierTP<2, 2>::extract_facet<2>(const int) const;
template std::shared_ptr<BezierTP<1, 3>> BezierTP<2, 3>::extract_facet<2>(const int) const;
template std::shared_ptr<BezierTP<2, 1>> BezierTP<3, 1>::extract_facet<3>(const int) const;
template std::shared_ptr<BezierTP<2, 2>> BezierTP<3, 2>::extract_facet<3>(const int) const;
template std::shared_ptr<BezierTP<2, 3>> BezierTP<3, 3>::extract_facet<3>(const int) const;

template std::shared_ptr<BezierTP<1, 1>> BezierTP<1, 1>::compose<1>(const BezierTP<1, 1> &) const;
template std::shared_ptr<BezierTP<2, 1>> BezierTP<1, 1>::compose<2>(const BezierTP<2, 1> &) const;
template std::shared_ptr<BezierTP<3, 1>> BezierTP<1, 1>::compose<3>(const BezierTP<3, 1> &) const;
template std::shared_ptr<BezierTP<1, 1>> BezierTP<2, 1>::compose<1>(const BezierTP<1, 2> &) const;
template std::shared_ptr<BezierTP<2, 1>> BezierTP<2, 1>::compose<2>(const BezierTP<2, 2> &) const;
template std::shared_ptr<BezierTP<3, 1>> BezierTP<2, 1>::compose<3>(const BezierTP<3, 2> &) const;
template std::shared_ptr<BezierTP<1, 1>> BezierTP<3, 1>::compose<1>(const BezierTP<1, 3> &) const;
template std::shared_ptr<BezierTP<2, 1>> BezierTP<3, 1>::compose<2>(const BezierTP<2, 3> &) const;
template std::shared_ptr<BezierTP<3, 1>> BezierTP<3, 1>::compose<3>(const BezierTP<3, 3> &) const;

template std::shared_ptr<BezierTP<1, 2>> BezierTP<1, 2>::compose<1>(const BezierTP<1, 1> &) const;
template std::shared_ptr<BezierTP<2, 2>> BezierTP<1, 2>::compose<2>(const BezierTP<2, 1> &) const;
template std::shared_ptr<BezierTP<3, 2>> BezierTP<1, 2>::compose<3>(const BezierTP<3, 1> &) const;
template std::shared_ptr<BezierTP<1, 2>> BezierTP<2, 2>::compose<1>(const BezierTP<1, 2> &) const;
template std::shared_ptr<BezierTP<2, 2>> BezierTP<2, 2>::compose<2>(const BezierTP<2, 2> &) const;
template std::shared_ptr<BezierTP<3, 2>> BezierTP<2, 2>::compose<3>(const BezierTP<3, 2> &) const;
template std::shared_ptr<BezierTP<1, 2>> BezierTP<3, 2>::compose<1>(const BezierTP<1, 3> &) const;
template std::shared_ptr<BezierTP<2, 2>> BezierTP<3, 2>::compose<2>(const BezierTP<2, 3> &) const;
template std::shared_ptr<BezierTP<3, 2>> BezierTP<3, 2>::compose<3>(const BezierTP<3, 3> &) const;

template std::shared_ptr<BezierTP<1, 3>> BezierTP<1, 3>::compose<1>(const BezierTP<1, 1> &) const;
template std::shared_ptr<BezierTP<2, 3>> BezierTP<1, 3>::compose<2>(const BezierTP<2, 1> &) const;
template std::shared_ptr<BezierTP<3, 3>> BezierTP<1, 3>::compose<3>(const BezierTP<3, 1> &) const;
template std::shared_ptr<BezierTP<1, 3>> BezierTP<2, 3>::compose<1>(const BezierTP<1, 2> &) const;
template std::shared_ptr<BezierTP<2, 3>> BezierTP<2, 3>::compose<2>(const BezierTP<2, 2> &) const;
template std::shared_ptr<BezierTP<3, 3>> BezierTP<2, 3>::compose<3>(const BezierTP<3, 2> &) const;
template std::shared_ptr<BezierTP<1, 3>> BezierTP<3, 3>::compose<1>(const BezierTP<1, 3> &) const;
template std::shared_ptr<BezierTP<2, 3>> BezierTP<3, 3>::compose<2>(const BezierTP<2, 3> &) const;
template std::shared_ptr<BezierTP<3, 3>> BezierTP<3, 3>::compose<3>(const BezierTP<3, 3> &) const;


}// namespace qugar::impl