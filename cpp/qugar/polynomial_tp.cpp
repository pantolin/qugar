// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------


//! @file polynomial_tp.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tensor-product polynomial class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <qugar/polynomial_tp.hpp>

#include <qugar/bbox.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace qugar::impl {

template<int dim, int range>
PolynomialTP<dim, range>::PolynomialTP(const TensorSizeTP<dim> &order)
  : PolynomialTP(order, CoefsType{ qugar::numbers::zero })
{}

template<int dim, int range>
PolynomialTP<dim, range>::PolynomialTP(const TensorSizeTP<dim> &order, const CoefsType &value)
  : order_(order), coefs_(static_cast<std::size_t>(order.size()), value)
{
  static_assert(0 < dim, "Parametric dimension must be > 0.");
#ifndef NDEBUG
  for (int dir = 0; dir < dim; ++dir) {
    assert(order_(dir) > 0);
  }
#endif//  NDEBUG
}


template<int dim, int range>
PolynomialTP<dim, range>::PolynomialTP(const TensorSizeTP<dim> &order, const std::vector<CoefsType> &coefs)
  : order_(order), coefs_(coefs)
{
  static_assert(0 < dim, "Parametric dimension must be > 0.");
#ifndef NDEBUG
  for (int dir = 0; dir < dim; ++dir) {
    assert(order_(dir) > 0);
  }
  assert(static_cast<std::size_t>(order.size()) == coefs_.size());
#endif//  NDEBUG
}

template<int dim, int range> std::size_t PolynomialTP<dim, range>::get_num_coefs() const
{
  return static_cast<std::size_t>(this->order_.size());
}

template<int dim, int range> auto PolynomialTP<dim, range>::get_coefs() const -> const std::vector<CoefsType> &
{
  return coefs_;
}

template<int dim, int range> const TensorSizeTP<dim> &PolynomialTP<dim, range>::get_order() const
{
  return order_;
}

template<int dim, int range> int PolynomialTP<dim, range>::get_order(const int dir) const
{
  return this->order_(dir);
}

template<int dim, int range> int PolynomialTP<dim, range>::get_degree(const int dir) const
{
  return this->get_order(dir) - 1;
}

template<int dim, int range> auto PolynomialTP<dim, range>::get_coef(const int index) const -> const CoefsType &
{
  return at(coefs_, index);
}

template<int dim, int range> auto PolynomialTP<dim, range>::get_coef(const int index) -> CoefsType &
{
  // NOLINTBEGIN (cppcoreguidelines-pro-type-const-cast)
  const auto *this_const = const_cast<const PolynomialTP<dim, range> *>(this);
  return const_cast<CoefsType &>(this_const->get_coef(index));
  // NOLINTEND (cppcoreguidelines-pro-type-const-cast)
}

template<int dim, int range>
auto PolynomialTP<dim, range>::get_coef(const TensorIndexTP<dim> &index) const -> const CoefsType &
{
  return this->get_coef(index.flat(this->order_));
}

template<int dim, int range> auto PolynomialTP<dim, range>::get_coef(const TensorIndexTP<dim> &index) -> CoefsType &
{
  // NOLINTBEGIN (cppcoreguidelines-pro-type-const-cast)
  const auto *this_const = const_cast<const PolynomialTP<dim, range> *>(this);
  return const_cast<CoefsType &>(this_const->get_coef(index));
  // NOLINTEND (cppcoreguidelines-pro-type-const-cast)
}

template<int dim, int range>
void PolynomialTP<dim, range>::transform_image(const BoundBox<range> &old_domain, const BoundBox<range> &new_domain)
{
  for (auto &point : coefs_) {
    if constexpr (range == 1) {
      point = (point - old_domain.min(0)) * new_domain.length(0) / old_domain.length(0) + new_domain.min(0);
    } else {
      point = old_domain.scale_to_new_domain(new_domain, point);
    }
  }
}

template<int dim, int range>
void PolynomialTP<dim, range>::coefs_linear_transform(const real scale, const CoefsType &shift)
{
  for (auto &coef : this->coefs_) {
    coef = scale * coef + shift;
  }
}

// Instantiations

template class PolynomialTP<1, 1>;
template class PolynomialTP<1, 2>;
template class PolynomialTP<1, 3>;

template class PolynomialTP<2, 1>;
template class PolynomialTP<2, 2>;
template class PolynomialTP<2, 3>;

template class PolynomialTP<3, 1>;
template class PolynomialTP<3, 2>;
template class PolynomialTP<3, 3>;


}// namespace qugar::impl
