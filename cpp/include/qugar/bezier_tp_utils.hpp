// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_BEZIER_TP_UTILS_HPP
#define QUGAR_IMPL_BEZIER_TP_UTILS_HPP

//! @file bezier_tp_utils.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of tensor-product Bezier utils.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bezier_tp.hpp>

namespace qugar::impl {

//! @brief Evaluates the Bernstein polynomials of the given order.
//!
//! @tparam T Type of the input coordinate.
//! @param point Evaluation point.
//! @param order Order of the polynomial.
//! @param values Computed basis values.
template<typename T> void evaluate_Bernstein_value(const T &point, const int order, std::vector<T> &values);

//! @brief Evaluates the Bernstein polynomials of the given order (or its derivative).
//!
//! @tparam T Type of the input coordinate.
//! @tparam V Type of the output vector of basis values.
//! @param point Evaluation point.
//! @param order Order of the polynomial.
//! @param der Order of the derivative to be computed. If 0, the value itself is computed.
//! @param values Computed basis values.
template<typename T> void evaluate_Bernstein(const T &point, const int order, int der, std::vector<T> &values);


//! @brief Product of two Beziers.
//!
//! @param lhs First Bezier to multiply.
//! @param rhs Second Bezier to multiply.
//! @return Product of Beziers.
template<int dim, int range>
[[nodiscard]] std::shared_ptr<BezierTP<dim, range>> Bezier_product(const BezierTP<dim, range> &lhs,
  const BezierTP<dim, range> &rhs);

//! @brief Computes the composition of two Beziers as rhs(lhs)
//!
//! @tparam dim Dimension of the resultant Bezier.
//! @tparam range Range of the resultant Bezier.
//! @tparam dim2 Common dimension between Beziers.
//! @param lhs Bezier to be composed.
//! @param rhs Composing Bezier.
//! @return Resultant composition.
template<int dim, int range, int dim2>
[[nodiscard]] std::shared_ptr<BezierTP<dim, range>> Bezier_composition(const BezierTP<dim2, range> &lhs,
  const BezierTP<dim, dim2> &rhs);


}// namespace qugar::impl

#endif// QUGAR_IMPL_BEZIER_TP_UTIL_HPP