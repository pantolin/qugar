// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_LAGRANGE_TP_UTILS_HPP
#define QUGAR_IMPL_LAGRANGE_TP_UTILS_HPP

//! @file lagrange_tp_utils.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of tensor-product Lagrange utils.
//! @date 2025-01-10
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/point.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <vector>

namespace qugar::impl {

//! @brief Evaluates the Lagrange basis polynomials in 1D at a given point.
//!
//! This function computes the values of the Lagrange basis polynomials of a specified order at a given point.
//! The basis can be evaluated using either Chebyshev nodes (of 2nd kind) or standard equidistant nodes.
//!
//! @param point The point at which to evaluate the Lagrange basis polynomials.
//! @param order The order of the Lagrange basis polynomials (degree + 1).
//! @param chebyshev A boolean flag indicating whether to use Chebyshev nodes (true) of 2nd kind or standard equidistant
//! nodes (false).
//! @param values A reference to a vector where the computed values of the Lagrange basis polynomials will be stored.
//! This vector will be resized to the order of the Lagrange basis polynomials.
void evaluate_Lagrange_basis_1D(real point, int order, bool chebyshev, std::vector<real> &values);

//! @brief Evaluates the first derivative of the Lagrange basis polynomial in 1D.
//!
//! This function computes the values of the first derivative of the Lagrange basis
//! polynomial at a given point for a specified order. The basis can be either
//! Chebyshev (of 2nd kind) or equidistant nodes.
//!
//! @param point The point at which to evaluate the derivative.
//! @param order The order of the Lagrange basis polynomial (degree + 1).
//! @param chebyshev A boolean flag indicating whether to use Chebyshev nodes (true) of 2nd kind
//!                  or standard equidistant nodes (false).
//! @param values A reference to a vector where the computed derivative values will
//!               be stored. This vector will be resized to the order of the Lagrange basis polynomials.
void evaluate_Lagrange_basis_der_1D(real point, int order, bool chebyshev, std::vector<real> &values);


//! @brief Evaluates the tensor-product Lagrange basis functions at a given point.
//!
//! @param point The point at which to evaluate the Lagrange basis functions.
//! @param order The order (degree + 1) of the tensor product along each direction.
//! @param chebyshev A boolean flag indicating whether to use Chebyshev nodes (true) of 2nd kind
//!                  or standard equidistant nodes (false).
//! @param basis A reference to a vector where the computed derivative values will
//!               be stored. This vector will be resized to the order of the Lagrange basis polynomials.
template<int dim>
void evaluate_Lagrange_basis(const Point<dim> &point,
  const TensorSizeTP<dim> &order,
  bool chebyshev,
  std::vector<real> &basis);

//! @brief Evaluates the derivative of the Lagrange basis functions at a given point, along all directions.
//!
//! @param point The point at which to evaluate the derivative of the Lagrange basis functions.
//! @param order The order (degree + 1) of the tensor product along each direction.
//! @param chebyshev A boolean flag indicating whether to use Chebyshev nodes (true) of 2nd kind
//!                  or standard equidistant nodes (false).
//! @param basis_ders A vector to store the computed derivatives of the Lagrange basis functions along all directions.
//! The vectors along each direction will be resized to the order of the Lagrange basis polynomials.
template<int dim>
void evaluate_Lagrange_derivative(const Point<dim> &point,
  const TensorSizeTP<dim> &order,
  bool chebyshev,
  Vector<std::vector<real>, dim> &basis_ders);


}// namespace qugar::impl

#endif// QUGAR_IMPL_LAGRANGE_TP_UTIL_HPP