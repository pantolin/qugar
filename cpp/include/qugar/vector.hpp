// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_VECTOR_HPP

#define QUGAR_VECTOR_HPP

//! @file point.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Definition and implementation of Point class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <algoim/uvector.hpp>

namespace qugar {

//! @brief Class representing a vector.
//!
//! @tparam dim Dimension of point.
//! @tparam T Vector type.
template<typename T, int dim> using Vector = ::algoim::uvector<T, dim>;

using ::algoim::norm;

using ::algoim::sqrnorm;

using ::algoim::remove_component;

using ::algoim::set_component;

using ::algoim::add_component;

using ::algoim::max;

using ::algoim::min;

using ::algoim::argmax;

using ::algoim::argmin;

using ::algoim::dot;

using ::algoim::prod;

using ::algoim::all;

using ::algoim::any;


//! Permutes the directions of a given vector by reversing the order of its elements.
//!
//! @note This function is useful for transforming points for some Algoim's algorithms
//! that use counter-lexicographical ordering (as coefficients of Bezier polynomials).
//!
//! @tparam T The type of the elements in the vector.
//! @tparam dim The dimension of the vector.
//! @param vec The input vector to be permuted.
//! @return A new vector with its directions permuted.
template<typename T, int dim> Vector<T, dim> permute_vector_directions(const Vector<T, dim> &vec)
{
  Vector<T, dim> permuted;
  for (int dir = 0; dir < dim; ++dir) {
    permuted(dir) = vec(dim - dir - 1);
  }
  return permuted;
}


}// namespace qugar

#endif// QUGAR_VECTOR_HPP