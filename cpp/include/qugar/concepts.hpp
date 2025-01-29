#ifndef QUGAR_LIBRARY_CONCEPTS_HPP
#define QUGAR_LIBRARY_CONCEPTS_HPP

//! @file concepts.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of concepts.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

namespace qugar {

//! @brief Checks if the given dimension is 2 or 3.
template<int dim, int dim_aux>
concept Is2Dor3D = dim == dim_aux && (dim == 2 || dim == 3);

//! @brief Checks if the given dimension is 1.
template<int dim, int dim_aux>
concept Is1D = dim == dim_aux && dim == 1;

}// namespace qugar

#endif// QUGAR_LIBRARY_CONCEPTS_HPP