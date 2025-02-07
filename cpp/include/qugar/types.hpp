#ifndef QUGAR_LIBRARY_TYPES_HPP
#define QUGAR_LIBRARY_TYPES_HPP

//! @file types.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of types.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <cstddef>

// Adding this include causes an unexplicable clang-diagnosis-error.
// #include <Standard_TypeDef.hxx>

namespace qugar {

using real = double;
using index = std::ptrdiff_t;

}// namespace qugar

#endif// QUGAR_LIBRARY_TYPES_HPP