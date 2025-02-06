#ifndef QUGAR_LIBRARY_NUMBERS_HPP
#define QUGAR_LIBRARY_NUMBERS_HPP

//! @file numbers.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of constant values.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/types.hpp>

#include <limits>
#include <numbers>

//! Numbers' namespace.
namespace qugar::numbers {

//! @brief Real zero value.
constexpr real zero{ 0.0 };
//! @brief Real one value.
constexpr real one{ 1.0 };
//! @brief Real two value.
constexpr real two{ 2.0 };
//! @brief Real three value.
constexpr real three{ 3.0 };
//! @brief Real four value.
constexpr real four{ 4.0 };
//! @brief Real five value.
constexpr real five{ 5.0 };
//! @brief Real six value.
constexpr real six{ 6.0 };
//! @brief Real seven value.
constexpr real seven{ 7.0 };
//! @brief Real eight value.
constexpr real eight{ 8.0 };
//! @brief Real nine value.
constexpr real nine{ 9.0 };
//! @brief Real one over two value.
constexpr real half{ 0.5 };
//! @brief Real one third value.
constexpr real one_third{ 1.0 / 3.0 };
//! @brief Real two thirds value.
constexpr real two_thirds{ 2.0 / 3.0 };
//! @brief Real four thirds value.
constexpr real four_thirds{ 4.0 / 3.0 };
//! @brief Real one quarter value.
constexpr real one_quarter{ 0.25 };
//! @brief Real three quarters value.
constexpr real three_quarters{ 0.75 };
//! @brief Pi.
constexpr real pi{ std::numbers::pi_v<real> };
//! @brief Infinity.
constexpr real infty{ std::numeric_limits<real>::infinity() };
//! @brief Machine epsilon.
constexpr real eps = std::numeric_limits<real>::epsilon();
//! @brief Near machine epsilon (10 times the machine precision.
constexpr real near_eps = real{ 10.0 } * eps;

}// namespace qugar::numbers

#endif// QUGAR_LIBRARY_NUMBERS_HPP