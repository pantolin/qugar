#ifndef QUGAR_LIBRARY_UTILS_HPP
#define QUGAR_LIBRARY_UTILS_HPP

//! @file utils.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Utility functions.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/types.hpp>


#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>

namespace qugar {

// narrow_cast(): a searchable way to do narrowing casts of values
template<class T, class V> constexpr T narrow_cast(V &&val) noexcept
{
  return static_cast<T>(std::forward<V>(val));
}

//
// at() - Bounds-checked way of accessing builtin arrays, std::array, std::vector
//
template<class T, std::size_t N>
// NOLINTNEXTLINE (cppcoreguidelines-avoid-c-arrays)
constexpr T &at(T (&arr)[N], const index ind)
{
  assert(ind >= 0 && ind < narrow_cast<index>(N));
  return arr[narrow_cast<std::size_t>(ind)];
}

template<class Cont> constexpr auto at(Cont &cont, const index ind) -> decltype(cont[cont.size()])
{
  assert(ind >= 0 && ind < narrow_cast<index>(cont.size()));
  using size_type = decltype(cont.size());
  return cont[narrow_cast<size_type>(ind)];
}

template<class T> constexpr T at(const std::initializer_list<T> cont, const index ind)
{
  assert(ind >= 0 && ind < narrow_cast<index>(cont.size()));
  return *(cont.begin() + ind);
}

template<class T, std::size_t extent = std::dynamic_extent>
constexpr auto at(std::span<T, extent> spn, const index ind) -> decltype(spn[spn.size()])
{
  assert(ind >= 0 && ind < narrow_cast<index>(spn.size()));
  return spn[narrow_cast<std::size_t>(ind)];
}

}// namespace qugar

#endif// QUGAR_LIBRARY_UTILS_HPP