// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file bezier_test_utils.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Utilities for testing BezierTP class.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#ifndef QUGAR_TEST_BEZIER_TEST_UTILS_HPP
#define QUGAR_TEST_BEZIER_TEST_UTILS_HPP

#include "random_generator.hpp"

#include <qugar/bezier_tp.hpp>
#include <qugar/monomials_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tolerance.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

template<int dim, int range>
// NOLINTBEGIN (bugprone-easily-swappable-parameters)
std::shared_ptr<qugar::impl::BezierTP<dim, range>> create_random_bezier(const int min_order = 1,
  const int max_order = 5,
  const qugar::real min_coord = qugar::numbers::zero,
  const qugar::real max_coord = qugar::numbers::one)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  assert(min_order <= max_order);
  const bool avoid_rep = (max_order - min_order + 1) >= dim;

  const auto order = qugar::rand::create_random_tensor_size<dim>(min_order, max_order, avoid_rep);
  const auto n_coefs = static_cast<std::size_t>(order.size());

  if constexpr (range == 1) {
    const auto coefs = qugar::rand::create_random_values(n_coefs);
    return std::make_shared<qugar::impl::BezierTP<dim, range>>(order, coefs);
  } else {// if constexpr (range >) {
    const auto coefs = qugar::rand::create_random_real_vectors<range>(n_coefs, min_coord, max_coord);
    return std::make_shared<qugar::impl::BezierTP<dim, range>>(order, coefs);
  }
}

template<int dim, int range>
// NOLINTBEGIN (bugprone-easily-swappable-parameters)
std::shared_ptr<qugar::impl::BezierTP<dim, range>> create_constant_random_bezier(const int min_order = 1,
  const int max_order = 5,
  const qugar::real min_coord = qugar::numbers::zero,
  const qugar::real max_coord = qugar::numbers::one)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  assert(min_order <= max_order);
  const bool avoid_rep = (max_order - min_order + 1) >= dim;

  const auto order = qugar::rand::create_random_tensor_size<dim>(min_order, max_order, avoid_rep);
  const auto n_coefs = static_cast<std::size_t>(order.size());

  if constexpr (range == 1) {
    const auto rand_value = qugar::rand::create_random_values(1).front();
    const std::vector<qugar::real> coefs(n_coefs, rand_value);
    return std::make_shared<qugar::impl::BezierTP<dim, range>>(order, coefs);
  } else {
    const auto rand_pt = qugar::rand::create_random_real_vector<range>(min_coord, max_coord);
    const std::vector<qugar::Point<range>> coefs(n_coefs, rand_pt);
    return std::make_shared<qugar::impl::BezierTP<dim, range>>(order, coefs);
  }
}

template<int dim, int range>
// NOLINTBEGIN (bugprone-easily-swappable-parameters)
std::shared_ptr<qugar::impl::MonomialsTP<dim, range>> create_random_monomials(const int min_order = 1,
  const int max_order = 5,
  const qugar::real min_coord = qugar::numbers::zero,
  const qugar::real max_coord = qugar::numbers::one)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  assert(min_order <= max_order);
  const bool avoid_rep = (max_order - min_order + 1) >= dim;

  const auto order = qugar::rand::create_random_tensor_size<dim>(min_order, max_order, avoid_rep);
  const auto n_coefs = static_cast<std::size_t>(order.size());

  if constexpr (range == 1) {
    const auto coefs = qugar::rand::create_random_values(n_coefs);
    return std::make_shared<qugar::impl::MonomialsTP<dim, range>>(order, coefs);
  } else {// if constexpr (range >) {
    const auto coefs = qugar::rand::create_random_real_vectors<range>(n_coefs, min_coord, max_coord);
    return std::make_shared<qugar::impl::MonomialsTP<dim, range>>(order, coefs);
  }
}

template<typename T> bool compare_values(const T &ref_value, const T &value, const qugar::Tolerance &tol)
{
  if constexpr (std::is_same_v<T, qugar::real>) {
    return (tol.equal(value, ref_value));
  } else {
    return (tol.coincident(ref_value, value));
  }
};

template<typename T, int dim>
bool compare_vectors(const qugar::Vector<T, dim> &ref_value,
  const qugar::Vector<T, dim> &value,
  const qugar::Tolerance &tol)
{
  for (int dir = 0; dir < dim; ++dir) {
    if (!compare_values(ref_value(dir), value(dir), tol)) {
      return false;
    }
  }
  return true;
};

#endif// QUGAR_TEST_BEZIER_TEST_UTILS_HPP