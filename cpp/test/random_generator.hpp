// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_TEST_RANDOM_GENERATOR_HPP
#define QUGAR_TEST_RANDOM_GENERATOR_HPP

//! @file random_generator.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Definition of random number generators for test purposes.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <catch2/catch_get_random_seed.hpp>

#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <cassert>
#include <memory>
#include <random>

namespace qugar::rand {

// NOLINTNEXTLINE (cert-err58-cpp)
static thread_local std::mt19937 rand_gen(Catch::getSeed());

std::function<qugar::real(void)> create_real_uniform_values_generator(const qugar::real lower, const qugar::real upper)
{
  assert(lower <= upper);
  const auto distr = std::make_shared<std::uniform_real_distribution<qugar::real>>(lower, upper);
  return [distr]() { return distr->operator()(rand_gen); };
}

std::function<int(void)> create_int_uniform_values_generator(const int lower, const int upper)
{
  assert(lower <= upper);
  const auto distr = std::make_shared<std::uniform_int_distribution<int>>(lower, upper);
  return [distr]() { return distr->operator()(rand_gen); };
}


//! @brief Create a real vector with random coordinates
//! in the interval defined by @p min_coord and @p max_coord.
//!
//! @tparam dim Dimension of the vector.
//! @param min_coord Lower bound for the vector coordinates.
//! @param max_coord Upper bound for the vector coordinates.
//! @return Generated vector.
template<int dim>
Vector<real, dim> create_random_real_vector(const real min_coord = numbers::zero, const real max_coord = numbers::one)
{
  const auto generator = create_real_uniform_values_generator(min_coord, max_coord);
  assert(min_coord <= max_coord);
  if constexpr (dim == 1) {
    return Vector<real, 1>(generator());
  } else if constexpr (dim == 2) {
    return Vector<real, 2>(generator(), generator());
  } else {// if constexpr (dim == 3) {
    static_assert(dim == 3, "Invalid dimension.");
    return Vector<real, 3>(generator(), generator(), generator());
  }
}


//! @brief Create a vector of random real vectors.
//!
//! @tparam dim Dimension of the points.
//! @param n_vectors Number of vectors to generate.
//! @param min_coord Lower bound coordinates.
//! @param max_coord Upper bound coordinates.
//! @return Generated vectors.
template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
std::vector<qugar::Vector<qugar::real, dim>>
  create_random_real_vectors(const std::size_t n_vectors, const Point<dim> &min_coords, const Point<dim> &max_coords)
{

  std::vector<qugar::Vector<qugar::real, dim>> vectors;
  vectors.reserve(n_vectors);

  std::array<std::function<qugar::real(void)>, dim> generators;
  for (int dir = 0; dir < dim; ++dir) {
    assert(min_coords(dir) <= max_coords(dir));
    qugar::at(generators, dir) = create_real_uniform_values_generator(min_coords(dir), max_coords(dir));
  }

  for (std::size_t i = 0; i < n_vectors; ++i) {
    if constexpr (dim == 1) {
      vectors.emplace_back(generators[0]());
    } else if constexpr (dim == 2) {
      vectors.emplace_back(generators[0](), generators[1]());
    } else {// if constexpr (dim == 3) {
      static_assert(dim == 3, "Invalid dimension.");
      vectors.emplace_back(generators[0](), generators[1](), generators[2]());
    }
  }

  return vectors;
}

//! @brief Create a vector of random real vectors.
//!
//! @tparam dim Dimension of the points.
//! @param n_vectors Number of vectors to generate.
//! @param min_coord Lower bound for the point coordinates (constant for all point coordinate)
//! @param max_coord Upper bound for the point coordinates (constant for all point coordinate)
//! @return Generated vectors.
template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
std::vector<qugar::Vector<qugar::real, dim>> create_random_real_vectors(const std::size_t n_vectors,
  const real min_coord = numbers::zero,
  const real max_coord = numbers::one)
{
  return create_random_real_vectors(n_vectors, Point<dim>(min_coord), Point<dim>(max_coord));
}


//! @brief Create a point with random coordinates
//! in the interval defined by @p min_coord and @p max_coord.
//!
//! @tparam dim Dimension of the point.
//! @param min_coord Lower bound for the point coordinates.
//! @param max_coord Upper bound for the point coordinates.
//! @return Generated point.
template<int dim>
Point<dim> create_random_point(const real min_coord = numbers::zero, const real max_coord = numbers::one)
{
  return create_random_real_vectors<dim>(1, min_coord, max_coord).front();
}

//! @brief Create a vector of random values.
//!
//! @param n_vals Number of values to generate.
//! @param lower Lower bound for the values.
//! @param upper Upper bound for the values.
//! @return Generated values.
std::vector<real>
  // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
  create_random_values(const std::size_t n_vals, const real lower = numbers::zero, const real upper = numbers::one)
{
  const auto generator = create_real_uniform_values_generator(lower, upper);

  std::vector<real> values(n_vals);
  std::generate(values.begin(), values.end(), generator);

  return values;
}


//! @brief Create a tensor sized constainer with random values.
//!
//! @tparam dim Dimension of the container.
//! @param lower Lower bound (included) for the values in the tensor.
//! @param upper Upper bound (included) for the values in the tensor.
//! @param avoid_repetitions If True, no values in the tensor present
//! no repetitions.
//! @return Generated random tensor.
template<int dim>
TensorSizeTP<dim> create_random_tensor_size(const int lower, const int upper, const bool avoid_repetitions)
{
  TensorSizeTP<dim> vals;
  if (avoid_repetitions) {
    assert((upper - lower + 1) >= dim);

    // Create a sequence of numbers in the interval [lower, upper]
    std::vector<int> numbers;
    for (int i = lower; i <= upper; ++i) {
      numbers.push_back(i);
    }
    std::shuffle(numbers.begin(), numbers.end(), rand_gen);
    // gcc does not understand that numbers size is greater or equal than dim
#pragma GCC diagnostic ignored "-Wnull-dereference"
    std::copy_n(numbers.cbegin(), dim, vals.data());
  } else {
    const auto generator = create_real_uniform_values_generator(lower, upper);
    std::generate(vals.data(), vals.data() + dim, generator);
  }
  return vals;
}


}// namespace qugar::rand


#endif// QUGAR_TEST_RANDOM_GENERATOR_HPP