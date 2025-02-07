// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file bezier_tp_utils.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tensor-product Bezier utils.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bezier_tp_utils.hpp>

#include <qugar/bezier_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/utils.hpp>

#include <algoim/binomial.hpp>
#include <algoim/interval.hpp>

#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace qugar::impl {


namespace alg = ::algoim;

template<int dim> using Interval = alg::Interval<dim>;

namespace {
  template<int range>
  std::shared_ptr<BezierTP<1, range>> product(const BezierTP<1, range> &lhs, const BezierTP<1, range> &rhs)
  {
    static const int dim = 1;

    using CoefsType = typename BezierTP<dim, range>::CoefsType;

    const auto compute_aux_vector = [](const auto &bezier, auto &vec) {
      vec.resize(bezier.get_num_coefs());

      const auto binomials_u =
        std::span(alg::Binomial::row(bezier.get_degree(0)), static_cast<std::size_t>(bezier.get_order(0)));

      std::transform(
        binomials_u.begin(), binomials_u.end(), bezier.get_coefs().cbegin(), vec.begin(), std::multiplies<CoefsType>());
    };

    std::vector<CoefsType> aux_lhs;
    std::vector<CoefsType> aux_rhs;
    compute_aux_vector(lhs, aux_lhs);
    compute_aux_vector(rhs, aux_rhs);

    TensorSizeTP<dim> new_order;
    for (int dir = 0; dir < dim; ++dir) {
      new_order(dir) = lhs.get_degree(dir) + rhs.get_degree(dir) + 1;
    }
    std::vector<CoefsType> new_coefs(static_cast<std::size_t>(new_order.size()), CoefsType(numbers::zero));

    // Note: this is the most expensive part of the algorithm.
    // It is a full 3D linear convolution.
    // Probably it can be speed up using FFT algorithms
    // (or AVX extensions?).

    auto lhs_it = aux_lhs.cbegin();
    for (int i = 0; i < lhs.get_order(0); ++i) {
      //   // Note: this is the most expensive part.
      const CoefsType &c0 = *lhs_it++;

      auto new_coef_i = new_coefs.begin() + i;
      for (const auto &rhs_coef : aux_rhs) {
        *new_coef_i++ += c0 * rhs_coef;
      }
    }// i

    const auto binomials_u = std::span(alg::Binomial::row(new_order(0) - 1), static_cast<std::size_t>(new_order(0)));

    auto new_coef = new_coefs.begin();
    for (const auto &bin_u : binomials_u) {
      *new_coef++ /= bin_u;
    }

    return std::make_shared<BezierTP<dim, range>>(new_order, new_coefs);
  }

  template<int range>
  std::shared_ptr<BezierTP<2, range>> product(const BezierTP<2, range> &lhs, const BezierTP<2, range> &rhs)
  {
    static const int dim = 2;

    using CoefsType = typename BezierTP<dim, range>::CoefsType;

    TensorSizeTP<dim> new_order;
    for (int dir = 0; dir < dim; ++dir) {
      new_order(dir) = lhs.get_degree(dir) + rhs.get_degree(dir) + 1;
    }

    const int o0 = new_order(0);
    const int o1 = new_order(1);

    std::vector<CoefsType> new_coefs(static_cast<std::size_t>(new_order.size()), CoefsType(numbers::zero));

    const auto compute_aux_vector = [](const auto &bezier, auto &vec) {
      const auto binomials_u =
        std::span(alg::Binomial::row(bezier.get_degree(0)), static_cast<std::size_t>(bezier.get_order(0)));

      const auto binomials_v =
        std::span(alg::Binomial::row(bezier.get_degree(1)), static_cast<std::size_t>(bezier.get_order(1)));

      const auto &coefs = bezier.get_coefs();

      vec.resize(bezier.get_num_coefs());
      auto vec_ij = vec.begin();

      // Lexicographical.
      for (int j = 0, ij = 0; j < bezier.get_order(1); ++j) {
        for (int i = 0; i < bezier.get_order(0); ++i, ++ij) {
          *vec_ij++ = at(coefs, ij) * at(binomials_u, i) * at(binomials_v, j);
        }
      }// j
    };

    std::vector<CoefsType> aux_lhs;
    std::vector<CoefsType> aux_rhs;
    compute_aux_vector(lhs, aux_lhs);
    compute_aux_vector(rhs, aux_rhs);

    auto lhs_it = aux_lhs.data();

    // Note: this is the most expensive part of the algorithm.
    // It is a full 3D linear convolution.
    // Probably it can be speed up using FFT algorithms
    // (or AVX extensions?).

    for (int j = 0; j < lhs.get_order(1); ++j) {
      for (int i = 0; i < lhs.get_order(0); ++i) {
        // Note: this is the most expensive part.
        const CoefsType &lhs_coef = *lhs_it++;
        auto rhs_it = aux_rhs.data();

        // NOLINTNEXTLINE (readability-identifier-length)
        for (int m = 0; m < rhs.get_order(1); ++m) {
          auto new_coef = new_coefs.begin() + ((j + m) * o0) + i;
          // NOLINTNEXTLINE (readability-identifier-length)
          for (int l = 0; l < rhs.get_order(0); ++l) {
            *new_coef++ += lhs_coef * *rhs_it++;
          }// l
        }// m
      }// i
    }// j

    const auto binomials_u = std::span(alg::Binomial::row(o0 - 1), static_cast<std::size_t>(o0));
    const auto binomials_v = std::span(alg::Binomial::row(o1 - 1), static_cast<std::size_t>(o1));

    // Lexicographical
    auto new_coef = new_coefs.begin();
    for (int j = 0; j < o1; ++j) {
      for (int i = 0; i < o0; ++i) {
        *new_coef++ /= at(binomials_u, i) * at(binomials_v, j);
      }// i
    }// j

    return std::make_shared<BezierTP<dim, range>>(new_order, new_coefs);
  }

  template<int range>
  // NOLINTNEXTLINE (readability-function-cognitive-complexity)
  std::shared_ptr<BezierTP<3, range>> product(const BezierTP<3, range> &lhs, const BezierTP<3, range> &rhs)
  {
    static const int dim = 3;

    using CoefsType = typename BezierTP<dim, range>::CoefsType;

    TensorSizeTP<dim> new_order;
    for (int dir = 0; dir < dim; ++dir) {
      new_order(dir) = lhs.get_degree(dir) + rhs.get_degree(dir) + 1;
    }

    const int o0 = new_order(0);
    const int o1 = new_order(1);
    const int o2 = new_order(2);

    std::vector<CoefsType> new_coefs(static_cast<std::size_t>(new_order.size()), CoefsType(numbers::zero));

    const auto compute_aux_vector = [](const auto &bezier, auto &vec) {
      const auto binomials_u =
        std::span(alg::Binomial::row(bezier.get_degree(0)), static_cast<std::size_t>(bezier.get_order(0)));
      const auto binomials_v =
        std::span(alg::Binomial::row(bezier.get_degree(1)), static_cast<std::size_t>(bezier.get_order(1)));
      const auto binomials_w =
        std::span(alg::Binomial::row(bezier.get_degree(2)), static_cast<std::size_t>(bezier.get_order(2)));

      auto coefs_ijk = bezier.get_coefs().cbegin();

      vec.resize(bezier.get_num_coefs());
      auto vec_ijk = vec.begin();

      for (int k = 0; k < bezier.get_order(2); ++k) {
        for (int j = 0; j < bezier.get_order(1); ++j) {
          for (int i = 0; i < bezier.get_order(0); ++i) {
            *vec_ijk++ = *coefs_ijk++ * at(binomials_u, i) * at(binomials_v, j) * at(binomials_w, k);
          }
        }// j
      }// k
    };

    std::vector<CoefsType> aux_lhs;
    std::vector<CoefsType> aux_rhs;

    compute_aux_vector(lhs, aux_lhs);
    compute_aux_vector(rhs, aux_rhs);

    auto lhs_it = aux_lhs.cbegin();

    // Note: this is the most expensive part of the algorithm.
    // It is a full 3D linear convolution.
    // Probably it can be speed up using FFT algorithms
    // (or AVX extensions?).

    for (int k = 0; k < lhs.get_order(2); ++k) {
      for (int j = 0; j < lhs.get_order(1); ++j) {
        for (int i = 0; i < lhs.get_order(0); ++i) {

          // Note: this is the most expensive part.
          const CoefsType &lhs_coef = *lhs_it++;
          auto rhs_it = aux_rhs.cbegin();

          // NOLINTNEXTLINE (readability-identifier-length)
          for (int n = 0; n < rhs.get_order(2); ++n) {
            // NOLINTNEXTLINE (readability-identifier-length)
            for (int m = 0; m < rhs.get_order(1); ++m) {
              auto new_coef = new_coefs.begin() + ((k + n) * o0) * o1 + ((j + m) * o0) + i;
              // NOLINTNEXTLINE (readability-identifier-length)
              for (int l = 0; l < rhs.get_order(0); ++l) {
                *new_coef++ += lhs_coef * *rhs_it++;
              }// l
            }// m
          }// n

        }// i
      }// j
    }// k

    const auto binomials_u = std::span(alg::Binomial::row(o0 - 1), static_cast<std::size_t>(o0));
    const auto binomials_v = std::span(alg::Binomial::row(o1 - 1), static_cast<std::size_t>(o1));
    const auto binomials_w = std::span(alg::Binomial::row(o2 - 1), static_cast<std::size_t>(o2));

    auto new_coef = new_coefs.begin();
    for (int k = 0; k < o2; ++k) {
      for (int j = 0; j < o1; ++j) {
        for (int i = 0; i < o0; ++i) {
          *new_coef++ /= at(binomials_u, i) * at(binomials_v, j) * at(binomials_w, k);
        }// i
      }// j
    }// k

    return std::make_shared<BezierTP<dim, range>>(new_order, new_coefs);
  }

}// namespace

template<typename T> void evaluate_Bernstein_value(const T &point, const int order, std::vector<T> &values)
{
  const auto binom = std::span(alg::Binomial::row(order - 1), static_cast<std::size_t>(order));

  const T one(numbers::one);

  T val{ one };
  auto value = values.begin();
  for (int i = 0; i < order; ++i) {
    *value++ = val * at(binom, i);
    val *= point;
  }

  val = one;
  auto value_r = values.begin() + order - 1;
  for (int i = 0; i < order; ++i) {
    *value_r-- *= val;
    val *= one - point;
  }
}

// NOLINTNEXTLINE (misc-no-recursion)
template<typename T> void evaluate_Bernstein(const T &point, const int order, int der, std::vector<T> &values)
{
  assert(der >= 0);
  assert(order > 0);

  if (der == 0) {
    evaluate_Bernstein_value(point, order, values);
  }

  else if (order <= der) {
    values.assign(static_cast<std::size_t>(order), T{ numbers::zero });
  }

  else {
    evaluate_Bernstein(point, order - 1, der - 1, values);

    const int degree = order - 1;

    auto value = values.begin() + order - 1;

    *value-- = degree * *std::prev(value);

    for (int i = degree - 1; i > 0; --i) {
      *value-- = degree * (*std::prev(value) - *value);
    }
    *value *= -degree;
  }
}

template<int dim, int range>
std::shared_ptr<BezierTP<dim, range>> Bezier_product(const BezierTP<dim, range> &lhs, const BezierTP<dim, range> &rhs)
{
  return product(lhs, rhs);
}

// Instantiations


template std::shared_ptr<BezierTP<1, 1>> Bezier_product(const BezierTP<1, 1> &, const BezierTP<1, 1> &);
template std::shared_ptr<BezierTP<1, 2>> Bezier_product(const BezierTP<1, 2> &, const BezierTP<1, 2> &);
template std::shared_ptr<BezierTP<1, 3>> Bezier_product(const BezierTP<1, 3> &, const BezierTP<1, 3> &);

template std::shared_ptr<BezierTP<2, 1>> Bezier_product(const BezierTP<2, 1> &, const BezierTP<2, 1> &);
template std::shared_ptr<BezierTP<2, 2>> Bezier_product(const BezierTP<2, 2> &, const BezierTP<2, 2> &);
template std::shared_ptr<BezierTP<2, 3>> Bezier_product(const BezierTP<2, 3> &, const BezierTP<2, 3> &);

template std::shared_ptr<BezierTP<3, 1>> Bezier_product(const BezierTP<3, 1> &, const BezierTP<3, 1> &);
template std::shared_ptr<BezierTP<3, 2>> Bezier_product(const BezierTP<3, 2> &, const BezierTP<3, 2> &);
template std::shared_ptr<BezierTP<3, 3>> Bezier_product(const BezierTP<3, 3> &, const BezierTP<3, 3> &);

template void evaluate_Bernstein_value(const real &, const int, std::vector<real> &);
template void evaluate_Bernstein_value(const Interval<1> &, const int, std::vector<Interval<1>> &);
template void evaluate_Bernstein_value(const Interval<2> &, const int, std::vector<Interval<2>> &);
template void evaluate_Bernstein_value(const Interval<3> &, const int, std::vector<Interval<3>> &);

template void evaluate_Bernstein(const real &, const int, int, std::vector<real> &);
template void evaluate_Bernstein(const Interval<1> &, const int, int, std::vector<Interval<1>> &);
template void evaluate_Bernstein(const Interval<2> &, const int, int, std::vector<Interval<2>> &);
template void evaluate_Bernstein(const Interval<3> &, const int, int, std::vector<Interval<3>> &);


}// namespace qugar::impl