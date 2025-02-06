// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file lagrange_tp_utils.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tensor-product Lagrange utils.
//! @version 0.0.2
//! @date 2025-01-10
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/lagrange_tp_utils.hpp>

#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/bernstein.hpp>

#include <cassert>
#include <cstddef>
#include <functional>
#include <vector>


namespace qugar::impl {


// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
void evaluate_Lagrange_basis_1D(const real point, const int order, const bool chebyshev, std::vector<real> &values)
{
  assert(1 < order);

  values.clear();
  values.reserve(static_cast<std::size_t>(order));

  std::function<real(const int)> get_coord;
  if (chebyshev) {
    get_coord = [order](const int ind) -> real { return static_cast<real>(ind) / (order - 1); };
  } else {
    get_coord = [order](const int ind) -> real { return ::algoim::bernstein::modifiedChebyshevNode(ind, order); };
  }

  for (int i = 0; i < order; ++i) {
    real product = numbers::one;
    for (int j = 0; j < order; ++j) {
      if (i != j) {
        product *= (point - get_coord(j)) / (get_coord(i) - get_coord(j));
      }
    }
    values.push_back(product);
  }
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
void evaluate_Lagrange_basis_der_1D(const real point, const int order, const bool chebyshev, std::vector<real> &values)
{
  assert(1 < order);

  values.clear();
  values.reserve(static_cast<std::size_t>(order));

  std::function<real(const int)> get_coord;
  if (chebyshev) {
    get_coord = [order](const int ind) -> real { return static_cast<real>(ind) / (order - 1); };
  } else {
    get_coord = [order](const int ind) -> real { return ::algoim::bernstein::modifiedChebyshevNode(ind, order); };
  }

  for (int i = 0; i < order; ++i) {
    real product = numbers::zero;
    for (int j = 0; j < order; ++j) {
      if (i != j) {
        real term = numbers::one / (get_coord(i) - get_coord(j));
        for (int k = 0; k < order; ++k) {
          if (k != i && k != j) {
            term *= (point - get_coord(k)) / (get_coord(i) - get_coord(k));
          }
        }
        product += term;
      }
    }
    values.push_back(product);
  }
}


template<int dim>
void evaluate_Lagrange_basis(const Point<dim> &point,
  const TensorSizeTP<dim> &order,
  const bool chebyshev,
  std::vector<real> &basis)
{
  Vector<std::vector<real>, dim> basis_dir;
  for (int dir = 0; dir < dim; ++dir) {
    evaluate_Lagrange_basis_1D(point(dir), order(dir), chebyshev, basis_dir(dir));
  }

  basis.resize(static_cast<std::size_t>(TensorSizeTP<dim>(order).size()));
  for (const auto &tid : TensorIndexRangeTP<dim>(order)) {

    auto &val = at(basis, tid.flat(order));
    val = numbers::one;

    for (int dir = 0; dir < dim; ++dir) {
      const auto &basis_dir_i = basis_dir(dir);
      val *= at(basis_dir_i, tid(dir));
    }
  }
}

template<int dim>
void evaluate_Lagrange_derivative(const Point<dim> &point,
  const TensorSizeTP<dim> &order,
  const bool chebyshev,
  Vector<std::vector<real>, dim> &basis_ders)
{
  Vector<std::vector<real>, dim> basis_dir;
  Vector<std::vector<real>, dim> basis_der_dir;
  for (int dir = 0; dir < dim; ++dir) {
    evaluate_Lagrange_basis_1D(point(dir), order(dir), chebyshev, basis_dir(dir));
    evaluate_Lagrange_basis_der_1D(point(dir), order(dir), chebyshev, basis_der_dir(dir));
  }

  for (int dir_der = 0; dir_der < dim; ++dir_der) {
    auto &output = basis_ders(dir_der);
    output.resize(static_cast<std::size_t>(TensorSizeTP<dim>(order).size()));

    for (const auto &tid : TensorIndexRangeTP<dim>(order)) {

      auto &val = at(output, tid.flat(order));
      val = numbers::one;

      for (int dir = 0; dir < dim; ++dir) {
        if (dir == dir_der) {
          const auto &basis_dir_i = basis_der_dir(dir);
          val *= at(basis_dir_i, tid(dir));
        } else {
          const auto &basis_dir_i = basis_dir(dir);
          val *= at(basis_dir_i, tid(dir));
        }
      }
    }
  }
}

// Instantiations

template void evaluate_Lagrange_basis<1>(const Point<1> &, const TensorSizeTP<1> &, const bool, std::vector<real> &);
template void evaluate_Lagrange_basis<2>(const Point<2> &, const TensorSizeTP<2> &, const bool, std::vector<real> &);
template void evaluate_Lagrange_basis<3>(const Point<3> &, const TensorSizeTP<3> &, const bool, std::vector<real> &);

template void
  evaluate_Lagrange_derivative(const Point<1> &, const TensorSizeTP<1> &, const bool, Vector<std::vector<real>, 1> &);
template void
  evaluate_Lagrange_derivative(const Point<2> &, const TensorSizeTP<2> &, const bool, Vector<std::vector<real>, 2> &);
template void
  evaluate_Lagrange_derivative(const Point<3> &, const TensorSizeTP<3> &, const bool, Vector<std::vector<real>, 3> &);


}// namespace qugar::impl