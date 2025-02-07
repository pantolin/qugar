// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file bezier_tp_utils_composition.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tensor-product Bezier composition class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bezier_tp_utils.hpp>

#include <qugar/bezier_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <algoim/binomial.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace qugar::impl {

namespace alg = ::algoim;

namespace {

  template<int dim, int range>
  std::shared_ptr<BezierTP<dim, 1>> extract_component(const BezierTP<dim, range> &bezier, const int comp_id)
  {
    if constexpr (range == 1) {
      return std::make_shared<BezierTP<dim, 1>>(bezier);
    } else {
      static_cast<void>(0 <= comp_id && comp_id < range);

      std::vector<real> coefs_0;
      coefs_0.reserve(bezier.get_num_coefs());
      for (const auto &coef : bezier.get_coefs()) {
        coefs_0.push_back(coef(comp_id));
      }

      return std::make_shared<BezierTP<dim, 1>>(bezier.get_order(), coefs_0);
    }
  }

  template<int dim> std::shared_ptr<BezierTP<dim, 1>> create_one_minus(const BezierTP<dim, 1> &bezier)
  {
    const auto one_minus = std::make_shared<BezierTP<dim, 1>>(bezier);
    one_minus->coefs_linear_transform(-numbers::one, numbers::one);
    return one_minus;
  }

  template<int dim>
  std::vector<std::shared_ptr<BezierTP<dim, 1>>>
    // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
    compute_powers(const BezierTP<dim, 1> &bezier, const int order)
  {
    assert(order > 1);

    const int degree = order - 1;

    std::vector<std::shared_ptr<BezierTP<dim, 1>>> powers;
    powers.reserve(static_cast<std::size_t>(degree));

    const auto power_0 = std::make_shared<BezierTP<dim, 1>>(bezier);
    powers.push_back(power_0);

    for (int i = 1; i < degree; ++i) {
      const auto &prev_power = *powers.back();
      powers.push_back(Bezier_product(*power_0, prev_power));
    }

    return powers;
  }


  template<int dim>
  std::vector<std::shared_ptr<BezierTP<dim, 1>>> compose_with_Bernstein_bases_1D(const BezierTP<dim, 1> &bezier,
    const int order)
  {
    assert(order > 0);

    std::vector<std::shared_ptr<BezierTP<dim, 1>>> compositions;
    compositions.clear();
    compositions.reserve(static_cast<std::size_t>(order));

    if (order > 1) {
      const auto one_minus = create_one_minus(bezier);
      const auto powers_0 = compute_powers(bezier, order);
      const auto powers_1 = compute_powers(*one_minus, order);

      const int degree = order - 1;
      const auto binomials = std::span(alg::Binomial::row(degree), static_cast<std::size_t>(order));

      compositions.push_back(powers_1.back());
      for (int i = 1; i < degree; ++i) {
        const auto comp_i = *at(powers_0, i - 1) * *at(powers_1, degree - 1 - i);

        const auto &bin_i = at(binomials, i);
        comp_i->coefs_linear_transform(bin_i, numbers::zero);

        compositions.push_back(comp_i);
      }
      compositions.push_back(powers_0.back());

    } else {// order == 1
      const auto ones_bzr = std::make_shared<BezierTP<dim, 1>>(TensorSizeTP<dim>(1), std::vector<real>{ numbers::one });
      compositions.push_back(ones_bzr);
    }

    return compositions;
  }


}// namespace

template<int dim, int range, int dim2>
// NOLINTNEXTLINE (readability-function-cognitive-complexity)
std::shared_ptr<BezierTP<dim, range>> Bezier_composition(const BezierTP<dim2, range> &lhs,
  const BezierTP<dim, dim2> &rhs)
{
  static_assert(1 <= dim && dim <= 3);
  static_assert(1 <= dim2 && dim2 <= 3);
  static_assert(1 <= range && range <= 3);

  using Bases = std::vector<std::shared_ptr<BezierTP<dim, 1>>>;
  std::array<Bases, dim2> bases{};
  for (int dir2 = 0; dir2 < dim2; ++dir2) {
    const auto rhs_comp = extract_component(rhs, dir2);
    at(bases, dir2) = compose_with_Bernstein_bases_1D(*rhs_comp, lhs.get_order(dir2));
  }

  TensorSizeTP<dim> comp_order;
  for (int dir = 0; dir < dim; ++dir) {
    comp_order(dir) = 1;
    for (const auto &basis : bases) {
      comp_order(dir) += basis.front()->get_degree(dir);
    }
  }
  using CoefsType = typename BezierTP<dim, range>::CoefsType;
  std::vector<CoefsType> comp_coefs(static_cast<std::size_t>(comp_order.size()), CoefsType{ numbers::zero });

  auto coef_lhs_it = lhs.get_coefs().cbegin();

  const auto compute = [&coef_lhs_it, &comp_coefs](const auto &basis) {
    const auto &coef_lhs = *coef_lhs_it++;
    auto comp_coef = comp_coefs.begin();
    for (const auto &basis_coef : basis->get_coefs()) {
      *comp_coef++ += basis_coef * coef_lhs;
    }
  };

  if constexpr (dim2 == 1) {
    for (const auto &basis_u : at(bases, 0)) {
      compute(basis_u);
    }
  } else if constexpr (dim2 == 2) {
    for (const auto &basis_v : at(bases, 1)) {
      for (const auto &basis_u : at(bases, 0)) {
        const auto basis_uv = *basis_u * *basis_v;
        compute(basis_uv);
      }
    }
  } else {// if constexpr (dim2 == 3)
    for (const auto &basis_w : at(bases, 2)) {
      for (const auto &basis_v : at(bases, 1)) {
        const auto basis_vw = *basis_w * *basis_v;
        for (const auto &basis_u : at(bases, 0)) {
          const auto basis_uvw = *basis_u * *basis_vw;
          compute(basis_uvw);
        }
      }
    }
  }

  return std::make_shared<BezierTP<dim, range>>(comp_order, comp_coefs);
}

// Instantiations


template std::shared_ptr<BezierTP<1, 1>> Bezier_composition(const BezierTP<1, 1> &, const BezierTP<1, 1> &);
template std::shared_ptr<BezierTP<1, 1>> Bezier_composition(const BezierTP<2, 1> &, const BezierTP<1, 2> &);
template std::shared_ptr<BezierTP<1, 1>> Bezier_composition(const BezierTP<3, 1> &, const BezierTP<1, 3> &);

template std::shared_ptr<BezierTP<1, 2>> Bezier_composition(const BezierTP<1, 2> &, const BezierTP<1, 1> &);
template std::shared_ptr<BezierTP<1, 2>> Bezier_composition(const BezierTP<2, 2> &, const BezierTP<1, 2> &);
template std::shared_ptr<BezierTP<1, 2>> Bezier_composition(const BezierTP<3, 2> &, const BezierTP<1, 3> &);

template std::shared_ptr<BezierTP<1, 3>> Bezier_composition(const BezierTP<1, 3> &, const BezierTP<1, 1> &);
template std::shared_ptr<BezierTP<1, 3>> Bezier_composition(const BezierTP<2, 3> &, const BezierTP<1, 2> &);
template std::shared_ptr<BezierTP<1, 3>> Bezier_composition(const BezierTP<3, 3> &, const BezierTP<1, 3> &);

template std::shared_ptr<BezierTP<2, 1>> Bezier_composition(const BezierTP<1, 1> &, const BezierTP<2, 1> &);
template std::shared_ptr<BezierTP<2, 1>> Bezier_composition(const BezierTP<2, 1> &, const BezierTP<2, 2> &);
template std::shared_ptr<BezierTP<2, 1>> Bezier_composition(const BezierTP<3, 1> &, const BezierTP<2, 3> &);

template std::shared_ptr<BezierTP<2, 2>> Bezier_composition(const BezierTP<1, 2> &, const BezierTP<2, 1> &);
template std::shared_ptr<BezierTP<2, 2>> Bezier_composition(const BezierTP<2, 2> &, const BezierTP<2, 2> &);
template std::shared_ptr<BezierTP<2, 2>> Bezier_composition(const BezierTP<3, 2> &, const BezierTP<2, 3> &);

template std::shared_ptr<BezierTP<2, 3>> Bezier_composition(const BezierTP<1, 3> &, const BezierTP<2, 1> &);
template std::shared_ptr<BezierTP<2, 3>> Bezier_composition(const BezierTP<2, 3> &, const BezierTP<2, 2> &);
template std::shared_ptr<BezierTP<2, 3>> Bezier_composition(const BezierTP<3, 3> &, const BezierTP<2, 3> &);

template std::shared_ptr<BezierTP<3, 1>> Bezier_composition(const BezierTP<1, 1> &, const BezierTP<3, 1> &);
template std::shared_ptr<BezierTP<3, 1>> Bezier_composition(const BezierTP<2, 1> &, const BezierTP<3, 2> &);
template std::shared_ptr<BezierTP<3, 1>> Bezier_composition(const BezierTP<3, 1> &, const BezierTP<3, 3> &);

template std::shared_ptr<BezierTP<3, 2>> Bezier_composition(const BezierTP<1, 2> &, const BezierTP<3, 1> &);
template std::shared_ptr<BezierTP<3, 2>> Bezier_composition(const BezierTP<2, 2> &, const BezierTP<3, 2> &);
template std::shared_ptr<BezierTP<3, 2>> Bezier_composition(const BezierTP<3, 2> &, const BezierTP<3, 3> &);

template std::shared_ptr<BezierTP<3, 3>> Bezier_composition(const BezierTP<1, 3> &, const BezierTP<3, 1> &);
template std::shared_ptr<BezierTP<3, 3>> Bezier_composition(const BezierTP<2, 3> &, const BezierTP<3, 2> &);
template std::shared_ptr<BezierTP<3, 3>> Bezier_composition(const BezierTP<3, 3> &, const BezierTP<3, 3> &);

}// namespace qugar::impl