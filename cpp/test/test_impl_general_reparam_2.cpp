// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_general_reparam_1.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 0 for Implicit general (non-polynomial) functions reparameterization for grids.
//! @date 2025-01-15
//!
//! @copyright Copyright (c) 2025-present

#include "reparam_test_utils.hpp"

#include <qugar/bbox.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/point.hpp>
#include <qugar/primitive_funcs_lib.hpp>
#include <qugar/tpms_lib.hpp>
#include <qugar/types.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <memory>


// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Fischer Koch S reparameterization in 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  static const int dim = 3;

  const auto gyroid = std::make_shared<tpms::FischerKochS<3>>(qugar::Vector<real, dim>(2., 2., 2.));

  const int n_elems_dir = 2;
  const int order{ 3 };

  const BoundBox<dim> domain_01;


  const auto grid = std::make_shared<CartGridTP<dim>>(
    domain_01, std::array<std::size_t, 3>({ { n_elems_dir, n_elems_dir, n_elems_dir } }));

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam<dim, false>(gyroid,
    grid,
    order,
    780170,
    2054052,
    17784,
  // TODO: to be fixed.
#ifndef NDEBUG
    393229,
    393132,
    536207171676,
    4563175623,
#else
    393230,
    393172,
    536207183152,
    4563176610,
#endif
    Point<3>{ 0.500195469777085955, 0.500185229121587582, 0.500248055822314019 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}
