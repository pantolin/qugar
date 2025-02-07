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


// // NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
// TEST_CASE("General reparameterization for plane 2D in a grid", "[impl]")
// {
//   using namespace qugar;
//   using namespace qugar::impl;

//   static const int dim = 2;

//   const Point<dim> origin{ 0.5, 0.5 };
//   const Point<dim> normal{ 2.0, 1.0 };
//   const auto plane = std::make_shared<funcs::Plane<dim>>(origin, normal);

//   const int n_elems_dir = 4;
//   const int order{ 4 };

//   const BoundBox<dim> domain_01;

//   const auto grid =
//     std::make_shared<CartGridTP<dim>>(domain_01, std::array<std::size_t, 2>({ { n_elems_dir, n_elems_dir } }));

//   // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
//   test_reparam<dim, false>(
//     plane, grid, order, 106, 160, 100, 54, 53, 5297, 3333, Point<2>{ 0.29245283018867929, 0.43632075471698106 });
//   test_reparam<dim, true>(plane, grid, order, 13, 16, 0, 6, 0, 61, 0, Point<2>{ 0.5, 0.5 });
//   // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
// }


// // NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
// TEST_CASE("General reparameterization for sphere 2D in a grid", "[impl]")
// {
//   using namespace qugar;
//   using namespace qugar::impl;

//   static const int dim = 2;

//   const Point<dim> origin{ 0.5, 0.5 };
//   const real radius{ 0.4 };
//   const auto sphere = std::make_shared<funcs::Sphere<dim>>(radius, origin);

//   const int n_elems_dir = 4;
//   const int order{ 4 };

//   BoundBox<dim> domain;
//   domain.set(origin, Point<dim>(1.0, 1.0));

//   const auto grid =
//     std::make_shared<CartGridTP<dim>>(domain, std::array<std::size_t, 2>({ { n_elems_dir, n_elems_dir } }));

//   // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
//   test_reparam<dim, false>(
//     sphere, grid, order, 161, 240, 148, 79, 81, 12120, 7504, Point<2>{ 0.69809115940959476, 0.69840201181060246 });
//   test_reparam<dim, true>(
//     sphere, grid, order, 22, 28, 0, 10, 0, 184, 0, Point<2>{ 0.7531802200751827, 0.75290288681705375 });
//   // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
// }

// // NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
// TEST_CASE("General reparameterization for plane 3D in a grid", "[impl]")
// {
//   using namespace qugar;
//   using namespace qugar::impl;

//   static const int dim = 3;

//   const Point<dim> origin{ 0.75, 0.75, 0.75 };
//   const Point<dim> normal{ 1.0, 2.0, 3.0 };
//   const auto plane = std::make_shared<funcs::Plane<dim>>(origin, normal);

//   const int n_elems_dir = 4;
//   const int order{ 4 };

//   const BoundBox<dim> domain_01;

//   const auto grid = std::make_shared<CartGridTP<dim>>(
//     domain_01, std::array<std::size_t, 3>({ { n_elems_dir, n_elems_dir, n_elems_dir } }));

//   // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
//   test_reparam<dim, false>(plane,
//     grid,
//     order,
//     2365,
//     4480,
//     1224,
//     1180,
//     1198,
//     3298920,
//     878133,
//     Point<3>{ 0.49887244538407344, 0.49998238195912598, 0.50408738548273424 });
//   test_reparam<dim, true>(plane,
//     grid,
//     order,
//     130,
//     208,
//     120,
//     71,
//     69,
//     8785,
//     4956,
//     Point<3>{ 0.5807692307692307, 0.72403846153846152, 0.82371794871794835 });
//   // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
// }


// // NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
// TEST_CASE("General reparameterization for sphere 3D in a grid", "[impl]")
// {
//   using namespace qugar;
//   using namespace qugar::impl;

//   static const int dim = 3;

//   const Point<dim> origin{ 0.5, 0.5, 0.5 };
//   const real radius{ 0.4 };
//   const auto sphere = std::make_shared<funcs::Sphere<dim>>(radius, origin);

//   const int n_elems_dir = 4;
//   const int order{ 4 };

//   // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
//   BoundBox<dim> domain;
//   domain.set(Point<dim>(0.2, 0.5, 0.5), Point<dim>(1.0, 1.0, 1.0));

//   const auto grid = std::make_shared<CartGridTP<dim>>(
//     domain, std::array<std::size_t, 3>({ { n_elems_dir, n_elems_dir, n_elems_dir } }));

//   test_reparam<dim, false>(sphere,
//     grid,
//     order,
//     3560,
//     6208,
//     1344,
//     1814,
//     1795,
//     7239487,
//     1505827,
//     Point<3>{ 0.52116600022550597, 0.68439215188013247, 0.68461205493195154 });
//   test_reparam<dim, true>(sphere,
//     grid,
//     order,
//     654,
//     1008,
//     512,
//     328,
//     313,
//     215946,
//     104707,
//     Point<3>{ 0.5401765755373088, 0.70911875389866863, 0.70914108224542027 });
//   // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
// }

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("General reparameterization for gyroid 3D in a grid", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  static const int dim = 3;

  const auto gyroid = std::make_shared<tpms::Schoen<3>>(qugar::Vector<real, dim>(2., 2., 2.));

  const int n_elems_dir = 5;
  const int order{ 4 };

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const BoundBox<dim> domain_01;

  const auto grid = std::make_shared<CartGridTP<dim>>(
    domain_01, std::array<std::size_t, 3>({ { n_elems_dir, n_elems_dir, n_elems_dir } }));

  test_reparam<dim, false>(gyroid,
    grid,
    order,
    275728,
    470720,
    18928,
    137480,
  // TODO: to be fixed.
#ifndef NDEBUG
    138821,
    43051359976,
    1737929347,
#else
    138822,
    43051360339,
    1737930621,
#endif
    Point<3>{ 0.49999581355805495, 0.49994953471206927, 0.50034259284691829 });
  test_reparam<dim, true>(gyroid,
    grid,
    order,
    41450,
    61168,
    11712,
    20714,
  // TODO: to be fixed.
#ifndef NDEBUG
    20548,
    842939545,
    159279841,
#else
    20549,
    842940515,
    159280812,
#endif
    Point<3>{ 0.50267206966729439, 0.5048150852384049, 0.5032492131051628 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}