// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_general_reparam_0.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 0 for Implicit general (non-polynomial) functions reparameterization.
//! @version 0.0.1
//! @date 2025-01-08
//!
//! @copyright Copyright (c) 2025-present

#include "reparam_test_utils.hpp"

#include <qugar/bbox.hpp>
#include <qugar/point.hpp>
#include <qugar/primitive_funcs_lib.hpp>
#include <qugar/types.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <memory>


// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("General reparameterization for plane 2D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  static const int dim = 2;

  const Point<dim> origin{ 0.5, 0.5 };
  const Point<dim> normal{ 2.0, 1.0 };
  const auto plane = std::make_shared<funcs::Plane<dim>>(origin, normal);

  const int order{ 4 };

  const BoundBox<dim> domain_01;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam_general<dim, false>(plane, domain_01, order, 16, 16, 16, 7, 7, 77, 78, Point<2>{ 0.25, 0.5 });
  test_reparam_general<dim, true>(plane, domain_01, order, 4, 4, 0, 1, 0, 3, 0, Point<2>{ 0.5, 0.5 });

  test_reparam_general_facet<dim>(plane, domain_01, 0, order, 4, 4, 0, 1, 0, 1, 0, Point<2>{ 0, 0.5 });
  test_reparam_general_facet<dim>(plane, domain_01, 1, order, 0, 0, 0, 0, 0, 0, 0, Point<2>{ 0, 0 });
  test_reparam_general_facet<dim>(plane, domain_01, 2, order, 4, 4, 0, 1, 0, 3, 0, Point<2>{ 0.375, 0 });
  test_reparam_general_facet<dim>(plane, domain_01, 3, order, 4, 4, 0, 1, 0, 1, 0, Point<2>{ 0.125, 1 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}


// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("General reparameterization for sphere 2D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  static const int dim = 2;

  const Point<dim> origin{ 0.5, 0.5 };
  const real radius{ 0.4 };
  const auto sphere = std::make_shared<funcs::Sphere<dim>>(radius, origin);

  const int order{ 4 };

  BoundBox<dim> domain;
  domain.set(origin, Point<dim>(1.0, 1.0));

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam_general<dim, false>(
    sphere, domain, order, 39, 48, 24, 19, 17, 584, 270, Point<2>{ 0.67963005297814727, 0.68418444051578464 });
  test_reparam_general<dim, true>(
    sphere, domain, order, 7, 8, 0, 3, 0, 12, 0, Point<2>{ 0.7325408618676954, 0.75419238430541147 });

  test_reparam_general_facet<dim>(sphere, domain, 0, order, 4, 4, 0, 1, 0, 1, 0, Point<2>{ 0.5, 0.69999999999999996 });
  test_reparam_general_facet<dim>(sphere, domain, 1, order, 0, 0, 0, 0, 0, 0, 0, Point<2>{ 0, 0 });
  test_reparam_general_facet<dim>(sphere, domain, 2, order, 4, 4, 0, 1, 0, 3, 0, Point<2>{ 0.69999999999999996, 0.5 });
  test_reparam_general_facet<dim>(sphere, domain, 3, order, 0, 0, 0, 0, 0, 0, 0, Point<2>{ 0, 0 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("General reparameterization for plane 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  static const int dim = 3;

  const Point<dim> origin{ 0.75, 0.75, 0.75 };
  const Point<dim> normal{ 1.0, 2.0, 3.0 };
  const auto plane = std::make_shared<funcs::Plane<dim>>(origin, normal);

  const int order{ 4 };

  const BoundBox<dim> domain_01;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam_general<dim, false>(plane,
    domain_01,
    order,
    112,
    128,
    68,
    55,
    55,
    3788,
    2523,
    Point<3>{ 0.50000000000000011, 0.49999999999999983, 0.45238095238095233 });
  test_reparam_general<dim, true>(
    plane, domain_01, order, 16, 16, 16, 7, 7, 66, 78, Point<3>{ 0.5, 0.75, 0.83333333333333326 });

  test_reparam_general_facet<dim>(
    plane, domain_01, 0, order, 28, 32, 24, 13, 13, 278, 215, Point<3>{ 0, 0.6071428571428571, 0.47619047619047628 });
  test_reparam_general_facet<dim>(
    plane, domain_01, 1, order, 28, 32, 24, 13, 13, 267, 217, Point<3>{ 1, 0.39285714285714285, 0.42857142857142855 });
  test_reparam_general_facet<dim>(plane, domain_01, 2, order, 16, 16, 16, 7, 7, 77, 78, Point<3>{ 0.5, 0, 0.5 });
  test_reparam_general_facet<dim>(
    plane, domain_01, 3, order, 16, 16, 16, 7, 7, 77, 78, Point<3>{ 0.5, 1, 0.33333333333333337 });
  test_reparam_general_facet<dim>(plane, domain_01, 4, order, 16, 16, 16, 7, 7, 66, 78, Point<3>{ 0.5, 0.5, 0 });
  test_reparam_general_facet<dim>(plane, domain_01, 5, order, 16, 16, 16, 7, 7, 66, 78, Point<3>{ 0.5, 0.25, 1 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}


// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("General reparameterization for sphere 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  static const int dim = 3;

  const Point<dim> origin{ 0.5, 0.5, 0.5 };
  const real radius{ 0.4 };
  const auto sphere = std::make_shared<funcs::Sphere<dim>>(radius, origin);


  const int order{ 4 };

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  BoundBox<dim> domain;
  domain.set(Point<dim>(0.2, 0.5, 0.5), Point<dim>(1.0, 1.0, 1.0));

  test_reparam_general<dim, false>(sphere,
    domain,
    order,
    802,
    1216,
    84,
    410,
    405,
    320771,
    22487,
    Point<3>{ 0.40438775251169379, 0.66607421497736186, 0.67406699390339997 });
  test_reparam_general<dim, true>(sphere,
    domain,
    order,
    158,
    208,
    48,
    79,
    76,
    10757,
    2366,
    Point<3>{ 0.43978666418277523, 0.70693181636921709, 0.71301704690085466 });

  test_reparam_general_facet<dim>(sphere,
    domain,
    0,
    order,
    83,
    112,
    40,
    44,
    45,
    3192,
    1170,
    Point<3>{ 0.19999999999999968, 0.61692424583772576, 0.62637687495600103 });
  test_reparam_general_facet<dim>(sphere, domain, 1, order, 0, 0, 0, 0, 0, 0, 0, Point<3>{ 0, 0, 0 });
  test_reparam_general_facet<dim>(
    sphere, domain, 2, order, 39, 48, 24, 19, 16, 581, 252, Point<3>{ 0.49662045825899526, 0.5, 0.68916423402871385 });
  test_reparam_general_facet<dim>(sphere, domain, 3, order, 0, 0, 0, 0, 0, 0, 0, Point<3>{ 0, 0, 0 });
  test_reparam_general_facet<dim>(
    sphere, domain, 4, order, 39, 48, 24, 19, 16, 581, 252, Point<3>{ 0.49662045825899526, 0.68916423402871385, 0.5 });
  test_reparam_general_facet<dim>(sphere, domain, 5, order, 0, 0, 0, 0, 0, 0, 0, Point<3>{ 0, 0, 0 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}