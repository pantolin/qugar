// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file test_impl_impl_poly_reparam_0.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Test 0 for Implicit Bezier polynomial reparameterization.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "reparam_test_utils.hpp"

#include <qugar/bbox.hpp>
#include <qugar/point.hpp>
#include <qugar/primitive_funcs_lib.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <memory>


// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier reparameterization for plane 2D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<2> origin{ 0.75, 0.65 };
  const Point<2> normal{ 1.0, 2.0 };
  const auto bzr = std::make_shared<qugar::impl::funcs::PlaneBzr<2>>(origin, normal);

  const int order{ 4 };

  const BoundBox<2> domain_01;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam_Bezier<2, false>(
    bzr, domain_01, order, 28, 32, 24, 13, 13, 267, 217, Point<2>{ 0.30714285714285711, 0.43214285714285716 });
  test_reparam_Bezier<2, true>(
    bzr, domain_01, order, 4, 4, 0, 1, 0, 1, 0, Point<2>{ 0.52499999999999991, 0.76250000000000007 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier reparameterization for sphere 2D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<2> center{ 0.5, 0.5 };
  const auto bzr = std::make_shared<qugar::impl::funcs::SphereBzr<2>>(0.45, center);

  const int order{ 8 };

  const BoundBox<2> domain_01;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam_Bezier<2, false>(bzr,
    domain_01,
    order,
#ifdef WITH_LAPACK
    57,
    64,
    24,
    24,
    34,
    1113,
    502,
    Point<2>{ 0.5, 0.55526315789473679 });
#else
    64,
    64,
    32,
    31,
    31,
    1333,
    686,
    Point<2>{ 0.49999999999999994, 0.50000000000000011 });
#endif

  test_reparam_Bezier<2, true>(bzr,
    domain_01,
    order,
#ifdef WITH_LAPACK
    15,
    16,
    0,
    6,
    0,
    51,
    0,
    Point<2>{ 0.5, 0.53000000000000003 });
#else
    16,
    16,
    0,
    7,
    0,
    58,
    0,
    Point<2>{ 0.5, 0.500000000000000111 });
#endif

  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier reparameterization for plane 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<3> origin{ 0.75, 0.75, 0.75 };
  const Point<3> normal{ 1.0, 2.0, 3.0 };
  const auto bzr = std::make_shared<qugar::impl::funcs::PlaneBzr<3>>(origin, normal);

  const int order{ 4 };

  const BoundBox<3> domain_01;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam_Bezier<3, false>(bzr,
    domain_01,
    order,
    112,
    128,
    68,
    55,
    55,
    4285,
    2538,
    Point<3>{ 0.50000000000000011, 0.49999999999999983, 0.45238095238095233 });
  test_reparam_Bezier<3, true>(
    bzr, domain_01, order, 16, 16, 16, 7, 7, 66, 78, Point<3>{ 0.5, 0.75, 0.83333333333333326 });
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace,readability-function-cognitive-complexity)
TEST_CASE("Bezier reparameterization for sphere 3D", "[impl]")
{
  using namespace qugar;
  using namespace qugar::impl;

  const Point<3> center{ 0.5, 0.5, 0.5 };
  const auto bzr = std::make_shared<qugar::impl::funcs::SphereBzr<3>>(0.45, center);

  const int order{ 8 };

  const BoundBox<3> domain_01;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  test_reparam_Bezier<3, false>(bzr,
    domain_01,
    order,
#ifdef WITH_LAPACK
    463,
    512,
    0,
    223,
    0,
    59871,
    0,
    Point<3>{ 0.50236724669387545, 0.5185225506305885, 0.5 });
#else
    470,
    512,
    0,
    231,
    0,
    62053,
    0,
    Point<3>{ 0.51410601560676028, 0.507529772076287689, 0.499999999999999944 });
#endif

  test_reparam_Bezier<3, true>(bzr,
    domain_01,
    order,
#ifdef WITH_LAPACK
    121,
    128,
    0,
    58,
    0,
    4166,
    0,
    Point<3>{ 0.50129402032971004, 0.51012507785355654, 0.5 });
#else
    122,
    128,
    0,
    60,
    0,
    4236,
    0,
    Point<3>{ 0.507763263858521552, 0.504144019760954132, 0.5 });
#endif

  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}
