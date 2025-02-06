// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file quad.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of nanobind wrappers for quadrature related data structures.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "wrapper_utils.hpp"

#include <qugar/quadrature.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// NOLINTBEGIN (misc-include-cleaner)
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
// NOLINTEND (misc-include-cleaner)

namespace nb = nanobind;

namespace qugar::wrappers {

namespace {

  // Declare class for quadratures.
  template<int dim> void declare_quadratures(nb::module_ &module)
  {
    using Quad = Quadrature<dim>;

    const std::string pyclass_name_cut_cells{ std::string("Quad_") + std::to_string(dim) + "D" };
    nb::class_<Quad>(module, pyclass_name_cut_cells.c_str(), "Quadrature object")
      .def_prop_ro(
        "points",
        [](Quad &quad) -> qugar::wrappers::npPointConstArray<dim> { return get_points_array<dim>(quad.points()); },
        nb::rv_policy::reference_internal,
        "Quadrature point coordinates.")
      .def_prop_ro(
        "weights",
        [](Quad &quad) {
          using Array = nb::ndarray<const real, nb::numpy, nb::shape<-1>, nb::c_contig>;
          const auto &weights = quad.weights();
          return Array(weights.data(), { weights.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Quadrature weights.");

    module.def(
      "create_Gauss_quad_01",
      [](const std::array<int, dim> &n_pts_dir) { return Quad::create_Gauss_01(n_pts_dir); },
      nb::arg("n_pts_dir"));
  }

}// namespace

// NOLINTNEXTLINE (misc-use-internal-linkage)
void quad(nanobind::module_ &module)
{
  declare_quadratures<1>(module);
  declare_quadratures<2>(module);
  declare_quadratures<3>(module);
}


}// namespace qugar::wrappers
