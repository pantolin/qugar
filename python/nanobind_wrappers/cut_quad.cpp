// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file cut_quad.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of nanobind wrappers for quadrature related data structures.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/cut_quadrature.hpp>
#include <qugar/point.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include "wrapper_utils.hpp"

#include <memory>
#include <span>
#include <string>

#include <nanobind/nanobind.h>
// NOLINTBEGIN (misc-include-cleaner)
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
// NOLINTEND (misc-include-cleaner)

namespace nb = nanobind;

namespace qugar::wrappers {

namespace {


  // Declare class for cut quadratures.
  template<int dim> void declare_cut_quadratures(nb::module_ &module)
  {
    using CutCellsQuad = CutCellsQuad<dim>;


    const std::string pyclass_name_cut_cells{ std::string("CutCellsQuad_") + std::to_string(dim) + "D" };
    nb::class_<CutCellsQuad>(module, pyclass_name_cut_cells.c_str(), "CutCellsQuad object")
      .def_prop_ro(
        "cells",
        [](CutCellsQuad &quad) {
          using Array = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.cells.data(), { quad.cells.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Cell ids.")
      .def_prop_ro(
        "n_pts_per_entity",
        [](CutCellsQuad &quad) {
          using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.n_pts_per_cell.data(), { quad.n_pts_per_cell.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Number of quadratures points per cell.")
      .def_prop_ro(
        "points",
        [](CutCellsQuad &quad) { return get_points_array<dim>(quad.points); },
        nb::rv_policy::reference_internal,
        "Quadrature point coordinates.")
      .def_prop_ro(
        "weights",
        [](CutCellsQuad &quad) {
          using Array = nb::ndarray<const real, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.weights.data(), { quad.weights.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Quadrature weights.");

    using CutIsoBoundsQuad = CutIsoBoundsQuad<dim - 1>;

    const std::string pyclass_name_cut_iso_bounds_cells{ std::string("CutIsoBoundsQuad_") + std::to_string(dim - 1)
                                                         + "D" };
    nb::class_<CutIsoBoundsQuad>(module, pyclass_name_cut_iso_bounds_cells.c_str(), "CutIsoBoundsQuad object")
      .def_prop_ro(
        "cells",
        [](CutIsoBoundsQuad &quad) {
          using Array = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.cells.data(), { quad.cells.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Cell ids.")
      .def_prop_ro(
        "facets",
        [](CutIsoBoundsQuad &quad) {
          using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.local_facet_ids.data(), { quad.local_facet_ids.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Local facet ids (referred to the cells).")
      .def_prop_ro(
        "n_pts_per_entity",
        [](CutIsoBoundsQuad &quad) {
          using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.n_pts_per_facet.data(), { quad.n_pts_per_facet.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Number of quadratures points per facet.")
      .def_prop_ro(
        "points",
        [](CutIsoBoundsQuad &quad) { return get_points_array<dim - 1>(quad.points); },
        nb::rv_policy::reference_internal,
        "Quadrature point coordinates.")
      .def_prop_ro(
        "weights",
        [](CutIsoBoundsQuad &quad) {
          using Array = nb::ndarray<const real, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.weights.data(), { quad.weights.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Quadrature weights.");


    using CutUnfBoundsQuad = CutUnfBoundsQuad<dim>;

    const std::string pyclass_name_cut_unf_bounds{ std::string("CutUnfBoundsQuad_") + std::to_string(dim) + "D" };
    nb::class_<CutUnfBoundsQuad>(module, pyclass_name_cut_unf_bounds.c_str(), "CutUnfBoundsQuad object")
      .def_prop_ro(
        "cells",
        [](CutUnfBoundsQuad &quad) {
          using Array = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.cells.data(), { quad.cells.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Cell ids.")
      .def_prop_ro(
        "n_pts_per_entity",
        [](CutUnfBoundsQuad &quad) {
          using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.n_pts_per_cell.data(), { quad.n_pts_per_cell.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Number of quadratures points per cell.")
      .def_prop_ro(
        "points",
        [](CutUnfBoundsQuad &quad) { return get_points_array<dim>(quad.points); },
        nb::rv_policy::reference_internal,
        "Quadrature point coordinates.")
      .def_prop_ro(
        "weights",
        [](CutUnfBoundsQuad &quad) {
          using Array = nb::ndarray<const real, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(quad.weights.data(), { quad.weights.size() }, nb::handle());
        },
        nb::rv_policy::reference_internal,
        "Quadrature weights.")
      .def_prop_ro(
        "normals",
        [](CutUnfBoundsQuad &quad) { return get_points_array<dim>(quad.normals); },
        nb::rv_policy::reference_internal,
        "Unit normal vectors at the quadrature points.");
  }


  void declare_immersed_status(nb::module_ &module)
  {
    nb::enum_<ImmersedStatus>(module, "ImmersedStatus")
      .value("cut", ImmersedStatus::cut)
      .value("full", ImmersedStatus::full)
      .value("empty", ImmersedStatus::empty)
      .export_values();
  }

  template<int dim> void create_quadrature(nb::module_ &module)
  {
    using CellsArray = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>;
    using FacetsArray = nb::ndarray<const int, nb::numpy, nb::shape<-1>>;

    module.def(
      "create_quadrature",
      [](const UnfittedDomain<dim> &unf_domain, const CellsArray &cells_py, const int n_pts_dir) {
        const std::span<const std::int64_t> cells_span(cells_py.data(), cells_py.size());
        const std::vector<std::int64_t> cells(cells_span.begin(), cells_span.end());

        return create_quadrature<dim>(unf_domain, cells, n_pts_dir);
      },
      nb::arg("unf_domain"),
      nb::arg("cells"),
      nb::arg("n_pts_dir"));
  }

  template<int dim> void create_unfitted_bound_quadrature(nb::module_ &module)
  {
    using CellsArray = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>;

    module.def(
      "create_unfitted_bound_quadrature",
      [](const UnfittedDomain<dim> &unf_domain,
        const CellsArray &cells_py,
        const int n_pts_dir,
        const bool include_facet_unf_bdry,
        const bool exclude_ext_bdry) {
        const std::span<const std::int64_t> cells_span(cells_py.data(), cells_py.size());
        const std::vector<std::int64_t> cells(cells_span.begin(), cells_span.end());

        return create_unfitted_bound_quadrature<dim>(
          unf_domain, cells, n_pts_dir, include_facet_unf_bdry, exclude_ext_bdry);
      },
      nb::arg("unf_domain"),
      nb::arg("cells"),
      nb::arg("n_pts_dir"),
      nb::arg("include_facet_unf_bdry"),
      nb::arg("exclude_ext_bdry"));
  }


  template<int dim> void create_facets_quadrature(nb::module_ &module)
  {
    using CellsArray = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>;
    using FacetsArray = nb::ndarray<const int, nb::numpy, nb::shape<-1>>;

    module.def(
      "create_interior_facets_quadrature",
      [](const UnfittedDomain<dim> &unf_domain,
        const CellsArray &cells_py,
        const FacetsArray &facets_py,
        const int n_pts_dir) {
        const std::span<const std::int64_t> cells_span(cells_py.data(), cells_py.size());
        const std::vector<std::int64_t> cells(cells_span.begin(), cells_span.end());

        const std::span<const int> facets_span(facets_py.data(), facets_py.size());
        const std::vector<int> facets(facets_span.begin(), facets_span.end());

        return create_interior_facets_quadrature<dim>(unf_domain, cells, facets, n_pts_dir);
      },
      nb::arg("unf_domain"),
      nb::arg("cells"),
      nb::arg("facets"),
      nb::arg("n_pts_dir"));

    module.def(
      "create_exterior_facets_quadrature",
      [](const UnfittedDomain<dim> &unf_domain,
        const CellsArray &cells_py,
        const FacetsArray &facets_py,
        const int n_pts_dir) {
        const std::span<const std::int64_t> cells_span(cells_py.data(), cells_py.size());
        const std::vector<std::int64_t> cells(cells_span.begin(), cells_span.end());

        const std::span<const int> facets_span(facets_py.data(), facets_py.size());
        const std::vector<int> facets(facets_span.begin(), facets_span.end());

        return create_exterior_facets_quadrature<dim>(unf_domain, cells, facets, n_pts_dir);
      },
      nb::arg("unf_domain"),
      nb::arg("cells"),
      nb::arg("facets"),
      nb::arg("n_pts_dir"));
  }

}// namespace

// NOLINTNEXTLINE (misc-use-internal-linkage)
void cut_quad(nanobind::module_ &module)
{
  declare_immersed_status(module);

  declare_cut_quadratures<2>(module);
  declare_cut_quadratures<3>(module);

  create_quadrature<2>(module);
  create_quadrature<3>(module);

  create_unfitted_bound_quadrature<2>(module);
  create_unfitted_bound_quadrature<3>(module);

  create_facets_quadrature<2>(module);
  create_facets_quadrature<3>(module);
}


}// namespace qugar::wrappers
