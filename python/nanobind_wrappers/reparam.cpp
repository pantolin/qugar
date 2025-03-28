// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file reparam.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of nanobind wrappers for generation of reparameterizations.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "wrapper_utils.hpp"

#include <qugar/reparam_mesh.hpp>

#include <cassert>
#include <span>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// NOLINTBEGIN (misc-include-cleaner)
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
// NOLINTEND (misc-include-cleaner)

namespace nb = nanobind;

namespace qugar::wrappers {

namespace {

  auto get_conn_array(const std::vector<std::size_t> &conn, const std::size_t n_pts_per_cell)
  {
    using ConnArray = nb::ndarray<const std::size_t, nb::numpy, nb::shape<-1, -1>, nb::c_contig>;
    if (conn.empty()) {
      return ConnArray(nullptr, { 0, n_pts_per_cell }, nb::handle());
    } else {
      assert(conn.size() % n_pts_per_cell == 0);
      const auto n_cells = conn.size() / n_pts_per_cell;
      return ConnArray(conn.data(), { n_cells, n_pts_per_cell }, nb::handle());
    }
  }

  template<int dim, int range> void declare_reparam_mesh(nb::module_ &module)
  {
    static_cast<void>(module);

    using ReparamMesh = qugar::ReparamMesh<dim, range>;

    const std::string pyclass_name_cut_cells{ std::string("ReparamMesh_") + std::to_string(dim) + "_"
                                              + std::to_string(range) };
    nb::class_<ReparamMesh>(module, pyclass_name_cut_cells.c_str(), "ReparamMesh object")
      .def_prop_ro(
        "dim",
        [](ReparamMesh & /*mesh*/) { return dim; },
        nb::rv_policy::reference_internal,
        "Mesh parametric dimension.")
      .def_prop_ro(
        "range",
        [](ReparamMesh & /*mesh*/) { return range; },
        nb::rv_policy::reference_internal,
        "Mesh geometric dimension.")
      .def_prop_ro(
        "order", [](ReparamMesh &mesh) { return mesh.get_order(); }, nb::rv_policy::reference_internal, "Mesh order.")
      .def_prop_ro(
        "chebyshev",
        [](ReparamMesh &mesh) { return mesh.use_Chebyshev(); },
        nb::rv_policy::reference_internal,
        "Whether Chebyshev nodes are used.")
      .def_prop_ro(
        "points",
        [](ReparamMesh &mesh) { return get_points_array<range>(mesh.get_points()); },
        nb::rv_policy::reference_internal,
        "Mesh points.")
      .def_prop_ro(
        "cells_conn",
        [](ReparamMesh &mesh) { return get_conn_array(mesh.get_connectivity(), mesh.get_num_points_per_cell()); },
        nb::rv_policy::reference_internal,
        "Mesh cells connectivity.")
      .def_prop_ro(
        "wirebasket_conn",
        [](ReparamMesh &mesh) {
          return get_conn_array(mesh.get_wires_connectivity(), static_cast<std::size_t>(mesh.get_order()));
        },
        nb::rv_policy::reference_internal,
        "Wirebasket connectivity.");
  }

  template<int dim> void create_reparam(nb::module_ &module)
  {
    using CellsArray = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>;

    module.def(
      "create_reparameterization",
      [](const qugar::UnfittedDomain<dim> &unf_domain, const CellsArray &cells_py, const int n_pts_dir) {
        const std::span<const std::int64_t> cells_span(cells_py.data(), cells_py.size());
        const std::vector<std::int64_t> cells(cells_span.begin(), cells_span.end());

        return qugar::create_reparameterization<dim, false>(unf_domain, cells, n_pts_dir);
      },
      nb::arg("unf_domain"),
      nb::arg("cells"),
      nb::arg("n_pts_dir"));

    module.def(
      "create_reparameterization_levelset",
      [](const qugar::UnfittedDomain<dim> &unf_domain, const CellsArray &cells_py, const int n_pts_dir) {
        const std::span<const std::int64_t> cells_span(cells_py.data(), cells_py.size());
        const std::vector<std::int64_t> cells(cells_span.begin(), cells_span.end());

        return qugar::create_reparameterization<dim, true>(unf_domain, cells, n_pts_dir);
      },
      nb::arg("unf_domain"),
      nb::arg("cells"),
      nb::arg("n_pts_dir"));


    module.def(
      "create_reparameterization",
      [](const qugar::UnfittedDomain<dim> &unf_domain, const int n_pts_dir) {
        return qugar::create_reparameterization<dim, false>(unf_domain, n_pts_dir);
      },
      nb::arg("unf_domain"),
      nb::arg("n_pts_dir"));


    module.def(
      "create_reparameterization_levelset",
      [](const qugar::UnfittedDomain<dim> &unf_domain, const int n_pts_dir) {
        return qugar::create_reparameterization<dim, true>(unf_domain, n_pts_dir);
      },
      nb::arg("unf_domain"),
      nb::arg("n_pts_dir"));
  }


}// namespace

// NOLINTNEXTLINE (misc-use-internal-linkage)
void reparam(nanobind::module_ &module)
{
  declare_reparam_mesh<1, 2>(module);
  declare_reparam_mesh<2, 2>(module);
  declare_reparam_mesh<2, 3>(module);
  declare_reparam_mesh<3, 3>(module);

  create_reparam<2>(module);
  create_reparam<3>(module);
}


}// namespace qugar::wrappers
