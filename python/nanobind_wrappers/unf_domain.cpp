// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_quad.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of nanobind wrappers for generation of quadratures
//! for implicitly defined domains.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/unfitted_domain.hpp>

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

  // Declare class for unfitted implicit domains.
  template<int dim> void declare_unfitted_domain(nb::module_ &module)
  {
    using UnfDomain = UnfittedDomain<dim>;

    const auto make_tuple = [](const std::vector<int> &cells, const std::vector<int> &facets) {
      using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
      const auto cells_py = Array(cells.data(), { cells.size() }, nb::handle());
      const auto facets_py = Array(facets.data(), { facets.size() }, nb::handle());
      return nb::make_tuple(cells_py, facets_py);
    };

    const std::string pyclass_name{ std::string("UnfittedDomain_") + std::to_string(dim) + "D" };
    nb::class_<UnfDomain>(module, pyclass_name.c_str(), "UnfittedDomain object")
      .def_prop_ro(
        "dim", [](UnfDomain & /*domain*/) { return dim; }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "grid", [](UnfDomain &domain) { return domain.get_grid(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "full_cells",
        [](UnfDomain &domain) {
          using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(domain.get_full_cells().data(), { domain.get_full_cells().size() }, nb::handle());
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "empty_cells",
        [](UnfDomain &domain) {
          using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(domain.get_empty_cells().data(), { domain.get_empty_cells().size() }, nb::handle());
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "cut_cells",
        [](UnfDomain &domain) {
          using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
          return Array(domain.get_cut_cells().data(), { domain.get_cut_cells().size() }, nb::handle());
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "empty_facets",
        [&make_tuple](UnfDomain &domain) {
          std::vector<int> cell_ids;
          std::vector<int> local_facet_ids;
          domain.get_empty_facets(cell_ids, local_facet_ids);

          return make_tuple(cell_ids, local_facet_ids);
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "full_facets",
        [&make_tuple](UnfDomain &domain) {
          std::vector<int> cell_ids;
          std::vector<int> local_facet_ids;
          domain.get_full_facets(cell_ids, local_facet_ids);

          return make_tuple(cell_ids, local_facet_ids);
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "cut_facets",
        [&make_tuple](UnfDomain &domain) {
          std::vector<int> cell_ids;
          std::vector<int> local_facet_ids;
          domain.get_cut_facets(cell_ids, local_facet_ids);

          return make_tuple(cell_ids, local_facet_ids);
        },
        nb::rv_policy::reference_internal);
  }

  // Declare class for unfitted implicit domains.
  template<int dim> void declare_unfitted_impl_domain(nb::module_ &module)
  {
    using UnfittedImplDomain = impl::UnfittedImplDomain<dim>;
    using UnfDomain = UnfittedDomain<dim>;

    const auto make_tuple = [](const std::vector<int> &cells, const std::vector<int> &facets) {
      using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
      const auto cells_py = Array(cells.data(), { cells.size() }, nb::handle());
      const auto facets_py = Array(facets.data(), { facets.size() }, nb::handle());
      return nb::make_tuple(cells_py, facets_py);
    };

    const std::string pyclass_name{ std::string("UnfittedImplDomain_") + std::to_string(dim) + "D" };
    nb::class_<UnfittedImplDomain, UnfDomain>(module, pyclass_name.c_str(), "UnfittedImplDomain object")
      .def_prop_ro(
        "impl_func",
        [](UnfittedImplDomain &domain) { return domain.get_impl_func(); },
        nb::rv_policy::reference_internal);
  }

  template<int dim> void create_unfitted_impl_domain(nb::module_ &module)
  {
    module.def(
      "create_unfitted_impl_domain",
      [](const std::shared_ptr<const impl::ImplicitFunc<dim>> phi, const std::shared_ptr<const CartGridTP<dim>> grid) {
        assert(phi != nullptr);
        assert(grid != nullptr);
        return std::make_shared<impl::UnfittedImplDomain<dim>>(phi, grid);
      },
      nb::arg("impl_func"),
      nb::arg("grid"));

    using CellsArray = nb::ndarray<const int, nb::numpy, nb::shape<-1>>;

    module.def(
      "create_unfitted_impl_domain",
      [](const std::shared_ptr<const impl::ImplicitFunc<dim>> phi,
        const std::shared_ptr<const CartGridTP<dim>> grid,
        const CellsArray &cells_py) {
        assert(phi != nullptr);
        assert(grid != nullptr);
        const std::span<const int> cells_span(cells_py.data(), cells_py.size());
        const std::vector<int> cells(cells_span.begin(), cells_span.end());
        return std::make_shared<impl::UnfittedImplDomain<dim>>(phi, grid, cells);
      },
      nb::arg("impl_func"),
      nb::arg("grid"),
      nb::arg("cells"));

    module.def(
      "create_unfitted_impl_domain",
      [](const std::shared_ptr<const impl::ImplicitFunc<dim>> phi,
        const std::shared_ptr<const CartGridTP<dim>> grid,
        const std::vector<int> &cells) {
        assert(phi != nullptr);
        assert(grid != nullptr);
        return std::make_shared<impl::UnfittedImplDomain<dim>>(phi, grid, cells);
      },
      nb::arg("impl_func"),
      nb::arg("grid"),
      nb::arg("cells"));
  }


}// namespace

void unf_domain(nanobind::module_ &module)
{
  declare_unfitted_domain<2>(module);
  declare_unfitted_domain<3>(module);

  declare_unfitted_impl_domain<2>(module);
  declare_unfitted_impl_domain<3>(module);

  create_unfitted_impl_domain<2>(module);
  create_unfitted_impl_domain<3>(module);
}


}// namespace qugar::wrappers
