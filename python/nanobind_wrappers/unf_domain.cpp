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
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "wrapper_utils.hpp"

#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/unfitted_domain.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// NOLINTBEGIN (misc-include-cleaner)
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
// NOLINTEND (misc-include-cleaner)

namespace nb = nanobind;

namespace qugar::wrappers {

namespace {

  // Declare class for unfitted implicit domains.
  template<int dim> void declare_unfitted_domain(nb::module_ &module)
  {
    using UnfDomain = UnfittedDomain<dim>;

    const auto make_tuple = [](const std::vector<std::int64_t> &cells, const std::vector<int> &facets) {
      using Array64 = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>, nb::c_contig>;
      using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
      const auto cells_py = Array64(cells.data(), { cells.size() }, nb::handle());
      const auto facets_py = Array(facets.data(), { facets.size() }, nb::handle());
      return nb::make_tuple(cells_py, facets_py);
    };

    const auto get_cells =
      [&make_tuple](const auto &accessor_0,
        const auto &accessor_1,
        const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py) {
        std::vector<std::int64_t> cell_ids;

        if (target_cell_ids_py.has_value()) {

          const std::span<const std::int64_t> cells_span(
            target_cell_ids_py.value().data(), target_cell_ids_py.value().size());

          // Is this copy avoidable?
          accessor_0(std::vector<std::int64_t>(cells_span.begin(), cells_span.end()), cell_ids);
        } else {
          accessor_1(cell_ids);
        }

        return as_nbarray(std::move(cell_ids));
      };

    const auto get_facets =
      [&make_tuple](const auto &accessor_0,
        const auto &accessor_1,
        const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py,
        const std::optional<nb::ndarray<const int, nb::numpy, nb::shape<-1>>> &target_local_facet_ids_py) {
        std::vector<std::int64_t> cell_ids;
        std::vector<int> local_facet_ids;

        if (target_cell_ids_py.has_value()) {
          assert(target_local_facets_ids_py.has_value());

          const std::span<const std::int64_t> cells_span(
            target_cell_ids_py.value().data(), target_cell_ids_py.value().size());
          const std::span<const int> local_facets_span(
            target_local_facet_ids_py.value().data(), target_local_facet_ids_py.value().size());

          // Is this copy avoidable?
          accessor_0(std::vector<std::int64_t>(cells_span.begin(), cells_span.end()),
            std::vector<int>(local_facets_span.begin(), local_facets_span.end()),
            cell_ids,
            local_facet_ids);
        } else {
          accessor_1(cell_ids, local_facet_ids);
        }

        return make_tuple(cell_ids, local_facet_ids);
      };


    const std::string pyclass_name{ std::string("UnfittedDomain_") + std::to_string(dim) + "D" };
    nb::class_<UnfDomain>(module, pyclass_name.c_str(), "UnfittedDomain object")
      .def_prop_ro(
        "dim", [](const UnfDomain & /*domain*/) { return dim; }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "grid", [](const UnfDomain &domain) { return domain.get_grid(); }, nb::rv_policy::reference_internal)
      .def_prop_ro("num_total_cells", [](const UnfDomain &domain) { return domain.get_num_total_cells(); })
      .def_prop_ro(
        "has_facets_with_unf_bdry", [](const UnfDomain &domain) { return domain.has_facets_with_unf_bdry(); })
      .def(
        "get_full_cells",
        [&get_cells](const UnfDomain &domain,
          const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py) {
          const auto accessor_0 = [&domain](const auto &target_cell_ids, auto &cell_ids) {
            domain.get_full_cells(target_cell_ids, cell_ids);
          };
          const auto accessor_1 = [&domain](auto &cell_ids) { domain.get_full_cells(cell_ids); };

          return get_cells(accessor_0, accessor_1, target_cell_ids_py);
        },
        nb::arg("target_cell_ids") = nb::none())
      .def(
        "get_empty_cells",
        [&get_cells](const UnfDomain &domain,
          const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py) {
          const auto accessor_0 = [&domain](const auto &target_cell_ids, auto &cell_ids) {
            domain.get_empty_cells(target_cell_ids, cell_ids);
          };
          const auto accessor_1 = [&domain](auto &cell_ids) { domain.get_empty_cells(cell_ids); };

          return get_cells(accessor_0, accessor_1, target_cell_ids_py);
        },
        nb::arg("target_cell_ids") = nb::none())
      .def(
        "get_cut_cells",
        [&get_cells](const UnfDomain &domain,
          const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py) {
          const auto accessor_0 = [&domain](const auto &target_cell_ids, auto &cell_ids) {
            domain.get_cut_cells(target_cell_ids, cell_ids);
          };
          const auto accessor_1 = [&domain](auto &cell_ids) { domain.get_cut_cells(cell_ids); };

          return get_cells(accessor_0, accessor_1, target_cell_ids_py);
        },
        nb::arg("target_cell_ids") = nb::none())
      .def(
        "get_empty_facets",
        [&get_facets](const UnfDomain &domain,
          const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py,
          const std::optional<nb::ndarray<const int, nb::numpy, nb::shape<-1>>> &target_local_facet_ids_py) {
          const auto accessor_0 =
            [&domain](
              const auto &target_cell_ids, const auto &target_local_facet_ids, auto &cell_ids, auto &local_facet_ids) {
              domain.get_empty_facets(target_cell_ids, target_local_facet_ids, cell_ids, local_facet_ids);
            };
          const auto accessor_1 = [&domain](auto &cell_ids, auto &local_facet_ids) {
            domain.get_empty_facets(cell_ids, local_facet_ids);
          };

          return get_facets(accessor_0, accessor_1, target_cell_ids_py, target_local_facet_ids_py);
        },
        nb::arg("target_cell_ids") = nb::none(),
        nb::arg("target_local_facet_ids") = nb::none())
      .def(
        "get_full_facets",
        [&get_facets](const UnfDomain &domain,
          const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py,
          const std::optional<nb::ndarray<const int, nb::numpy, nb::shape<-1>>> &target_local_facet_ids_py) {
          const auto accessor_0 =
            [&domain](
              const auto &target_cell_ids, const auto &target_local_facet_ids, auto &cell_ids, auto &local_facet_ids) {
              domain.get_full_facets(target_cell_ids, target_local_facet_ids, cell_ids, local_facet_ids);
            };
          const auto accessor_1 = [&domain](auto &cell_ids, auto &local_facet_ids) {
            domain.get_full_facets(cell_ids, local_facet_ids);
          };

          return get_facets(accessor_0, accessor_1, target_cell_ids_py, target_local_facet_ids_py);
        },
        nb::arg("target_cell_ids") = nb::none(),
        nb::arg("target_local_facet_ids") = nb::none())
      .def(
        "get_cut_facets",
        [&get_facets](const UnfDomain &domain,
          const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py,
          const std::optional<nb::ndarray<const int, nb::numpy, nb::shape<-1>>> &target_local_facet_ids_py) {
          const auto accessor_0 =
            [&domain](
              const auto &target_cell_ids, const auto &target_local_facet_ids, auto &cell_ids, auto &local_facet_ids) {
              domain.get_cut_facets(target_cell_ids, target_local_facet_ids, cell_ids, local_facet_ids);
            };
          const auto accessor_1 = [&domain](auto &cell_ids, auto &local_facet_ids) {
            domain.get_cut_facets(cell_ids, local_facet_ids);
          };

          return get_facets(accessor_0, accessor_1, target_cell_ids_py, target_local_facet_ids_py);
        },
        nb::arg("target_cell_ids") = nb::none(),
        nb::arg("target_local_facet_ids") = nb::none())
      .def(
        "get_unf_bdry_facets",
        [&get_facets](const UnfDomain &domain,
          const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py,
          const std::optional<nb::ndarray<const int, nb::numpy, nb::shape<-1>>> &target_local_facet_ids_py) {
          const auto accessor_0 =
            [&domain](
              const auto &target_cell_ids, const auto &target_local_facet_ids, auto &cell_ids, auto &local_facet_ids) {
              domain.get_unfitted_facets(target_cell_ids, target_local_facet_ids, cell_ids, local_facet_ids);
            };
          const auto accessor_1 = [&domain](auto &cell_ids, auto &local_facet_ids) {
            domain.get_unfitted_facets(cell_ids, local_facet_ids);
          };

          return get_facets(accessor_0, accessor_1, target_cell_ids_py, target_local_facet_ids_py);
        },
        nb::arg("target_cell_ids") = nb::none(),
        nb::arg("target_local_facet_ids") = nb::none())
      .def(
        "get_full_unf_bdry_facets",
        [&get_facets](const UnfDomain &domain,
          const std::optional<nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>> &target_cell_ids_py,
          const std::optional<nb::ndarray<const int, nb::numpy, nb::shape<-1>>> &target_local_facet_ids_py) {
          const auto accessor_0 =
            [&domain](
              const auto &target_cell_ids, const auto &target_local_facet_ids, auto &cell_ids, auto &local_facet_ids) {
              domain.get_full_unfitted_facets(target_cell_ids, target_local_facet_ids, cell_ids, local_facet_ids);
            };
          const auto accessor_1 = [&domain](auto &cell_ids, auto &local_facet_ids) {
            domain.get_full_unfitted_facets(cell_ids, local_facet_ids);
          };

          return get_facets(accessor_0, accessor_1, target_cell_ids_py, target_local_facet_ids_py);
        },
        nb::arg("target_cell_ids") = nb::none(),
        nb::arg("target_local_facet_ids") = nb::none());
  }

  // Declare class for unfitted implicit domains.
  template<int dim> void declare_unfitted_impl_domain(nb::module_ &module)
  {
    using UnfittedImplDomain = impl::UnfittedImplDomain<dim>;
    using UnfDomain = UnfittedDomain<dim>;

    const auto make_tuple = [](const std::vector<std::int64_t> &cells, const std::vector<int> &facets) {
      using Array64 = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>, nb::c_contig>;
      using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
      const auto cells_py = Array64(cells.data(), { cells.size() }, nb::handle());
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

    using CellsArray = nb::ndarray<const std::int64_t, nb::numpy, nb::shape<-1>>;

    module.def(
      "create_unfitted_impl_domain",
      [](const std::shared_ptr<const impl::ImplicitFunc<dim>> phi,
        const std::shared_ptr<const CartGridTP<dim>> grid,
        const CellsArray &cells_py) {
        assert(phi != nullptr);
        assert(grid != nullptr);
        const std::span<const std::int64_t> cells_span(cells_py.data(), cells_py.size());
        const std::vector<std::int64_t> cells(cells_span.begin(), cells_span.end());
        return std::make_shared<impl::UnfittedImplDomain<dim>>(phi, grid, cells);
      },
      nb::arg("impl_func"),
      nb::arg("grid"),
      nb::arg("cells"));

    module.def(
      "create_unfitted_impl_domain",
      [](const std::shared_ptr<const impl::ImplicitFunc<dim>> phi,
        const std::shared_ptr<const CartGridTP<dim>> grid,
        const std::vector<std::int64_t> &cells) {
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
