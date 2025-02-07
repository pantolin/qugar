// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file common.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of nanobind wrappers for general classes as
//! Cartesian bounding box class and Cartesian grid class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "wrapper_utils.hpp"

#include <qugar/affine_transf.hpp>
#include <qugar/bbox.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/point.hpp>
#include <qugar/ref_system.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>

#include <array>
#include <memory>
#include <span>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// NOLINTBEGIN (misc-include-cleaner)
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
// NOLINTEND (misc-include-cleaner)

namespace nb = nanobind;

namespace qugar::wrappers {

namespace {

  template<int dim> void declare_bound_box(nb::module_ &module)
  {
    using BBox = qugar::BoundBox<dim>;

    const std::string pyclass_name{ std::string("BoundBox_") + std::to_string(dim) + "D" };
    nb::class_<BBox>(module, pyclass_name.c_str(), "BoundBox object")
      .def(
        "as_array",
        [](BBox &bbox) {
          // TODO: to get rid of this copy.
          Vector<real, dim * 2> data;
          for (int dir = 0; dir < dim; ++dir) {
            data(dir * 2) = bbox.min(dir);
            data(dir * 2 + 1) = bbox.max(dir);
          }

          using Array = nb::ndarray<const real, nb::numpy, nb::shape<dim, 2>, nb::c_contig>;
          return Array(data.data(), { dim, 2 }, nb::handle()).cast();
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "min_corner",
        [](BBox &bbox) {
          using Array = nb::ndarray<const real, nb::numpy, nb::shape<dim>, nb::c_contig>;
          return Array(bbox.min_corner().data(), { dim }, nb::handle());
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "max_corner",
        [](BBox &bbox) {
          using Array = nb::ndarray<const real, nb::numpy, nb::shape<dim>, nb::c_contig>;
          return Array(bbox.max_corner().data(), { dim }, nb::handle());
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "mid_point",
        [](BBox &bbox) {
          using Array = nb::ndarray<const real, nb::numpy, nb::shape<dim>, nb::c_contig>;
          return Array(bbox.mid_point().data(), { dim }, nb::handle());
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro("volume", &BBox::volume)
      .def("length", [](BBox &box, const int dir) { return box.length(dir); }, nb::rv_policy::reference_internal);


    module.def(
      "create_bound_box",
      [](const std::array<real, dim> &min, const std::array<real, dim> &max) {
        return std::make_shared<BoundBox<dim>>(min, max);
      },
      nb::arg("min_corner"),
      nb::arg("max_corner"));

    module.def(
      "create_bound_box",
      [](const npPointConst<dim> &min, const npPointConst<dim> &max) {
        Point<dim> min_pt;
        Point<dim> max_pt;
        for (int dir = 0; dir < dim; ++dir) {
          min_pt(dir) = min(dir);
          max_pt(dir) = max(dir);
        }
        return std::make_shared<BoundBox<dim>>(min_pt, max_pt);
      },
      nb::arg("min_corner"),
      nb::arg("max_corner"));
  }

  template<int dim> void declare_cartesian_grid_tp(nb::module_ &module)
  {
    using Grid = CartGridTP<dim>;
    using BreaksArray = nb::ndarray<const real, nb::numpy, nb::shape<-1>>;

    const std::string pyclass_name{ std::string("CartGridTP_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Grid>(module, pyclass_name.c_str(), "CartGridTP object")
      .def_prop_ro("domain", &Grid::get_domain)
      .def_prop_ro(
        "dim", [](Grid & /*grid*/) { return dim; }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "num_cells_dir",
        [](Grid &grid) {
          const auto n_cells = grid.get_num_cells_dir().as_Vector();
          using Array = nb::ndarray<const int, nb::numpy, nb::shape<dim>, nb::c_contig>;
          return Array(n_cells.data(), { dim }).cast();
        },
        nb::rv_policy::reference_internal)
      .def("get_cell_domain", &Grid::get_cell_domain)
      .def(
        "get_boundary_cells",
        // TODO: to check here value of facet_id
        [](Grid &grid, const int facet_id) {
          const auto cells = grid.get_boundary_cells(facet_id);
          using Array = nb::ndarray<const int, nb::numpy, nb::shape<-1>, nb::c_contig>;
          // There is a copy here.
          return Array(cells.data(), { cells.size() }).cast();
        },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "cell_breaks",
        [](Grid &grid) {
          using Array = nb::ndarray<const real, nb::numpy, nb::shape<-1>, nb::c_contig>;
          std::array<Array, dim> breaks_py;
          for (int dir = 0; dir < dim; ++dir) {
            const auto &breaks = grid.get_breaks(dir);
            at(breaks_py, dir) = Array(breaks.data(), { breaks.size() }, nb::handle());
          }
          return breaks_py;
        },
        nb::rv_policy::reference_internal);


    module.def(
      "create_cart_grid",
      [](const BoundBox<dim> &bbox, const std::array<std::size_t, dim> &n_cells) {
        return std::make_shared<CartGridTP<dim>>(bbox, n_cells);
      },
      nb::arg("bound_box"),
      nb::arg("n_cells_dir"));

    module.def(
      "create_cart_grid",
      [](const std::array<std::vector<real>, dim> &breaks) { return std::make_shared<CartGridTP<dim>>(breaks); },
      nb::arg("breaks"));

    module.def(
      "create_cart_grid",
      [](const std::array<BreaksArray, dim> &breaks_py) {
        std::array<std::vector<real>, dim> breaks;
        for (int dir = 0; dir < dim; ++dir) {
          const auto &breaks_py_dir = at(breaks_py, dir);
          const std::span<const real> breaks_span(breaks_py_dir.data(), breaks_py_dir.size());
          at(breaks, dir).assign(breaks_span.begin(), breaks_span.end());
        }

        return std::make_shared<CartGridTP<dim>>(breaks);
      },
      nb::arg("breaks"));
  }

  template<int dim> Point<dim> transform_point(const npPointConst<dim> &point_py)
  {
    Point<dim> point;
    for (int dir = 0; dir < dim; ++dir) {
      point(dir) = point_py(dir);
    }
    return point;
  }

  template<int dim> void declare_affine_transf(nanobind::module_ &module)
  {
    using AffineTransf = impl::AffineTransf<dim>;
    constexpr int n_coefs = AffineTransf::n_coefs;

    const std::string pyclass_name{ std::string("AffineTransf_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<AffineTransf>(module, pyclass_name.c_str(), "Affine transformation");

    module.def("create_affine_transformation", []() { return std::make_shared<AffineTransf>(); });

    module.def(
      "create_affine_transformation",
      [](const npPointConst<dim> &origin_py) {
        const auto origin = transform_point<dim>(origin_py);
        return std::make_shared<AffineTransf>(origin);
      },
      nb::arg("origin"));

    module.def(
      "create_affine_transformation",
      [](const npPointConst<dim> &origin_py, const real scale) {
        const auto origin = transform_point<dim>(origin_py);
        return std::make_shared<AffineTransf>(origin, scale);
      },
      nb::arg("origin"),
      nb::arg("scale"));

    // NOLINTNEXTLINE (bugprone-branch-clone)
    if constexpr (dim == 2) {
      module.def(
        "create_affine_transformation",
        // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
        [](const npPointConst<dim> &origin_py, const npPointConst<dim> &axis_x_py) {
          const auto origin = transform_point<dim>(origin_py);
          const auto axis_x = transform_point<dim>(axis_x_py);
          return std::make_shared<AffineTransf>(origin, axis_x);
        },
        nb::arg("origin"),
        nb::arg("axis_x"));

      module.def(
        "create_affine_transformation",
        // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
        [](const npPointConst<dim> &origin_py,
          const npPointConst<dim> &axis_x_py,
          const real scale_x,
          const real scale_y) {
          const auto origin = transform_point<dim>(origin_py);
          const auto axis_x = transform_point<dim>(axis_x_py);
          return std::make_shared<AffineTransf>(origin, axis_x, scale_x, scale_y);
        },
        nb::arg("origin"),
        nb::arg("axis_x"),
        nb::arg("scale_x"),
        nb::arg("scale_y"));
    } else {// if constexpr (dim == 3)
      module.def(
        "create_affine_transformation",
        // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
        [](const npPointConst<dim> &origin_py, const npPointConst<dim> &axis_x_py, const npPointConst<dim> &axis_y_py) {
          const auto origin = transform_point<dim>(origin_py);
          const auto axis_x = transform_point<dim>(axis_x_py);
          const auto axis_y = transform_point<dim>(axis_y_py);
          return std::make_shared<AffineTransf>(origin, axis_x, axis_y);
        },
        nb::arg("origin"),
        nb::arg("axis_x"),
        nb::arg("axis_y"));

      module.def(
        "create_affine_transformation",
        // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
        [](const npPointConst<dim> &origin_py,
          const npPointConst<dim> &axis_x_py,
          const npPointConst<dim> &axis_y_py,
          const real scale_x,
          const real scale_y,
          const real scale_z) {
          const auto origin = transform_point<dim>(origin_py);
          const auto axis_x = transform_point<dim>(axis_x_py);
          const auto axis_y = transform_point<dim>(axis_y_py);
          return std::make_shared<AffineTransf>(origin, axis_x, axis_y, scale_x, scale_y, scale_z);
        },
        nb::arg("origin"),
        nb::arg("axis_x"),
        nb::arg("axis_y"),
        nb::arg("scale_x"),
        nb::arg("scale_y"),
        nb::arg("scale_z"));
    }

    module.def(
      "create_affine_transformation",
      // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
      [](const nb::ndarray<real, nb::shape<n_coefs>> &coefs_py) {
        Vector<real, n_coefs> coefs;
        for (int i = 0; i < n_coefs; ++i) {
          coefs(i) = coefs_py(i);
        }

        return std::make_shared<AffineTransf>(coefs);
      },
      nb::arg("coefs"));
  }

  template<int dim> void declare_ref_system(nanobind::module_ &module)
  {
    using RefSystem = impl::RefSystem<dim>;

    const std::string pyclass_name{ std::string("RefSystem_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<RefSystem>(module, pyclass_name.c_str(), "Affine transformation");

    module.def("create_ref_system", []() { return std::make_shared<RefSystem>(); });

    module.def(
      "create_ref_system",
      [](const npPointConst<dim> &origin_py) {
        const auto origin = transform_point<dim>(origin_py);
        return std::make_shared<RefSystem>(origin);
      },
      nb::arg("origin"));


    // if constexpr (dim == 2) {
    module.def(
      "create_ref_system",
      // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
      [](const npPointConst<dim> &origin_py, const npPointConst<dim> &axis_py) {
        const auto origin = transform_point<dim>(origin_py);
        const auto axis = transform_point<dim>(axis_py);
        return std::make_shared<RefSystem>(origin, axis);
      },
      nb::arg("origin"),
      nb::arg("axis"));

    if constexpr (dim == 3) {
      module.def(
        "create_ref_system",
        // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
        [](const npPointConst<dim> &origin_py, const npPointConst<dim> &axis_x_py, const npPointConst<dim> &axis_y_py) {
          const auto origin = transform_point<dim>(origin_py);
          const auto axis_x = transform_point<dim>(axis_x_py);
          const auto axis_y = transform_point<dim>(axis_y_py);
          return std::make_shared<RefSystem>(origin, axis_x, axis_y);
        },
        nb::arg("origin"),
        nb::arg("axis_x"),
        nb::arg("axis_y"));
    }
  }

}// namespace

// NOLINTNEXTLINE (misc-use-internal-linkage)
void common(nanobind::module_ &module)
{
  declare_bound_box<2>(module);
  declare_bound_box<3>(module);

  declare_cartesian_grid_tp<2>(module);
  declare_cartesian_grid_tp<3>(module);

  declare_affine_transf<2>(module);
  declare_affine_transf<3>(module);

  declare_ref_system<2>(module);
  declare_ref_system<3>(module);
}


}// namespace qugar::wrappers
