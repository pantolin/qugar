// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_reparam_mesh.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of reparameterization class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_reparam_mesh.hpp>

#include <qugar/bbox.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/lagrange_tp_utils.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/reparam_mesh.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <ranges>
#include <span>
#include <vector>

namespace qugar::impl {

namespace {

  template<int dim> real compute_determinant(const std::array<Point<dim>, dim> &Jacobian)
  {
    if constexpr (dim == 3) {
      const auto &jac_0 = at(Jacobian, 0);
      const auto &jac_1 = at(Jacobian, 1);
      const auto &jac_2 = at(Jacobian, 2);

      return (jac_0(0) * (jac_1(1) * jac_2(2) - jac_1(2) * jac_2(1)))
             + (jac_0(1) * (jac_1(2) * jac_2(0) - jac_1(0) * jac_2(2)))
             + (jac_0(2) * (jac_1(0) * jac_2(1) - jac_1(1) * jac_2(0)));
    } else {
      const auto &jac_0 = at(Jacobian, 0);
      const auto &jac_1 = at(Jacobian, 1);

      return (jac_0(0) * jac_1(1)) - (jac_0(1) * jac_1(0));
    }
  }

  Point<3> cross_product(const Point<3> &lhs, const Point<3> &rhs)
  {
    return Point<3>{ (lhs(1) * rhs(2)) - (lhs(2) * rhs(1)),
      (lhs(2) * rhs(0)) - (lhs(0) * rhs(2)),
      (lhs(0) * rhs(1)) - (lhs(1) * rhs(0)) };
  }


}// namespace

template<int dim, int range>
ImplReparamMesh<dim, range>::ImplReparamMesh(const int order) : ReparamMesh<dim, range>(order)
{}


template<int dim, int range>
void ImplReparamMesh<dim, range>::generate_wirebasket(
  const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> &impl_funcs,
  const BoundBox<range> &domain,
  const Tolerance &tol)
{
  const auto n_cells = this->get_num_cells();

  std::vector<int> cell_ids;
  cell_ids.reserve(n_cells);

  for (std::size_t cell_id = 0; cell_id < n_cells; ++cell_id) {
    cell_ids.push_back(static_cast<int>(cell_id));
  }

  this->generate_wirebasket(impl_funcs, cell_ids, domain, tol);
}


template<int dim, int range>
void ImplReparamMesh<dim, range>::generate_wirebasket(
  const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> &impl_funcs,
  const std::vector<int> &cell_ids,
  const BoundBox<range> &domain,
  const Tolerance &tol)
{
  // This function is buggy for more than one implicit function.
  // For more than one implicit function, the function should be
  // consider how the functions are combined, e.g., union, intersection, etc.
  assert(impl_funcs.size() == 1);

  const auto check_face = [&impl_funcs, &tol](const Point<range> &point) -> bool {
    return std::ranges::any_of(
      impl_funcs, [&tol, &point](const auto &impl_func) { return on_levelset<range>(impl_func, point, tol); });
  };

  const auto check_internal = [&impl_funcs, &tol](const Point<range> &point) -> bool {
    constexpr int n_zeros = range - 1;

    return impl_funcs.size() >= (range - 1) && std::ranges::count_if(impl_funcs, [&tol, &point](const auto &impl_func) {
      return on_levelset<range>(impl_func, point, tol);
    }) >= n_zeros;
  };

  return this->generate_wirebasket(cell_ids, domain, tol, check_face, check_internal);
}

template<int dim, int range>
// NOLINTNEXTLINE (readability-function-cognitive-complexity)
void ImplReparamMesh<dim, range>::generate_wirebasket(const std::vector<int> &cell_ids,
  const BoundBox<range> &domain,
  const Tolerance &tol,
  // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
  const std::function<bool(const Point<range> &)> &check_face,
  const std::function<bool(const Point<range> &)> &check_internal)
{
  if constexpr (1 < dim) {
    constexpr int n_edges = dim == 2 ? 4 : 12;

    for (const auto &cell_id : cell_ids) {

      for (int edge_id = 0; edge_id < n_edges; ++edge_id) {

        const auto edge_point_ids = this->get_edge_points(cell_id, edge_id);

        if (this->check_subentity_degenerate(edge_point_ids, tol)) {
          continue;
        }

        // Checking if edge belongs to a domain's edge.
        auto add_edge = this->template check_edge_in_subdomain<range - 1>(edge_point_ids, domain, tol);

        if (!add_edge) {
          if (this->template check_edge_in_subdomain<1>(edge_point_ids, domain, tol)) {
            // Edge belongs to a domain's face.
            add_edge = std::ranges::all_of(edge_point_ids, [&check_face, this](const auto &pt_id) {
              const auto &point = at(this->points_, static_cast<int>(pt_id));
              return check_face(point);
            });
          } else {
            // Edge doesn't belong neither to an edge nor to a face.
            // Then, we check if edge corresponds to an internal zero-levelset.
            add_edge = std::ranges::all_of(edge_point_ids, [&check_internal, this](const auto &pt_id) {
              const auto &point = at(this->points_, static_cast<int>(pt_id));
              return check_internal(point);
            });
          }
        }

        if (add_edge) {
          this->wires_connectivity_.insert(
            this->wires_connectivity_.end(), edge_point_ids.cbegin(), edge_point_ids.cend());
        }
      }
    }

  }// if constexpr (1 < dim)
}


template<int dim, int range>
template<int dim_aux>
  requires(dim_aux == dim && range == dim)
void ImplReparamMesh<dim, range>::orient_cells_positively(const std::vector<int> &cell_ids)
{
  const TensorSizeTP<dim> order(this->order_);
  Vector<std::vector<real>, dim> derivatives;

  constexpr bool chebyshev = false;

  const Point<dim> mid_pt{ numbers::half };
  evaluate_Lagrange_derivative<dim>(mid_pt, order, chebyshev, derivatives);

  for (const auto &cell_id : cell_ids) {

    assert(0 <= cell_id && static_cast<std::size_t>(cell_id) < this->get_num_cells());

    const auto offset = static_cast<std::size_t>(cell_id) * this->get_num_points_per_cell();

    const auto cell_points = std::span(this->connectivity_).subspan(offset, this->get_num_points_per_cell());

    std::array<Point<dim>, dim> jacobian;
    for (int i = 0; i < order.size(); ++i) {
      const auto pt_id = at(cell_points, i);
      const auto &point = at(this->points_, static_cast<int>(pt_id));

      for (int dir = 0; dir < dim; ++dir) {
        at(jacobian, dir) += point * at(derivatives(dir), i);
      }
    }
    if (compute_determinant<dim>(jacobian) < 0) {
      this->permute_cell_directions(static_cast<std::size_t>(cell_id));
    }
  }
}

template<int dim, int range>
template<int dim_aux>
  requires(dim_aux == dim && range == dim)
void ImplReparamMesh<dim, range>::orient_cells_positively()
{
  const std::ranges::iota_view<int, int> rng{ 0, static_cast<int>(this->get_num_cells()) };
  const std::vector<int> cell_ids(rng.begin(), rng.end());
  this->orient_cells_positively(cell_ids);
}

template<int dim, int range>
template<int dim_aux>
  requires(dim_aux == dim && range == (dim + 1))
void ImplReparamMesh<dim, range>::orient_levelset_cells_positively(const std::vector<int> &cell_ids,
  const std::function<Point<range>(const Point<range> &)> &outer_normal_computer)
{
  const TensorSizeTP<dim> order(this->order_);
  std::vector<real> basis;
  Vector<std::vector<real>, dim> derivatives;

  constexpr bool chebyshev = false;

  const Point<dim> mid_pt{ numbers::half };

  evaluate_Lagrange_basis(mid_pt, order, chebyshev, basis);
  evaluate_Lagrange_derivative<dim>(mid_pt, order, chebyshev, derivatives);

  for (const auto &cell_id : cell_ids) {
    assert(0 <= cell_id && static_cast<std::size_t>(cell_id) < this->get_num_cells());

    const auto offset = static_cast<std::size_t>(cell_id) * this->get_num_points_per_cell();

    const auto cell_points = std::span(this->connectivity_).subspan(offset, this->get_num_points_per_cell());

    Point<range> eval_point;

    std::array<Point<range>, dim> jacobian;
    for (int i = 0; i < order.size(); ++i) {
      const auto pt_id = at(cell_points, i);
      const auto &point = at(this->points_, static_cast<int>(pt_id));

      for (int dir = 0; dir < dim; ++dir) {
        at(jacobian, dir) += point * at(derivatives(dir), i);
      }
      eval_point += point * at(basis, i);
    }

    Point<range> normal;
    if constexpr (dim == 2) {
      normal = cross_product(at(jacobian, 0), at(jacobian, 1));
    } else {
      static_assert(dim == 1, "Invalid dimension.");
      normal = Point<2>{ at(jacobian, 0)(1), -at(jacobian, 0)(0) };
    }

    normal /= norm(normal);

    const auto target_normal = outer_normal_computer(eval_point);
    if (dot(normal, target_normal) < 0) {
      this->permute_cell_directions(static_cast<std::size_t>(cell_id));
    }
  }
}

template<int dim, int range>
template<int dim_aux>
  requires(dim_aux == dim && range == (dim + 1))
void ImplReparamMesh<dim, range>::orient_levelset_cells_positively(
  const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> &impl_funcs,
  const std::vector<int> &cell_ids)
{
  const auto outer_normal_computer = [&impl_funcs](const Point<range> &param_pt) -> Point<range> {
    const auto func = std::ranges::min_element(impl_funcs, [&param_pt](const auto &lhs_wrap, const auto &rhs_wrap) {
      const ImplicitFunc<range> &lhs = lhs_wrap.get();
      const ImplicitFunc<range> &rhs = rhs_wrap.get();
      return lhs(param_pt) < rhs(param_pt);
    });

    auto normal = func->get().grad(param_pt);
    normal /= norm(normal);

    return normal;
  };

  return orient_levelset_cells_positively(cell_ids, outer_normal_computer);
}

template<int dim, int range>
template<int dim_aux>
  requires(dim_aux == dim && range == (dim + 1))
void ImplReparamMesh<dim, range>::orient_levelset_cells_positively(
  const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> &impl_funcs)
{
  const std::ranges::iota_view<int, int> rng{ 0, static_cast<int>(this->get_num_cells()) };
  const std::vector<int> cell_ids(rng.begin(), rng.end());
  this->orient_levelset_cells_positively(impl_funcs, cell_ids);
}

template<int dim, int range>
template<int dim_aux>
  requires(dim_aux == dim && range == (dim + 1))
void ImplReparamMesh<dim, range>::orient_facet_cells_positively(const int local_facet_id,
  const std::vector<int> &cell_ids)
{
  assert(0 <= local_facet_id && local_facet_id < 2 * range);

  const auto dir = get_facet_constant_dir<range>(local_facet_id);
  const auto side = get_facet_side<range>(local_facet_id);
  Point<range> normal{ numbers::zero };
  normal(dir) = side == 0 ? -numbers::one : numbers::one;

  const auto outer_normal_computer = [normal](const Point<range> & /*param_pt*/) -> Point<range> { return normal; };

  this->orient_levelset_cells_positively(cell_ids, outer_normal_computer);
}

template<int dim, int range>
template<int dim_aux>
  requires(dim_aux == dim && range == (dim + 1))
void ImplReparamMesh<dim, range>::orient_facet_cells_positively(const int local_facet_id)
{
  const std::ranges::iota_view<int, int> rng{ 0, static_cast<int>(this->get_num_cells()) };
  const std::vector<int> cell_ids(rng.begin(), rng.end());

  this->orient_facet_cells_positively(local_facet_id, cell_ids);
}


// Instantiations.


template class ImplReparamMesh<1, 2>;
template class ImplReparamMesh<2, 2>;
template class ImplReparamMesh<2, 3>;
template class ImplReparamMesh<3, 3>;

template void ImplReparamMesh<2, 2>::orient_cells_positively<2>();
template void ImplReparamMesh<3, 3>::orient_cells_positively<3>();

template void ImplReparamMesh<2, 2>::orient_cells_positively<2>(const std::vector<int> &);
template void ImplReparamMesh<3, 3>::orient_cells_positively<3>(const std::vector<int> &);

template void ImplReparamMesh<1, 2>::orient_levelset_cells_positively<1>(
  const std::vector<std::reference_wrapper<const ImplicitFunc<2>>> &);
template void ImplReparamMesh<2, 3>::orient_levelset_cells_positively<2>(
  const std::vector<std::reference_wrapper<const ImplicitFunc<3>>> &);

template void ImplReparamMesh<1, 2>::orient_levelset_cells_positively<1>(
  const std::vector<std::reference_wrapper<const ImplicitFunc<2>>> &,
  const std::vector<int> &);
template void ImplReparamMesh<2, 3>::orient_levelset_cells_positively<2>(
  const std::vector<std::reference_wrapper<const ImplicitFunc<3>>> &,
  const std::vector<int> &);

template void ImplReparamMesh<1, 2>::orient_facet_cells_positively<1>(const int);
template void ImplReparamMesh<2, 3>::orient_facet_cells_positively<2>(const int);
template void ImplReparamMesh<1, 2>::orient_facet_cells_positively<1>(const int, const std::vector<int> &);
template void ImplReparamMesh<2, 3>::orient_facet_cells_positively<2>(const int, const std::vector<int> &);

}// namespace qugar::impl