// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file reparam_mesh.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of reparameterization class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/reparam_mesh.hpp>

#include <qugar/bbox.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/impl_reparam.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/point.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/unfitted_domain.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/bernstein.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qugar {

namespace {

  // NOLINTNEXTLINE (misc-no-recursion)
  constexpr int int_pow(const int base, const int exp)
  {
    return (exp == 0) ? 1 : base * int_pow(base, exp - 1);
  }

  template<int dim> int get_VTK_cell_type(const int order)
  {
    assert(1 < order);
    // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
    if (order == 2) {
      return dim == 1 ? 3 : dim == 2 ? 9 : 12;
    } else {
      return dim == 1 ? 68 : dim == 2 ? 70 : 72;
    }
    // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
  }

  //! @brief Helper function for creating a re-enumeration from lexicographical
  //! to VTK ordering for dim-dimensional tensor-product cells of the given order.
  //!
  //! We denote here lexicographical as the ordering in which the
  //! component 0 iterates the fastest, i.e., is the inner-most, and
  //! the component dim-1 iterates the slowest, i.e., the outer-most.
  //!
  //! @tparam dim Dimension of the cell.
  //! @param order Cell order along the N directions.
  //! @return Vector storing the generated mask.
  // NOLINTNEXTLINE (readability-function-cognitive-complexity)
  template<int dim> std::vector<std::size_t> create_VTU_conn_mask(const int order);

  template<> std::vector<std::size_t> create_VTU_conn_mask<1>(const int order)
  {
    assert(1 < order);

    std::vector<std::size_t> mask;
    mask.reserve(static_cast<std::size_t>(order));

    // Vertices
    mask.push_back(0);
    mask.push_back(static_cast<std::size_t>(order - 1));

    // Internal points.
    for (int i = 1; i < (order - 1); ++i) {
      mask.push_back(static_cast<std::size_t>(i));
    }

    return mask;
  }

  template<> std::vector<std::size_t> create_VTU_conn_mask<2>(const int order)
  {
    assert(1 < order);
    const TensorSizeTP<2> order_tp(order);

    std::vector<std::size_t> mask;
    mask.reserve(static_cast<std::size_t>(order_tp.size()));

    const auto to_flat = [order_tp](const int i_0, const int i_1) {
      return static_cast<std::size_t>(TensorIndexTP<2>(i_0, i_1).flat(order_tp));
    };
    // Vertices
    mask.push_back(to_flat(0, 0));
    mask.push_back(to_flat(order_tp(0) - 1, 0));
    mask.push_back(to_flat(order_tp(0) - 1, order_tp(1) - 1));
    mask.push_back(to_flat(0, order_tp(1) - 1));

    // Edges
    for (int i = 1; i < (order_tp(0) - 1); ++i) {
      mask.push_back(to_flat(i, 0));
    }
    for (int j = 1; j < (order_tp(1) - 1); ++j) {
      mask.push_back(to_flat(order_tp(0) - 1, j));
    }
    for (int i = 1; i < (order_tp(0) - 1); ++i) {
      mask.push_back(to_flat(i, order_tp(1) - 1));
    }
    for (int j = 1; j < (order_tp(1) - 1); ++j) {
      mask.push_back(to_flat(0, j));
    }

    // Internal
    for (int j = 1; j < (order_tp(1) - 1); ++j) {
      for (int i = 1; i < (order_tp(0) - 1); ++i) {
        mask.push_back(to_flat(i, j));
      }
    }

    return mask;
  }

  // NOLINTNEXTLINE (readability-function-cognitive-complexity)
  template<> std::vector<std::size_t> create_VTU_conn_mask<3>(const int order)
  {
    assert(1 < order);
    const TensorSizeTP<3> order_tp(order);

    std::vector<std::size_t> mask;
    mask.reserve(static_cast<std::size_t>(order_tp.size()));

    const auto to_flat = [order_tp](const int i_0, const int i_1, const int i_2) {
      return static_cast<std::size_t>(TensorIndexTP<3>(i_0, i_1, i_2).flat(order_tp));
    };

    // Vertices
    mask.push_back(to_flat(0, 0, 0));
    mask.push_back(to_flat(order_tp(0) - 1, 0, 0));
    mask.push_back(to_flat(order_tp(0) - 1, order_tp(1) - 1, 0));
    mask.push_back(to_flat(0, order_tp(1) - 1, 0));
    mask.push_back(to_flat(0, 0, order_tp(2) - 1));
    mask.push_back(to_flat(order_tp(0) - 1, 0, order_tp(2) - 1));
    mask.push_back(to_flat(order_tp(0) - 1, order_tp(1) - 1, order_tp(2) - 1));
    mask.push_back(to_flat(0, order_tp(1) - 1, order_tp(2) - 1));

    // Edges
    for (int i = 1; i < (order_tp(0) - 1); ++i) {
      mask.push_back(to_flat(i, 0, 0));
    }
    for (int j = 1; j < (order_tp(1) - 1); ++j) {
      mask.push_back(to_flat(order_tp(0) - 1, j, 0));
    }
    for (int i = 1; i < (order_tp(0) - 1); ++i) {
      mask.push_back(to_flat(i, order_tp(1) - 1, 0));
    }
    for (int j = 1; j < (order_tp(1) - 1); ++j) {
      mask.push_back(to_flat(0, j, 0));
    }
    for (int i = 1; i < (order_tp(0) - 1); ++i) {
      mask.push_back(to_flat(i, 0, order_tp(2) - 1));
    }
    for (int j = 1; j < (order_tp(1) - 1); ++j) {
      mask.push_back(to_flat(order_tp(0) - 1, j, order_tp(2) - 1));
    }
    for (int i = 1; i < (order_tp(0) - 1); ++i) {
      mask.push_back(to_flat(i, order_tp(1) - 1, order_tp(2) - 1));
    }
    for (int j = 1; j < (order_tp(1) - 1); ++j) {
      mask.push_back(to_flat(0, j, order_tp(2) - 1));
    }
    for (int k = 1; k < (order_tp(2) - 1); ++k) {
      mask.push_back(to_flat(0, 0, k));
    }
    for (int k = 1; k < (order_tp(2) - 1); ++k) {
      mask.push_back(to_flat(order_tp(0) - 1, 0, k));
    }
    for (int k = 1; k < (order_tp(2) - 1); ++k) {
      mask.push_back(to_flat(0, order_tp(1) - 1, k));
    }
    for (int k = 1; k < (order_tp(2) - 1); ++k) {
      mask.push_back(to_flat(order_tp(0) - 1, order_tp(1) - 1, k));
    }

    // Faces
    for (int k = 1; k < (order_tp(2) - 1); ++k) {
      for (int j = 1; j < (order_tp(1) - 1); ++j) {
        mask.push_back(to_flat(0, j, k));
      }
    }
    for (int k = 1; k < (order_tp(2) - 1); ++k) {
      for (int j = 1; j < (order_tp(1) - 1); ++j) {
        mask.push_back(to_flat(order_tp(0) - 1, j, k));
      }
    }
    for (int k = 1; k < (order_tp(2) - 1); ++k) {
      for (int i = 1; i < (order_tp(0) - 1); ++i) {
        mask.push_back(to_flat(i, 0, k));
      }
    }
    for (int k = 1; k < (order_tp(2) - 1); ++k) {
      for (int i = 1; i < (order_tp(0) - 1); ++i) {
        mask.push_back(to_flat(i, order_tp(1) - 1, k));
      }
    }
    for (int j = 1; j < (order_tp(1) - 1); ++j) {
      for (int i = 1; i < (order_tp(0) - 1); ++i) {
        mask.push_back(to_flat(i, j, 0));
      }
    }
    for (int j = 1; j < (order_tp(1) - 1); ++j) {
      for (int i = 1; i < (order_tp(0) - 1); ++i) {
        mask.push_back(to_flat(i, j, order_tp(2) - 1));
      }
    }

    // Internal
    for (int k = 1; k < (order_tp(2) - 1); ++k) {
      for (int j = 1; j < (order_tp(1) - 1); ++j) {
        for (int i = 1; i < (order_tp(0) - 1); ++i) {
          mask.push_back(to_flat(i, j, k));
        }
      }
    }

    return mask;
  }

  template<int dim, int range>
  void write_reparam_to_VTK_cells(const std::vector<Point<range>> &points,
    const std::vector<std::size_t> &connectivity,
    const int order,
    const std::string &filename)
  {
    static_assert(1 <= dim && dim <= 3, "Invalid dimension.");
    static_assert(1 <= range && range <= 3, "Invalid dimension.");

    assert(1 < order);

    const auto n_pts_per_cell = static_cast<std::size_t>(TensorSizeTP<dim>(order).size());
    assert(connectivity.size() % n_pts_per_cell == 0);

    const auto n_cells = connectivity.size() / n_pts_per_cell;
    const auto n_pts = points.size();

    std::ofstream stream(filename + ".vtu");

    stream << R"(<?xml version="1.0"?>)" << "\n";
    stream << R"(<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">)" << "\n";
    stream << "<UnstructuredGrid>\n";
    stream << R"(<Piece NumberOfPoints=")" << n_pts << R"(" NumberOfCells=")" << n_cells << R"(">)" << "\n";
    stream << "<Points>\n";
    stream << R"(  <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">)";

    stream << std::setprecision(std::numeric_limits<real>::max_digits10);
    for (const auto &point : points) {
      if constexpr (range == 1) {
        stream << point(0) << " 0 0 ";
      } else if constexpr (range == 2) {
        stream << point(0) << ' ' << point(1) << " 0 ";
      } else {
        stream << point(0) << ' ' << point(1) << ' ' << point(2) << ' ';
      }
    }
    stream << "</DataArray>\n";
    stream << "</Points>\n";

    stream << "<Cells>\n";
    stream << R"(  <DataArray type="UInt64" Name="connectivity" format="ascii">)";

    const auto mask = create_VTU_conn_mask<dim>(order);

    for (std::size_t cell_id = 0; cell_id < n_cells; ++cell_id) {
      const auto cell = std::span(connectivity).subspan(cell_id * n_pts_per_cell, n_pts_per_cell);
      for (const auto &point_id : mask) {
        stream << cell[point_id] << ' ';
      }
    }
    stream << "</DataArray>\n";

    stream << R"(  <DataArray type="UInt64" Name="offsets" format="ascii">)";
    for (std::size_t cell_id = 0; cell_id < n_cells; ++cell_id) {
      stream << (cell_id + 1) * n_pts_per_cell << ' ';
    }
    stream << "</DataArray>\n";

    const int cell_type = get_VTK_cell_type<dim>(order);
    stream << R"(  <DataArray type="Int32" Name="types" format="ascii">)";
    for (std::size_t cell_id = 0; cell_id < n_cells; ++cell_id) {
      stream << cell_type << ' ';
    }
    stream << "</DataArray>\n";

    stream << "</Cells>\n";

    stream << "</Piece>\n";
    stream << "</UnstructuredGrid>\n";
    stream << "</VTKFile>\n";

    stream.close();
  }

}// namespace

template<int dim, int range> ReparamMesh<dim, range>::ReparamMesh(const int order) : order_(order), points_()
{
  assert(1 < this->order_);
}

template<int dim, int range> void ReparamMesh<dim, range>::merge(const ReparamMesh<dim, range> &mesh)
{
  assert(this->order_ == mesh.order_);

  const auto &new_points = mesh.get_points();
  const auto pts_offset = this->points_.size();
  this->points_.insert(this->points_.end(), new_points.cbegin(), new_points.cend());

  this->connectivity_.reserve(this->connectivity_.size() + mesh.connectivity_.size());
  for (const auto &pt_id : mesh.connectivity_) {
    this->connectivity_.push_back(pt_id + pts_offset);
  }

  this->wires_connectivity_.reserve(this->wires_connectivity_.size() + mesh.wires_connectivity_.size());
  for (const auto &pt_id : mesh.wires_connectivity_) {
    this->wires_connectivity_.push_back(pt_id + pts_offset);
  }
}

template<int dim, int range>
template<int aux_dim>
  requires(aux_dim == dim && dim == range)
void ReparamMesh<dim, range>::add_full_cell(const BoundBox<dim> &domain, const bool wirebasket)
{
  const auto n_pts_per_cell = this->get_num_points_per_cell();
  this->points_.reserve(this->points_.size() + n_pts_per_cell);
  this->connectivity_.reserve(this->connectivity_.size() + n_pts_per_cell);

  const bool chebyshev = this->use_Chebyshev();

  const auto generate_point_in_01 = [this, chebyshev](const int ind) {
    if (chebyshev) {
      return ::algoim::bernstein::modifiedChebyshevNode(ind, this->order_);
    } else {
      return real(ind) / real(this->order_ - 1);
    }
  };

  Point<range> point;
  for (const auto &tid : TensorIndexRangeTP<dim>(this->order_)) {
    for (int dir = 0; dir < range; ++dir) {
      point(dir) = domain.min(dir) + (domain.length(dir) * generate_point_in_01(tid(dir)));
    }
    this->connectivity_.push_back(this->points_.size());
    this->points_.push_back(point);
  }

  const auto cell_id = static_cast<int>(this->get_num_cells()) - 1;

  if (wirebasket) {

    constexpr int n_edges = dim == 2 ? 4 : 12;

    for (int edge_id = 0; edge_id < n_edges; ++edge_id) {

      const auto edge_points_ids = this->get_edge_points(cell_id, edge_id);
      this->wires_connectivity_.insert(
        this->wires_connectivity_.end(), edge_points_ids.cbegin(), edge_points_ids.cend());
    }
  }
}

template<int dim, int range>
template<int aux_dim>
  requires(aux_dim == dim && dim == range)
void ReparamMesh<dim, range>::add_full_cells(const CartGridTP<dim> &grid,
  const std::vector<int> &cell_ids,
  const bool wirebasket)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  const auto n_pts_per_cell = this->get_num_points_per_cell();
  const auto n_pts = n_pts_per_cell * cell_ids.size();
  this->points_.reserve(this->points_.size() + n_pts);
  this->connectivity_.reserve(this->connectivity_.size() + n_pts);

  if (wirebasket) {
    const std::size_t n_edges_per_cell = dim == 2 ? 4 : 12;
    const auto n_wirebasket_pts = n_edges_per_cell * static_cast<std::size_t>(this->order_) * cell_ids.size();
    this->wires_connectivity_.reserve(this->wires_connectivity_.size() + n_wirebasket_pts);
  }

  for (const auto &cell_id : cell_ids) {
    const auto domain = grid.get_cell_domain(cell_id);
    this->add_full_cell(domain, wirebasket);
  }
}


template<int dim, int range> std::size_t ReparamMesh<dim, range>::get_num_cells() const
{
  const auto n_pts_per_cell = this->get_num_points_per_cell();
  assert(this->connectivity_.size() % n_pts_per_cell == 0);
  return this->connectivity_.size() / n_pts_per_cell;
}

template<int dim, int range> std::size_t ReparamMesh<dim, range>::get_num_points() const
{
  return this->points_.size();
}

template<int dim, int range> std::size_t ReparamMesh<dim, range>::get_num_points_per_cell() const
{
  return static_cast<std::size_t>(int_pow(this->order_, dim));
}

template<int dim, int range> const std::vector<Point<range>> &ReparamMesh<dim, range>::get_points() const
{
  return this->points_;
}

template<int dim, int range> const std::vector<std::size_t> &ReparamMesh<dim, range>::get_connectivity() const
{
  return this->connectivity_;
}

template<int dim, int range> const std::vector<std::size_t> &ReparamMesh<dim, range>::get_wires_connectivity() const
{
  return this->wires_connectivity_;
}

template<int dim, int range> int ReparamMesh<dim, range>::get_order() const
{
  return this->order_;
}

template<int dim, int range> void ReparamMesh<dim, range>::reserve_cells(const std::size_t n_new_cells)
{
  const auto n_pts_per_cell = this->get_num_points_per_cell();
  const auto n_new_points = n_new_cells * n_pts_per_cell;

  this->points_.reserve(this->points_.size() + n_new_points);
  this->connectivity_.reserve(this->connectivity_.size() + n_new_points);
}

template<int dim, int range> int ReparamMesh<dim, range>::allocate_cells(const std::size_t n_new_cells)
{
  const auto n_pts_per_cell = this->get_num_points_per_cell();
  const auto n_new_points = n_new_cells * n_pts_per_cell;
  this->points_.reserve(this->points_.size() + n_new_points);

  const auto n_cells = this->connectivity_.size() / n_pts_per_cell;
  this->connectivity_.resize(this->connectivity_.size() + n_new_points);

  return static_cast<int>(n_cells);
}

template<int dim, int range> bool ReparamMesh<dim, range>::use_Chebyshev(const int order)
{
  return order >= chebyshev_order;
}

template<int dim, int range> bool ReparamMesh<dim, range>::use_Chebyshev() const
{
  return use_Chebyshev(this->order_);
}

template<int dim, int range>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
void ReparamMesh<dim, range>::insert_cell_point(const Point<range> &point, const int cell_id, const int pt_id)
{
  const auto n_pts_per_cell = static_cast<int>(this->get_num_points_per_cell());
  const auto con_offset = cell_id * n_pts_per_cell;

  const auto con_id = con_offset + pt_id;
  at(this->connectivity_, con_id) = this->points_.size();

  this->points_.push_back(point);
}


template<int dim, int range>
bool ReparamMesh<dim, range>::check_subentity_degenerate(const std::vector<std::size_t> &points_ids,
  const Tolerance &tol) const
{
  assert(!points_ids.empty());

  const auto &point_0 = at(this->points_, static_cast<int>(points_ids.front()));

  return std::ranges::all_of(points_ids, [point_0, tol, this](const auto &pt_id) {
    const auto &point = at(this->points_, static_cast<int>(pt_id));
    return tol.coincident(point_0, point);
  });
}

template<int dim, int range>
void ReparamMesh<dim, range>::sort_edge_points(std::span<std::size_t> edge_points_ids) const
{
  // Ordering edge points such that the first point is smaller than the last one.
  bool reverse = edge_points_ids.back() < edge_points_ids.front();

  // If both are the same, the ids of the second points are compared.
  if (edge_points_ids.front() == edge_points_ids.back() && this->order_ > 2) {
    reverse = at(edge_points_ids, this->order_ - 2) < at(edge_points_ids, 1);
  }

  if (reverse) {
    std::ranges::reverse(edge_points_ids);
  }
}

template<int dim, int range>
template<int aux_dim>
  requires(aux_dim == dim && 1 < dim)
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
std::vector<std::size_t> ReparamMesh<dim, range>::get_edge_points(const int cell_id, const int edge_id) const
{
#ifndef NDEBUG
  constexpr int n_edges = dim == 2 ? 4 : 12;
  assert(0 <= edge_id && edge_id < n_edges);
#endif// NDEBUG

  const auto const_dirs = impl::get_edge_constant_dirs<dim>(edge_id);
  const auto sides = impl::get_edge_sides<dim>(edge_id);

  TensorIndexTP<dim> lower_bound(0);
  TensorIndexTP<dim> upper_bound(this->order_);

  for (int i = 0; i < (dim - 1); ++i) {
    if (sides(i) == 0) {
      upper_bound(const_dirs(i)) = 1;
    } else {
      lower_bound(const_dirs(i)) = this->order_ - 1;
    }
  }

  const TensorSizeTP<dim> order(this->order_);

  std::vector<std::size_t> point_ids;
  point_ids.reserve(static_cast<std::size_t>(this->order_));

  const auto offset = cell_id * static_cast<int>(this->get_num_points_per_cell());

  for (const auto tid : TensorIndexRangeTP<dim>(lower_bound, upper_bound)) {
    const auto pt_id = offset + tid.flat(order);
    point_ids.push_back(at(this->connectivity_, pt_id));
  }

  this->sort_edge_points(point_ids);

  return point_ids;
}

template<int dim, int range>
template<int sub_dim>
bool ReparamMesh<dim, range>::check_edge_in_subdomain(const std::vector<std::size_t> &edge_points_ids,
  const BoundBox<range> &domain,
  const Tolerance &tol) const
{
  const auto get_bounds = [&domain, tol, this](const std::size_t pt_id) {
    const auto &point = at(this->points_, static_cast<int>(pt_id));

    Vector<int, range> bounds{ -1 };
    for (int dir = 0; dir < range; ++dir) {
      if (tol.equal(point(dir), domain.min(dir))) {
        bounds(dir) = 0;
      } else if (tol.equal(point(dir), domain.max(dir))) {
        bounds(dir) = 1;
      }
    }
    return bounds;
  };

  const auto bounds_0 = get_bounds(edge_points_ids.front());

  return std::ranges::none_of(edge_points_ids, [bounds_0, get_bounds](const auto &pt_id) {
    const auto bounds = get_bounds(pt_id);
    int n_bounds{ 0 };
    for (int dir = 0; dir < range; ++dir) {
      if (bounds_0(dir) >= 0 && bounds_0(dir) == bounds(dir)) {
        ++n_bounds;
      }
    }
    return n_bounds < sub_dim;
  });
}

template<int dim, int range> void ReparamMesh<dim, range>::merge_coincident_points(const Tolerance &tol)
{
  if (points_.size() < 2) {
    return;
  }

  std::vector<std::size_t> points_map;
  make_points_unique(this->points_, tol, points_map);

  std::transform(connectivity_.begin(), connectivity_.end(), connectivity_.begin(), [&points_map](std::size_t pt_id) {
    return points_map.at(pt_id);
  });

  std::transform(wires_connectivity_.begin(),
    wires_connectivity_.end(),
    wires_connectivity_.begin(),
    [&points_map](std::size_t pt_id) { return points_map.at(pt_id); });

  this->purge_duplicate_wirebasket_edges();
}

template<int dim, int range>
void ReparamMesh<dim, range>::scale_points(const std::vector<int> &point_ids,
  const BoundBox<range> &old_domain,
  const BoundBox<range> &new_domain)
{
  Point<range> scale;
  Point<range> shift;
  for (int dir = 0; dir < range; ++dir) {
    scale(dir) = new_domain.length(dir) / old_domain.length(dir);
    shift(dir) = new_domain.min(dir) - old_domain.min(dir) * scale(dir);
  }

  for (const auto &point_id : point_ids) {
    auto &point = at(this->points_, point_id);
    for (int dir = 0; dir < range; ++dir) {
      point(dir) = scale(dir) * point(dir) + shift(dir);
    }
  }
}


template<int dim, int range>
void ReparamMesh<dim, range>::scale_points(const BoundBox<range> &old_domain, const BoundBox<range> &new_domain)
{
  const auto rng = std::ranges::iota_view<int, int>{ 0, static_cast<int>(this->get_num_points()) };
  const std::vector<int> point_ids(rng.begin(), rng.end());

  this->scale_points(point_ids, old_domain, new_domain);
}


template<int dim, int range> void ReparamMesh<dim, range>::sort_wirebasket_edges()
{
  // We first sort the indices every single edge, possibly reversing them.
  const auto order = static_cast<std::size_t>(this->order_);
  const auto n_edges = this->wires_connectivity_.size() / order;

  const auto edges = std::span<std::size_t>(this->wires_connectivity_);

  for (std::size_t wire_id = 0; wire_id < n_edges; ++wire_id) {
    const auto edge = edges.subspan(wire_id * order, order);
    this->sort_edge_points(edge);
  }


  // Then, we sort the all the edges in the wirebasket respect to each other.
  const auto rng = std::ranges::iota_view<std::size_t, std::size_t>{ 0, n_edges };
  std::vector<std::size_t> indices(rng.begin(), rng.end());

  std::ranges::sort(indices, [&edges, order](const std::size_t lhs, const std::size_t rhs) {
    const auto l_edge = edges.subspan(lhs * order, order);
    const auto r_edge = edges.subspan(rhs * order, order);
    for (std::size_t dir = 0; dir < order; ++dir) {
      if (l_edge[dir] != r_edge[dir]) {
        return l_edge[dir] < r_edge[dir];
      }
    }
    return false;
  });

  const auto conn_old = this->wires_connectivity_;
  const auto old_edges = std::span<const std::size_t>(conn_old);

  for (std::size_t i = 0; i < n_edges; ++i) {
    const auto edge = edges.subspan(i * order, order);
    const auto old_edge = old_edges.subspan(indices[i] * order, order);
    std::ranges::copy(old_edge, edge.begin());
  }
}

template<int dim, int range> void ReparamMesh<dim, range>::purge_duplicate_wirebasket_edges()
{
  this->sort_wirebasket_edges();

  std::vector<std::size_t> new_wires_conn;
  new_wires_conn.reserve(this->wires_connectivity_.size());

  const auto order_t = static_cast<std::ptrdiff_t>(this->order_);

  auto it_0 = this->wires_connectivity_.cbegin();
  const auto end = this->wires_connectivity_.cend();
  bool repetition{ false };
  for (auto it = it_0; it != end;) {

    const auto edge_0 = std::span<const std::size_t>(it, it + order_t);

    repetition = false;
    for (it += order_t; it < end; it += order_t) {
      const auto edge_1 = std::span<const std::size_t>(it, it + order_t);
      if (std::ranges::equal(edge_0, edge_1)) {
        if (!repetition) {
          repetition = true;
          new_wires_conn.insert(new_wires_conn.end(), it_0, it);
        }
      } else {
        if (repetition) {
          it_0 = it;
        }
        break;
      }
    }
  }
  if (!repetition) {
    new_wires_conn.insert(new_wires_conn.end(), it_0, end);
  }

  this->wires_connectivity_ = new_wires_conn;
}


template<int dim, int range> void ReparamMesh<dim, range>::write_VTK_file(const std::string &filename) const
{
  write_reparam_to_VTK_cells<dim, range>(this->points_, this->connectivity_, this->order_, filename);

  const auto wirebasket_filename = filename + "_wb";
  write_reparam_to_VTK_cells<1, range>(this->points_, this->wires_connectivity_, this->order_, wirebasket_filename);

  std::ofstream stream(filename + ".vtmb");
  stream << R"(<?xml version="1.0"?>)" << "\n";
  stream << R"(<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">)" << "\n";
  stream << "  <vtkMultiBlockDataSet>\n";

  stream << R"(    <DataSet index="0" name="interior" file=")" << filename << R"(.vtu"/>)" << "\n";
  stream << R"(    <DataSet index="1" name="boundary" file=")" << wirebasket_filename << R"(.vtu"/>)" << "\n";

  stream << "  </vtkMultiBlockDataSet>\n";
  stream << "</VTKFile>\n";
  stream.close();
}


template<int dim, int range> void ReparamMesh<dim, range>::permute_cell_directions(const std::size_t cell_id)
{
  assert(cell_id < this->get_num_cells());

  const TensorSizeTP<dim> order(this->order_);
  const auto offset = static_cast<int>(cell_id * this->get_num_points_per_cell());

  const auto swap = [offset, this, &order](const TensorIndexTP<dim> &tid_0, const TensorIndexTP<dim> &tid_1) {
    std::swap(at(this->connectivity_, offset + tid_0.flat(order)), at(this->connectivity_, offset + tid_1.flat(order)));
  };

  if constexpr (dim == 1) {
    for (int i = 0; i < this->order_ / 2; ++i) {
      swap(TensorIndexTP<1>(i), TensorIndexTP<1>(this->order_ - 1 - i));
    }
  } else {
    for (const auto &tid_0 : TensorIndexRangeTP<dim>(order)) {
      if (tid_0(0) <= tid_0(1)) {
        continue;
      }
      auto tid_1 = tid_0;
      std::swap(tid_1(0), tid_1(1));

      swap(tid_0, tid_1);
    }
  }
}

template<int dim, bool levelset>
std::shared_ptr<const ReparamMesh<levelset ? dim - 1 : dim, dim>>
  create_reparameterization(const UnfittedDomain<dim> &unf_domain, const int n_pts_dir)
{
  const auto *unf_impl_domain = dynamic_cast<const impl::UnfittedImplDomain<dim> *>(&unf_domain);
  assert(unf_impl_domain != nullptr);
  return impl::create_reparameterization<dim, levelset>(*unf_impl_domain, n_pts_dir);
}

template<int dim, bool levelset>
std::shared_ptr<const ReparamMesh<levelset ? dim - 1 : dim, dim>>
  create_reparameterization(const UnfittedDomain<dim> &unf_domain, const std::vector<int> &cells, const int n_pts_dir)
{
  const auto *unf_impl_domain = dynamic_cast<const impl::UnfittedImplDomain<dim> *>(&unf_domain);
  assert(unf_impl_domain != nullptr);
  return impl::create_reparameterization<dim, levelset>(*unf_impl_domain, cells, n_pts_dir);
}

// Instantiations.


template class ReparamMesh<1, 2>;
template class ReparamMesh<2, 2>;
template class ReparamMesh<2, 3>;
template class ReparamMesh<3, 3>;

template void ReparamMesh<2, 2>::add_full_cell<2>(const BoundBox<2> &, const bool);
template void ReparamMesh<3, 3>::add_full_cell<3>(const BoundBox<3> &, const bool);

template void ReparamMesh<2, 2>::add_full_cells<2>(const CartGridTP<2> &, const std::vector<int> &, const bool);
template void ReparamMesh<3, 3>::add_full_cells<3>(const CartGridTP<3> &, const std::vector<int> &, const bool);

template bool ReparamMesh<2, 2>::check_edge_in_subdomain<1>(const std::vector<std::size_t> &,
  const BoundBox<2> &,
  const Tolerance &) const;

template bool ReparamMesh<2, 3>::check_edge_in_subdomain<1>(const std::vector<std::size_t> &,
  const BoundBox<3> &,
  const Tolerance &) const;
template bool ReparamMesh<2, 3>::check_edge_in_subdomain<2>(const std::vector<std::size_t> &,
  const BoundBox<3> &,
  const Tolerance &) const;

template bool ReparamMesh<3, 3>::check_edge_in_subdomain<1>(const std::vector<std::size_t> &,
  const BoundBox<3> &,
  const Tolerance &) const;
template bool ReparamMesh<3, 3>::check_edge_in_subdomain<2>(const std::vector<std::size_t> &,
  const BoundBox<3> &,
  const Tolerance &) const;

template std::vector<std::size_t> ReparamMesh<2, 2>::get_edge_points<2>(const int, const int) const;
template std::vector<std::size_t> ReparamMesh<2, 3>::get_edge_points<2>(const int, const int) const;
template std::vector<std::size_t> ReparamMesh<3, 3>::get_edge_points<3>(const int, const int) const;

template std::shared_ptr<const ReparamMesh<1, 2>> create_reparameterization<2, true>(const UnfittedDomain<2> &,
  const int);
template std::shared_ptr<const ReparamMesh<2, 3>> create_reparameterization<3, true>(const UnfittedDomain<3> &,
  const int);
template std::shared_ptr<const ReparamMesh<2, 2>> create_reparameterization<2, false>(const UnfittedDomain<2> &,
  const int);
template std::shared_ptr<const ReparamMesh<3, 3>> create_reparameterization<3, false>(const UnfittedDomain<3> &,
  const int);

template std::shared_ptr<const ReparamMesh<1, 2>>
  create_reparameterization<2, true>(const UnfittedDomain<2> &, const std::vector<int> &, const int);
template std::shared_ptr<const ReparamMesh<2, 3>>
  create_reparameterization<3, true>(const UnfittedDomain<3> &, const std::vector<int> &, const int);

template std::shared_ptr<const ReparamMesh<2, 2>>
  create_reparameterization<2, false>(const UnfittedDomain<2> &, const std::vector<int> &, const int);
template std::shared_ptr<const ReparamMesh<3, 3>>
  create_reparameterization<3, false>(const UnfittedDomain<3> &, const std::vector<int> &, const int);

}// namespace qugar