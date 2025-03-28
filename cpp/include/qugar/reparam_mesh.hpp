// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_REPARAM_MESH_HPP
#define QUGAR_REPARAM_MESH_HPP

//! @file reparam_mesh.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of reparameterization class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/unfitted_domain.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>


namespace qugar {

//! @brief Class for storing an implicit domain reparameterization
//! using Lagrange cells.
//!
//! This reparameterization is non-conforming (it may contain hanging nodes),
//! but is intended to be used for visualization purposes.
//!
//! It stores the cells as a list of point and the connectivity.
//! It also stores the connectivity for the wirebasket of the reparameterization.
//!
//! @tparam dim Parametric dimension of the reparameterization.
//! @tparam range Range (or physical dimension) of the reparameterization.
template<int dim, int range> class ReparamMesh
{
public:
  //! Order for deciding if Chebyshev nodes are used for defining the Lagrange polynomials
  //! instead of equally spaced nodes.
  //! If the reparameterization order is greater than this value, Chebyshev nodes are used.
  static const int chebyshev_order = 7;

  //! @brief Constructor.
  //!
  //! @param order Reparameterization order (number of points per direction).
  explicit ReparamMesh(int order);

  //! @brief Default virtual destructor.
  virtual ~ReparamMesh() = default;

protected:
  //! Reparameterization order (number of points per direction).
  int order_;
  //! Vector of points.
  std::vector<Point<range>> points_;
  //! Cells' connectivity.
  std::vector<std::size_t> connectivity_;
  //! Cells wires' connectivity.
  std::vector<std::size_t> wires_connectivity_;

public:
  //! @brief Merges the given mesh into the current mesh.
  //!
  //! This function takes another mesh as input and merges it with the current mesh.
  //! The points, connectivity, and wires connectivity are appended, shifting the indices
  //! by the number of points in the current mesh.
  //!
  //! @note Coincident points are not merged and duplicated wires are not purged.
  //! @warning Both meshes must have the same order.
  //!
  //! @param mesh The mesh to be merged with the current mesh.
  void merge(const ReparamMesh<dim, range> &mesh);

  //! @brief Adds a full cell to the mesh corresponding to the given domain.
  //!
  //! @param domain The bounding box that defines the domain to which the full cell will be added.
  //! @param wirebasket Whether wirebasket for each cell should be added.
  template<int aux_dim = dim>
    requires(aux_dim == dim && dim == range)
  void add_full_cell(const BoundBox<dim> &domain, bool wirebasket);

  //! @brief Adds full cells to the mesh corresponding to the grid.
  //!
  //! @param grid Cartesian grid that define the cells to be added.
  //! @param cell_ids List of cells to add.
  //! @param wirebasket Whether wirebasket for each cell should be added.
  //!
  //! @note No coincidents points are merged. If needed, `merge_coincident_points`
  //! should be called after this method.
  template<int aux_dim = dim>
    requires(aux_dim == dim && dim == range)
  void add_full_cells(const CartGridTP<dim> &grid, const std::vector<std::int64_t> &cell_ids, bool wirebasket);

  //! @brief Determines whether the Chebyshev nodes are used, or equally spaced nodes.
  //!
  //! Chebyshev nodes should be used for high-order reparameterizations if the
  //! reparameterization order is greater or equal than @ref chebyshev_order.
  //!
  //! @return true if the Chebyshev method is used, false otherwise.
  [[nodiscard]] bool use_Chebyshev() const;

  //! @brief Determines whether the Chebyshev nodes are used, or equally spaced nodes.
  //!
  //! Chebyshev nodes should be used for high-order reparameterizations if the
  //! reparameterization @p order is greater or equal than @ref chebyshev_order.
  //!
  //! @return order Reparameterization order.
  //! @return true if the Chebyshev method is used, false otherwise.
  [[nodiscard]] static bool use_Chebyshev(int order);

  //! @brief This method sets the given point @p point, with index @p pt_id,
  //! to the cell designed by @p cell_id.
  //!
  //! @param point Point to be set.
  //! @param cell_id Flat index of the cell in which the point is set.
  //! @param pt_id Flat index of the reparameterization point referred to the cell.
  void insert_cell_point(const Point<range> &point, int cell_id, int pt_id);

  //! @brief Merges coincident points in the reparameterization up to tolerance.
  //!
  //! @param tol Tolerance to be used in the comparisons between points.
  //!
  //! @note The connecitivity of cells (and wirebasket) is updated accordingly,
  //! and duplicated wirebasket edges are purged.
  void merge_coincident_points(const Tolerance &tol);

  //! @brief Scales points from an old domain to a new domain.
  //!
  //! This function takes two bounding boxes representing the old and new domains,
  //! and scales the points from the old to the new one.
  //!
  //! @param point_ids List of points to scale.
  //! @param old_domain The bounding box representing the old domain.
  //! @param new_domain The bounding box representing the new domain.
  void scale_points(const std::vector<int> &point_ids,
    const BoundBox<range> &old_domain,
    const BoundBox<range> &new_domain);

  //! @brief Scales points from an old domain to a new domain.
  //!
  //! This function takes two bounding boxes representing the old and new domains,
  //! and scales the points from the old to the new one.
  //!
  //! @param old_domain The bounding box representing the old domain.
  //! @param new_domain The bounding box representing the new domain.
  void scale_points(const BoundBox<range> &old_domain, const BoundBox<range> &new_domain);

  //! @brief Retrieves the points the reparameterization points.
  //!
  //! @return A constant reference to a vector containing points.
  [[nodiscard]] const std::vector<Point<range>> &get_points() const;

  //! @brief Retrieves the reparameterization connectivity.
  //!
  //! @return A constant reference to the connectivity vector.
  [[nodiscard]] const std::vector<std::size_t> &get_connectivity() const;

  //! @brief Retrieves the wirebasket reparameterization connectivity.
  //!
  //! @return A constant reference to the wirebasket connectivity vector.
  [[nodiscard]] const std::vector<std::size_t> &get_wires_connectivity() const;

  //! @brief Retrieves the reparameterization's order.
  //!
  //! @return int Reparameterization order.
  [[nodiscard]] int get_order() const;

  //! @brief Retrieves the number of reparameterization cells.
  //!
  //! @return The number of reparameterization cells.
  [[nodiscard]] std::size_t get_num_cells() const;

  //! @brief Retrieves the number of points.
  //!
  //! @return The number of points.
  [[nodiscard]] std::size_t get_num_points() const;

  //! @brief Retrieves the number of points per cell, that depends on the reparameterization order.
  //!
  //! @return The number of points per cell as a std::size_t.
  [[nodiscard]] std::size_t get_num_points_per_cell() const;

  //! @brief Writes the reparameterization data to a VTK file.
  //!
  //! This function takes a reparameterization object and writes its data to a file
  //! in VTK format, which can be used for visualization in tools like ParaView.
  //!
  //! If the repameterization contains a wirebasket mesh, the generated VTK files is composed
  //! of three files: a file with extension .vtmb that calls two .vtu files (one for the internal
  //! reparameterization), and one for the wirebasket.
  //!
  //! If no wirebasket mesh is present, the generated VTK file is a single .vtu file for the internal
  //! reparameterization.
  //!
  //! The reparameterizations of both the internal and wirebasket meshes is written as high-order Lagrange
  //! VTK cells.
  //!
  //! @param filename The name of the file to which the data will be written, without extension.
  void write_VTK_file(const std::string &filename) const;

  //! @brief Permutes the directions of the given cell.
  //!
  //! If dimension is greater or equal than 2, the first two direction are permuted.
  //! Otherwise, if the dimension is 1, the first direction is reversed.
  //!
  //! @param cell_id The identifier of the cell whose directions are to be permuted.
  void permute_cell_directions(std::size_t cell_id);

  //! @brief Reserves memory for a specified number of cells.
  //!
  //! It only reserves memory, no resize performed, for poitns and cells connectivity.
  //! Not for wires.
  //!
  //! @param n_new_cells The number of cells to allocate memory for.
  void reserve_cells(std::size_t n_new_cells);

  //! @brief Allocates memory for a specified number of cells.
  //!
  //! For the points, it just reserves memory for the new points. Thus, new points
  //! should be pushed back.
  //!
  //! However, for the cells connectivity, it resizes the array, ready to insert
  //! new indices at the corresponding positions.
  //!
  //! @param n_new_cells The number of cells to allocate memory for.
  //! @return Id of the first allocated cell.
  int allocate_cells(std::size_t n_new_cells);

protected:
  //! @brief Checks if a reparameterization cell's edge belongs to a subentity
  //! of a @p domain.
  //!
  //! @tparam sub_dim Parametric dimension of the subdomain.
  //! @param edge_points_ids Vector of point ids corresponding to the edge.
  //! @param domain Bounding box of the domain being considered.
  //! @param tol Tolerance to be used if points belong to the wirebasket.
  //! @return True if the edge sits on the subentity of dimension @p sub_dim of @p domain,
  //! false otherwise.
  template<int sub_dim>
  [[nodiscard]] bool check_edge_in_subdomain(const std::vector<std::size_t> &edge_points_ids,
    const BoundBox<range> &domain,
    const Tolerance &tol) const;

  //! @brief Checks if a reparameterization cell's sub-entity is degenerate (it has zero length).
  //!
  //! @param points_ids Vector of point ids corresponding to the sub-entity.
  //! @param tol Tolerance to decide if two points are coincident.
  //! @return True if the subentity is degenerate, false otherwise.
  [[nodiscard]] bool check_subentity_degenerate(const std::vector<std::size_t> &points_ids, const Tolerance &tol) const;

  //! @brief Gets the ids of the points of an edge of a reparameterization cell.
  //!
  //! @param cell_id Id of the reparameterization cell to consider.
  //! @param edge_id Edge id of the reparameterization cell to consider.
  //! @return Vector of point ids of the edge. They are sorted such that the first point
  //! is smaller than the last one. If both are the same, the ids of the second points
  //! (from the begining and the end) are compared.
  template<int aux_dim = dim>
    requires(aux_dim == dim && 1 < dim)
  [[nodiscard]] std::vector<std::size_t> get_edge_points(int cell_id, int edge_id) const;

  //! @brief Sorts the given edge points by their IDs.
  //!
  //! They are sorted such that the first point is smaller than the last one.
  //! If both indices are equal, the ids of the second points
  //! (from the begining and the end) are compared.
  //!
  //! @param edge_points_ids A span of edge point IDs to be sorted.
  void sort_edge_points(std::span<std::size_t> edge_points_ids) const;

  void sort_wirebasket_edges();

  void purge_duplicate_wirebasket_edges();
};

template<int dim, bool levelset>
std::shared_ptr<const ReparamMesh<levelset ? dim - 1 : dim, dim>>
  create_reparameterization(const UnfittedDomain<dim> &unf_domain, int n_pts_dir);

template<int dim, bool levelset>
std::shared_ptr<const ReparamMesh<levelset ? dim - 1 : dim, dim>> create_reparameterization(
  const UnfittedDomain<dim> &unf_domain,
  const std::vector<std::int64_t> &cells,
  int n_pts_dir);


}// namespace qugar

#endif// QUGAR_REPARAM_MESH_HPP
