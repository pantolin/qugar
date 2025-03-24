// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_LIBRARY_CART_GRID_TP_HPP
#define QUGAR_LIBRARY_CART_GRID_TP_HPP

//! @file cart_grid_tp.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Definition of Cartesian grid class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>
#include <qugar/point.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace qugar {


//! @brief Class representing a <tt>dim</tt>-dimensional Cartesian tensor-product grid.
//!
//! @tparam dim Dimension of the grid's domain.
template<int dim> class CartGridTP
{
  //! Point type.
  using PointType = Point<dim>;

public:
  //! @name Constructors
  //@{
  //! @brief Construct a new @ref CartGridTP object from its breaks.
  //! @param breaks Breaks defining the cell intervals along the @p dim parametric directions.
  explicit CartGridTP(const std::array<std::vector<real>, dim> &breaks);

  //! @brief Construct a new @ref CartGridTP object from a domain and the number of intervals per
  //! direction.
  //! @param domain Domain of the grid.
  //! @param n_intrvs_dir Number of cell intervals along the @p dim parametric directions.
  CartGridTP(const BoundBox<dim> &domain, const std::array<std::size_t, dim> &n_intrvs_dir);

  //! @brief Construct a new @ref CartGridTP object from a [0,1] domain and the number of intervals per
  //! direction.
  //! @param n_intrvs_dir Number of cell intervals along the @p dim parametric directions.
  explicit CartGridTP(const std::array<std::size_t, dim> &n_intrvs_dir);
  //@}

  //! @name Query methods
  //@{
  //! @brief Gets the breaks along the given direction @p dir.
  //!
  //! @param dir Direction along which the breaks are extracted.
  //! @return Breaks along @p dir.
  [[nodiscard]] const std::vector<real> &get_breaks(int dir) const;

  //! @brief Gets the grid's domain.
  //! @return Grid's domain.
  [[nodiscard]] const BoundBox<dim> &get_domain() const;

  //! @brief Get the id of the cell the given @p point belongs to.
  //!
  //! @param point Point to query.
  //! @param tolerance Tolerance to be used in checkings.
  //! @return Id of the cell. If the @p point belongs to more than one cell,
  //! it returns the lowest index of the neighbor cells.
  [[nodiscard]] std::int64_t get_cell_id(const PointType &point, const Tolerance &tolerance = Tolerance()) const;

  //! @brief Checks if the given @p point is on the boundary of two cells (up to @p tolerance).
  //!
  //! @param point Point to query.
  //! @param tolerance Tolerance to be used in checkings.
  //! @return If the point is not at any boundary, the returned <tt>optiona</tt> type has no value.
  //! Otherwise, it contains the index of the constant direction (from <tt>0</tt> to <tt>dim-1</tt>) of the boundary.
  //! For instance, in a 2D grid, if the point lays in a vertical boundary between two cells, it returns 0, but
  //! if the boundary is horizontal, returns 1.
  [[nodiscard]] std::optional<int> at_cells_boundary(const PointType &point,
    const Tolerance &tolerance = Tolerance()) const;

  //! @brief Checks if a cell's facet is on the grids boundary.
  //!
  //! @param cell_id Id of the cell whose facet is checked.
  //! @param local_facet_id Id of the local facet of the cell.
  //! @return bool Whether the face is on the grid's boundary.
  [[nodiscard]] bool on_boundary(std::int64_t cell_id, int local_facet_id) const;

  //! @brief Gets the list of cells belonging to given facet of the grid.
  //!
  //! @param facet_id Id of the grid facet. It must be in the range [0, dim*2).
  //! @return List of cells on the facet.
  [[nodiscard]] std::vector<std::int64_t> get_boundary_cells(int facet_id) const;

  //! @brief Gets the flat index of a grid cell from the tensor index.
  //!
  //! @param tid Tensor cell index to transform.
  //! @return Flat cell index.
  [[nodiscard]] std::int64_t to_flat(const TensorIndexTP<dim> &tid) const;

  //! @brief Gets the tensor index of a grid cell from the flat index.
  //!
  //! @param fid Flat cell index to transform.
  //! @return Tensor cell index.
  [[nodiscard]] TensorIndexTP<dim> to_tensor(std::int64_t fid) const;

  //! @brief Gets the number of cells per direction.
  //!
  //! @return Number of cells per direction.
  [[nodiscard]] TensorSizeTP<dim> get_num_cells_dir() const;


  //! @brief Gets the total number of cell.
  //!
  //! @return Total number of cells.
  [[nodiscard]] std::size_t get_num_cells() const;

  //! @brief Gets an cell's domain.
  //!
  //! @param cell_fid Flat id of the cell.
  //! @return Bounding box of the cell's domain.
  [[nodiscard]] BoundBox<dim> get_cell_domain(std::int64_t cell_fid) const;

  //@}


private:
  //! @name Members
  //@{
  //! @brief Breaks definining the cell intervals along the @p dim parametric directions.
  std::array<std::vector<real>, dim> breaks_;
  //! @brief Domain of the grid.
  qugar::BoundBox<dim> domain_;
  //! Indices range.
  TensorIndexRangeTP<dim> range_;
  //@}
};

//! @brief Subgrid of a Cartesian grid TP.
//! It is a subset of the cells of a given grid.
//!
//! @tparam dim Parametric dimension.
template<int dim> class SubCartGridTP
{
public:
  //! @brief Constructor.
  //!
  //! @param grid Parent grid.
  //! @param indices_start Start indices of the coordinates of the parent grid.
  //! @param indices_end End indices of the coordinates of the parent grid.
  SubCartGridTP(const CartGridTP<dim> &grid,
    const TensorIndexTP<dim> &indices_start,
    const TensorIndexTP<dim> &indices_end);

  //! @brief Constructor.
  //!
  //! @param grid Parent grid.
  //! @param indices_range Indices range.
  SubCartGridTP(const CartGridTP<dim> &grid, const TensorIndexRangeTP<dim> &indices_range);

  //! @brief Constructor.
  //! Creates a subgrid containing the full grid.
  //!
  //! @param grid Parent grid.
  explicit SubCartGridTP(const CartGridTP<dim> &grid);

private:
  //! Parent grid.
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-const-or-ref-data-members)
  const CartGridTP<dim> &grid_;
  //! Indices range.
  TensorIndexRangeTP<dim> range_;

public:
  //! @brief Gets the number of cells (spans) per direction
  //! of the subgrid.
  //!
  //! The ordering is such that dimension 0 is inner-most, i.e.,
  //! iterates the fastest, while dimension dim-1 is outer-most and iterates the slowest.
  //!
  //! @return Number of cells per direction.
  [[nodiscard]] TensorSizeTP<dim> get_num_cells_dir() const;

  //! @brief Gets the flat index of a grid cells from the tensor index.
  //!
  //! @param tid Tensor cell index to transform.
  //! @return Flat cell index.
  [[nodiscard]] std::int64_t to_flat(const TensorIndexTP<dim> &tid) const;

  //! @brief Gets the total number of cells of the subgrid.
  //!
  //! @return Total number of cells.
  [[nodiscard]] std::size_t get_num_cells() const;

  //! @brief Checks if the subgrid has only one cell.
  //!
  //! @return True if it has only one cell, false otherwise.
  [[nodiscard]] bool is_unique_cell() const;

  //! @brief Gets a range describing the range of the subrid.
  //! @return Tensor-product indices range.
  [[nodiscard]] const TensorIndexRangeTP<dim> &get_range() const;

  //! @brief Splits the current subgrid along the direction
  //! with a largest number of cells.
  //!
  //! @return Two generated subgrid wrapped in shared pointers.
  [[nodiscard]] std::array<std::shared_ptr<const SubCartGridTP<dim>>, 2> split() const;

  //! @brief Creates the bounding box of the subgrid's domain.
  //!
  //! @return Bounding box of the subgrid.
  [[nodiscard]] BoundBox<dim> get_domain() const;

  //! @brief Gets the parent grid.
  //!
  //! @return Constant reference to the parent grid.
  [[nodiscard]] const CartGridTP<dim> &get_grid() const;

  //! @brief Gets the single cell in the subgrid.
  //!
  //! @return Single cell in the subgrid.
  [[nodiscard]] std::int64_t get_single_cell() const;
};

}// namespace qugar

#endif// QUGAR_LIBRARY_CART_GRID_TP_HPP