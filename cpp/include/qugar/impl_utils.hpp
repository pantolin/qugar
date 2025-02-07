// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_UTILS_HPP
#define QUGAR_IMPL_UTILS_HPP

//! @file impl_utils.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of utility functions for implicit geometries.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/domain_function.hpp>
#include <qugar/point.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <utility>
#include <vector>


namespace qugar::impl {

//! @brief Checks if a @p point belongs to the levelset of an implicit
//! function @p phi.
//!
//! @tparam dim Dimension of the function and point.
//! @param phi Implicit function to be tested.
//! @param point Point to be tested.
//! @param tol Tolerance to be used for checking if the value of
//! @p phi is (close to) zero. If not provided, default tolerance is
//! used.
//! @return Whether the point belong to the levelset of @p phi.
template<int dim>
[[nodiscard]] bool on_levelset(const ImplicitFunc<dim> &phi, const Point<dim> &point, Tolerance tol = Tolerance());

//! @brief Gets the constant direction of the local facet.
//!
//! The constant direction is computed as the floor of division by 2
//! of @p local_facet_id.
//!
//! @param local_facet_id Id of the facet referred to a cell. It must be
//! a value in the range [0, 2*dim[.
//! @return Constant direction in the range [0,dim[.
template<int dim> [[nodiscard]] int get_facet_constant_dir(int local_facet_id);

//! @brief Gets the side of the facet. Either 0 or 1.
//!
//! I.e., it returns if the local facet corresponds to the side of the
//! cell's bounding box with minimum coordinate along the constant
//! direction of the facet (0), or the maximum (1).
//!
//! Side 0 correspond to even values of @p local_facet_id, and side 1
//! to odd values.
//!
//! @param local_facet_id Id of the facet referred to a cell. It must be
//! a value in the range [0, 2*dim[.
//! @return Side of the facet. Either 0 or 1.
template<int dim> [[nodiscard]] int get_facet_side(int local_facet_id);

//! @brief Get the local facet ID for a given const direction and side.
//!
//! This function returns the local facet ID based on the specified constant
//! direction and side. It is used in the context of a multi-dimensional
//! space where facets (or faces) of a geometric entity are identified.
//!
//! @tparam dim The dimension of the space.
//! @param const_dir The constant direction for which the local facet ID is to be determined.
//! @param side The side (0 or 1) in the specified direction.
//! @return The local facet ID corresponding to the given direction and side.
template<int dim> [[nodiscard]] int get_local_facet_id(int const_dir, int side);

//! @brief Gets the constant directions of the local edge.
//!
//! The edges are assumed to follow a lexicographical numbering.
//!
//! @param local_edge_id Id of the edge referred to a cell.
//! @return Constant directions in the range [0,dim[.
template<int dim> [[nodiscard]] Vector<int, dim - 1> get_edge_constant_dirs(int local_edge_id);

//! @brief Gets the sides of the edge. Either 0 or 1 along each constant direction.
//!
//! Along a constant direction, it returns if the local edge corresponds
//! to the side of the cell's bounding box with minimum coordinate along the constant
//! direction of the facet (0), or the maximum (1).
//!
//!
//! @param local_edge_id Id of the edge referred to a cell.
//! @return Sides of the edge.
template<int dim> [[nodiscard]] Vector<int, dim - 1> get_edge_sides(int local_edge_id);


template<int dim, int range> class ImplReparamMesh;


//! @brief Struct for storing and managing computed roots and intervals
//! of an implicit function.
//! @tparam dim Dimension of the point at which the intervals are computed.
template<int dim> struct RootsIntervals
{
  /// List of roots.
  std::vector<real> roots;
  /// Point at which roots are computed.
  Point<dim> point;
  /// Restrictions to which root correspond to (there is a one to correspondence).
  std::vector<int> func_ids;
  /// Flags indicating if the intervals defined by two consecutive roots are active.
  std::vector<bool> active_intervals;

private:
  std::vector<std::pair<real, int>> roots_ids;

public:
  //! @brief Default constructor.
  RootsIntervals();

  //! @brief Clears the container to the initial state.
  void clear();

  //! @brief Adds a new root.
  //!
  //! @param root New root to be added.
  //! @param func_id If of the restriction to which the @p root belongs to.
  //!
  //! @note The roots are neither sorted nor adjusted (for degeneracies) after
  //! appending the new root.
  void add_root(real root, int func_id);

  //! @brief Checks whether the container is empty
  //!
  //! @return True if empty, i.e. there are no roots, false otherwise.
  [[nodiscard]] bool empty() const;

  //! @brief Gets the number of roots in the container.
  //!
  //! @return Number of roots.
  [[nodiscard]] int get_num_roots() const;

  //! @brief Sorts (in increasing order) the roots in the container
  //! and the according restriction indices func_ids.
  void sort_roots();

  //! @brief Adjust the container roots by sorting them
  //! and forcing near roots (up to a @p _olerance) to be coincident.
  //!
  //! @param tol Tolerance to be used in the comparisons between roots.
  //! @param x0 Start of the interval to which the roots belong to.
  //! @param x1 End of the interval to which the roots belong to.
  void adjust_roots(const Tolerance &tol, real x0, real x1);
};
}// namespace qugar::impl

#endif// QUGAR_IMPL_UTILS_HPP
