// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_REPARAM_MESH_HPP
#define QUGAR_IMPL_REPARAM_MESH_HPP

//! @file impl_reparam_mesh.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of reparameterization class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/point.hpp>
#include <qugar/reparam_mesh.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/tolerance.hpp>

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>


namespace qugar::impl {

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
template<int dim, int range> class ImplReparamMesh : public ReparamMesh<dim, range>
{
public:
  //! @brief Constructor.
  //!
  //! @param order Reparameterization order (number of points per direction).
  explicit ImplReparamMesh(int order);


  //! @brief Generates the wirebasket for all the cells of the current reparameterization.
  //!
  //! @param impl_funcs Implicit functions defining the domain.
  //! @param domain Domain (bounding box) to which the reparameterization corresponds.
  //! @param tol Tolerance to be used if points belong to the wirebasket.
  void generate_wirebasket(const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> &impl_funcs,
    const BoundBox<range> &domain,
    const Tolerance &tol);

  //! @brief Generates the wirebasket for the given @p cell_ids of the current reparameterization.
  //!
  //! @param impl_funcs List of implicit functions defining the domain.
  //! @param cell_ids List of cells ids to be considered for extracting the wirebasket.
  //! @param domain Domain (bounding box) to which the reparameterization corresponds.
  //! @param tol Tolerance to be used if points belong to the wirebasket.
  void generate_wirebasket(const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> &impl_funcs,
    const std::vector<int> &cell_ids,
    const BoundBox<range> &domain,
    const Tolerance &tol);

private:
  //! @brief Generates the wirebasket for the current reparameterization.
  //!
  //! @param cell_ids List of cells ids to be considered for extracting the wirebasket.
  //! @param domain Domain (bounding box) to which the reparameterization corresponds.
  //! @param tol Tolerance to be used if points belong to the wirebasket.
  //! @param check_face Function returning true if an edge in a domain's face must
  //!        be included in the wirebasket.
  //! @param check_internal Function returning true if an edge that does not correspond
  //!        to neither a face nor an edge of @p domain must be included in the wirebasket.
  void generate_wirebasket(const std::vector<int> &cell_ids,
    const BoundBox<range> &domain,
    const Tolerance &tol,
    const std::function<bool(const Point<range> &)> &check_face,
    const std::function<bool(const Point<range> &)> &check_internal);

public:
  //! @brief Reorients the cells in the mesh to ensure they are oriented positively.
  //!
  //! This function modifies the orientation of the cells (by modifying its connectivity)
  //! in the mesh so that all cells have a positive orientation.
  //!
  //! Positive orientation is considered the one such that determinant of the Jacobian
  //! (evaluated at the cell's mid-point) is positive.
  //!
  //! This is typically required for consistency in
  //! computational geometry and finite element methods, where the orientation of cells
  //! can affect the results of calculations.
  //!
  //! @note The wirebasket connectivity is not modified at all.
  template<int dim_aux = dim>
    requires(dim_aux == dim && range == dim)
  void orient_cells_positively();

  //! @brief Reorients the cells in the mesh to ensure they are oriented positively.
  //!
  //! This function modifies the orientation of the cells (by modifying its connectivity)
  //! in the mesh so that all cells have a positive orientation.
  //!
  //! Positive orientation is considered the one such that determinant of the Jacobian
  //! (evaluated at the cell's mid-point) is positive.
  //!
  //! This is typically required for consistency in
  //! computational geometry and finite element methods, where the orientation of cells
  //! can affect the results of calculations.
  //!
  //! @param cell_ids List of cells ids to be considered for reorientation.
  //!
  //! @note The wirebasket connectivity is not modified at all.
  template<int dim_aux = dim>
    requires(dim_aux == dim && range == dim)
  void orient_cells_positively(const std::vector<int> &cell_ids);

private:
  //! @brief Reorients the levelset cells in the mesh to ensure their normal is an outer normal.
  //!
  //! This function modifies the orientation of the levelset cells (by modifying its connectivity)
  //! in the mesh so that all cells have a positive orientation.
  //!
  //! Positive orientation is considered the one such that the cell normal
  //! (evaluated at the cell's mid-point) goes along the normal provided by the given
  //! function at that point. I.e., the dot product between both normals is positive.
  //!
  //! This is typically required for consistency in computational geometry and finite
  //! element methods, where the orientation of cells can affect the results of calculations.
  //!
  //! @param cell_ids List of cells ids to be considered for reorientation.
  //! @param outer_normal_computer Function providing the expected outer normal (to compare with)
  //! at a given point.
  //!
  //! @note The wirebasket connectivity is not modified at all.
  template<int dim_aux = dim>
    requires(dim_aux == dim && range == (dim + 1))
  void orient_levelset_cells_positively(const std::vector<int> &cell_ids,
    const std::function<Point<range>(const Point<range> &)> &outer_normal_computer);

public:
  //! @brief Reorients the levelset cells in the mesh to ensure their normal is an outer normal.
  //!
  //! This function modifies the orientation of the levelset cells (by modifying its connectivity)
  //! in the mesh so that all cells have a positive orientation.
  //!
  //! Positive orientation is considered the one such that the cell normal
  //! (evaluated at the cell's mid-point) goes along the normal provided by the given
  //! implicit functions at that point. I.e., the dot product between both normals is positive.
  //!
  //! Among all the provided functions, the one whose value at the mid-point is minimum is chosen
  //! for evaluating its normals.
  //!
  //! This is typically required for consistency in computational geometry and finite
  //! element methods, where the orientation of cells can affect the results of calculations.
  //!
  //! @param impl_funcs List of functions that define the implicit domain and according
  //! to which the orientation is computed.
  //!
  //! @note The wirebasket connectivity is not modified at all.
  template<int dim_aux = dim>
    requires(dim_aux == dim && range == (dim + 1))
  void orient_levelset_cells_positively(
    const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> &impl_funcs);

  //! @brief Reorients the levelset cells in the mesh to ensure their normal is an outer normal.
  //!
  //! This function modifies the orientation of the levelset cells (by modifying its connectivity)
  //! in the mesh so that all cells have a positive orientation.
  //!
  //! Positive orientation is considered the one such that the cell normal
  //! (evaluated at the cell's mid-point) goes along the normal provided by the given
  //! implicit functions at that point. I.e., the dot product between both normals is positive.
  //!
  //! Among all the provided functions, the one whose value at the mid-point is minimum is chosen
  //! for evaluating its normals.
  //!
  //! This is typically required for consistency in computational geometry and finite
  //! element methods, where the orientation of cells can affect the results of calculations.
  //!
  //! @param impl_funcs List of functions that define the implicit domain and according
  //! to which the orientation is computed.
  //!
  //! @param cell_ids List of cells ids to be considered for reorientation.
  //!
  //! @note The wirebasket connectivity is not modified at all.
  template<int dim_aux = dim>
    requires(dim_aux == dim && range == (dim + 1))
  void orient_levelset_cells_positively(
    const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> &impl_funcs,
    const std::vector<int> &cell_ids);

  //! @brief Orients the cells reparameterization a domain facet positively.
  //!
  //! This function ensures that the cells reparameterizing a facet are oriented positively
  //! according to the specified local facet ID of a hyperrectangular domain.
  //! I.e., the cells normals are oriented towards the exterior of the domain.
  //!
  //! @param local_facet_id The local ID of the facet to be oriented positively.
  template<int dim_aux = dim>
    requires(dim_aux == dim && range == (dim + 1))
  void orient_facet_cells_positively(int local_facet_id);

  //! @brief Orients the cells reparameterization a domain facet positively.
  //!
  //! This function ensures that the cells reparameterizing a facet are oriented positively
  //! according to the specified local facet ID of a hyperrectangular domain.
  //! I.e., the cells normals are oriented towards the exterior of the domain.
  //!
  //! @param local_facet_id The local ID of the facet to be oriented positively.
  //! @param cell_ids List of cells ids to be considered for reorientation.
  template<int dim_aux = dim>
    requires(dim_aux == dim && range == (dim + 1))
  void orient_facet_cells_positively(int local_facet_id, const std::vector<int> &cell_ids);
};


}// namespace qugar::impl

#endif// QUGAR_IMPL_REPARAM_MESH_HPP
