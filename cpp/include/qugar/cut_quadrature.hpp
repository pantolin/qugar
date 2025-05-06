// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_LIBRARY_CUT_QUADRATURE_HPP
#define QUGAR_LIBRARY_CUT_QUADRATURE_HPP

//! @file cut_quadrature.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Definition of cut quadrature for unfitted domains.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <qugar/point.hpp>
#include <qugar/types.hpp>
#include <qugar/unfitted_domain.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>


namespace qugar {

enum ImmersedStatus : std::uint8_t { cut, full, empty };

template<int dim> struct CutCellsQuad
{
  //! @brief List of cut cells.
  std::vector<std::int64_t> cells;
  //! @brief List of number of quadrature points per cut cell.
  std::vector<int> n_pts_per_cell;

  //! @brief Vector of points.
  std::vector<Point<dim>> points;
  //! @brief Vector of weights.
  std::vector<real> weights;

  void reserve(const int n_cells, const int n_tot_pts);
};

template<int dim> struct CutIsoBoundsQuad
{
  //! @brief List of cut cells.
  std::vector<std::int64_t> cells;
  //! @brief List of (local) facet ids associated to the vector cells.
  std::vector<int> local_facet_ids;
  //! @brief List of number of quadrature points per cut facet.
  std::vector<int> n_pts_per_facet;

  //! @brief Vector of points.
  std::vector<Point<dim>> points;
  //! @brief Vector of weights.
  std::vector<real> weights;

  void reserve(const int n_cells, const int n_tot_pts);
};

template<int dim> struct CutUnfBoundsQuad
{
  //! @brief List of cut cells.
  std::vector<std::int64_t> cells;
  //! @brief List of number of quadrature points per cut cell.
  std::vector<int> n_pts_per_cell;

  //! @brief Vector of points.
  std::vector<Point<dim>> points;
  //! @brief Vector of weights.
  std::vector<real> weights;
  //! @brief Vector of normals.
  std::vector<Point<dim>> normals;

  void reserve(const int n_cells, const int n_tot_pts);
};

/**
 * @brief Creates quadrature for cells.
 *
 * For cut cells, the generated quadrature correspond to the interior part of the cell.
 * For full cells, the quadrature is the standard one, and for empty cells, no quadrature is created
 * (the generated quadrature is empty except for the number of points that is set to 0).
 *
 * @tparam dim Dimension of the domain.
 * @param unf_domain  Unitted domain.
 * @param cells Cells for which the quadrature is created.
 * @param n_pts_dir Number of points in each direction for generated custom quadratures.
 * @return Generated quadrature.
 */
template<int dim>
std::shared_ptr<const CutCellsQuad<dim>>
  create_quadrature(const UnfittedDomain<dim> &unf_domain, const std::vector<std::int64_t> &cells, int n_pts_dir);

/**
 * @brief Creates a quadrature for the unfitted boundary.
 * @tparam dim Dimension of the domain.
 * @param unf_domain  Unitted domain.
 * @param cells Cells for which the quadrature is created.
 * @param n_pts_dir Number of points in each direction for generated custom quadratures.
 * @param include_facet_unf_bdry  If true, the quadrature includes the parts of the unfitted boundary
 *                                that belong to the cells' facets.
 * @param exclude_ext_bdry If the previous parameter is true, and this is one is also true, the parts
 *                         of the unfitted boundary that belong to the external facets are not included.
 * @return Generated quadrature.
 */
template<int dim>
std::shared_ptr<const CutUnfBoundsQuad<dim>> create_unfitted_bound_quadrature(const UnfittedDomain<dim> &unf_domain,
  const std::vector<std::int64_t> &cells,
  int n_pts_dir,
  bool include_facet_unf_bdry,
  bool exclude_ext_bdry);

/**
 * @brief Creates quadrature for interior integrals.
 *
 * It creates quadratures for the internal part of interior facets.
 * I.e., fully internal facets or the cut (interior) part of cut facets.
 * Unfitted boundaries laying on the facet are not considered.
 *
 * @warning The provided facets are not checked to be interior facets. It is the caller's responsibility to
 *          provide the correct facets.
 *
 * @tparam dim Dimension of the domain.
 * @param unf_domain Unfitted domain.
 * @param cells Cells for which the quadrature is created.
 * @param facets Local facets for which the quadrature is created (this vector must be of the same size as cells).
 * @param n_pts_dir Number of points in each direction for generated custom quadratures.
 * @return Generated quadratures.
 */
template<int dim>
std::shared_ptr<const CutIsoBoundsQuad<dim - 1>> create_facets_quadrature_interior_integral(
  const UnfittedDomain<dim> &unf_domain,
  const std::vector<std::int64_t> &cells,
  const std::vector<int> &facets,
  int n_pts_dir);

/**
 * @brief Creates quadrature for exterior integrals.
 *
 * It creates quadrature for the active part of the facets that belong
 * to the external boundary of the domain.
 * The external boundary of the domain may be the external boundary of the grid or the unfitted boundary.
 *
 * @tparam dim Dimension of the domain.
 * @param unf_domain Unfitted domain.
 * @param cells Cells for which the quadrature is created.
 * @param facets Local facets for which the quadrature is created (this vector must be of the same size as cells).
 * @param n_pts_dir Number of points in each direction for generated custom quadratures.
 * @return Generated quadratures.
 */
template<int dim>
std::shared_ptr<const CutIsoBoundsQuad<dim - 1>> create_facets_quadrature_exterior_integral(
  const UnfittedDomain<dim> &unf_domain,
  const std::vector<std::int64_t> &cells,
  const std::vector<int> &facets,
  int n_pts_dir);


}// namespace qugar

#endif// QUGAR_LIBRARY_CUT_QUADRATURE_HPP