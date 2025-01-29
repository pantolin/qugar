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
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <qugar/point.hpp>
#include <qugar/types.hpp>
#include <qugar/unfitted_domain.hpp>

#include <vector>

#include <cstddef>


namespace qugar {

enum ImmersedStatus { cut, full, empty };

template<int dim> struct CutCellsQuad
{
  //! @brief List of cut cells.
  std::vector<int> cells;
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
  std::vector<int> cells;
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

template<int dim> struct CutIntBoundsQuad
{
  //! @brief List of cut cells.
  std::vector<int> cells;
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

template<int dim>
std::shared_ptr<const CutCellsQuad<dim>>
  create_quadrature(const UnfittedDomain<dim> &unf_domain, const std::vector<int> &cells, int n_pts_dir);

template<int dim>
std::shared_ptr<const CutIntBoundsQuad<dim>>
  create_interior_bound_quadrature(const UnfittedDomain<dim> &unf_domain, const std::vector<int> &cells, int n_pts_dir);

template<int dim>
std::shared_ptr<const CutIsoBoundsQuad<dim - 1>> create_facets_quadrature(const UnfittedDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const std::vector<int> &facets,
  int n_pts_dir);


}// namespace qugar

#endif// QUGAR_LIBRARY_CUT_QUADRATURE_HPP