// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_REPARAM_GENERAL_HPP
#define QUGAR_IMPL_REPARAM_GENERAL_HPP

//! @file impl_reparam_general.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of tools for creating reparameterizations of general functions.
//!
//! @date 2025-01-04
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_reparam_mesh.hpp>

#include <memory>

namespace qugar::impl {


//! @brief Reparameterizes the domain defined by a general implicit function.
//!
//! @note The generated reparameterization has a wirebasket associated, but coincident were not merged.
//!
//!
//! @tparam dim Parametric dimension of the function.
//! @tparam S Flag indicating if the reparameterization must
//!         be performed only for the levelset surface (true), i.e.,
//!         the manifold where the either of the two Bezier functions are
//!         equal to 0, or the subregion (false) between those surfaces,
//!         where both Bezier functions are negative.
//! @param func Function to reparameterize.
//! @param domain Domain to reparameterize.
//! @param order Order of the reparameterization (number of points
//!        per direction in each reparameterization cell).
//! @return Generated reparameterization.
template<int dim, bool S = false>
std::shared_ptr<ImplReparamMesh<S ? dim - 1 : dim, dim>>
  reparam_general(const ImplicitFunc<dim> &func, const BoundBox<dim> &domain, int order);

//! @brief Reparameterizes the domain defined by a general implicit function.
//!
//! @note The generated reparameterization has a wirebasket associated, but coincident were not merged.
//!
//!
//! @tparam dim Parametric dimension of the function.
//! @tparam S Flag indicating if the reparameterization must
//!         be performed only for the levelset surface (true), i.e.,
//!         the manifold where the either of the two Bezier functions are
//!         equal to 0, or the subregion (false) between those surfaces,
//!         where both Bezier functions are negative.
//! @param func Function to reparameterize.
//! @param domain Domain to reparameterize.
//! @param reparam Reparameterization container to which new generated cells
//! are appended to.
template<int dim, bool S = false>
void reparam_general(const ImplicitFunc<dim> &func,
  const BoundBox<dim> &domain,
  ImplReparamMesh<S ? dim - 1 : dim, dim> &reparam);

//! @brief Reparameterizes a face of domain defined by an implicit function.
//! It reparameterizes one of the 2*dim faces of the domain.
//!
//! @note The generated reparameterization has a wirebasket associated, but coincident were not merged.
//!
//! @tparam dim Parametric dimension of the function.
//! @param func Function to reparameterize.
//! @param domain Domain to reparameterize.
//! @param facet_id Id of the face to reparameterize. It must be a value in the range [0, 2*dim[.
//! @param order Order of the reparameterization (number of points
//!        per direction in each reparameterization cell).
//! @return Generated reparameterization.
template<int dim>
std::shared_ptr<ImplReparamMesh<dim - 1, dim>>
  reparam_general_facet(const ImplicitFunc<dim> &func, const BoundBox<dim> &domain, int facet_id, int order);

//! @brief Reparameterizes a face of domain defined by an implicit function.
//! It reparameterizes one of the 2*dim faces of the domain.
//!
//! @note The generated reparameterization has a wirebasket associated, but coincident were not merged.
//!
//! @tparam dim Parametric dimension of the function.
//! @param func Function to reparameterize.
//! @param domain Domain to reparameterize.
//! @param facet_id Id of the face to reparameterize. It must be a value in the range [0, 2*dim[.
//! @param reparam Reparameterization container to which new generated cells
//! are appended to.
template<int dim>
void reparam_general_facet(const ImplicitFunc<dim> &func,
  const BoundBox<dim> &domain,
  int facet_id,
  ImplReparamMesh<dim - 1, dim> &reparam);

}// namespace qugar::impl

#endif// QUGAR_IMPL_REPARAM_GENERAL_HPP
