// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_unfitted_domain.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of UnfittedImplDomain class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_unfitted_domain.hpp>

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/cut_quadrature.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/unfitted_domain.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/hyperrectangle.hpp>
#include <algoim/interval.hpp>
#include <algoim/quadrature_general.hpp>
#include <algoim/quadrature_multipoly.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qugar::impl {


namespace alg = ::algoim;

namespace {


  //! @brief Checks if the subgrid intersects with the target cells.
  //!
  //! This function determines whether any of the target cells are within the range
  //! of the given subgrid by checking if at least one cell ID from the target cells
  //! exists within the subgrid's range.
  //!
  //! @tparam dim The dimensionality of the grid.
  //! @param subgrid The subgrid to check for intersection with target cells.
  //! @param target_cells A vector of cell indices to test for intersection.
  //! @return `true` if any target cell is within the subgrid's range; `false` otherwise.
  template<int dim> bool intersect(const SubCartGridTP<dim> &subgrid, const std::vector<int> &target_cells)
  {
    const auto &range = subgrid.get_range();
    return std::any_of(
      target_cells.cbegin(), target_cells.cend(), [&range](const int cell_id) { return range.is_in_range(cell_id); });
  }


  //! Inserts cell IDs from a subgrid into a container.
  //!
  //! @param subgrid The subgrid from which to extract cell IDs.
  //! @param target_cells Optional vector of target cell IDs. If provided, only these cells
  //!                     within the subgrid's range will be inserted into the container.
  //! @param container The container to store the cell IDs.
  //!
  //! If `target_cells` is provided, this function inserts the cell IDs from `target_cells`
  //! that are within the range of `subgrid` into `container`. If `target_cells` is not provided,
  //! all cell IDs from the `subgrid` are inserted into `container`.
  template<int dim>
  void insert_cells(const SubCartGridTP<dim> &subgrid,
    const std::optional<std::vector<int>> &target_cells,
    std::vector<int> &container)
  {
    const auto &range = subgrid.get_range();

    if (target_cells.has_value()) {
      const auto &cells = target_cells.value();
      assert(intersect(subgrid, cells));

      container.reserve(container.size() + cells.size());
      copy_if(cells.cbegin(), cells.cend(), std::back_inserter(container), [&range](const auto cell_id) {
        return range.is_in_range(cell_id);
      });

    } else {
      container.reserve(container.size() + static_cast<std::size_t>(subgrid.get_num_cells()));
      for (auto it = range.cbegin(); it != range.cend(); ++it) {
        container.push_back(subgrid.to_flat(*it));
      }
    }
  }

  template<int dim>
  void insert_facets(const SubCartGridTP<dim> &subgrid,
    const std::optional<std::vector<int>> &target_cells,
    const ImmersedFacetStatus facet_value,
    std::unordered_map<int, typename UnfittedImplDomain<dim>::FacetsStatus> &cells_facets)
  {
    const auto &range = subgrid.get_range();

    typename UnfittedImplDomain<dim>::FacetsStatus facets;
    facets.fill(facet_value);

    if (target_cells.has_value()) {
#ifndef NDEBUG
      const auto &cells = target_cells.value();
      assert(intersect(subgrid, cells));
#endif// NDEBUG


      for (auto it = range.cbegin(); it != range.cend(); ++it) {
        const auto cell_id = subgrid.to_flat(*it);
        if (range.is_in_range(cell_id)) {
          cells_facets.emplace(cell_id, facets);
        }
      }

    } else {

      for (auto it = range.cbegin(); it != range.cend(); ++it) {
        const auto cell_id = subgrid.to_flat(*it);
        cells_facets.emplace(cell_id, facets);
      }
    }
  }

  template<int dim>
  std::function<FuncSign(const BoundBox<dim> &)> create_compute_sign_function(const ImplicitFunc<dim> &phi)
  {
    return [&phi](const BoundBox<dim> &domain) {
      Vector<alg::Interval<dim>, dim> xint;
      const auto mid_pt = domain.mid_point();
      for (int dir = 0; dir < dim; ++dir) {
        const auto beta = set_component<real, dim>(numbers::zero, dir, numbers::one);
        xint(dir) = alg::Interval<dim>(mid_pt(dir), beta);
        // We slightly enlarge the domain to deal with edge cases.
        alg::Interval<dim>::delta(dir) = numbers::half * domain.length(dir) * (numbers::one + numbers::near_eps);
      }

      const auto res = phi(xint);
      if (res.uniformSign()) {
        if (res.alpha < 0.0) {
          return FuncSign::negative;
        } else {
          return FuncSign::positive;
        }
      } else {
        return FuncSign::undetermined;
      }
    };
  }

  enum ImmersedStatusTmp : std::uint8_t { cut, full, empty, full_int_bdry };

  template<int dim> using QuadRule = typename ::algoim::QuadratureRule<dim>;

  template<int dim> ImmersedStatusTmp classify_cell_from_quad(const QuadRule<dim> &quad, const BoundBox<dim> &domain)
  {
    const auto cell_vol = quad.sumWeights();

    const Tolerance default_tol;
    if (default_tol.equal(cell_vol, numbers::zero)) {
      // It is an empty cell but with an interior boundary (on the cell's boundary).
      // We consider that the interior boundary belongs to a neighbor cell.
      return ImmersedStatusTmp::empty;
    } else if (default_tol.equal(cell_vol, domain.volume())) {
      // Full cell but with an interior boundary (on a cell's boundary).
      return ImmersedStatusTmp::full_int_bdry;
    } else {
      // Cut cell with an interior boundary
      return ImmersedStatusTmp::cut;
    }
  }


  template<int dim> ImmersedStatusTmp classify_cell_general(const ImplicitFunc<dim> &phi, const BoundBox<dim> &domain)
  {
    constexpr int n_pts_dir{ 1 };
    const auto alg_domain = domain.to_hyperrectangle();

    const auto srf_quad = ::algoim::quadGen(phi, alg_domain, dim, 0, n_pts_dir);

    if (srf_quad.nodes.empty()) {
      if (phi(alg_domain.midpoint()) < numbers::zero) {
        return ImmersedStatusTmp::full;
      } else {
        return ImmersedStatusTmp::empty;
      }
    }

    const auto quad = ::algoim::quadGen(phi, alg_domain, -1, 0, n_pts_dir);
    return classify_cell_from_quad(quad, domain);
  }

  template<int dim> ImmersedStatusTmp classify_cell_Bezier(const BezierTP<dim> &bezier, const BoundBox<dim> &domain)
  {
    BezierTP<dim> bzr_domain(bezier);
    bzr_domain.rescale_domain(domain);

    switch (bezier.sign()) {
    case FuncSign::positive:
      return ImmersedStatusTmp::empty;
    case FuncSign::negative:
      return ImmersedStatusTmp::full;
    default:
      break;
    }

    constexpr auto quad_strategy = alg::QuadStrategy::AlwaysGL;
    constexpr int n_pts_dir{ 1 };
    const BoundBox<dim> domain_0_1(numbers::zero, numbers::one);

    alg::ImplicitPolyQuadrature<dim> ipquad(bzr_domain.get_xarray());

    // Detecting if the cell presents an interior boundary.
    bool has_int_bdry{ false };
    ipquad.integrate_surf(quad_strategy,
      n_pts_dir,
      [&has_int_bdry](const Vector<real, dim> & /*point*/,
        const real /*weight*/,
        const Vector<real, dim> & /*normal*/) { has_int_bdry = true; });

    if (!has_int_bdry) {
      // Does not have interior boundary: determines again the sign by evaluating
      // the Bezier at the domain's mid-point.
      if (bzr_domain(domain_0_1.mid_point()) < numbers::zero) {
        return ImmersedStatusTmp::full;
      } else {
        return ImmersedStatusTmp::empty;
      }
    }

    // Creates a quadrature for the negative part of the function.
    QuadRule<dim> quad;
    ipquad.integrate(
      quad_strategy, n_pts_dir, [&bzr_domain, &quad](const Vector<real, dim> &perm_point, const real weight) {
        const auto point = permute_vector_directions(perm_point);
        if (bzr_domain(point) < numbers::zero) {
          quad.nodes.emplace_back(point, weight);
        }
      });

    return classify_cell_from_quad(quad, domain_0_1);
  }

  template<int dim> ImmersedStatusTmp classify_cell(const ImplicitFunc<dim> &phi, const BoundBox<dim> &domain)
  {
    if (is_bezier(phi)) {
      const auto &bezier = dynamic_cast<const BezierTP<dim> &>(phi);
      return classify_cell_Bezier(bezier, domain);
    } else {
      return classify_cell_general(phi, domain);
    }
  }


  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  template<int dim>
  ImmersedFacetStatus classify_facet_full_interior_boundary(const ImmersedFacetStatus facet_status,
    const alg::HyperRectangle<real, dim> &domain,
    const int local_facet_id,
    const real cell_facet_vol,
    const int n_pts)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {

    // Classifying full interior boundaries.
    if (n_pts == 1
        && (facet_status == ImmersedFacetStatus::int_bdry || facet_status == ImmersedFacetStatus::ext_bdry)) {
      const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
      const real domain_facet_vol = prod(remove_component(domain.extent(), const_dir));

      const Tolerance tol;
      const Tolerance rel_tol{ tol.value() * 1.0e3 };
      const bool is_full_facet = tol.equal_rel(cell_facet_vol, domain_facet_vol, rel_tol);

      if (is_full_facet) {
        if (facet_status == ImmersedFacetStatus::int_bdry) {
          return ImmersedFacetStatus::full_int_bdry;
        } else {// if (facet_status == ImmersedFacetStatus::ext_bdry) {
          return ImmersedFacetStatus::full_ext_bdry;
        }
      }
    }

    return facet_status;
  }

  template<int dim>
  ImmersedFacetStatus classify_facet_from_quad(const ImplicitFunc<dim> &phi,
    const alg::HyperRectangle<real, dim> &domain,
    const int local_facet_id,
    const QuadRule<dim> &facet_quad)
  {
    const Tolerance default_tol;

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);

    int ext_bdry{ false };
    int int_bdry{ false };
    int cut_facet{ false };
    for (const auto &node : facet_quad.nodes) {
      if (on_levelset(phi, node.x, default_tol)) {

        auto normal = phi.grad(node.x);
        normal /= norm(normal);
        const auto normal_comp = normal(const_dir);
        if (!default_tol.equal(std::abs(normal_comp), numbers::one)) {
          continue;
        }

        if (side == 0 ? default_tol.is_negative(normal_comp) : default_tol.is_positive(normal_comp)) {
          int_bdry = 1;
        } else {
          ext_bdry = 1;
        }
      } else {
        cut_facet = 1;
      }
    }
    constexpr std::array<ImmersedFacetStatus, 8> values{ ImmersedFacetStatus::empty,
      ImmersedFacetStatus::ext_bdry,
      ImmersedFacetStatus::int_bdry,
      ImmersedFacetStatus::int_bdry_ext_bdry,
      ImmersedFacetStatus::cut,
      ImmersedFacetStatus::cut_ext_bdry,
      ImmersedFacetStatus::cut_int_bdry,
      ImmersedFacetStatus::cut_int_bdry_ext_bdry };

    const int index = ext_bdry + (int_bdry * 2) + (cut_facet * 4);
    const auto facet_status = at(values, index);


    const real cell_facet_vol = facet_quad.sumWeights();
    const int n_pts = static_cast<int>(facet_quad.nodes.size());

    return classify_facet_full_interior_boundary(facet_status, domain, local_facet_id, cell_facet_vol, n_pts);
  }


  template<int dim>
  ImmersedFacetStatus
    classify_facet_general(const ImplicitFunc<dim> &phi, const SubCartGridTP<dim> &subgrid, const int local_facet_id)
  {
    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    const auto domain = subgrid.get_domain().to_hyperrectangle();
    constexpr int n_pts_dir{ 1 };
    const auto facet_quad = ::algoim::quadGen(phi, domain, const_dir, side, n_pts_dir);

    return classify_facet_from_quad(phi, domain, local_facet_id, facet_quad);
  }

  template<int dim>
  ImmersedFacetStatus
    classify_facet_Bezier(const BezierTP<dim> &bezier, const BoundBox<dim> &domain, const int local_facet_id)
  {
    BezierTP<dim> bzr_domain(bezier);
    bzr_domain.rescale_domain(domain);
    const auto bzr_facet = bzr_domain.extract_facet(local_facet_id);

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    constexpr int n_pts_dir{ 1 };
    constexpr auto quad_strategy = alg::QuadStrategy::AlwaysGL;

    alg::ImplicitPolyQuadrature<dim - 1> ipquad(bzr_facet->get_xarray());

    QuadRule<dim> facet_quad;

    ipquad.integrate(quad_strategy,
      n_pts_dir,
      [&facet_quad, &bzr_facet, side, const_dir](const Vector<real, dim - 1> &perm_point, const real weight) {
        const auto point = permute_vector_directions(perm_point);
        if (bzr_facet->operator()(point) < numbers::zero) {
          const auto sup_point = add_component(point, const_dir, static_cast<real>(side));
          facet_quad.nodes.emplace_back(sup_point, weight);
        }
      });

    const BoundBox<dim> domain_01;
    return classify_facet_from_quad(bzr_domain, domain_01.to_hyperrectangle(), local_facet_id, facet_quad);
  }

  template<int dim>
  ImmersedFacetStatus classify_facet(const ImplicitFunc<dim> &phi,
    const SubCartGridTP<dim> &subgrid,
    const int local_facet_id,
    const ImmersedStatusTmp cell_status)
  {
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    assert(0 <= local_facet_id && local_facet_id < dim * 2);

    if (cell_status == ImmersedStatusTmp::empty) {
      return ImmersedFacetStatus::empty;
    } else if (cell_status == ImmersedStatusTmp::full) {
      return ImmersedFacetStatus::full;
    }

    // NOTE: if needed, this function could speed-up by looking at the
    // facet state of neighbor cells. However, this involves a more
    // complicated algorithm that may not be worth it.

    if (is_bezier(phi)) {
      const auto &bzr = dynamic_cast<const BezierTP<dim> &>(phi);
      return classify_facet_Bezier(bzr, subgrid.get_domain(), local_facet_id);
    } else {
      return classify_facet_general(phi, subgrid, local_facet_id);
    }
  }

  template<int dim>
  std::pair<ImmersedStatus, typename UnfittedImplDomain<dim>::FacetsStatus>
    classify_cut_cell_and_facets(const ImplicitFunc<dim> &phi, const SubCartGridTP<dim> &subgrid)
  {
    assert(subgrid.is_unique_cell());

    const auto cell_status = classify_cell(phi, subgrid.get_domain());

    typename UnfittedImplDomain<dim>::FacetsStatus facets{};
    if (cell_status == ImmersedStatusTmp::full) {
      return std::make_pair(ImmersedStatus::full, facets);
    } else if (cell_status == ImmersedStatusTmp::empty) {
      return std::make_pair(ImmersedStatus::empty, facets);
    }

    for (int local_facet_id = 0; local_facet_id < dim * 2; ++local_facet_id) {
      at(facets, local_facet_id) = classify_facet(phi, subgrid, local_facet_id, cell_status);
    }

    if (cell_status == ImmersedStatusTmp::cut) {
      return std::make_pair(ImmersedStatus::cut, facets);
    }

    // Looking for cells whose facets are either full, or full interior
    // boundaries if the facet is on a boundary. In that case, we
    // consider the cell as full.

    assert(cell_status == ImmersedStatusTmp::full_int_bdry);

    for (int local_facet_id = 0; local_facet_id < dim * 2; ++local_facet_id) {

      const auto facet = at(facets, local_facet_id);

      if (facet == ImmersedFacetStatus::full_int_bdry) {
        const auto cell_id = subgrid.get_single_cell();
        const auto &grid = subgrid.get_grid();
        if (!grid.on_boundary(cell_id, local_facet_id)) {
          return std::make_pair(ImmersedStatus::cut, facets);
        }
      } else if (facet != ImmersedFacetStatus::full) {
        return std::make_pair(ImmersedStatus::cut, facets);
      }
    }

    return std::make_pair(ImmersedStatus::full, facets);
  }


}// namespace


template<int dim>
UnfittedImplDomain<dim>::UnfittedImplDomain(const FuncPtr phi, const GridPtr grid) : UnfittedDomain<dim>(grid)
{
  assert(phi != nullptr);

  this->phi_ = phi;
  const std::optional<std::vector<int>> cells;
  const auto compute_sign = create_compute_sign_function<dim>(*phi);
  this->create_decomposition(SubCartGridTP<dim>(*grid), compute_sign, cells);
  this->sort();
}


template<int dim>
UnfittedImplDomain<dim>::UnfittedImplDomain(const FuncPtr phi, const GridPtr grid, const std::vector<int> &cells)
  : UnfittedDomain<dim>(grid)
{
  assert(phi != nullptr);

  this->phi_ = phi;
  const auto compute_sign = create_compute_sign_function<dim>(*phi);
  this->create_decomposition(SubCartGridTP<dim>(*grid), compute_sign, cells);
  this->sort();
}

template<int dim>
// NOLINTNEXTLINE (misc-no-recursion)
void UnfittedImplDomain<dim>::create_decomposition(const SubCartGridTP<dim> &subgrid,
  const std::function<FuncSign(const BoundBox<dim> &)> &func_sign,
  const std::optional<std::vector<int>> &target_cells)
{
  if (target_cells.has_value() && !intersect(subgrid, target_cells.value())) {
    return;
  }

  const auto domain = subgrid.get_domain();
  const auto sign = func_sign(domain);

  if (sign == FuncSign::undetermined && !subgrid.is_unique_cell()) {
    for (const auto &subgrid_half : subgrid.split()) {
      this->create_decomposition(*subgrid_half, func_sign, target_cells);
    }
  } else {
    switch (sign) {
    case FuncSign::undetermined: {
      const auto [status, facets] = classify_cut_cell_and_facets(*this->phi_, subgrid);
      const auto cell_id = subgrid.get_single_cell();
      switch (status) {
      case ImmersedStatus::cut:
        this->cut_cells_.push_back(cell_id);
        break;
      case ImmersedStatus::empty:
        this->empty_cells_.push_back(cell_id);
        break;
      case ImmersedStatus::full:
        this->full_cells_.push_back(cell_id);
        break;
      }
      this->facets_status_.emplace(cell_id, facets);
      return;
    }
    case FuncSign::positive: {
      insert_cells(subgrid, target_cells, this->empty_cells_);
      insert_facets(subgrid, target_cells, ImmersedFacetStatus::empty, this->facets_status_);
      return;
    }
    case FuncSign::negative: {
      insert_cells(subgrid, target_cells, this->full_cells_);
      insert_facets(subgrid, target_cells, ImmersedFacetStatus::full, this->facets_status_);
      return;
    }
    }
  }
}


template<int dim> auto UnfittedImplDomain<dim>::get_impl_func() const -> FuncPtr
{
  return this->phi_;
}


// Instantiations
template class UnfittedImplDomain<2>;
template class UnfittedImplDomain<3>;

}// namespace qugar::impl
