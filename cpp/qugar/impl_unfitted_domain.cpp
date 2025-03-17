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

  //! @brief Checks if the subgrid intersects with the target cells.
  //!
  //! This function determines whether any of the target cells are within the range
  //! of the given subgrid by checking if at least one cell ID from the target cells
  //! exists within the subgrid's range.
  //!
  //! @p target_cells is an optional vector of cell indices to test for intersection.
  //! If `target_cells` is not provided, this function returns `true` (it assumes that
  //! all the cells are considered.)
  //!
  //! @tparam dim The dimensionality of the grid.
  //! @param subgrid The subgrid to check for intersection with target cells.
  //! @param target_cells An optional vector of cell indices to test for intersection.
  //! @return `true` if any target cell is within the subgrid's range, or no target cells are defined;
  //! `false` otherwise.
  template<int dim>
  bool intersect(const SubCartGridTP<dim> &subgrid, const std::optional<std::vector<int>> &target_cells)
  {
    if (target_cells.has_value()) {
      return intersect(subgrid, target_cells.value());
    } else {
      return true;
    }
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
    const auto &cells_range = subgrid.get_range();

    if (target_cells.has_value()) {
      const auto &cells = target_cells.value();
      assert(intersect(subgrid, cells));

      container.reserve(container.size() + cells.size());
      copy_if(cells.cbegin(), cells.cend(), std::back_inserter(container), [&cells_range](const auto cell_id) {
        return cells_range.is_in_range(cell_id);
      });

    } else {
      container.reserve(container.size() + static_cast<std::size_t>(subgrid.get_num_cells()));
      for (const auto &cell_tid : cells_range) {
        container.push_back(subgrid.to_flat(cell_tid));
      }
    }
  }

  template<int dim>
  void insert_facets(const SubCartGridTP<dim> &subgrid,
    const std::optional<std::vector<int>> &target_cells,
    const ImmersedFacetStatus facet_value,
    std::unordered_map<int, typename UnfittedImplDomain<dim>::FacetsStatus> &cells_facets)
  {
    const auto &cells_range = subgrid.get_range();

    typename UnfittedImplDomain<dim>::FacetsStatus facets;
    facets.fill(facet_value);

    if (target_cells.has_value()) {
#ifndef NDEBUG
      const auto &cells = target_cells.value();
      assert(intersect(subgrid, cells));
#endif// NDEBUG

      for (const auto &cell_tid : cells_range) {
        const auto cell_id = subgrid.to_flat(cell_tid);
        if (cells_range.is_in_range(cell_id)) {
          cells_facets.emplace(cell_id, facets);
        }
      }
    } else {

      for (const auto &cell_tid : cells_range) {
        const auto cell_id = subgrid.to_flat(cell_tid);
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
        alg::Interval<dim>::delta(dir) = numbers::half * domain.length(dir);
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

  enum ImmersedStatusTmp : std::uint8_t { cut, full, empty, full_unf_bdry };

  template<int dim> using QuadRule = typename ::algoim::QuadratureRule<dim>;

  template<int dim> ImmersedStatusTmp classify_cell_from_quad(const QuadRule<dim> &quad, const BoundBox<dim> &domain)
  {
    const auto cell_vol = quad.sumWeights();

    const Tolerance default_tol;
    if (default_tol.equal(cell_vol, numbers::zero)) {
      // It is an empty cell but with an unfitted boundary (on the cell's boundary).
      // We consider that the unfitted boundary belongs to a neighbor cell.
      return ImmersedStatusTmp::empty;
    } else if (default_tol.equal(cell_vol, domain.volume())) {
      // Full cell but with an unfitted boundary (on a cell's boundary).
      return ImmersedStatusTmp::full_unf_bdry;
    } else {
      // Cut cell with an unfitted boundary
      return ImmersedStatusTmp::cut;
    }
  }


  template<int dim> ImmersedStatusTmp classify_cell_general(const ImplicitFunc<dim> &phi, const BoundBox<dim> &domain)
  {
    constexpr int n_pts_dir{ 1 };
    const auto alg_domain = domain.to_hyperrectangle();
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

    alg::ImplicitPolyQuadrature<dim> ipquad(bzr_domain.get_xarray());

    // Creates a quadrature for the negative part of the function.
    QuadRule<dim> quad;
    ipquad.integrate(
      quad_strategy, n_pts_dir, [&bzr_domain, &quad](const Vector<real, dim> &perm_point, const real weight) {
        const auto point = permute_vector_directions(perm_point);
        if (bzr_domain(point) < numbers::zero) {
          quad.nodes.emplace_back(point, weight);
        }
      });

    const BoundBox<dim> domain_0_1(numbers::zero, numbers::one);
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
  ImmersedFacetStatus classify_facet_full_unfitted_boundary(const ImmersedFacetStatus facet_status,
    const alg::HyperRectangle<real, dim> &domain,
    const int local_facet_id,
    const real cell_facet_vol,
    const int n_pts)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const real domain_facet_vol = prod(remove_component(domain.extent(), const_dir));

    const Tolerance tol;
    const Tolerance rel_tol{ tol.value() * 1.0e3 };
    const bool is_full_facet = tol.equal_rel(cell_facet_vol, domain_facet_vol, rel_tol);

    if (is_full_facet) {
      switch (facet_status) {
      case ImmersedFacetStatus::unf_bdry:
      case ImmersedFacetStatus::ext_bdry: {
        if (n_pts == 1) {
          return facet_status == ImmersedFacetStatus::unf_bdry ? ImmersedFacetStatus::full_unf_bdry
                                                               : ImmersedFacetStatus::full_ext_bdry;
        }
        break;
      }
      case ImmersedFacetStatus::cut:
        return ImmersedFacetStatus::full;
      default:
        break;
      }
    }

    return facet_status;
  }

  template<int dim>
  ImmersedFacetStatus classify_facet_from_quad(const ImplicitFunc<dim> &phi,
    const alg::HyperRectangle<real, dim> &domain,
    const int local_facet_id,
    const ImmersedStatusTmp cell_status,
    const QuadRule<dim> &facet_quad)
  {
    assert(cell_status == ImmersedStatusTmp::cut || cell_status == ImmersedStatusTmp::full_unf_bdry);

    if (facet_quad.nodes.empty()) {
      return cell_status == ImmersedStatusTmp::full_unf_bdry ? ImmersedFacetStatus::full_unf_bdry
                                                             : ImmersedFacetStatus::empty;
    }

    const Tolerance default_tol;

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);

    bool ext_bdry{ false };
    bool unf_bdry{ false };
    bool cut_facet{ false };
    for (const auto &node : facet_quad.nodes) {
      if (on_levelset(phi, node.x, default_tol)) {

        const auto normal = phi.grad(node.x);
        const auto normal_comp = normal(const_dir) / norm(normal);
        if (!default_tol.equal(std::abs(normal_comp), numbers::one)) {
          continue;
        }

        if (side == 0 ? default_tol.is_negative(normal_comp) : default_tol.is_positive(normal_comp)) {
          unf_bdry = true;
        } else {
          ext_bdry = true;
        }
      } else {
        cut_facet = true;
      }
    }

    if (cell_status == ImmersedStatusTmp::full_unf_bdry) {
      if (facet_quad.nodes.empty() || unf_bdry || ext_bdry) {
        // Patological cases with external boundaries are overriden.
        return ImmersedFacetStatus::full_unf_bdry;
      } else {
        return ImmersedFacetStatus::full;
      }
    }

    constexpr std::array<ImmersedFacetStatus, 8> values{ ImmersedFacetStatus::empty,
      ImmersedFacetStatus::ext_bdry,
      ImmersedFacetStatus::unf_bdry,
      ImmersedFacetStatus::unf_bdry_ext_bdry,
      ImmersedFacetStatus::cut,
      ImmersedFacetStatus::cut_ext_bdry,
      ImmersedFacetStatus::cut_unf_bdry,
      ImmersedFacetStatus::cut_unf_bdry_ext_bdry };

    const int index = ext_bdry + (unf_bdry * 2) + (cut_facet * 4);
    auto facet_status = at(values, index);

    const real cell_facet_vol = facet_quad.sumWeights();
    const int n_pts = static_cast<int>(facet_quad.nodes.size());

    return classify_facet_full_unfitted_boundary(facet_status, domain, local_facet_id, cell_facet_vol, n_pts);
  }


  template<int dim>
  ImmersedFacetStatus classify_facet_general(const ImplicitFunc<dim> &phi,
    const SubCartGridTP<dim> &subgrid,
    const int local_facet_id,
    const ImmersedStatusTmp cell_status)
  {
    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    const auto domain = subgrid.get_domain().to_hyperrectangle();
    constexpr int n_pts_dir{ 1 };
    const auto facet_quad = ::algoim::quadGen(phi, domain, const_dir, side, n_pts_dir);

    return classify_facet_from_quad(phi, domain, local_facet_id, cell_status, facet_quad);
  }

  template<int dim>
  ImmersedFacetStatus classify_facet_Bezier(const BezierTP<dim> &bezier,
    const BoundBox<dim> &domain,
    const int local_facet_id,
    const ImmersedStatusTmp cell_status)
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
    return classify_facet_from_quad(bzr_domain, domain_01.to_hyperrectangle(), local_facet_id, cell_status, facet_quad);
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
      return classify_facet_Bezier(bzr, subgrid.get_domain(), local_facet_id, cell_status);
    } else {
      return classify_facet_general(phi, subgrid, local_facet_id, cell_status);
    }
  }

  template<int dim>
  std::pair<ImmersedStatus, typename UnfittedImplDomain<dim>::FacetsStatus>
    classify_cut_cell_and_facets(const ImplicitFunc<dim> &phi, const SubCartGridTP<dim> &subgrid)
  {
    assert(subgrid.is_unique_cell());

    const auto domain = subgrid.get_domain();

    const auto tmp_cell_status = classify_cell(phi, domain);

    typename UnfittedImplDomain<dim>::FacetsStatus facets{};
    if (tmp_cell_status == ImmersedStatusTmp::full) {
      facets.fill(ImmersedFacetStatus::full);
      return std::make_pair(ImmersedStatus::full, facets);
    } else if (tmp_cell_status == ImmersedStatusTmp::empty) {
      facets.fill(ImmersedFacetStatus::empty);
      return std::make_pair(ImmersedStatus::empty, facets);
    }

    assert(tmp_cell_status == ImmersedStatusTmp::cut || tmp_cell_status == ImmersedStatusTmp::full_unf_bdry);

    for (int local_facet_id = 0; local_facet_id < dim * 2; ++local_facet_id) {
      at(facets, local_facet_id) = classify_facet(phi, subgrid, local_facet_id, tmp_cell_status);
    }

    ImmersedStatus cell_status{ ImmersedStatus::cut };

    // Looking for full_unf_bdry cells whose facets are either full.
    // In that case, we consider the cell as full.
    if (tmp_cell_status == ImmersedStatusTmp::full_unf_bdry
        && std::ranges::all_of(facets, [](const auto &facet) { return facet == ImmersedFacetStatus::full; })) {
      cell_status = ImmersedStatus::full;
    }

    return std::make_pair(cell_status, facets);
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
  if (!intersect(subgrid, target_cells)) {
    return;
  }

  const auto domain = subgrid.get_domain();
  // We slightly enlarge the domain to deal with edge cases.
  const auto ext_domain = domain.extend(numbers::eps * 1000);
  const auto sign = func_sign(ext_domain);

  if (sign == FuncSign::undetermined) {
    if (subgrid.is_unique_cell()) {
      this->classify_undetermined_sign_cell(subgrid);
    } else {
      for (const auto &subgrid_half : subgrid.split()) {
        this->create_decomposition(*subgrid_half, func_sign, target_cells);
      }
    }
  } else {
    assert(sign == FuncSign::positive || sign == FuncSign::negative);

    std::vector<int> &cells = sign == FuncSign::positive ? this->empty_cells_ : this->full_cells_;
    const auto facet_status = sign == FuncSign::positive ? ImmersedFacetStatus::empty : ImmersedFacetStatus::full;
    insert_cells(subgrid, target_cells, cells);
    insert_facets(subgrid, target_cells, facet_status, this->facets_status_);
  }
}

template<int dim> void UnfittedImplDomain<dim>::classify_undetermined_sign_cell(const SubCartGridTP<dim> &subgrid)
{
  assert(subgrid.is_unique_cell());
  const auto [status, facets] = classify_cut_cell_and_facets(*this->phi_, subgrid);

  // NOLINTBEGIN (readability-avoid-nested-conditional-operator)
  std::vector<int> &cells = status == ImmersedStatus::cut
                              ? this->cut_cells_
                              : (status == ImmersedStatus::empty ? this->empty_cells_ : this->full_cells_);
  // NOLINTEND (readability-avoid-nested-conditional-operator)

  const auto cell_id = subgrid.get_single_cell();
  cells.push_back(cell_id);

  this->facets_status_.emplace(cell_id, facets);
}


template<int dim> auto UnfittedImplDomain<dim>::get_impl_func() const -> FuncPtr
{
  return this->phi_;
}


// Instantiations
template class UnfittedImplDomain<2>;
template class UnfittedImplDomain<3>;

}// namespace qugar::impl
