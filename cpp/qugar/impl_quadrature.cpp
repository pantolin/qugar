// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_quadrature.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of quadratures for general implicit functions on grids.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_quadrature.hpp>

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/cut_quadrature.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/quadrature.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/bernstein.hpp>
#include <algoim/quadrature_general.hpp>
#include <algoim/quadrature_multipoly.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <ranges>
#include <vector>


namespace qugar::impl {

namespace alg = ::algoim;

namespace {


  template<int dim> struct CutCellsQuadWrapper
  {
    explicit CutCellsQuadWrapper(CutCellsQuad<dim> &quad, const BoundBox<dim> &domain) : quad(quad), domain(domain) {}

    // NOLINTNEXTLINE (cppcoreguidelines-avoid-const-or-ref-data-members)
    CutCellsQuad<dim> &quad;
    BoundBox<dim> domain;

    // evalIntegrand records quadrature nodes when it is invoked by ImplicitIntegral
    void evalIntegrand(const Point<dim> &point, const real weight)
    {
      quad.points.push_back(domain.scale_to_0_1(point));
      quad.weights.push_back(weight / domain.volume());
    }
  };

  template<int dim> struct CutIntBoundsQuadWrapper
  {
    explicit CutIntBoundsQuadWrapper(CutIntBoundsQuad<dim> &quad,
      const ImplicitFunc<dim> &phi,
      const BoundBox<dim> &domain)
      : quad(quad), phi(phi), domain(domain)
    {}

    // NOLINTBEGIN (cppcoreguidelines-avoid-const-or-ref-data-members)
    CutIntBoundsQuad<dim> &quad;
    const ImplicitFunc<dim> &phi;
    const BoundBox<dim> &domain;
    // NOLINTEND (cppcoreguidelines-avoid-const-or-ref-data-members)

    // evalIntegrand records quadrature nodes when it is invoked by ImplicitIntegral
    void evalIntegrand(const Point<dim> &point, const real weight)
    {
      quad.points.push_back(domain.scale_to_0_1(point));

      // Unit normal in element's domain.
      Point<dim> normal = phi.grad(point);
      normal /= norm(normal);

      // Mapping the normal from element's domain to [0,1]^dim.
      // It is done left-multiplying the unit normal
      // in element domain with DF^-T, where F is the map
      // from the element domain to [0,1]^dim, and
      // DF = diag(1/dx, 1/dy, ...), where dx, dy, ... are
      // the lengths of the element's bounding box along
      // the different directions.
      for (int dir = 0; dir < dim; ++dir) {
        normal(dir) *= domain.length(dir);
      }
      const auto norm_normal = norm(normal);
      normal /= norm_normal;// This is the unit normal in [0,1]^dim

      quad.weights.push_back(weight * norm_normal / domain.volume());

      quad.normals.push_back(normal);
    }
  };

  template<int dim> struct CutIsoBoundsQuadWrapper
  {
    explicit CutIsoBoundsQuadWrapper(CutIsoBoundsQuad<dim> &quad, const BoundBox<dim> &face_domain, const int const_dir)
      : quad(quad), face_domain(face_domain), const_dir(const_dir)
    {}

    // NOLINTBEGIN (cppcoreguidelines-avoid-const-or-ref-data-members)
    CutIsoBoundsQuad<dim> &quad;
    // const ImplicitFunc<dim> &phi;
    const BoundBox<dim> &face_domain;
    // NOLINTEND (cppcoreguidelines-avoid-const-or-ref-data-members)
    int const_dir;

    // evalIntegrand records quadrature nodes when it is invoked by ImplicitIntegral
    void evalIntegrand(const Point<dim + 1> &point, const real weight)
    {
      const auto face_point = remove_component(point, const_dir);
      quad.points.push_back(face_domain.scale_to_0_1(face_point));
      quad.weights.push_back(weight / face_domain.volume());
    }
  };

  template<int dim, bool is_srf, typename T>
  void compute_quadrature_with_algoim_general(const ImplicitFunc<dim> &phi,
    const BoundBox<dim> &domain,
    const int n_pts_dir,
    T &quad_wrapper)
  {
    auto &quad = quad_wrapper.quad;
    const auto n_pts_0 = quad.points.size();

    constexpr auto n_psi = static_cast<unsigned int>(1) << static_cast<unsigned int>(dim - 1);
    std::array<alg::PsiCode<dim>, n_psi> psi;
    psi[0] = alg::PsiCode<dim>(0, -1);

    const Vector<bool, dim> free = true;

    alg::ImplicitIntegral<dim, dim, ImplicitFunc<dim>, T, is_srf>(
      phi, quad_wrapper, free, psi, 1, domain.to_hyperrectangle(), n_pts_dir);

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_cell.push_back(n_pts);
  }

  template<int dim, bool is_srf, typename T>
  void compute_quadrature_with_algoim_Bezier(const BezierTP<dim, 1> &bezier,
    const BoundBox<dim> &domain,
    const int n_pts_dir,
    T &quad)
  {
    BezierTP<dim> bzr_domain(bezier);
    bzr_domain.rescale_domain(domain);

    const auto n_pts_0 = quad.points.size();

    alg::ImplicitPolyQuadrature<dim> ipquad(bzr_domain.get_xarray());

    constexpr auto strategy = alg::QuadStrategy::AutoMixed;

    if constexpr (is_srf) {
      ipquad.integrate_surf(strategy,
        n_pts_dir,
        [&quad, &bzr_domain](
          const Point<dim> &perm_point_01, const real weight_01, const Point<dim> & /*aggreg_normal_01*/) {
          quad.points.push_back(permute_vector_directions(perm_point_01));
          quad.weights.push_back(weight_01);

          auto grad = ::algoim::bernstein::evalBernsteinPolyGradient(bzr_domain.get_xarray(), perm_point_01);
          grad /= norm(grad);
          quad.normals.push_back(permute_vector_directions(grad));
        });

    } else {
      ipquad.integrate(
        strategy, n_pts_dir, [&quad, &bzr_domain](const Point<dim> &perm_point_01, const real weight_01) {
          const auto point_01 = permute_vector_directions(perm_point_01);
          if (bzr_domain(point_01) < numbers::zero) {
            quad.points.push_back(point_01);
            quad.weights.push_back(weight_01);
          }
        });
    }

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_cell.push_back(n_pts);
  }

  template<int dim, bool is_srf, typename T>
  void compute_quadrature_with_algoim(const ImplicitFunc<dim> &phi,
    const BoundBox<dim> &domain,
    const int n_pts_dir,
    T &quad_wrapper)
  {
    if (is_bezier(phi)) {
      const auto &bezier = dynamic_cast<const BezierTP<dim, 1> &>(phi);
      compute_quadrature_with_algoim_Bezier<dim, is_srf>(bezier, domain, n_pts_dir, quad_wrapper.quad);
    } else {
      compute_quadrature_with_algoim_general<dim, is_srf>(phi, domain, n_pts_dir, quad_wrapper);
    }
  }

  template<int dim, typename T>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void purge_facet_points(const ImplicitFunc<dim> &func,
    const UnfittedImplDomain<dim> &unf_domain,
    const int cell_id,
    const int local_facet_id,
    const bool purge_internal_bdry,
    const bool purge_external_bdry,
    T &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    const bool int_bound = purge_internal_bdry && unf_domain.has_internal_boundary(cell_id, local_facet_id);
    const bool ext_bound = purge_external_bdry && unf_domain.has_external_boundary(cell_id, local_facet_id);
    if (!int_bound && !ext_bound) {
      return;
    }

    constexpr bool is_facet_quad = std::is_same_v<T, CutIsoBoundsQuad<dim - 1>>;

    const Tolerance tol;

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    const auto domain = unf_domain.get_grid()->get_cell_domain(cell_id);
    const auto facet_coord_0_1 = side == 0 ? numbers::zero : numbers::one;

    int n_pts{ 0 };
    if constexpr (is_facet_quad) {
      n_pts = quad.n_pts_per_facet.back();
    } else {
      n_pts = quad.n_pts_per_cell.back();
    }
    const auto n_tot_pts = static_cast<int>(quad.points.size());

    std::vector<int> points_to_purge;

    Point<dim> point;
    // NOLINTNEXTLINE (misc-const-correctness)
    bool pointing_outside{ true };
    for (int pt_id = n_tot_pts - n_pts; pt_id < n_tot_pts; ++pt_id) {

      const auto &pt_01 = at(quad.points, pt_id);

      if constexpr (is_facet_quad) {
        const Point<dim> sup_pt_01 = add_component(pt_01, const_dir, facet_coord_0_1);
        point = domain.scale_from_0_1(sup_pt_01);
      } else {

        if (!tol.equal(pt_01(const_dir), facet_coord_0_1)) {
          continue;
        }

        const auto normal = at(quad.normals, pt_id);
        const auto normal_comp = normal(const_dir);
        if (tol.smaller_than(std::abs(normal_comp), numbers::one)) {
          // Not normal to facet.
          continue;
        }

        pointing_outside = side == 0 ? tol.is_negative(normal_comp) : tol.is_positive(normal_comp);

        point = domain.scale_from_0_1(pt_01);
      }

      if (on_levelset(func, point, tol)) {
        if ((pointing_outside && int_bound) || (!pointing_outside && ext_bound)) {
          points_to_purge.push_back(pt_id);
        }
      }
    }

    // Sort in descending order
    std::ranges::sort(points_to_purge, std::ranges::greater());

    for (const auto &pt_id : points_to_purge) {
      quad.points.erase(quad.points.begin() + pt_id);
      quad.weights.erase(quad.weights.begin() + pt_id);
      if constexpr (!is_facet_quad) {
        quad.normals.erase(quad.normals.begin() + pt_id);
      }
    }

    if constexpr (is_facet_quad) {
      quad.n_pts_per_facet.back() -= points_to_purge.size();
    } else {
      quad.n_pts_per_cell.back() -= points_to_purge.size();
    }
  }


  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void compute_facet_quadrature_with_algoim_general(const ImplicitFunc<dim> &phi,
    const UnfittedImplDomain<dim> &unf_domain,
    const int cell_id,
    const int local_facet_id,
    const int n_pts_dir,
    CutIsoBoundsQuad<dim - 1> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    static_assert(dim == 2 || dim == 3, "Invalid dimension.");
    assert(0 < n_pts_dir);
    assert(unf_domain.is_cut_facet(cell_id, local_facet_id));

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);

    const auto grid = unf_domain.get_grid();
    const auto domain = grid->get_cell_domain(cell_id);
    const auto facet_domain = domain.slice(const_dir);

    CutIsoBoundsQuadWrapper<dim - 1> quad_wrapper(quad, facet_domain, const_dir);

    constexpr auto n_psi = static_cast<unsigned int>(1) << static_cast<unsigned int>(dim - 1);
    std::array<alg::PsiCode<dim>, n_psi> psi;
    psi[0] = alg::PsiCode<dim>(set_component<int, dim>(0, const_dir, side), -1);

    const auto free = set_component<int, dim>(true, const_dir, false);

    const auto n_pts_0 = quad.points.size();

    alg::ImplicitIntegral<dim - 1, dim, ImplicitFunc<dim>, CutIsoBoundsQuadWrapper<dim - 1>, false>(
      phi, quad_wrapper, free, psi, 1, domain.to_hyperrectangle(), n_pts_dir);

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_facet.push_back(n_pts);
  }

  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void compute_facet_quadrature_with_algoim_Bezier(const BezierTP<dim> &bezier,
    const UnfittedImplDomain<dim> &unf_domain,
    const int cell_id,
    const int local_facet_id,
    const int n_pts_dir,
    CutIsoBoundsQuad<dim - 1> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    static_assert(dim == 2 || dim == 3, "Invalid dimension.");

    assert(0 < n_pts_dir);
    assert(unf_domain.is_cut_facet(cell_id, local_facet_id));


    const auto grid = unf_domain.get_grid();
    const auto domain = grid->get_cell_domain(cell_id);

    BezierTP<dim> bzr_domain(bezier);
    bzr_domain.rescale_domain(domain);
    const auto bezier_facet = bzr_domain.extract_facet(local_facet_id);

    const auto n_pts_0 = quad.points.size();

    alg::ImplicitPolyQuadrature<dim - 1> ipquad(bezier_facet->get_xarray());

    constexpr auto strategy = alg::QuadStrategy::AutoMixed;
    ipquad.integrate(
      strategy, n_pts_dir, [&quad, bezier_facet](const Point<dim - 1> &perm_point_01, const real weight_01) {
        const auto point_01 = permute_vector_directions(perm_point_01);
        if (bezier_facet->operator()(point_01) < numbers::zero) {
          quad.points.push_back(point_01);
          quad.weights.push_back(weight_01);
        }
      });

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_facet.push_back(n_pts);
  }

  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void compute_facet_quadrature_with_algoim(const ImplicitFunc<dim> &phi,
    const UnfittedImplDomain<dim> &unf_domain,
    const int cell_id,
    const int local_facet_id,
    const int n_pts_dir,
    CutIsoBoundsQuad<dim - 1> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    if (is_bezier(phi)) {
      const auto &bezier = dynamic_cast<const BezierTP<dim, 1> &>(phi);
      compute_facet_quadrature_with_algoim_Bezier(bezier, unf_domain, cell_id, local_facet_id, n_pts_dir, quad);
    } else {
      compute_facet_quadrature_with_algoim_general(phi, unf_domain, cell_id, local_facet_id, n_pts_dir, quad);
    }
  }
}// namespace


template<int dim>
std::shared_ptr<const CutCellsQuad<dim>>
  create_quadrature(const UnfittedImplDomain<dim> &unf_domain, const std::vector<int> &cells, const int n_pts_dir)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(0 < n_pts_dir);

  // Estimation of number of points.
  const int n_cells = static_cast<int>(cells.size());
  const int n_quad_set_per_cell_estimate{ 2 };// This is an estimation.
  const int n_pts_per_quad_set = n_pts_dir * n_pts_dir * (dim == 3 ? n_pts_dir : 1);
  const int n_pts_per_cell_estimate = n_quad_set_per_cell_estimate * n_pts_per_quad_set;
  const int n_pts_estimate = n_cells * n_pts_per_cell_estimate;

  const auto quad = std::make_shared<CutCellsQuad<dim>>();
  quad->reserve(n_cells, n_pts_estimate);

  quad->cells = cells;
  auto &points = quad->points;
  auto &weights = quad->weights;
  auto &n_pts_per_cell = quad->n_pts_per_cell;

  const auto grid = unf_domain.get_grid();
  const auto phi = unf_domain.get_impl_func();

  const auto gauss_01 = Quadrature<dim>::create_Gauss_01(n_pts_dir);

  for (const int cell_id : cells) {

    if (unf_domain.is_cut_cell(cell_id)) {
      const auto domain = grid->get_cell_domain(cell_id);

      CutCellsQuadWrapper<dim> quad_wrapper(*quad, domain);
      compute_quadrature_with_algoim<dim, false>(*phi, domain, n_pts_dir, quad_wrapper);

    } else if (unf_domain.is_full_cell(cell_id)) {
      const auto &gauss_pt = gauss_01->points();
      points.insert(points.end(), gauss_pt.cbegin(), gauss_pt.cend());

      const auto &gauss_wg = gauss_01->weights();
      weights.insert(weights.end(), gauss_wg.cbegin(), gauss_wg.cend());

      n_pts_per_cell.push_back(n_pts_per_quad_set);
    } else {// empty
      n_pts_per_cell.push_back(0);
    }
  }

  return quad;
}

template<int dim>
std::shared_ptr<const CutIntBoundsQuad<dim>> create_interior_bound_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const int n_pts_dir)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(0 < n_pts_dir);

  // Estimation of number of points.
  const int n_cells = static_cast<int>(cells.size());
  const int n_quad_set_per_cell_estimate{ 2 };// This is an estimation.
  const int n_pts_per_quad_set = n_pts_dir * (dim == 3 ? n_pts_dir : 1);
  const int n_pts_per_cell_estimate = n_quad_set_per_cell_estimate * n_pts_per_quad_set;
  const int n_pts_estimate = n_cells * n_pts_per_cell_estimate;

  const auto quad = std::make_shared<CutIntBoundsQuad<dim>>();
  quad->reserve(n_cells, n_pts_estimate);

  quad->cells = cells;
  auto &n_pts_per_cell = quad->n_pts_per_cell;

  const auto grid = unf_domain.get_grid();
  const auto phi = unf_domain.get_impl_func();

  for (const int cell_id : cells) {

    if (unf_domain.is_empty_cell(cell_id) || unf_domain.is_full_cell(cell_id)) {
      n_pts_per_cell.push_back(0);
      continue;
    }

    const auto domain = grid->get_cell_domain(cell_id);

    CutIntBoundsQuadWrapper<dim> quad_wrapper(*quad, *phi, domain);
    compute_quadrature_with_algoim<dim, true>(*phi, domain, n_pts_dir, quad_wrapper);

    // Purging points in external boundaries that must be classified as facet points.
    constexpr int n_local_facets = dim * 2;
    for (int local_facet_id = 0; local_facet_id < n_local_facets; ++local_facet_id) {
      const bool has_int_bdry = unf_domain.has_internal_boundary_on_domain_boundary(cell_id, local_facet_id);
      const bool has_ext_bdry = unf_domain.has_external_boundary(cell_id, local_facet_id);
      purge_facet_points(*phi, unf_domain, cell_id, local_facet_id, has_int_bdry, has_ext_bdry, *quad);
    }
  }

  return quad;
}


template<int dim>
std::shared_ptr<const CutIsoBoundsQuad<dim - 1>> create_facets_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  const std::vector<int> &cells,
  const std::vector<int> &facets,
  const int n_pts_dir)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(cells.size() == facets.size());
  assert(0 < n_pts_dir);

  // Estimation of number of points.
  const int n_cells = static_cast<int>(cells.size());
  const int n_quad_set_per_facet_estimate{ 2 };// This is (almost surely) an overestimation.
  const int n_pts_per_quad_set = n_pts_dir * (dim == 3 ? n_pts_dir : 1);
  const int n_pts_per_facet_estimate = n_quad_set_per_facet_estimate * n_pts_per_quad_set;
  const int n_pts_estimate = n_cells * n_pts_per_facet_estimate;

  const auto quad = std::make_shared<CutIsoBoundsQuad<dim - 1>>();
  quad->reserve(n_cells, n_pts_estimate);

  quad->cells = cells;
  quad->local_facet_ids = facets;
  auto &points = quad->points;
  auto &weights = quad->weights;
  auto &n_pts_per_facet = quad->n_pts_per_facet;

  const auto gauss_01 = Quadrature<dim - 1>::create_Gauss_01(n_pts_dir);

  const auto phi = unf_domain.get_impl_func();

  auto facet_it = facets.cbegin();
  for (const int cell_id : cells) {
    const auto local_facet_id = *facet_it++;

    if (unf_domain.is_cut_facet(cell_id, local_facet_id)) {
      compute_facet_quadrature_with_algoim<dim>(*phi, unf_domain, cell_id, local_facet_id, n_pts_dir, *quad);

      constexpr bool purge_int_bdry{ true };
      constexpr bool purge_ext_bdry{ true };
      purge_facet_points(*phi, unf_domain, cell_id, local_facet_id, purge_int_bdry, purge_ext_bdry, *quad);
    }

    else if (unf_domain.is_empty_facet(cell_id, local_facet_id)) {
      n_pts_per_facet.push_back(0);
    }

    else {// if (unf_domain.is_full_facet(cell_id, local_facet_id)) {
      const auto &gauss_pt = gauss_01->points();
      points.insert(points.end(), gauss_pt.cbegin(), gauss_pt.cend());

      const auto &gauss_wg = gauss_01->weights();
      weights.insert(weights.end(), gauss_wg.cbegin(), gauss_wg.cend());

      n_pts_per_facet.push_back(n_pts_per_quad_set);
    }
  }

  return quad;
}

// Instantiations


template std::shared_ptr<const CutCellsQuad<2>>
  create_quadrature<2>(const UnfittedImplDomain<2> &, const std::vector<int> &, const int);
template std::shared_ptr<const CutCellsQuad<3>>
  create_quadrature<3>(const UnfittedImplDomain<3> &, const std::vector<int> &, const int);

template std::shared_ptr<const CutIntBoundsQuad<2>>
  create_interior_bound_quadrature<2>(const UnfittedImplDomain<2> &, const std::vector<int> &, const int);
template std::shared_ptr<const CutIntBoundsQuad<3>>
  create_interior_bound_quadrature<3>(const UnfittedImplDomain<3> &, const std::vector<int> &, const int);

template std::shared_ptr<const CutIsoBoundsQuad<1>> create_facets_quadrature<2>(const UnfittedImplDomain<2> &,
  const std::vector<int> &,
  const std::vector<int> &,
  const int);
template std::shared_ptr<const CutIsoBoundsQuad<2>> create_facets_quadrature<3>(const UnfittedImplDomain<3> &,
  const std::vector<int> &,
  const std::vector<int> &,
  const int);

}// namespace qugar::impl