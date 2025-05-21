// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file reparam_general.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tools for creating reparameterizations of general functions.
//!
//! @date 2025-01-04
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_reparam_general.hpp>

#include <qugar/bbox.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_reparam_mesh.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/bernstein.hpp>
#include <algoim/hyperrectangle.hpp>
#include <algoim/interval.hpp>
#include <algoim/quadrature_general.hpp>
#include <algoim/uvector.hpp>

#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <ranges>
#include <type_traits>
#include <unordered_map>
#include <vector>


namespace qugar::impl {

namespace alg = ::algoim;

//! @brief dim-dimensional reparameterization of an range-dimensional function restricted to given
//! implicitly defined domains.
//! This is just a modification of the class ImplicitIntegral (quadrature_general.hpp)
//! for generating reparameterizations.
//!
//! @tparam dim Parametric dimension of the current sub-domain.
//! @tparam range Parametric dimension of the final domain.
//! @tparam R Type of auxiliary data structure for storing either the generated
//!         reparameterization (when T == true), or the higher-dimension
//!         class instance.
//! @tparam S Flag indicating if the reparameterization must
//!         be performed only for the levelset surface (true), i.e.,
//!         the manifold where the any of the polynomials is equal to 0,
//!         or the subregion volume (false) between those manifolds
//!         where all the polynomials are negative.
//! @tparam T Flag indicating if this one is the highest dimension last class instance,
//!         who is actually in charge of storing the generated reparameterization.
template<int dim, int range, typename R, bool S, bool T = true> struct ImplicitGeneralReparam
{
  static constexpr int n_max_subdivs = 16;
  // NOLINTNEXTLINE (hicpp-signed-bitwise)
  static constexpr int n_facets = 1 << (range - 1);

  using Self = ImplicitGeneralReparam<dim, range, R, S, T>;

  // NOLINTNEXTLINE (cppcoreguidelines-avoid-const-or-ref-data-members)
  const ImplicitFunc<range> &phi_;
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-const-or-refdata-members)
  R &reparam_;
  Vector<bool, range> free_;
  Vector<alg::PsiCode<range>, n_facets> psi_;
  int psi_count_;
  //! Implicit function domain.
  alg::HyperRectangle<real, range> xrange_;
  //! Reparameterization order.
  int order_;
  //! Whether (2nd kind) Chebyshev nodes are used in the quadrature (if true), or equally spaced nodes (if false).
  bool use_Chebyshev_;
  //! Elimination axis/height direction.
  int dir_k_;
  Vector<alg::Interval<range>, range> xint_;

  //! Reference domains intervals (defined as roots intervals)
  std::unordered_map<TensorIndexTP<range>, RootsIntervals<range>> ref_intervals_;
  //! Points at which the reference domains intervals were computed.
  std::unordered_map<TensorIndexTP<range>, Vector<real, range>> ref_intervals_points_;

  //! Map of reparameterization cells (cell tensor index to flat index).
  std::conditional_t<T, std::unordered_map<TensorIndexTP<range>, int>, void *> cells_indices_map_;

  //! Prunes the given set of functions by checking for the existence of the interface. If a function is
  //! uniformly positive or negative and is consistent with specified sign, it can be removed. If a
  //! function is uniformly positive or negative but inconsistent with specified sign, the domain of
  //! integration is empty.
  bool prune()
  {
    for (int i = 0; i < this->psi_count_;) {
      for (int dir = 0; dir < range; ++dir) {
        if (!this->free_(dir)) {
          this->xint_(dir).alpha = this->xrange_.side(this->psi_(i).side(dir))(dir);
        }
      }
      const alg::Interval<range> res = this->phi_(this->xint_);
      if (res.uniformSign()) {
        if ((res.alpha >= 0.0 && this->psi_(i).sign() >= 0) || (res.alpha <= 0.0 && this->psi_(i).sign() <= 0)) {
          --this->psi_count_;
          std::swap(this->psi_(i), this->psi_(this->psi_count_));
        } else {
          return false;
        }
      } else {
        ++i;
      }
    }
    return true;
  }

  //! @brief Clears recursively the map of reparameterization elements.
  //! It clears the map of reparameterization elements of the highest dimension.
  void clear_reparam_elems_map()
  {
    if constexpr (T) {
      this->cells_indices_map_.clear();
    } else {
      this->reparam_.clear_reparam_elems_map();
    }
  }

  //! @brief Clears recursively the reference intervals.
  //! It clears the intervals for the current dimension and higher ones.
  void clear_ref_intervals()
  {
    this->ref_intervals_.clear();
    this->ref_intervals_points_.clear();
    if constexpr (!T) {
      this->reparam_.clear_ref_intervals();
    }
  }

  //! @brief Computes the tolerance to be used in calculations.
  //!
  //! @return Computed tolerance.
  [[nodiscard]] Tolerance compute_tolerance() const
  {
    Tolerance tol;
    tol.update(tol.value() * this->xrange_.extent(this->dir_k_));
    return tol;
  }

  //! @brief Restricts the coordinates of the point @p point according
  //! to the @p psi_id-th face restriction.
  //!
  //! @param psi_id Id of the restriction.
  //! @param point Point to constrain.
  void set_psi_bounds(const int psi_id, Vector<real, range> &point) const
  {
    for (int dir = 0; dir < range; ++dir) {
      if (!this->free_(dir)) {
        point(dir) = this->xrange_.side(this->psi_(psi_id).side(dir))(dir);
      }
    }
  }

  //! @brief Computes the roots of the function phi_ at point @p point,
  //! along the direction dir_k_, and under the constrains of the
  //! @p psi_id-th face restriction.
  //!
  //! @param point Point at which the roots are computed.
  //! @param psi_id Id of the restriction.
  //! @return Computed roots.
  [[nodiscard]] std::vector<real> compute_roots(Vector<real, range> point, const int psi_id) const
  {
    this->set_psi_bounds(psi_id, point);

    // point is now valid in all variables except dir_k_
    std::vector<real> roots;
    algoim::detail::rootFind<ImplicitFunc<range>, range>(this->phi_,
      point,
      this->dir_k_,
      this->xrange_.min(this->dir_k_),
      this->xrange_.max(this->dir_k_),
      roots,
      dim > 1);

    return roots;
  }

  //! @brief Checks if the interval defined by the coordinates @p x0 and @p x1
  //! along dir_k_ is active or not.
  //! To be active, the interval must not have zero length and the function phi_
  //! evaluated inside the interval must be negative.
  //!
  //!
  //! @param point Partially restricted point at which the function phi_ is evaluated.
  //! @param x0 First end of the interval along dir_k_.
  //! @param x1 Second end of the interval along dir_k_.
  //! @return True if the interval is active, false otherwise.
  [[nodiscard]] bool check_active_interval(Vector<real, range> &point, const real x0, const real x1) const
  {
    const auto tol = this->compute_tolerance();

    if (tol.equal(x1, x0)) {
      return false;
    }

    bool okay = true;
    point(this->dir_k_) = numbers::half * (x0 + x1);
    for (int i = 0; i < this->psi_count_ && okay; ++i) {
      this->set_psi_bounds(i, point);
      okay &= this->phi_(point) > 0.0 ? (this->psi_(i).sign() >= 0) : (this->psi_(i).sign() <= 0);
    }
    return okay;
  }

  //! @brief Computes all the intervals (between consecutive roots)
  //! at point @p point along direction dir_k_.
  //!
  //! @param point Point at which intervals are computed.
  //! @param intervals Computed intervals.
  void compute_all_intervals(Vector<real, range> point, RootsIntervals<range> &intervals) const
  {
    intervals.clear();
    intervals.point = point;
    // auto &roots = intervals.roots;

    const real x0 = this->xrange_.min(this->dir_k_);
    const real x1 = this->xrange_.max(this->dir_k_);

    intervals.add_root(x0, -1);
    intervals.add_root(x1, -1);

    for (int i = 0; i < this->psi_count_; ++i) {
      for (const auto root : this->compute_roots(point, i)) {
        intervals.add_root(root, i);
      }
    }

    // In rare cases, degenerate segments can be found, filter out with a tolerance
    intervals.adjust_roots(this->compute_tolerance(), x0, x1);

    if (!S) {
      const auto n_int = intervals.get_num_roots() - 1;
      assert(0 < n_int);

      intervals.active_intervals.resize(static_cast<std::size_t>(n_int));

      for (int i = 0; i < n_int; ++i) {
        const real xx0 = at(intervals.roots, i);
        const real xx1 = at(intervals.roots, i + 1);
        at(intervals.active_intervals, i) = this->check_active_interval(point, xx0, xx1);
      }
    }
  }

  //! @brief Computes recursively the reference intervals for the current
  //! dimension and the ones above.
  //!
  //! @param point Point at which the intervals are computed.
  //! @param cell_tid Element tensor index.
  void compute_ref_intervals(Vector<real, range> point, TensorIndexTP<range> cell_tid)
  {
    if constexpr (dim == 1 && !T) {
      reparam_.clear_ref_intervals();
    }

    static thread_local RootsIntervals<range> intervals;
    this->compute_all_intervals(point, intervals);

    this->ref_intervals_.emplace(cell_tid, intervals);
    this->ref_intervals_points_.emplace(cell_tid, point);


    // Loop over segments of divided interval
    const auto n_int = static_cast<int>(intervals.active_intervals.size());
    for (int i = 0; i < n_int; ++i) {
      if (!at(intervals.active_intervals, i)) {
        continue;
      }

      const real x0 = at(intervals.roots, i);
      const real x1 = at(intervals.roots, i + 1);

      cell_tid(this->dir_k_) = i;
      point(this->dir_k_) = numbers::half * (x0 + x1);
      if constexpr (!T) {
        reparam_.compute_ref_intervals(point, cell_tid);
      }
    }// i
  }

  real interpolate_coord(const int ind, const int dir)
  {
    return this->xrange_.min(dir) + (this->xrange_.extent(dir) * generate_point_in_01(ind));
  };

  template<int dim_aux>
  void insert_point_reparam(const Point<range> &point,
    const TensorIndexTP<range> &cell_tid,
    const TensorIndexTP<dim_aux> &pt_tid)
  {
    int cell_id{ 0 };

    const auto it = this->cells_indices_map_.find(cell_tid);
    if (it != this->cells_indices_map_.cend()) {
      cell_id = it->second;
    } else {
      cell_id = this->reparam_.allocate_cells(1);
      this->cells_indices_map_.emplace(cell_tid, cell_id);
    }

    const auto pt_id = static_cast<int>(pt_tid.flat(TensorSizeTP<dim_aux>(this->order_)));

    this->reparam_.insert_cell_point(point, cell_id, pt_id);
  }

  //! @brief Reparameterizes the current domain to be the entire dim-dimensional cube.
  void tensor_product_reparam_full()
  {
    static_assert(!S && T, "Invalid dimension.");

    if constexpr (dim == range) {
      constexpr bool wirebasket = false;
      this->reparam_.add_full_cell(BoundBox<range>(this->xrange_), wirebasket);
    } else {
      Point<range> point;

      int fix_dir{ -1 };
      for (int dir = 0; dir < range; ++dir) {
        if (!this->free_(dir)) {
          fix_dir = dir;
          point(fix_dir) = this->xrange_.side(this->psi_(0).side(dir))(dir);
          break;
        }
      }
      assert(fix_dir >= 0);

      const TensorIndexTP<range> cell_tid{ 0 };
      for (const auto &pt_tid : TensorIndexRangeTP<dim>(this->order_)) {

        for (int dir = 0, local_dir = 0; dir < range; ++dir) {
          if (dir != fix_dir) {
            point(dir) = interpolate_coord(pt_tid(local_dir), dir);
            ++local_dir;
          }
        }

        this->insert_point_reparam(point, cell_tid, pt_tid);
      }
    }
  }

  //! @brief Reparameterizes the current domain to be the entire dim-dimensional cube.
  void tensor_product_reparam_base()
  {
    static_assert(!S && !T, "Invalid dimension.");

    Vector<real, range> x;
    for (int dir = 0; dir < range; ++dir) {
      if (this->free_(dir)) {
        x(dir) = this->xrange_.midpoint(dir);
      }
    }
    reparam_.clear_ref_intervals();
    reparam_.compute_ref_intervals(x, TensorIndexTP<range>(0));


    for (const auto &sub_tid : TensorIndexRangeTP<dim>(this->order_)) {
      Vector<real, range> xx;
      TensorIndexTP<range> pt_tid;
      for (int dir = 0, k = 0; dir < range; ++dir) {
        if (this->free_(dir)) {
          xx(dir) = interpolate_coord(sub_tid(k), dir);
          pt_tid(dir) = sub_tid(k);
          ++k;
        }
      }
      reparam_.process(xx, TensorIndexTP<range>(0), pt_tid);
    }
  }

  //! @brief Returns the roots of the reference intervals associated to a given
  //! restriction.
  //!
  //! @param ref_intervals Reference intervals.
  //! @param psi_id Id of the restriction.
  //! @return Roots of the reference intervals associated to the given restriction.
  [[nodiscard]] std::vector<real> get_ref_intervals_roots(const RootsIntervals<range> &ref_intervals,
    const int psi_id) const
  {
    std::vector<real> ref_roots;

    for (int i = 0; i < ref_intervals.get_num_roots(); ++i) {
      if (at(ref_intervals.func_ids, i) == psi_id) {
        ref_roots.push_back(at(ref_intervals.roots, psi_id));
      }
    }

    return ref_roots;
  }

  //! @brief Computes all the roots in an interval, associated to a given
  //! restriction, using as reference a set of reference intervals.
  //!
  //! @param ref_intervals Reference intervals.
  //! @param psi_id Id of the restriction.
  //! @param point Point at which new intervals are computed.
  //! @return Computed roots.
  //!
  //! @warning This method is not bulletproof, and it may fail in corner cases.
  [[nodiscard]] std::vector<real>
    compute_similar_roots(const RootsIntervals<range> &ref_intervals, const int psi_id, Vector<real, range> point) const
  {
    auto ref_roots = get_ref_intervals_roots(ref_intervals, psi_id);

    if (ref_roots.empty()) {
      return ref_roots;
    }

    auto roots = this->compute_roots(point, psi_id);

    const auto tol = this->compute_tolerance();

    // Filtering out roots near x0 and x1.
    const auto x0 = this->xrange_.min(this->dir_k_);
    const auto x1 = this->xrange_.max(this->dir_k_);
    const auto it = std::remove_if(roots.begin(), roots.end(), [tol, x0, x1](const auto &root) {
      return tol.equal(root, x0) || tol.equal(root, x1);
    });
    roots.erase(it, roots.end());

    const auto n_ref_roots = ref_roots.size();
    auto n_roots = roots.size();

    if (n_roots == n_ref_roots) {
      return roots;
    } else if (n_roots > n_ref_roots) {
      return ref_roots;// Backup strategy
    }
    // else // n_roots < n_ref_roots

    this->set_psi_bounds(psi_id, point);
    const auto val_0 = this->phi_(algoim::set_component(point, this->dir_k_, x0));
    const auto val_1 = this->phi_(algoim::set_component(point, this->dir_k_, x1));

    bool root_0{ false };
    bool root_1{ false };

    // Check if the points x0 or x1 are roots of the function
    // up to different tolerances.
    for (int i = 0; i < 4; ++i) {
      const Tolerance new_tol(tol.value() * pow(10.0, i));
      root_0 = new_tol.is_zero(val_0);
      root_1 = new_tol.is_zero(val_1);
      if (root_0 || root_1) {
        break;
      }
    }

    if (root_0 ^ root_1) {
      roots.insert(roots.end(), n_ref_roots - n_roots, root_0 ? x0 : x1);
    } else if (root_0 && root_1) {
      if ((n_roots + 2) == n_ref_roots) {
        roots.push_back(x0);
        roots.push_back(x1);
      } else if ((n_roots + 1) == n_ref_roots) {
        // We decide if inserting x0 and x1 based on the function signs.
        auto xx = ref_intervals.point;
        this->set_psi_bounds(psi_id, xx);
        xx(this->dir_k_) = numbers::half * (x0 + ref_roots.front());
        const auto sign = this->phi_(xx) > 0;

        std::sort(roots.begin(), roots.end());
        const real xmid = roots.empty() ? x1 : roots.front();
        point(this->dir_k_) = numbers::half * (x0 + xmid);
        const auto new_sign = this->phi_(point) > 0;

        roots.push_back(sign == new_sign ? x1 : x0);
      }
    }

    if (roots.size() == n_ref_roots) {
      return roots;
    } else {
      return ref_roots;// Backup strategy
    }
  }

  //! @brief Computes all the intervals (between consecutive roots)
  //! at point @p point along direction dir_k_, taking as reference the provided
  //! @p ref_intervals.
  //!
  //! This method allows to deal with degenerate intervals by comparing
  //! the computed ones with the ones obtained at a point without degeneracies.
  //!
  //! @param ref_intervals Reference intervals.
  //! @param point Point at which new intervals are computed.
  //! @param Computed intervals.
  //!
  //! @warning This method is not bulletproof, and it may fail in corner cases.
  // NOLINTNEXTLINE (readability-function-cognitive-complexity)
  void compute_similar_intervals(const RootsIntervals<range> &ref_intervals,
    Vector<real, range> point,
    RootsIntervals<range> &intervals) const
  {
    intervals.clear();
    intervals.point = point;

    const auto n_roots = ref_intervals.get_num_roots();
    if (n_roots == 2) {
      if constexpr (!S) {
        intervals = ref_intervals;
      }
      return;
    }


    for (int psi_id = 0; psi_id < this->psi_count_; ++psi_id) {
      const auto new_roots_i = compute_similar_roots(ref_intervals, psi_id, point);

      for (const auto root : new_roots_i) {
        intervals.add_root(root, psi_id);
      }
    }

    const auto x0 = this->xrange_.min(this->dir_k_);
    const auto x1 = this->xrange_.max(this->dir_k_);

    if constexpr (!S) {
      intervals.add_root(x0, -1);
      intervals.add_root(x1, -1);
      intervals.active_intervals = ref_intervals.active_intervals;
      assert(intervals.get_num_roots() == ref_intervals.get_num_roots());
    }

    const auto tol = this->compute_tolerance();
    intervals.adjust_roots(tol, x0, x1);
  }

  //! @brief Computes all the intervals (between consecutive roots)
  //! at point @p point along direction dir_k_.
  //! In the 1D, it just computes the intervals without any reference.
  //! For higher dimensions, the stored reference intervals are used.
  //!
  //! @param point Point at which new intervals are computed.
  //! @param cell_tid Element tensor index (for selecting the reference intervals).
  //! @param Computed intervals.
  void compute_intervals(Vector<real, range> point,
    const TensorIndexTP<range> &cell_tid,
    RootsIntervals<range> &intervals) const
  {
    const auto &ref_int = this->ref_intervals_.at(cell_tid);
    if constexpr (dim == 1) {
      intervals = ref_int;
    } else {// if (1 < dim)
      compute_similar_intervals(ref_int, point, intervals);
    }
  }

  [[nodiscard]] real generate_point_in_01(int ind)
  {
    assert(0 <= ind && ind < this->order_);
    if (this->use_Chebyshev_) {
      return alg::bernstein::modifiedChebyshevNode(ind, this->order_);
    } else {
      return static_cast<real>(ind) / static_cast<real>(this->order_ - 1);
    }
  }

  //! @brief This method performs the reparameterization in a recursive way by
  //! computing the points coordinates along the different dimensions.
  //!
  //! @param point Partially computed point to be used in the reparameterization.
  //! @param cell_tid (Partial) tensor index of the element being reparameterized.
  //! @param pt_tid (Partial) tensor index of the reparameterization point.
  // NOLINTNEXTLINE (readability-function-cognitive-complexity)
  void process(Vector<real, range> point, TensorIndexTP<range> cell_tid, TensorIndexTP<range> pt_tid)
  {
    if constexpr (dim == 1) {
      this->compute_ref_intervals(point, cell_tid);
    }

    static thread_local RootsIntervals<range> intervals;
    this->compute_intervals(point, cell_tid, intervals);

    if constexpr (S) {
      static_assert(T, "Invalid dimension.");

      for (const auto root : intervals.roots) {
        point(this->dir_k_) = root;
        cell_tid(this->dir_k_) = 0;
        const TensorIndexTP<range - 1> srf_pt_tid(remove_component(pt_tid, this->dir_k_));
        this->insert_point_reparam(point, cell_tid, srf_pt_tid);
      }
    } else {
      // Loop over segments of divided interval
      const auto n_int = static_cast<int>(intervals.active_intervals.size());
      for (int i = 0; i < n_int; ++i) {
        if (!at(intervals.active_intervals, i)) {
          continue;
        }

        const real x0 = at(intervals.roots, i);
        const real x1 = at(intervals.roots, i + 1);

        cell_tid(this->dir_k_) = i;

        for (int j = 0; j < this->order_; ++j) {
          point(this->dir_k_) = x0 + (x1 - x0) * generate_point_in_01(j);
          pt_tid(this->dir_k_) = j;
          if constexpr (T && dim < range)// Face
          {
            int fix_dir{ -1 };
            for (int dir = 0; dir < range; ++dir) {
              if (!this->free_(dir)) {
                fix_dir = dir;
              }
            }
            this->set_psi_bounds(0, point);


            const auto face_pt_tid = TensorIndexTP<range - 1>(remove_component(pt_tid, fix_dir));
            if constexpr (T) {
              this->insert_point_reparam(point, cell_tid, face_pt_tid);
            } else {
              reparam_.process(point, cell_tid, face_pt_tid);
            }
          } else {
            if constexpr (T) {
              this->insert_point_reparam(point, cell_tid, pt_tid);
            } else {
              reparam_.process(point, cell_tid, pt_tid);
            }
          }
        }
      }// i
    }
  }

  //! Main calling engine; parameters with underscores are copied upon entry but modified internally in the ctor
  // NOLINTNEXTLINE (misc-no-recursion)
  ImplicitGeneralReparam(const ImplicitFunc<range> &phi,
    R &reparam,
    const Vector<bool, range> &free,
    const Vector<alg::PsiCode<range>, n_facets> &psi,
    int psi_count,
    const alg::HyperRectangle<real, range> &xrange,
    // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
    int order,
    int level = 0)
    : phi_(phi), reparam_(reparam), free_(free), psi_(psi), psi_count_(psi_count), xrange_(xrange), order_(order),
      use_Chebyshev_(ImplReparamMesh<range, range>::use_Chebyshev(order))
  {
    // For the one-dimensional base case, evaluate the bottom-level integral.
    if constexpr (dim == 1) {
      for (int dir = 0; dir < range; ++dir) {
        if (this->free_(dir)) {
          this->dir_k_ = dir;
        }
      }
      process(real(0.0), TensorIndexTP<range>(0), TensorIndexTP<range>(0));
      return;
    }

    // Establish interval bounds for prune() and remaining part of ctor.
    for (int dir = 0; dir < range; ++dir) {
      if (this->free_(dir)) {
        this->xint_(dir) = alg::Interval<range>(this->xrange_.midpoint(dir), set_component<real, range>(0.0, dir, 1.0));
        alg::Interval<range>::delta(dir) = this->xrange_.extent(dir) * numbers::half;
      } else {
        this->xint_(dir) = alg::Interval<range>(real(0.0));// this->xint_(dir).delta will be set per psi function
        alg::Interval<range>::delta(dir) = real(0.0);
      }
    }

    // Prune list of psi functions: if prune procedure returns false, then the domain of integration is empty.
    if (!prune()) {
      return;
    }

    // If all psi functions were pruned, then the volumetric integral domain is the entire hyperrectangle.
    if (this->psi_count_ == 0) {
      if constexpr (!S) {
        if constexpr (T) {
          this->tensor_product_reparam_full();
        } else {// if (!T)
          this->tensor_product_reparam_base();
        }
      }
      return;
    }

    // Among all monotone height function directions, choose the one that makes the associated height function look as
    // flat as possible. This is a modification to the criterion presented in [R. Saye, High-Order Quadrature Methods
    // for Implicitly Defined Surfaces and Volumes in Hyperrectangles, SIAM J. Sci. Comput., Vol. 37, No. 2, pp.
    // A993-A1019, http://dx.doi.org/10.1137/140966290].
    this->dir_k_ = -1;
    real max_quan = 0.0;
    for (int dir = 0; dir < range; ++dir) {
      if (!this->free_(dir)) {
        this->xint_(dir).alpha = this->xrange_.side(this->psi_(0).side(dir))(dir);
      }
    }
    Vector<alg::Interval<range>, range> grad = this->phi_.grad(this->xint_);
    for (int dir = 0; dir < range; ++dir) {
      // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
      if (this->free_(dir) && std::abs(grad(dir).alpha) > 1.001 * grad(dir).maxDeviation()) {
        const real quan = std::abs(grad(dir).alpha) * this->xrange_.extent(dir);
        if (quan > max_quan) {
          max_quan = quan;
          this->dir_k_ = dir;
        }
      }
    }

    // Check compatibility with all implicit functions whilst simultaneously constructing new implicit functions.
    Vector<alg::PsiCode<range>, n_facets> new_psi;
    int new_psi_count = 0;
    for (int i = 0; i < this->psi_count_; ++i) {
      // Evaluate gradient in an interval
      for (int dir = 0; dir < range; ++dir) {
        if (!this->free_(dir)) {
          this->xint_(dir).alpha = this->xrange_.side(this->psi_(i).side(dir))(dir);
        }
      }
      const auto new_grad = this->phi_.grad(this->xint_);

      // Determine if derivative in dir_k_ direction is bounded away from zero.
      const bool direction_okay = this->dir_k_ != -1 && new_grad(this->dir_k_).uniformSign();
      if (!direction_okay) {
        if (level < n_max_subdivs) {
          // Direction is not a good one, divide the domain into two along the biggest free_ extent
          real max_ext = 0.0;
          int ind = -1;
          for (int dir = 0; dir < range; ++dir) {
            if (this->free_(dir)) {
              const real ext = this->xrange_.extent(dir);
              if (ext > max_ext) {
                max_ext = ext;
                ind = dir;
              }
            }
          }
          assert(ind >= 0);
          const real xmid = this->xrange_.midpoint(ind);
          ImplicitGeneralReparam<dim, range, R, S, T>(this->phi_,
            reparam,
            this->free_,
            this->psi_,
            this->psi_count_,
            alg::HyperRectangle<real, range>(this->xrange_.min(), set_component(this->xrange_.max(), ind, xmid)),
            this->order_,
            level + 1);
          this->clear_reparam_elems_map();

          ImplicitGeneralReparam<dim, range, R, S, T>(this->phi_,
            reparam,
            this->free_,
            this->psi_,
            this->psi_count_,
            alg::HyperRectangle<real, range>(set_component(this->xrange_.min(), ind, xmid), this->xrange_.max()),
            this->order_,
            level + 1);
          return;
        } else {
          // Halt subdivision because we have recursively subdivided too deep; evaluate level set functions at
          // the centre of box and check compatibility with signs.
          Vector<real, range> xpoint = this->xrange_.midpoint();
          bool okay = true;
          for (int j = 0; j < n_facets && okay; ++j) {
            for (int dir = 0; dir < range; ++dir) {
              if (!this->free_(dir)) {
                xpoint(dir) = this->xrange_.side(this->psi_(j).side(dir))(dir);
              }
            }
            okay &= this->phi_(xpoint) >= 0.0 ? (this->psi_(j).sign() >= 0) : (this->psi_(j).sign() <= 0);
          }
          if (okay) {
            if constexpr (S) {
              assert(dim == range);
            } else {
              if constexpr (T) {
                this->tensor_product_reparam_full();
              } else {// if (!T)
                this->tensor_product_reparam_base();
              }
            }
          }
          return;
        }
      }

      // Direction is okay - build restricted level set functions and determine the appropriate signs
      int bottom_sign{ 0 };
      int top_sign{ 0 };
      algoim::detail::determineSigns<S>(grad(this->dir_k_).alpha > 0.0, this->psi_(i).sign(), bottom_sign, top_sign);
      new_psi(new_psi_count++) = alg::PsiCode<range>(this->psi_(i), this->dir_k_, 0, bottom_sign);
      new_psi(new_psi_count++) = alg::PsiCode<range>(this->psi_(i), this->dir_k_, 1, top_sign);
      assert(new_psi_count <= n_facets);
    }

    // Dimension reduction call
    assert(this->dir_k_ != -1);
    ImplicitGeneralReparam<dim - 1, range, Self, false, false>(this->phi_,
      *this,
      set_component(this->free_, this->dir_k_, false),
      new_psi,
      new_psi_count,
      this->xrange_,
      this->order_);
  }

  static void orient_cells(const ImplicitFunc<range> &phi, const std::vector<int> &cell_ids, R &reparam)
  {
    if constexpr (T) {
      if constexpr (S) {
        const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> phi_vec{ phi };
        reparam.orient_levelset_cells_positively(phi_vec, cell_ids);

      } else if constexpr (dim == range) {
        reparam.orient_cells_positively(cell_ids);
      }
    }
  }

  //! @brief Reparameterizes the domain defined implicitly by a single C1 scalar function.
  //! The interior of the domain is the subregion where that function has a negative value.
  //!
  //! @param Implicit function to reparameterize.
  //! @param domain Domain in which the reparameterization is performed.
  //! @param Reparamterization mesh to which new generated cells are appended to.
  static void reparameterize(const ImplicitFunc<range> &phi, const BoundBox<range> &domain, R &reparam)
  {
    Vector<alg::PsiCode<range>, n_facets> psi;
    psi(0) = alg::PsiCode<range>(0, -1);

    const Vector<bool, range> free = true;

    static const int param_dim = S ? dim - 1 : dim;
    const auto n_cells_est = param_dim == 2 ? 3 : 13;// Estimation
    reparam.reserve_cells(n_cells_est);

    const auto order = reparam.get_order();

    const auto n_cells_0 = static_cast<int>(reparam.get_num_cells());
    const ImplicitGeneralReparam<range, range, R, S> impl(
      phi, reparam, free, psi, 1, domain.to_hyperrectangle(), order);
    const auto n_cells_1 = static_cast<int>(reparam.get_num_cells());

    const auto rng = std::ranges::iota_view<int, int>{ n_cells_0, n_cells_1 };
    const std::vector<int> cell_ids(rng.begin(), rng.end());

    orient_cells(phi, cell_ids, reparam);

    const Tolerance tol(1.0e4 * numbers::eps);

    const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> funcs_vec{ phi };
    reparam.generate_wirebasket(funcs_vec, cell_ids, domain, tol);
  }

  static void reparameterize_facet(const ImplicitFunc<range> &phi,
    const BoundBox<range> &domain,
    // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
    const int facet_id,
    R &reparam)
  {
    static_assert(!S, "Not allowed.");

    const auto n_cells_est = (dim - 1) == 2 ? 3 : 2;
    reparam.reserve_cells(n_cells_est);

    const auto const_dir = get_facet_constant_dir<range>(facet_id);
    const auto side = get_facet_side<range>(facet_id);

    Vector<alg::PsiCode<range>, n_facets> psi;
    psi(0) = alg::PsiCode<range>(set_component<int, range>(0, const_dir, side), -1);

    Vector<bool, range> free = true;
    free(const_dir) = false;

    const auto order = reparam.get_order();

    const auto n_cells_0 = static_cast<int>(reparam.get_num_cells());
    const ImplicitGeneralReparam<range - 1, range, R, S> impl(
      phi, reparam, free, psi, 1, domain.to_hyperrectangle(), order);
    const auto n_cells_1 = static_cast<int>(reparam.get_num_cells());

    const auto rng = std::ranges::iota_view<int, int>{ n_cells_0, n_cells_1 };
    const std::vector<int> cell_ids(rng.begin(), rng.end());

    reparam.orient_facet_cells_positively(facet_id, cell_ids);

    const Tolerance tol(1.0e4 * numbers::eps);

    const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> funcs_vec{ phi };
    reparam.generate_wirebasket(funcs_vec, cell_ids, domain, tol);
  }
};


//! Partial specialization on dim=0 as a dummy base case for the compiler
template<int range, typename R, bool S, bool T> struct ImplicitGeneralReparam<0, range, R, S, T>
{
  // NOLINTNEXTLINE (hicpp-signed-bitwise)
  static constexpr int n_facets = 1 << (range - 1);
  // NOLINTBEGIN (hicpp-named-parameter,readability-named-parameter)
  ImplicitGeneralReparam(const ImplicitFunc<range> &,
    R &,
    const Vector<bool, range> &,
    const Vector<alg::PsiCode<range>, n_facets> &,
    int,
    const alg::HyperRectangle<real, range> &,
    int)
  // NOLINTEND (hicpp-named-parameter,readability-named-parameter)
  {}
};


template<int dim, bool S>
std::shared_ptr<ImplReparamMesh<S ? dim - 1 : dim, dim>>
  reparam_general(const ImplicitFunc<dim> &func, const BoundBox<dim> &domain, const int order)
{
  using R = ImplReparamMesh<S ? dim - 1 : dim, dim>;
  const auto reparam = std::make_shared<R>(order);

  reparam_general<dim, S>(func, domain, *reparam);
  return reparam;
}

template<int dim, bool S>
void reparam_general(const ImplicitFunc<dim> &func,
  const BoundBox<dim> &domain,
  ImplReparamMesh<S ? dim - 1 : dim, dim> &reparam)
{
  using R = ImplReparamMesh<S ? dim - 1 : dim, dim>;
  ImplicitGeneralReparam<dim, dim, R, S>::reparameterize(func, domain, reparam);
}


template<int dim>
std::shared_ptr<ImplReparamMesh<dim - 1, dim>>
  // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
  reparam_general_facet(const ImplicitFunc<dim> &func, const BoundBox<dim> &domain, const int facet_id, const int order)
{
  using R = ImplReparamMesh<dim - 1, dim>;
  const auto reparam = std::make_shared<R>(order);

  reparam_general_facet(func, domain, facet_id, *reparam);
  return reparam;
}

template<int dim>
void reparam_general_facet(const ImplicitFunc<dim> &func,
  const BoundBox<dim> &domain,
  const int facet_id,
  ImplReparamMesh<dim - 1, dim> &reparam)
{
  using R = ImplReparamMesh<dim - 1, dim>;
  ImplicitGeneralReparam<dim - 1, dim, R, false>::reparameterize_facet(func, domain, facet_id, reparam);
}

// Instantiations.

template std::shared_ptr<ImplReparamMesh<2, 2>>
  reparam_general<2, false>(const ImplicitFunc<2> &, const BoundBox<2> &, const int);
template std::shared_ptr<ImplReparamMesh<1, 2>>
  reparam_general<2, true>(const ImplicitFunc<2> &, const BoundBox<2> &, const int);

template std::shared_ptr<ImplReparamMesh<3, 3>>
  reparam_general<3, false>(const ImplicitFunc<3> &, const BoundBox<3> &, const int);
template std::shared_ptr<ImplReparamMesh<2, 3>>
  reparam_general<3, true>(const ImplicitFunc<3> &, const BoundBox<3> &, const int);

template void reparam_general<2, false>(const ImplicitFunc<2> &, const BoundBox<2> &, ImplReparamMesh<2, 2> &);
template void reparam_general<2, true>(const ImplicitFunc<2> &, const BoundBox<2> &, ImplReparamMesh<1, 2> &);

template void reparam_general<3, false>(const ImplicitFunc<3> &, const BoundBox<3> &, ImplReparamMesh<3, 3> &);
template void reparam_general<3, true>(const ImplicitFunc<3> &, const BoundBox<3> &, ImplReparamMesh<2, 3> &);

template std::shared_ptr<ImplReparamMesh<1, 2>>
  reparam_general_facet<2>(const ImplicitFunc<2> &, const BoundBox<2> &, const int, const int);
template std::shared_ptr<ImplReparamMesh<2, 3>>
  reparam_general_facet<3>(const ImplicitFunc<3> &, const BoundBox<3> &, const int, const int);

template void
  reparam_general_facet<2>(const ImplicitFunc<2> &, const BoundBox<2> &, const int, ImplReparamMesh<1, 2> &);
template void
  reparam_general_facet<3>(const ImplicitFunc<3> &, const BoundBox<3> &, const int, ImplReparamMesh<2, 3> &);

}// namespace qugar::impl