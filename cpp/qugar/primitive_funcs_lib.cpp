// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file primitive_funcs_lib.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of a few primitive implicit functions ready to be consumed by Algoim.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/primitive_funcs_lib.hpp>

#include <qugar/bezier_tp.hpp>
#include <qugar/impl_funcs_lib_macros.hpp>
#include <qugar/monomials_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/ref_system.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <cassert>
#include <memory>
#include <vector>

namespace qugar::impl::funcs {


template<int dim>
SphereBase<dim>::SphereBase(const real radius, const Point<dim> &center) : radius_(radius), center_(center)
{
  assert(this->radius_ > numbers::zero);
}

template<int dim> Point<dim> SphereBase<dim>::get_default_center()
{
  return Point<dim>(numbers::zero);
}

template<int dim> real SphereBase<dim>::radius() const
{
  return this->radius_;
}

template<int dim> const Point<dim> &SphereBase<dim>::center() const
{
  return this->center_;
}

template<int dim> Sphere<dim>::Sphere(const real radius) : Sphere(radius, SphereBase<dim>::get_default_center()) {}

template<int dim> Sphere<dim>::Sphere(const real radius, const Point<dim> &center) : SphereBase<dim>(radius, center) {}

implement_impl_func(Sphere);

template<int dim> template<typename T> T Sphere<dim>::eval_(const Point<dim, T> &point) const
{
  return sqrnorm(point - this->center_) - (this->radius_ * this->radius_);
}

template<int dim> template<typename T> Vector<T, dim> Sphere<dim>::grad_(const Point<dim, T> &point) const
{
  return numbers::two * (point - this->center_);
}

template<int dim> template<typename T> auto Sphere<dim>::hessian_(const Point<dim, T> & /*point*/) const -> Hessian<T>
{
  if constexpr (dim == 2) {
    return Hessian<T>{ numbers::two, numbers::zero, numbers::two };
  } else {// if constexpr (dim == 3)
    return Hessian<T>{ numbers::two, numbers::zero, numbers::zero, numbers::two, numbers::zero, numbers::two };
  }
}

template<int dim>
SphereBzr<dim>::SphereBzr(const real radius) : SphereBzr(radius, SphereBase<dim>::get_default_center())
{}

template<int dim>
SphereBzr<dim>::SphereBzr(const real radius, const Point<dim> &center)
  : SphereBase<dim>(radius, center), BezierTP<dim, 1>(*create_monomials(radius, center))
{}


template<int dim>
std::shared_ptr<MonomialsTP<dim, 1>> SphereBzr<dim>::create_monomials(const real radius, const Point<dim> &center)
{
  assert(radius > numbers::zero);

  const auto monomials = std::make_shared<MonomialsTP<dim, 1>>(TensorSizeTP<dim>(3));
  monomials->get_coef(0) = -radius * radius;
  for (int dir = 0; dir < dim; ++dir) {
    TensorIndexTP<dim> tid{ 0 };
    monomials->get_coef(tid) += center(dir) * center(dir);

    tid(dir) = 1;
    monomials->get_coef(tid) = -numbers::two * center(dir);

    tid(dir) = 2;
    monomials->get_coef(tid) = numbers::one;
  }
  return monomials;
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
CylinderBase::CylinderBase(const real radius, const Point<3> &origin, const Point<3> &axis)
  : radius_(radius), origin_(origin), axis_(axis)
{
  assert(this->radius_ > numbers::zero);

  const auto norm_axis = norm(this->axis_);
  if (!Tolerance().is_zero(norm_axis)) {
    axis_ /= norm_axis;
  }
}

real CylinderBase::radius() const
{
  return this->radius_;
}

const Point<3> &CylinderBase::origin() const
{
  return this->origin_;
}

const Point<3> &CylinderBase::axis() const
{
  return this->axis_;
}

Point<3> CylinderBase::get_default_origin()
{
  return Point<3>{ 0.0, 0.0, 0.0 };
}

Point<3> CylinderBase::get_default_axis()
{
  return Point<3>{ 0.0, 0.0, 1.0 };
}

CylinderBzr::CylinderBzr(const real radius) : CylinderBzr(radius, CylinderBase::get_default_origin()) {}

CylinderBzr::CylinderBzr(const real radius, const Point<3> &origin)
  : CylinderBzr(radius, origin, CylinderBase::get_default_axis())
{}

CylinderBzr::CylinderBzr(const real radius, const Point<3> &origin, const Point<3> &axis)
  : CylinderBase(radius, origin, axis), BezierTP<3, 1>(*create_monomials(radius, origin, axis))
{}

std::shared_ptr<MonomialsTP<3, 1>>
  CylinderBzr::create_monomials(const real radius, const Point<3> &origin, const Point<3> &axis)
{
  assert(radius > numbers::zero);

  const auto monomials = std::make_shared<MonomialsTP<3, 1>>(TensorSizeTP<3>(3));

  const RefSystem<3> system(origin, axis);

  const auto &basis = system.get_basis();

  monomials->get_coef(0) = -radius * radius;

  for (int dir = 0; dir < 2; ++dir) {

    const auto &vec = at(basis, dir);

    const auto proj_origin = dot(vec, origin);

    monomials->get_coef(0) += proj_origin * proj_origin;

    for (int dir2 = 0; dir2 < 3; ++dir2) {
      TensorIndexTP<3> tid{ 0 };

      ++tid(dir2);
      monomials->get_coef(tid) += -numbers::two * vec(dir2) * proj_origin;

      for (int dir3 = 0; dir3 < 3; ++dir3) {
        ++tid(dir3);
        monomials->get_coef(tid) += vec(dir2) * vec(dir3);
        --tid(dir3);
      }
    }
  }

  return monomials;
}

Cylinder::Cylinder(const real radius) : Cylinder(radius, CylinderBase::get_default_origin()) {}

Cylinder::Cylinder(const real radius, const Point<3> &origin)
  : Cylinder(radius, origin, CylinderBase::get_default_axis())
{}

Cylinder::Cylinder(const real radius, const Point<3> &origin, const Point<3> &axis) : CylinderBase(radius, origin, axis)
{}

implement_impl_func_3D(Cylinder);

template<typename T> T Cylinder::eval_(const Point<3, T> &point) const
{
  const Point<3, T> diff = point - this->origin_;
  const T proj = dot(diff, this->axis_);
  // NOLINTNEXTLINE (readability-math-missing-parentheses)
  return dot(diff, diff) - proj * proj - (this->radius_ * this->radius_);
}

template<typename T> Vector<T, 3> Cylinder::grad_(const Point<3, T> &point) const
{
  const Point<3, T> diff = point - this->origin_;
  const T proj = dot(diff, this->axis_);
  // NOLINTNEXTLINE (readability-math-missing-parentheses)
  return numbers::two * diff - numbers::two * proj * this->axis_;
}

template<typename T> auto Cylinder::hessian_(const Point<3, T> & /*point*/) const -> Hessian<T>
{
  return Hessian<T>{ numbers::two, numbers::zero, numbers::zero, numbers::two, numbers::zero, numbers::two };
  Hessian<T> hess;
  for (int i = 0, ij = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j, ++ij) {
      const real Idt = (i == j) ? numbers::one : numbers::zero;
      hess(ij) = numbers::two * Idt - numbers::two * this->axis_(i) * this->axis_(j);
    }
  }
  return hess;
}

template<int dim>
EllipsoidBase<dim>::EllipsoidBase(const Point<dim> &semi_axes, const RefSystem<dim> &system)
  : semi_axes_(semi_axes), system_(system)
{
#ifndef NDEBUG
  for (int dir = 0; dir < dim; ++dir) {
    assert(semi_axes(dir) > numbers::zero);
  }
#endif// NDEBUG
}

template<int dim> const Point<dim> &EllipsoidBase<dim>::semi_axes() const
{
  return this->semi_axes_;
}

template<int dim> const RefSystem<dim> &EllipsoidBase<dim>::ref_system() const
{
  return this->system_;
}

template<int dim> RefSystem<dim> EllipsoidBase<dim>::get_default_system()
{
  return RefSystem<dim>{};
}

template<int dim>
EllipsoidBzr<dim>::EllipsoidBzr(const Point<dim> &semi_axes)
  : EllipsoidBzr<dim>(semi_axes, EllipsoidBase<dim>::get_default_system())
{}

template<int dim>
EllipsoidBzr<dim>::EllipsoidBzr(const Point<dim> &semi_axes, const RefSystem<dim> &system)
  : EllipsoidBase<dim>(semi_axes, system), BezierTP<dim, 1>(*create_monomials(semi_axes, system))
{}

template<int dim>
std::shared_ptr<MonomialsTP<dim, 1>> EllipsoidBzr<dim>::create_monomials(const Point<dim> &semi_axes,
  const RefSystem<dim> &system)
{
  const auto monomials = std::make_shared<MonomialsTP<dim, 1>>(TensorSizeTP<dim>(3));

  const auto &center = system.get_origin();
  const auto &basis = system.get_basis();

  monomials->get_coef(0) = -numbers::one;

  for (int dir = 0; dir < dim; ++dir) {
    assert(semi_axes(dir) > numbers::zero);

    const real coef = numbers::one / semi_axes(dir) / semi_axes(dir);

    const auto &vec = at(basis, dir);

    const auto proj_center = dot(vec, center);

    monomials->get_coef(0) += proj_center * proj_center * coef;

    for (int dir2 = 0; dir2 < dim; ++dir2) {
      TensorIndexTP<dim> tid{ 0 };

      ++tid(dir2);
      monomials->get_coef(tid) += -numbers::two * vec(dir2) * proj_center * coef;

      for (int dir3 = 0; dir3 < dim; ++dir3) {
        ++tid(dir3);
        monomials->get_coef(tid) += vec(dir2) * vec(dir3) * coef;
        --tid(dir3);
      }
    }
  }

  return monomials;
}

template<int dim>
Ellipsoid<dim>::Ellipsoid(const Point<dim> &semi_axes)
  : Ellipsoid<dim>(semi_axes, EllipsoidBase<dim>::get_default_system())
{}

template<int dim>
Ellipsoid<dim>::Ellipsoid(const Point<dim> &semi_axes, const RefSystem<dim> &system)
  : EllipsoidBase<dim>(semi_axes, system)
{}

implement_impl_func(Ellipsoid);

template<int dim> template<typename T> T Ellipsoid<dim>::eval_(const Point<dim, T> &point) const
{
  const Point<dim, T> diff = point - this->system_.get_origin();

  T val{ -numbers::one };
  for (int dir = 0; dir < dim; ++dir) {
    const auto proj = dot(at(this->system_.get_basis(), dir), diff) / this->semi_axes_(dir);
    val += proj * proj;
  }
  return val;
}

template<int dim> template<typename T> Vector<T, dim> Ellipsoid<dim>::grad_(const Point<dim, T> &point) const
{
  const Point<dim, T> diff = point - this->system_.get_origin();

  Vector<T, dim> grad{ numbers::zero };
  for (int dir = 0; dir < dim; ++dir) {
    const real coef = numbers::two / this->semi_axes_(dir) / this->semi_axes_(dir);
    const auto &axis = at(this->system_.get_basis(), dir);
    grad += (coef * dot(axis, diff)) * axis;
  }
  return grad;
}

template<int dim>
template<typename T>
auto Ellipsoid<dim>::hessian_(const Point<dim, T> & /*point*/) const -> Hessian<T>
{
  Hessian<T> hess{ numbers::zero };
  for (int dir = 0; dir < dim; ++dir) {
    const auto &axis = at(this->system_.get_basis(), dir);
    const real coef = numbers::two / this->semi_axes_(dir) / this->semi_axes_(dir);
    if constexpr (dim == 2) {
      hess(0) += coef * axis(0) * axis(0);
      hess(1) += coef * axis(0) * axis(1);
      hess(2) += coef * axis(1) * axis(1);
    } else {// if constexpr (dim == 3)
      hess(0) += coef * axis(0) * axis(0);
      hess(1) += coef * axis(0) * axis(1);
      hess(2) += coef * axis(0) * axis(2);
      hess(3) += coef * axis(1) * axis(1);
      hess(4) += coef * axis(1) * axis(2);
      // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
      hess(5) += coef * axis(2) * axis(2);
    }
  }
  return hess;
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
AnnulusBase::AnnulusBase(const real inner_radius, const real outer_radius, const Point<2> &center)
  : inner_radius_(inner_radius), outer_radius_(outer_radius), center_(center)
{
  assert(0.0 < this->inner_radius_ && this->inner_radius_ < this->outer_radius_);
}

real AnnulusBase::inner_radius() const
{
  return this->inner_radius_;
}

real AnnulusBase::outer_radius() const
{
  return this->outer_radius_;
}

const Point<2> &AnnulusBase::center() const
{
  return this->center_;
}

Point<2> AnnulusBase::get_default_center()
{
  return Point<2>{ 0.0, 0.0 };
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
AnnulusBzr::AnnulusBzr(const real inner_radius, const real outer_radius)
  : AnnulusBzr(inner_radius, outer_radius, AnnulusBase::get_default_center())
{}


// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
AnnulusBzr::AnnulusBzr(const real inner_radius, const real outer_radius, const Point<2> &center)
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  : AnnulusBase(inner_radius, outer_radius, center),
    BezierTP<2, 1>(*create_monomials(inner_radius, outer_radius, center))
{}


std::shared_ptr<MonomialsTP<2, 1>>
  // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
  AnnulusBzr::create_monomials(const real inner_radius, const real outer_radius, const Point<2> &center)
{
  assert(0.0 < inner_radius && inner_radius < outer_radius);

  const real Ri_2 = inner_radius * inner_radius;
  const real Ro_2 = outer_radius * outer_radius;
  const real o0 = center(0);
  const real o1 = center(1);
  const real o0_2 = o0 * o0;
  const real o1_2 = o1 * o1;


  // The coefficients below were computed with sympy as follows:
  // import sympy as sym

  // import sympy as sym

  // x0, x1 = sym.symbols('x0 x1')
  // o0, o1 = sym.symbols('o0 o1')
  // R, r = sym.symbols('R r')

  // y = sym.Matrix([[x0 - o0, x1 - o1]])
  // y2 = y.dot(y)
  // phi =(y2 - r * r) *(y2 - R * R) print(sym.expand(sym.collect(phi, [x0, x1])))

  // -R**2*o0**2 + 2*R**2*o0*x0 - R**2*o1**2 + 2*R**2*o1*x1 + R**2*r**2 - R**2*x0**2 - R**2*x1**2 + o0**4 - 4*o0**3*x0
  // + 2*o0**2*o1**2 - 4*o0**2*o1*x1 - o0**2*r**2 + 6*o0**2*x0**2 + 2*o0**2*x1**2 - 4*o0*o1**2*x0 + 8*o0*o1*x0*x1 +
  // 2*o0*r**2*x0 - 4*o0*x0**3 - 4*o0*x0*x1**2 + o1**4 - 4*o1**3*x1 - o1**2*r**2 + 2*o1**2*x0**2 + 6*o1**2*x1**2 +
  // 2*o1*r**2*x1 - 4*o1*x0**2*x1 - 4*o1*x1**3 - r**2*x0**2 - r**2*x1**2 + x0**4 + 2*x0**2*x1**2 + x1**4


  const auto monomials = std::make_shared<MonomialsTP<2, 1>>(TensorSizeTP<2>(5));

  monomials->get_coef(TensorIndexTP<2>{ 0, 0 }) = -Ro_2 * o0_2 - Ro_2 * o1_2 + Ro_2 * Ri_2 + o0_2 * o0_2
                                                  + numbers::two * o0_2 * o1_2 - o0_2 * Ri_2 + o1_2 * o1_2
                                                  - o1_2 * Ri_2;
  monomials->get_coef(TensorIndexTP<2>{ 1, 0 }) =
    numbers::two * Ro_2 * o0 - numbers::four * o0_2 * o0 - numbers::four * o0 * o1_2 + numbers::two * o0 * Ri_2;
  monomials->get_coef(TensorIndexTP<2>{ 0, 1 }) =
    numbers::two * Ro_2 * o1 - numbers::four * o0_2 * o1 - numbers::four * o1_2 * o1 + numbers::two * o1 * Ri_2;
  monomials->get_coef(TensorIndexTP<2>{ 1, 1 }) = numbers::eight * o0 * o1;
  monomials->get_coef(TensorIndexTP<2>{ 2, 0 }) = -Ro_2 + numbers::six * o0_2 + numbers::two * o1_2 - Ri_2;
  monomials->get_coef(TensorIndexTP<2>{ 0, 2 }) = -Ro_2 + numbers::two * o0_2 + numbers::six * o1_2 - Ri_2;
  monomials->get_coef(TensorIndexTP<2>{ 1, 2 }) = -numbers::four * o0;
  monomials->get_coef(TensorIndexTP<2>{ 2, 1 }) = -numbers::four * o1;
  monomials->get_coef(TensorIndexTP<2>{ 3, 0 }) = -numbers::four * o0;
  monomials->get_coef(TensorIndexTP<2>{ 0, 3 }) = -numbers::four * o1;
  monomials->get_coef(TensorIndexTP<2>{ 2, 2 }) = numbers::two;
  monomials->get_coef(TensorIndexTP<2>{ 4, 0 }) = numbers::one;
  monomials->get_coef(TensorIndexTP<2>{ 0, 4 }) = numbers::one;

  return monomials;
}

Annulus::Annulus(const real inner_radius, const real outer_radius)
  : Annulus(inner_radius, outer_radius, AnnulusBase::get_default_center())
{}

Annulus::Annulus(const real inner_radius, const real outer_radius, const Point<2> &center)
  : AnnulusBase(inner_radius, outer_radius, center)
{}

implement_impl_func_2D(Annulus);

template<typename T> T Annulus::eval_(const Point<2, T> &point) const
{
  const auto dist2 = sqrnorm(point - this->center_);
  const auto Ri2 = this->inner_radius_ * this->inner_radius_;
  const auto Ro2 = this->outer_radius_ * this->outer_radius_;
  return ((dist2 - Ri2 - Ro2) * dist2) + (Ri2 * Ro2);
}

template<typename T> Vector<T, 2> Annulus::grad_(const Point<2, T> &point) const
{
  const Point<2, T> diff = point - this->center_;
  const auto Ri2 = this->inner_radius_ * this->inner_radius_;
  const auto Ro2 = this->outer_radius_ * this->outer_radius_;
  return (numbers::four * sqrnorm(diff) - Ri2 - Ro2) * diff;
}

template<typename T> auto Annulus::hessian_(const Point<2, T> &point) const -> Hessian<T>
{
  const Point<2, T> diff = point - this->center_;
  const auto aux = numbers::four * sqrnorm(diff);
  return Hessian<T>{ aux + (numbers::eight * diff(0) * diff(0)),
    numbers::eight * diff(0) * diff(1),
    aux + (numbers::eight * diff(1) * diff(1)) };
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
TorusBase::TorusBase(const real major_radius, const real minor_radius, const Point<3> &center, const Point<3> &axis)
  : major_radius_(major_radius), minor_radius_(minor_radius), center_(center), axis_(axis)
{
  assert(0.0 < this->minor_radius_ && this->minor_radius_ < this->major_radius_);
  const auto norm_axis = norm(this->axis_);
  if (!Tolerance().is_zero(norm_axis)) {
    axis_ /= norm_axis;
  }
}

Point<3> TorusBase::get_default_center()
{
  return Point<3>{ 0.0, 0.0, 0.0 };
}

Point<3> TorusBase::get_default_axis()
{
  return Point<3>{ 0.0, 0.0, 1.0 };
}

real TorusBase::major_radius() const
{
  return this->major_radius_;
}

real TorusBase::minor_radius() const
{
  return this->minor_radius_;
}

const Point<3> &TorusBase::center() const
{
  return this->center_;
}

const Point<3> &TorusBase::axis() const
{
  return this->axis_;
}


// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
TorusBzr::TorusBzr(const real major_radius, const real minor_radius)
  : TorusBzr(major_radius, minor_radius, TorusBase::get_default_center())
{}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
TorusBzr::TorusBzr(const real major_radius, const real minor_radius, const Point<3> &center)
  : TorusBzr(major_radius, minor_radius, center, TorusBase::get_default_axis())
{}


// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
TorusBzr::TorusBzr(const real major_radius, const real minor_radius, const Point<3> &center, const Point<3> &axis)
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  : TorusBase(major_radius, minor_radius, center, axis),
    BezierTP<3, 1>(*create_monomials(major_radius, minor_radius, center, axis))
{}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
std::shared_ptr<MonomialsTP<3, 1>> TorusBzr::create_monomials(const real major_radius,
  const real minor_radius,
  // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
  const Point<3> &center,
  const Point<3> &axis)
{
  assert(0.0 < minor_radius && minor_radius < major_radius);

  const Point<3> unit_axis = axis / norm(axis);

  const auto o0 = center(0);
  const auto o1 = center(1);
  const auto o2 = center(2);
  const auto o0_2 = o0 * o0;
  const auto o1_2 = o1 * o1;
  const auto o2_2 = o2 * o2;

  const auto a0 = unit_axis(0);
  const auto a1 = unit_axis(1);
  const auto a2 = unit_axis(2);
  const auto a0_2 = a0 * a0;
  const auto a1_2 = a1 * a1;
  const auto a2_2 = a2 * a2;

  // NOLINTBEGIN (readability-identifier-length)
  const auto R = major_radius;
  const auto R_2 = R * R;
  const auto r = minor_radius;
  const auto r_2 = r * r;
  // NOLINTEND (readability-identifier-length)

  // The coefficients below were computed with sympy as follows:
  // import sympy as sym

  // x0, x1, x2 = sym.symbols('x0 x1 x2')
  // a0, a1, a2 = sym.symbols('a0 a1 a2')
  // o0, o1, o2 = sym.symbols('o0 o1 o2')
  // R, r = sym.symbols('R r')

  // y = sym.Matrix([[x0 - o0, x1 - o1, x2 - o2]])
  // a = sym.Matrix([[a0, a1, a2]])
  // y2 = y.dot(y)

  // ay = y.dot(a)

  // phi = y2 * y2 - 2 * (R * R + r * r) * y2 + 4 * R * R * ay * ay + (R * R - r * r) * (R * R - r * r)
  // print(sym.expand(sym.collect(phi, [ x0, x1, x2 ])))
  //
  // >>> R**4 + 4*R**2*a0**2*o0**2 - 8*R**2*a0**2*o0*x0 + 4*R**2*a0**2*x0**2 + 8*R**2*a0*a1*o0*o1 - 8*R**2*a0*a1*o0*x1
  // - 8*R**2*a0*a1*o1*x0 + 8*R**2*a0*a1*x0*x1 + 8*R**2*a0*a2*o0*o2 - 8*R**2*a0*a2*o0*x2 - 8*R**2*a0*a2*o2*x0 +
  // 8*R**2*a0*a2*x0*x2 + 4*R**2*a1**2*o1**2 - 8*R**2*a1**2*o1*x1 + 4*R**2*a1**2*x1**2 + 8*R**2*a1*a2*o1*o2 -
  // 8*R**2*a1*a2*o1*x2 - 8*R**2*a1*a2*o2*x1 + 8*R**2*a1*a2*x1*x2 + 4*R**2*a2**2*o2**2 - 8*R**2*a2**2*o2*x2 +
  // 4*R**2*a2**2*x2**2 - 2*R**2*o0**2 + 4*R**2*o0*x0 - 2*R**2*o1**2 + 4*R**2*o1*x1 - 2*R**2*o2**2 + 4*R**2*o2*x2 -
  // 2*R**2*r**2 - 2*R**2*x0**2 - 2*R**2*x1**2 - 2*R**2*x2**2 + o0**4 - 4*o0**3*x0 + 2*o0**2*o1**2 - 4*o0**2*o1*x1 +
  // 2*o0**2*o2**2 - 4*o0**2*o2*x2 - 2*o0**2*r**2 + 6*o0**2*x0**2 + 2*o0**2*x1**2 + 2*o0**2*x2**2 - 4*o0*o1**2*x0 +
  // 8*o0*o1*x0*x1 - 4*o0*o2**2*x0 + 8*o0*o2*x0*x2 + 4*o0*r**2*x0 - 4*o0*x0**3 - 4*o0*x0*x1**2 - 4*o0*x0*x2**2 + o1**4
  // - 4*o1**3*x1 + 2*o1**2*o2**2 - 4*o1**2*o2*x2 - 2*o1**2*r**2 + 2*o1**2*x0**2 + 6*o1**2*x1**2 + 2*o1**2*x2**2 -
  // 4*o1*o2**2*x1 + 8*o1*o2*x1*x2 + 4*o1*r**2*x1 - 4*o1*x0**2*x1 - 4*o1*x1**3 - 4*o1*x1*x2**2 + o2**4 - 4*o2**3*x2 -
  // 2*o2**2*r**2 + 2*o2**2*x0**2 + 2*o2**2*x1**2 + 6*o2**2*x2**2 + 4*o2*r**2*x2 - 4*o2*x0**2*x2 - 4*o2*x1**2*x2 -
  // 4*o2*x2**3 + r**4 - 2*r**2*x0**2 - 2*r**2*x1**2 - 2*r**2*x2**2 + x0**4 + 2*x0**2*x1**2 + 2*x0**2*x2**2 + x1**4 +
  // 2*x1**2*x2**2 + x2**4


  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  const auto monomials = std::make_shared<MonomialsTP<3, 1>>(TensorSizeTP<3>(5));

  monomials->get_coef(TensorIndexTP<3>{ 0, 0, 0 }) =
    R_2 * R_2 + 4 * R_2 * a0_2 * o0_2 + 8 * R_2 * a0 * a1 * o0 * o1 + 8 * R_2 * a0 * a2 * o0 * o2
    + 4 * R_2 * a1_2 * o1_2 + 8 * R_2 * a1 * a2 * o1 * o2 + 4 * R_2 * a2_2 * o2_2 - 2 * R_2 * o0_2 - 2 * R_2 * o1_2
    - 2 * R_2 * o2_2 - 2 * R_2 * r_2 + o0_2 * o0_2 + 2 * o0_2 * o1_2 + 2 * o0_2 * o2_2 - 2 * o0_2 * r_2 + o1_2 * o1_2
    + 2 * o1_2 * o2_2 - 2 * o1_2 * r_2 + o2_2 * o2_2 - 2 * o2_2 * r_2 + r_2 * r_2;
  monomials->get_coef(TensorIndexTP<3>{ 1, 0, 0 }) = -8 * R_2 * a0_2 * o0 - 8 * R_2 * a0 * a1 * o1
                                                     - 8 * R_2 * a0 * a2 * o2 + 4 * R_2 * o0 - 4 * o0_2 * o0
                                                     - 4 * o0 * o1_2 - 4 * o0 * o2_2 + 4 * o0 * r_2;
  monomials->get_coef(TensorIndexTP<3>{ 0, 1, 0 }) = -8 * R_2 * a0 * a1 * o0 - 8 * R_2 * a1_2 * o1
                                                     - 8 * R_2 * a1 * a2 * o2 + 4 * R_2 * o1 - 4 * o0_2 * o1
                                                     - 4 * o1_2 * o1 - 4 * o1 * o2_2 + 4 * o1 * r_2;
  monomials->get_coef(TensorIndexTP<3>{ 0, 0, 1 }) = -8 * R_2 * a0 * a2 * o0 - 8 * R_2 * a1 * a2 * o1
                                                     - 8 * R_2 * a2_2 * o2 + 4 * R_2 * o2 - 4 * o0_2 * o2
                                                     - 4 * o1_2 * o2 - 4 * o2_2 * o2 + 4 * o2 * r_2;
  monomials->get_coef(TensorIndexTP<3>{ 1, 1, 0 }) = 8 * R_2 * a0 * a1 + 8 * o0 * o1;
  monomials->get_coef(TensorIndexTP<3>{ 1, 0, 1 }) = 8 * o0 * o2 + 8 * R_2 * a0 * a2;
  monomials->get_coef(TensorIndexTP<3>{ 0, 1, 1 }) = 8 * R_2 * a1 * a2 + 8 * o1 * o2;
  monomials->get_coef(TensorIndexTP<3>{ 2, 0, 0 }) =
    4 * R_2 * a0_2 - 2 * R_2 + 6 * o0_2 + 2 * o1_2 + 2 * o2_2 - 2 * r_2;
  monomials->get_coef(TensorIndexTP<3>{ 0, 2, 0 }) =
    4 * R_2 * a1_2 - 2 * R_2 + 2 * o0_2 + 6 * o1_2 + 2 * o2_2 - 2 * r_2;
  monomials->get_coef(TensorIndexTP<3>{ 0, 0, 2 }) =
    4 * R_2 * a2_2 - 2 * R_2 + 2 * o0_2 + 6 * o2_2 - 2 * r_2 + 2 * o1_2;
  monomials->get_coef(TensorIndexTP<3>{ 3, 0, 0 }) = -4 * o0;
  monomials->get_coef(TensorIndexTP<3>{ 0, 3, 0 }) = -4 * o1;
  monomials->get_coef(TensorIndexTP<3>{ 0, 0, 3 }) = -4 * o2;
  monomials->get_coef(TensorIndexTP<3>{ 1, 2, 0 }) = -4 * o0;
  monomials->get_coef(TensorIndexTP<3>{ 1, 0, 2 }) = -4 * o0;
  monomials->get_coef(TensorIndexTP<3>{ 0, 1, 2 }) = -4 * o1;
  monomials->get_coef(TensorIndexTP<3>{ 2, 1, 0 }) = -4 * o1;
  monomials->get_coef(TensorIndexTP<3>{ 2, 0, 1 }) = -4 * o2;
  monomials->get_coef(TensorIndexTP<3>{ 0, 2, 1 }) = -4 * o2;
  monomials->get_coef(TensorIndexTP<3>{ 2, 2, 0 }) = 2;
  monomials->get_coef(TensorIndexTP<3>{ 2, 0, 2 }) = 2;
  monomials->get_coef(TensorIndexTP<3>{ 0, 2, 2 }) = 2;
  monomials->get_coef(TensorIndexTP<3>{ 4, 0, 0 }) = 1;
  monomials->get_coef(TensorIndexTP<3>{ 0, 4, 0 }) = 1;
  monomials->get_coef(TensorIndexTP<3>{ 0, 0, 4 }) = 1;
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)


  return monomials;
}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
Torus::Torus(const real major_radius, const real minor_radius)
  : Torus(major_radius, minor_radius, TorusBase::get_default_center())
{}

// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
Torus::Torus(const real major_radius, const real minor_radius, const Point<3> &center)
  : Torus(major_radius, minor_radius, center, TorusBase::get_default_axis())
{}


// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
Torus::Torus(const real major_radius, const real minor_radius, const Point<3> &center, const Point<3> &axis)
  // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
  : TorusBase(major_radius, minor_radius, center, axis)
{}

implement_impl_func_3D(Torus);

template<typename T> Point<3, T> Torus::compute_P_x_0(const Point<3, T> &point) const
{
  const Point<3, T> x_0 = point - this->center_;
  return x_0 - (dot(x_0, this->axis_) * this->axis_);
}

template<typename T> T Torus::eval_(const Point<3, T> &point) const
{
  const real R_2 = this->major_radius_ * this->major_radius_;
  const real r_2 = this->minor_radius_ * this->minor_radius_;

  const Point<3, T> diff = point - this->center_;
  const T dist2 = sqrnorm(diff);
  const T proj_diff = dot(diff, this->axis_);

  // NOLINTNEXTLINE (readability-math-missing-parentheses)
  return dist2 * dist2 - numbers::two * (R_2 + r_2) * dist2 + numbers::four * R_2 * proj_diff * proj_diff
         + ((R_2 - r_2) * (R_2 - r_2));
}

template<typename T> Vector<T, 3> Torus::grad_(const Point<3, T> &point) const
{
  const Point<3, T> diff = point - this->center_;
  const T dist2 = sqrnorm(diff);
  const T proj_diff = dot(diff, this->axis_);
  const real R_2 = this->major_radius_ * this->major_radius_;
  const real r_2 = this->minor_radius_ * this->minor_radius_;

  // NOLINTNEXTLINE (readability-math-missing-parentheses)
  return numbers::four * (dist2 - R_2 - r_2) * diff + numbers::eight * R_2 * proj_diff * this->axis_;
}

template<typename T> auto Torus::hessian_(const Point<3, T> &point) const -> Hessian<T>
{
  const Point<3, T> diff = point - this->center_;
  const T dist2 = sqrnorm(diff);
  const real R_2 = this->major_radius_ * this->major_radius_;
  const real r_2 = this->minor_radius_ * this->minor_radius_;

  const T aux_0 = numbers::four * (dist2 - R_2 - r_2);
  const T aux_1 = numbers::eight * R_2;

  Hessian<T> hess;
  for (int i = 0, ij = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j, ++ij) {
      const real Idt = (i == j) ? numbers::one : numbers::zero;
      hess(ij) = aux_0 * Idt + numbers::eight * diff(i) * diff(j) + aux_1 * this->axis_(i) * this->axis_(j);
    }
  }
  return hess;
}

ConstantBase::ConstantBase(const real value) : value_(value) {}

real ConstantBase::value() const
{
  return this->value_;
}


template<int dim> ConstantBzr<dim>::ConstantBzr() : ConstantBzr(ConstantBase::default_value){};

template<int dim>
ConstantBzr<dim>::ConstantBzr(const real value)
  : ConstantBase(value), qugar::impl::BezierTP<dim, 1>(TensorSizeTP<dim>(1), std::vector<real>({ value }))
{}

template<int dim> Constant<dim>::Constant() : Constant(ConstantBase::default_value){};

template<int dim> Constant<dim>::Constant(const real value) : ConstantBase(value) {}

implement_impl_func(Constant);

template<int dim> template<typename T> T Constant<dim>::eval_(const Point<dim, T> & /*point*/) const
{
  return T{ this->value_ };
}

template<int dim> template<typename T> Vector<T, dim> Constant<dim>::grad_(const Point<dim, T> & /*point*/) const
{
  return Vector<T, dim>(numbers::zero);
}

template<int dim> template<typename T> auto Constant<dim>::hessian_(const Point<dim, T> & /*point*/) const -> Hessian<T>
{
  return Hessian<T>(numbers::zero);
}


template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
PlaneBase<dim>::PlaneBase(const Point<dim> &origin, const Point<dim> &normal) : origin_(origin), normal_(normal)
{}

template<int dim> Point<dim> PlaneBase<dim>::get_default_origin()
{
  return Point<dim>(numbers::zero);
}

template<int dim> Point<dim> PlaneBase<dim>::get_default_normal()
{
  return set_component<real, dim>(numbers::zero, 0, numbers::one);
}

template<int dim> const Point<dim> &PlaneBase<dim>::origin() const
{
  return this->origin_;
}

template<int dim> const Point<dim> &PlaneBase<dim>::normal() const
{
  return this->normal_;
}

template<int dim>
PlaneBzr<dim>::PlaneBzr() : PlaneBzr(PlaneBase<dim>::get_default_origin(), PlaneBase<dim>::get_default_normal())
{}

template<int dim>
PlaneBzr<dim>::PlaneBzr(const Point<dim> &origin, const Point<dim> &normal)
  : PlaneBase<dim>(origin, normal), BezierTP<dim, 1>(*create_monomials(origin, normal))
{}

template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
std::shared_ptr<MonomialsTP<dim, 1>> PlaneBzr<dim>::create_monomials(const Point<dim> &origin, const Point<dim> &normal)
{
  const Point<dim> unit_normal = normal / norm(normal);

  const auto monomials = std::make_shared<MonomialsTP<dim, 1>>(TensorSizeTP<dim>(2));
  for (int dir = 0; dir < dim; ++dir) {
    TensorIndexTP<dim> tid{ 0 };
    monomials->get_coef(tid) -= origin(dir) * unit_normal(dir);

    tid(dir) = 1;
    monomials->get_coef(tid) = unit_normal(dir);
  }

  return monomials;
}


template<int dim>
Plane<dim>::Plane() : Plane(PlaneBase<dim>::get_default_origin(), PlaneBase<dim>::get_default_normal())
{}

template<int dim> Plane<dim>::Plane(const Point<dim> &origin, const Point<dim> &normal) : PlaneBase<dim>(origin, normal)
{}

implement_impl_func(Plane);

template<int dim> template<typename T> T Plane<dim>::eval_(const Point<dim, T> &point) const
{
  return dot(point - this->origin_, this->normal_);
}

template<int dim> template<typename T> Vector<T, dim> Plane<dim>::grad_(const Point<dim, T> & /*point*/) const
{
  return this->normal_;
}

template<int dim> template<typename T> auto Plane<dim>::hessian_(const Point<dim, T> & /*point*/) const -> Hessian<T>
{
  return Hessian<T>{ numbers::zero };
}


// Instantiations

template class SphereBase<2>;
template class SphereBase<3>;

template class Sphere<2>;
template class Sphere<3>;

template class SphereBzr<2>;
template class SphereBzr<3>;

template class EllipsoidBase<2>;
template class EllipsoidBase<3>;

template class EllipsoidBzr<2>;
template class EllipsoidBzr<3>;

template class Ellipsoid<2>;
template class Ellipsoid<3>;

template class PlaneBase<2>;
template class PlaneBase<3>;

template class PlaneBzr<2>;
template class PlaneBzr<3>;

template class Plane<2>;
template class Plane<3>;

template class Constant<2>;
template class Constant<3>;

template class ConstantBzr<2>;
template class ConstantBzr<3>;


}// namespace qugar::impl::funcs