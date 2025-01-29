// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_PRIMITIVE_FUNCS_LIB_HPP
#define QUGAR_IMPL_PRIMITIVE_FUNCS_LIB_HPP

//! @file primitive_funcs_lib.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of a primitive implicit functions ready to be consumed by Algoim.
//! @version 0.0.1
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bezier_tp.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_funcs_lib_macros.hpp>
#include <qugar/monomials_tp.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/ref_system.hpp>
#include <qugar/types.hpp>

#include <memory>

//! Namespace for defining implicit function examples.
//! These function are ready to be consumed by Algoim.
namespace qugar::impl::funcs {

//! @brief Dimension independent spherical function.
//! @note This is the base class for Bezier and general implementations.
//!
//! @tparam dim Parametric dimension.
template<int dim> class SphereBase
{
public:
  //! @brief Constructs a Sphere object with a specified @p center and @p radius.
  //!
  //! @param radius The radius of the sphere.
  //! @param center The center of the sphere.
  SphereBase(real radius, const Point<dim> &center);

  //! @brief Gets the radius of the sphere.
  //! @return real The sphere's radius.
  [[nodiscard]] real radius() const;

  //! @brief Gets the center of the sphere.
  //! @return real The sphere's center.
  [[nodiscard]] const Point<dim> &center() const;

protected:
  //! @brief Gets the default center of the sphere.
  //! It is set to the origin of the Cartesian coordinate system.
  [[nodiscard]] static Point<dim> get_default_center();

  //! Radius of the sphere.
  real radius_{ 0 };

  //! Center of the sphere.
  Point<dim> center_;
};


//! @brief Dimension independent spherical function.
//! The function is defined by its center and radius.
//! The function presents a negative sign around the center,
//! and positive far away. At a radius distance from the
//! center, the function vanishes.
//!
//! @note Non-Bezier version.
//!
//! @tparam dim Parametric dimension.
template<int dim>
class Sphere
  : public SphereBase<dim>
  , public ImplicitFunc<dim>
{
public:
  //! @brief Constructs a Sphere with the given @p radius and centered at the origin.
  //!
  //! @param radius The radius of the sphere.
  explicit Sphere(real radius);

  //! @brief Constructs a Sphere object with a specified @p center and @p radius.
  //!
  //! @param radius The radius of the sphere.
  //! @param center The center of the sphere.
  Sphere(real radius, const Point<dim> &center);

  declare_impl_func_virtual_interface;
};


//! @brief Dimension independent spherical function.
//! The function is defined by its center and radius.
//! The function presents a negative sign around the center,
//! and positive far away. At a radius distance from the
//! center, the function vanishes.
//!
//! @note Bezier version.
//!
//! @tparam dim Parametric dimension.
template<int dim>
class SphereBzr
  : public SphereBase<dim>
  , public BezierTP<dim, 1>
{
public:
  //! @brief Constructs a Sphere with the given @p radius and centered at the origin.
  //!
  //! @param radius The radius of the sphere.
  explicit SphereBzr(real radius);

  //! @brief Constructs a Sphere object with a specified @p center and @p radius.
  //!
  //! @param radius The radius of the sphere.
  //! @param center The center of the sphere.
  SphereBzr(real radius, const Point<dim> &center);

private:
  //! @brief Creates a polynomial representation based on monomials for the given center and radius.
  //!
  //! @param radius The radius of the sphere.
  //! @param center The center of the sphere.
  //! @return Polynomial defining the function expressed in a monomials base.
  [[nodiscard]] static std::shared_ptr<MonomialsTP<dim, 1>> create_monomials(real radius, const Point<dim> &center);
};

//! @brief Infinite cylinder base class.
//!
//! The cylinder is defined by its radius, origin, and axis.
//!
//! The function presents a negative sign around the cylinder's axis,
//! and positive far away. At a radius distance from the
//! cylinder's axis, the function vanishes.
class CylinderBase
{
public:
  //! @brief Constructor.
  //!
  //! @param radius Cylinder's radius.
  //! @param origin Cylinder's origin.
  //! @param axis Cylinder's axis.
  CylinderBase(real radius, const Point<3> &origin, const Point<3> &axis);


  //! @brief Gets the radius of the cylinder.
  //! @return real The cylinder's radius.
  [[nodiscard]] real radius() const;

  //! @brief Gets the origin of the cylinder.
  //! @return real The cylinder's origin.
  [[nodiscard]] const Point<3> &origin() const;

  //! @brief Gets the axis of the cylinder.
  //! @return real The cylinder's axis.
  [[nodiscard]] const Point<3> &axis() const;

protected:
  //! @brief Gets the default origin of the cylinder.
  //! It is set to the origin of the Cartesian coordinate system.
  [[nodiscard]] static Point<3> get_default_origin();

  //! @brief Gets the default axis of the cylinder.
  //! It is set to the z-axis of the Cartesian coordinate system.
  [[nodiscard]] static Point<3> get_default_axis();

  //! Radius of the cylinder.
  real radius_;

  //! Origin of the cylinder.
  Point<3> origin_;

  //! Axis of the axis.
  Point<3> axis_;
};

//! @brief Infinite cylinder.
//!
//! The cylinder is defined by its radius, origin, and axis.
//!
//! The function presents a negative sign around the cylinder's axis,
//! and positive far away. At a radius distance from the
//! cylinder's axis, the function vanishes.
//!
//! @note Bezier version.
class CylinderBzr
  : public CylinderBase
  , public BezierTP<3, 1>
{
public:
  //! @brief Constructor.
  //!
  //! Cylinder along z-axis.
  //! @param radius Cylinder's radius.
  explicit CylinderBzr(real radius);

  //! @brief Constructor.
  //!
  //! Cylinder with vertical axis at the given origin.
  //! @param radius Cylinder's radius.
  //! @param origin Cylinder's origin.
  CylinderBzr(real radius, const Point<3> &origin);

  //! @brief Constructor.
  //!
  //! @param radius Cylinder's radius.
  //! @param origin Cylinder's origin.
  //! @param axis Cylinder's axis.
  CylinderBzr(real radius, const Point<3> &origin, const Point<3> &axis);

  //! @brief Creates a polynomial representation based on monomials for the given center, radius, and axis.
  //!
  //! @param radius The radius of the cylinder.
  //! @param origin The center of the cylinder.
  //! @param axis The radius of the cylinder.
  [[nodiscard]] static std::shared_ptr<MonomialsTP<3, 1>>
    create_monomials(real radius, const Point<3> &origin, const Point<3> &axis);
};


//! @brief Infinite cylinder.
//!
//! The cylinder is defined by its radius, origin, and axis.
//!
//! The function presents a negative sign around the cylinder's axis,
//! and positive far away. At a radius distance from the
//! cylinder's axis, the function vanishes.
//!
//! @note Non-Bezier version.
class Cylinder
  : public CylinderBase
  , public ImplicitFunc<3>
{
public:
  static const int dim = 3;

  //! @brief Constructor.
  //!
  //! Cylinder along z-axis.
  //! @param radius Cylinder's radius.
  explicit Cylinder(real radius);

  //! @brief Constructor.
  //!
  //! Cylinder with vertical axis at the given origin.
  //! @param radius Cylinder's radius.
  //! @param origin Cylinder's origin.
  Cylinder(real radius, const Point<3> &origin);

  //! @brief Constructor.
  //!
  //! @param radius Cylinder's radius.
  //! @param origin Cylinder's origin.
  //! @param axis Cylinder's axis.
  Cylinder(real radius, const Point<3> &origin, const Point<3> &axis);

  declare_impl_func_virtual_interface;
};

//! @brief Dimension independent ellipsoidal function (base cass).
//! The function is defined by the ellipsoid's semi-axes and centered at the origin.
//! The function presents a negative sign around the origin,
//! and positive far away.
//!
//! @note This class is implemented as an orthotropic scaling of a sphere.
//!
//! @tparam dim Parametric dimension.
template<int dim> class EllipsoidBase
{
public:
  //! @brief Constructs an Ellipsoid object with specified semi-axes and reference system.
  //!
  //! @param semi_axes A Point object representing the semi-axes of the ellipsoid.
  //! @param system A RefSystem object representing the reference system in which the ellipsoid is defined.
  EllipsoidBase(const Point<dim> &semi_axes, const RefSystem<dim> &system);

  //! @brief Gets the semi-axes of the ellipsoid.
  //! @return real The ellipsoid's semi-axes.
  [[nodiscard]] const Point<dim> &semi_axes() const;

  //! @brief Gets the reference system of the ellipsoid.
  //! @return real The ellipsoid's reference system.
  [[nodiscard]] const RefSystem<dim> &ref_system() const;

protected:
  //! Ellipsoid's semi-axes.
  Point<dim> semi_axes_;

  //! Ellipsoid's reference system.
  RefSystem<dim> system_;

  //! @brief Gets the default reference system of the ellipsoid.
  //! @return Default (Cartesian) coordinate system.
  [[nodiscard]] static RefSystem<dim> get_default_system();
};

//! @brief Dimension independent ellipsoidal function.
//! The function is defined by the ellipsoid's semi-axes and centered at the origin.
//! The function presents a negative sign around the origin,
//! and positive far away.
//!
//! @note This class is implemented as an orthotropic scaling of a sphere.
//!
//! @tparam dim Parametric dimension.
//! @note Bezier version.
template<int dim>
class EllipsoidBzr
  : public EllipsoidBase<dim>
  , public BezierTP<dim, 1>
{
public:
  //! @brief Constructor.
  //!
  //! @param semi_axes Semi-axes length along the Cartesian axes.
  explicit EllipsoidBzr(const Point<dim> &semi_axes);

  //! @brief Constructs an Ellipsoid object with specified semi-axes and reference system.
  //!
  //! @param semi_axes A Point object representing the semi-axes of the ellipsoid.
  //! @param system A RefSystem object representing the reference system in which the ellipsoid is defined.
  EllipsoidBzr(const Point<dim> &semi_axes, const RefSystem<dim> &system);

private:
  //! @brief Creates a polynomial representation based on monomials for the given semi-axes and reference system.
  //!
  //! @param semi_axes A Point object representing the semi-axes of the ellipsoid.
  //! @param system A RefSystem object representing the reference system in which the ellipsoid is defined.
  [[nodiscard]] static std::shared_ptr<MonomialsTP<dim, 1>> create_monomials(const Point<dim> &semi_axes,
    const RefSystem<dim> &system);
};

//! @brief Dimension independent ellipsoidal function.
//! The function is defined by the ellipsoid's semi-axes and centered at the origin.
//! The function presents a negative sign around the origin,
//! and positive far away.
//!
//! @note This class is implemented as an orthotropic scaling of a sphere.
//!
//! @tparam dim Parametric dimension.
//! @note Non-Bezier version.
template<int dim>
class Ellipsoid
  : public EllipsoidBase<dim>
  , public ImplicitFunc<dim>
{
public:
  //! @brief Constructor.
  //!
  //! @param semi_axes Semi-axes length along the Cartesian axes.
  explicit Ellipsoid(const Point<dim> &semi_axes);

  //! @brief Constructs an Ellipsoid object with specified semi-axes and reference system.
  //!
  //! @param semi_axes A Point object representing the semi-axes of the ellipsoid.
  //! @param system A RefSystem object representing the reference system in which the ellipsoid is defined.
  Ellipsoid(const Point<dim> &semi_axes, const RefSystem<dim> &system);

  declare_impl_func_virtual_interface;
};

//! @brief 2D annulus base class.
//! The function is defined by the annulus center and major and inner radii.
//! The function presents a negative sign inside the annulus (between both boundaries),
//! and positive outside.
class AnnulusBase
{
public:
  //! @brief Constructor.
  //!
  //! @param inner_radius Inner radius of the annulus.
  //! @param outer_radius Outer radius of the annulus.
  //! @param center Annulus' center.
  AnnulusBase(real inner_radius, real outer_radius, const Point<2> &center);

  //! @brief Gets the center of the annulus.
  //! @return real The annulus's center.
  [[nodiscard]] const Point<2> &center() const;

  //! @brief Gets the inner radius of the annulus.
  //! @return real The annulus's inner radius.
  [[nodiscard]] real inner_radius() const;

  //! @brief Gets the outer radius of the annulus.
  //! @return real The annulus's outer radius.
  [[nodiscard]] real outer_radius() const;

protected:
  //! Inner radius of the annulus.
  real inner_radius_;

  //! Outer radius of the annulus.
  real outer_radius_;

  //! Center of the annulus.
  Point<2> center_;

  //! @brief Gets the default center of the cylinder.
  //! It is set to the center of the Cartesian coordinate system.
  [[nodiscard]] static Point<2> get_default_center();
};

//! @brief 2D annulus function.
//! The function is defined by the annulus center and outer and inner radii.
//! The function presents a negative sign inside the annulus (between both boundaries),
//! and positive outside.
//! @note Bezier version.
class AnnulusBzr
  : public AnnulusBase
  , public BezierTP<2, 1>
{
public:
  //! @brief Constructor.
  //!
  //! Center is set to (0,0).
  //!
  //! @param inner_radius Inner radius of the annulus.
  //! @param outer_radius Outer radius of the annulus.
  AnnulusBzr(real inner_radius, real outer_radius);

  //! @brief Constructor.
  //!
  //! @param inner_radius Inner radius of the annulus.
  //! @param outer_radius Outer radius of the annulus.
  //! @param center Annulus' center.
  AnnulusBzr(real inner_radius, real outer_radius, const Point<2> &center);

private:
  //! @brief Creates a polynomial representation based on monomials for the given radii and center.
  //!
  //! @param inner_radius Inner radius of the annulus.
  //! @param outer_radius Outer radius of the annulus.
  //! @param center Annulus' center.
  [[nodiscard]] static std::shared_ptr<MonomialsTP<2, 1>>
    create_monomials(real inner_radius, real outer_radius, const Point<2> &center);
};


//! @brief 2D annulus function.
//! The function is defined by the annulus center and outer and inner radii.
//! The function presents a negative sign inside the annulus (between both boundaries),
//! and positive outside.
//! @note non-Bezier version.
class Annulus
  : public AnnulusBase
  , public ImplicitFunc<2>
{
public:
  //! @brief Constructor.
  //!
  //! Center is set to (0,0).
  //!
  //! @param inner_radius Inner radius of the annulus.
  //! @param outer_radius Outer radius of the annulus.
  Annulus(real inner_radius, real outer_radius);

  //! @brief Constructor.
  //!
  //! @param inner_radius Inner radius of the annulus.
  //! @param outer_radius Outer radius of the annulus.
  //! @param center Annulus' center.
  Annulus(real inner_radius, real outer_radius, const Point<2> &center);

  declare_impl_func_virtual_interface_2D;
};

//! @brief 3D torus function base class.
//! The function is defined by the torus center, axis, and major and
//! inner radii.
//! The function presents a negative sign inside the torus, and positive
//! outside.
//!
class TorusBase
{
public:
  //! @brief Constructor.
  //!
  //! @param major_radius Radius of the major circle.
  //! @param minor_radius Radius of the inner circle.
  //! @param center Torus' center.
  //! @param axis Torus' axis (perpendicular to the major circle plane).
  TorusBase(real major_radius, real minor_radius, const Point<3> &center, const Point<3> &axis);

  //! @brief Gets the major radius of the torus.
  //! @return real The torus' major radius.
  [[nodiscard]] real major_radius() const;

  //! @brief Gets the minor radius of the torus.
  //! @return real The torus' minor radius.
  [[nodiscard]] real minor_radius() const;

  //! @brief Gets the center of the plane.
  //! @return real The plane's center.
  [[nodiscard]] const Point<3> &center() const;

  //! @brief Gets the axis of the plane.
  //! @return real The plane's axis.
  [[nodiscard]] const Point<3> &axis() const;

protected:
  //! Major radius of the torus.
  real major_radius_;

  //! Minor radius of the torus.
  real minor_radius_;

  //! Center of the torus.
  Point<3> center_;

  //! Axis of the torus.
  Point<3> axis_;

  //! @brief Gets the default center of the torus.
  //! It is set to the center of the Cartesian coordinate system.
  [[nodiscard]] static Point<3> get_default_center();

  //! @brief Gets the default axis of the torus.
  //! It is set to the z-axis of the Cartesian coordinate system.
  [[nodiscard]] static Point<3> get_default_axis();
};


//! @brief 3D torus function.
//! The function is defined by the torus center, axis, and major and
//! inner radii.
//! The function presents a negative sign inside the torus, and positive
//! outside.
//!
//! @note Bezier version.
class TorusBzr
  : public TorusBase
  , public BezierTP<3, 1>
{
public:
  //! @brief Constructor.
  //!
  //! Center is set to (0,0,0) and axis to (0,0,1).
  //!
  //! @param major_radius Radius of the major circle.
  //! @param minor_radius Radius of the inner circle.
  TorusBzr(real major_radius, real minor_radius);

  //! @brief Constructor.
  //!
  //! The axis is set to (0,0,1).
  //!
  //! @param major_radius Radius of the major circle.
  //! @param minor_radius Radius of the inner circle.
  //! @param center Torus' center.
  TorusBzr(real major_radius, real minor_radius, const Point<3> &center);


  //! @brief Constructor.
  //!
  //! @param major_radius Radius of the major circle.
  //! @param minor_radius Radius of the inner circle.
  //! @param center Torus' center.
  //! @param axis Torus' axis (perpendicular to the major circle plane).
  TorusBzr(real major_radius, real minor_radius, const Point<3> &center, const Point<3> &axis);

private:
  //! @brief Creates a polynomial representation based on monomials for the given radii, center, and axis.
  //!
  //! @param major_radius Radius of the major circle.
  //! @param minor_radius Radius of the inner circle.
  //! @param center Torus' center.
  //! @param axis Torus' axis (perpendicular to the major circle plane).
  [[nodiscard]] static std::shared_ptr<MonomialsTP<3, 1>>
    create_monomials(real major_radius, real minor_radius, const Point<3> &center, const Point<3> &axis);
};

//! @brief 3D torus function.
//! The function is defined by the torus center, axis, and major and
//! inner radii.
//! The function presents a negative sign inside the torus, and positive
//! outside.
//!
//! @note Non-Bezier version.
class Torus
  : public TorusBase
  , public ImplicitFunc<3>
{
public:
  //! @brief Constructor.
  //!
  //! Center is set to (0,0,0) and axis to (0,0,1).
  //!
  //! @param major_radius Radius of the major circle.
  //! @param minor_radius Radius of the inner circle.
  Torus(real major_radius, real minor_radius);

  //! @brief Constructor.
  //!
  //! The axis is set to (0,0,1).
  //!
  //! @param major_radius Radius of the major circle.
  //! @param minor_radius Radius of the inner circle.
  //! @param center Torus' center.
  Torus(real major_radius, real minor_radius, const Point<3> &center);


  //! @brief Constructor.
  //!
  //! @param major_radius Radius of the major circle.
  //! @param minor_radius Radius of the inner circle.
  //! @param center Torus' center.
  //! @param axis Torus' axis (perpendicular to the major circle plane).
  Torus(real major_radius, real minor_radius, const Point<3> &center, const Point<3> &axis);

  declare_impl_func_virtual_interface_3D;

  /**
   * @brief Computes the normal component of the a vector respect to the origin.
   *
   * Given a @p point, computes the relative vector respect to the origin, and
   * then computes the normal component of the vector respect to the torus' axis.
   *
   * @tparam T The type of the coordinates of the point.
   * @param point The input 3D point for which the P_x_0 value is to be computed.
   * @return A 3D point representing the computed P_x_0 value.
   */
  template<typename T> Point<3, T> compute_P_x_0(const Point<3, T> &point) const;
};

//! @brief Dimension independent constant function.
//! @tparam dim Parametric dimension.
class ConstantBase
{
public:
  static constexpr real default_value = numbers::half;

  //! @brief Constructor.
  //!
  //! @param value Constant value.
  explicit ConstantBase(real value);

  //! @brief Gets the constant value.
  //! @return real The constant value.
  [[nodiscard]] real value() const;

protected:
  //! Constant value.
  real value_{ 0.0 };
};

//! @brief Dimension independent constant function.
//! @tparam dim Parametric dimension.
//! @note Bezier version.
template<int dim>
class ConstantBzr
  : public ConstantBase
  , public BezierTP<dim, 1>
{
public:
  //! @brief Default constructor. Sets constant value to 0.5.
  ConstantBzr();

  //! @brief Constructor.
  //!
  //! @param value Constant value.
  explicit ConstantBzr(real value);
};

//! @brief Dimension independent constant function.
//! @tparam dim Parametric dimension.
//! @note Non-Bezier version.
template<int dim>
class Constant
  : public ConstantBase
  , public ImplicitFunc<dim>
{
public:
  //! @brief Default constructor. Sets constant value to 0.5.
  Constant();

  //! @brief Constructor.
  //!
  //! @param value Constant value.
  explicit Constant(real value);

  declare_impl_func_virtual_interface;
};

//! @brief Plane base class.
//!
//! This is a linear function whose value is zero at a line, and grows
//! linearly (positively or negatively) as you move away from the line.
//! @tparam dim Parametric dimension.
template<int dim> class PlaneBase
{
public:
  //! @brief Constructs a new plane function. The line is defined by an @p origin and
  //! a @p normal vector.
  PlaneBase(const Point<dim> &origin, const Point<dim> &normal);

  //! @brief Gets the origin of the plane.
  //! @return real The plane's origin.
  [[nodiscard]] const Point<dim> &origin() const;

  //! @brief Gets the normal of the plane.
  //! @return real The plane's normal.
  [[nodiscard]] const Point<dim> &normal() const;

protected:
  //! Origin of the (levelset) line.
  Point<dim> origin_;

  //! Normal to the (levelset) line.
  Point<dim> normal_;

  //! @brief Gets the default origin of the plane.
  //! It is set to the origin of the Cartesian coordinate system.
  //! @return The default origin.
  [[nodiscard]] static Point<dim> get_default_origin();

  //! @brief Gets the default normal vector of the plane.
  //! It is set to the x-axis.
  //! @return The default normal.
  [[nodiscard]] static Point<dim> get_default_normal();
};

//! @brief Plane function.
//!
//! This is a linear function whose value is zero at a line, and grows
//! linearly (positively or negatively) as you move away from the line.
//! @tparam dim Parametric dimension.
//! @note Bezier version.
template<int dim>
class PlaneBzr
  : public PlaneBase<dim>
  , public BezierTP<dim, 1>
{
public:
  //! @brief Constructs a new plane function. The line (levelset) is the line x=0.
  PlaneBzr();

  //! @brief Constructs a new plane function. The line is defined by an @p origin and
  //! a @p normal vector.
  PlaneBzr(const Point<dim> &origin, const Point<dim> &normal);

private:
  //! @brief Creates a polynomial representation based on monomials for the given origin a normal.
  //!
  //! @param origin Plane's origin.
  //! @param normal Plane's normal.
  [[nodiscard]] static std::shared_ptr<MonomialsTP<dim, 1>> create_monomials(const Point<dim> &origin,
    const Point<dim> &normal);
};

//! @brief Plane function.
//!
//! This is a linear function whose value is zero at a line, and grows
//! linearly (positively or negatively) as you move away from the line.
//! @tparam dim Parametric dimension.
//! @note Non-Bezier version.
template<int dim>
class Plane
  : public PlaneBase<dim>
  , public ImplicitFunc<dim>
{
public:
  //! @brief Constructs a new plane function. The line (levelset) is the line x=0.
  Plane();

  //! @brief Constructs a new plane function. The line is defined by an @p origin and
  //! a @p normal vector.
  Plane(const Point<dim> &origin, const Point<dim> &normal);

  declare_impl_func_virtual_interface;
};


}// namespace qugar::impl::funcs


#endif// QUGAR_IMPL_PRIMITIVE_FUNCS_LIB_HPP