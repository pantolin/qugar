// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_BEZIER_TP_HPP
#define QUGAR_IMPL_BEZIER_TP_HPP

//! @file bezier_tp.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of tensor-product Bezier class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/monomials_tp.hpp>
#include <qugar/polynomial_tp.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>

#include <algoim/xarray.hpp>

#include <memory>
#include <vector>

namespace qugar::impl {

template<int dim, int range> class MonomialsTP;

//! @brief dim-dimensional tensor-product Bezier polynomial function.
//!
//! @tparam dim Dimension of the parametric domain.
//! @tparam range Dimension of the image.
template<int dim, int range = 1> class BezierTP : public PolynomialTP<dim, range>
{
public:
  //! Parent type.
  using Parent = PolynomialTP<dim, range>;

  //! Coefs type.
  using CoefsType = typename Parent::CoefsType;

  //! Value type.
  template<typename T> using Value = typename Parent::template Value<T>;

  //! Gradient type.
  template<typename T> using Gradient = typename Parent::template Gradient<T>;

  //! Hessian type.
  template<typename T> using Hessian = typename Parent::template Hessian<T>;

  //! @brief Algoim's interval alias.
  template<int N> using Interval = ::algoim::Interval<N>;

  //! @brief Constructor.
  //!
  //! The coefficients vector is allocated by the constructor.
  //!
  //! @param order Order of the polynomial along each parametric direction.
  explicit BezierTP(const TensorSizeTP<dim> &order);

  //! @brief Constructs a constant value Bezier tensor product.
  //!
  //! @param order Order of the polynomial along each parametric direction.
  //! @param value The value to set for all the coefficients.
  BezierTP(const TensorSizeTP<dim> &order, const CoefsType &value);

  //! @brief Constructor.
  //!
  //! @param coefs Coefficients of the polynomial.
  //! The ordering is such that dimension 0 is inner-most, i.e.,
  //! iterates the fastest, while dimension dim-1 is outer-most and iterates the slowest.
  //! @param order Order of the polynomial along each parametric direction.
  BezierTP(const TensorSizeTP<dim> &order, const std::vector<CoefsType> &coefs);

  //! @brief Copy constructor.
  //!
  //! @param bezier Bezier to copy.
  BezierTP(const BezierTP<dim, range> &bezier);

  //! @brief Constructor.
  //!
  //! Constructs from monomials.
  //!
  //! @param monomials Monomials from which the Bezier is created.
  explicit BezierTP(const MonomialsTP<dim, range> &monomials);

  //! @brief Inherits the function call operator from the base class.
  //!
  //! Allows instances of this class to be called as functions, utilizing the
  //! (non-virtual) operator() implementation from `DomainFunc<dim, range>` for `Point`
  //! instances.
  using DomainFunc<dim, range>::operator();

  //! @brief Inherits the function grad method from the base class.
  //!
  //! Allows instances of this class to be called as functions, utilizing the
  //! (non-virtual) grad implementation from `DomainFunc<dim, range>` for `Point`
  //! instances.
  using DomainFunc<dim, range>::grad;


private:
  //! dim-dimensional array view of the coefficients.
  ::algoim::xarray<CoefsType, dim> coefs_xarray_;

public:
  //! @brief Evaluator operator.
  //!
  //! @param point Point at which the function is evaluated.
  //! @return Function value at @p point.
  [[nodiscard]] virtual Value<real> operator()(const Point<dim> &point) const final;

  //! @brief Evaluator operator.
  //!
  //! @param point Point at which the function is evaluated.
  //! @return Function value at @p point.
  [[nodiscard]] virtual Value<Interval<dim>> operator()(const Point<dim, Interval<dim>> &point) const final;

  //! @brief Gradient evaluator operator.
  //!
  //! @param point Point at which the function's gradient is evaluated.
  //! @return Function gradient at @p point.
  [[nodiscard]] virtual Gradient<real> grad(const Point<dim> &point) const final;

  //! @brief Gradient evaluator operator.
  //!
  //! @param point Point at which the function's gradient is evaluated.
  //! @return Function gradient at @p point.
  [[nodiscard]] virtual Gradient<Interval<dim>> grad(const Point<dim, Interval<dim>> &point) const final;

  //! @brief Hessian evaluator operator.
  //!
  //! @param point Point at which the function's hessian is evaluated.
  //! @return Function hessian at @p point.
  [[nodiscard]] Hessian<real> virtual hessian(const Point<dim> &point) const final;

private:
  //! @brief Evaluator operator.
  //!
  //! @tparam T Input and output's type.
  //! @param point Point at which the function is evaluated.
  //! @return Function value at @p point.
  template<typename T> [[nodiscard]] Value<T> eval_(const Point<dim, T> &point) const;

  //! @brief Gradient evaluator operator.
  //!
  //! @tparam T Input and output's type.
  //! @param point Point at which the function's gradient is evaluated.
  //! @return Function gradient at @p point.
  template<typename T> [[nodiscard]] Gradient<T> grad_(const Point<dim, T> &point) const;

  //! @brief Hessian evaluator operator.
  //!
  //! @tparam T Input and output's type.
  //! @param point Point at which the function's hessian is evaluated.
  //! @return Function hessian at @p point.
  template<typename T> [[nodiscard]] Hessian<T> hessian_(const Point<dim, T> &point) const;

public:
  //! @brief Evaluates a Bezier polynomial using the Casteljau's algorithm.
  //!
  //! For the algorithm details check:
  //!   https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
  //!
  //! The evaluation along the dimensions is performed by recursively calling this function.
  //!
  //! @tparam T Type of the input and output variables.
  //! @param point Evaluation point.
  //! @param coefs Iterator to the coefficients of the polynomial.
  //! Their ordering is the same as the coefficients members of the class.
  //! @param order Order of the polynomial along the dim dimensions.
  //! @return Value of the polynomial at @p point.
  template<typename T>
  static Value<T> casteljau(const Point<dim, T> &point,
    typename std::vector<CoefsType>::const_iterator &coefs,
    const Vector<int, dim> &order);

  //! @brief Evaluates the gradient Bezier polynomial using the Casteljau's algorithm.
  //!
  //! For the algorithm details check:
  //!   https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
  //!
  //! The evaluation along the dimensions is performed by recursively calling this function.
  //!
  //! @tparam T Type of the input and output variables.
  //! @param point Evaluation point.
  //! @param coefs Iterator to the coefficients of the polynomial.
  //! Their ordering is the same as the coefficients members of the class.
  //! @param order Order of the polynomial along the dim dimensions.
  //! @return The first component of the return vector corresponds to the value of the polynomial itself,
  //! while the gradient is stored in the following dim components.
  template<typename T>
  static Vector<Value<T>, dim + 1> casteljau_der(const Point<dim, T> &point,
    typename std::vector<CoefsType>::const_iterator &coefs,
    const Vector<int, dim> &order);

  //! @brief Gets a constant reference to the stored xarray view of the polynomial coefficients.
  //! @return Constant view of the polynomial coefficients.
  [[nodiscard]] const ::algoim::xarray<CoefsType, dim> &get_xarray() const;

  template<int dim_aux = dim>
    requires(dim == dim_aux && dim > 1)

  //! @brief Extracts a facet from the Bezier tensor product.
  //!
  //! This function extracts a facet of the Bezier tensor product specified by the local facet ID.
  //!
  //! @tparam dim The dimension of the Bezier tensor product.
  //! @tparam range The range type of the Bezier tensor product.
  //! @param local_facet_id The ID of the local facet to extract.
  //! @return A shared pointer to the extracted facet.
  [[nodiscard]] std::shared_ptr<BezierTP<dim - 1, range>> extract_facet(const int local_facet_id) const;

  //! @brief Raises the order of the Bezier tensor product.
  //!
  //! This function creates a new Bezier tensor product with an increased order
  //! as specified by the @p new_order parameter. The new Bezier will represent the
  //! same polynomial function as the original, but with a higher order.
  //!
  //! @param new_order The new order for each dimension of the tensor product. Along
  //! each dimension, the new order must be greater or equal than the current order.
  //! @return A shared pointer to the new Bezier tensor product with the raised order.
  [[nodiscard]] std::shared_ptr<BezierTP<dim, range>> raise_order(const TensorSizeTP<dim> &new_order) const;

  //! @brief Creates a new Bezier tensor product (TP) object that is the negation of the current object.
  //!
  //! This function returns a shared pointer to a new BezierTP object where each control point
  //! has the same value but with negative sign.
  //!
  //! @return A shared pointer to the negated BezierTP object.
  [[nodiscard]] std::shared_ptr<BezierTP<dim, range>> negate() const;

  //! @brief Recomputes (in-place) the Bezier coefficients for a new domain (that may not
  //! be [0, 1]).
  //!
  //! @param new_domain New domain for which the coefficientes are computed.
  template<int range_aux = range>
    requires(range_aux == range && range == 1)
  void rescale_domain(const BoundBox<dim> &new_domain);

  //! @brief Returns the sign of the function represented by the Bézier tensor-product
  //! using the properties of the control points convex hull.
  //!
  //! @return FuncSign The sign of the function.
  template<int range_aux = range>
    requires(range_aux == range && range == 1)
  [[nodiscard]] FuncSign sign() const;

  //! @brief Product of two Beziers.
  //!
  //! @param rhs Second Bezier to multiply.
  //! @return Product of Beziers.
  [[nodiscard]] std::shared_ptr<BezierTP<dim, range>> operator*(const BezierTP<dim, range> &rhs) const;

  //! @brief Addition of two Beziers.
  //!
  //! @param rhs Second Bezier to sum.
  //! @return Sum of Beziers. The resulting Bezier has the maximum order of the two Beziers
  //! along each dimension.
  [[nodiscard]] std::shared_ptr<BezierTP<dim, range>> operator+(const BezierTP<dim, range> &rhs) const;

  //! @brief Subtraction of two Beziers.
  //!
  //! @param rhs Bezier to subtract.
  //! @return Subtraction of Beziers. The resulting Bezier has the maximum order of the two Beziers
  //! along each dimension.
  [[nodiscard]] std::shared_ptr<BezierTP<dim, range>> operator-(const BezierTP<dim, range> &rhs) const;


  //! @brief Composes the current Bezier with the given Bezier @p rhs.
  //!
  //! @tparam sub_dim Subdimension of the right-hand-size Bezier.
  //! @param rhs Right-hand-side Bezier for the composition.
  //! @return Generated Bezier composition.
  template<int sub_dim>
  [[nodiscard]] std::shared_ptr<BezierTP<sub_dim, range>> compose(const BezierTP<sub_dim, dim> &rhs) const;
};

//! @brief Checks if a given DomainFunc object is of type BezierTP.
//!
//! This function uses dynamic_cast to determine if the provided DomainFunc
//! object can be cast to a BezierTP object. If the cast is successful, the
//! function returns true, indicating that the object is of type BezierTP.
//! Otherwise, it returns false.
//!
//! @tparam dim The dimension of the BezierTP object.
//! @tparam range The range of the BezierTP object.
//! @param func The DomainFunc object to be checked.
//! @return true if the object is of type BezierTP, false otherwise.
template<int dim, int range> static bool is_bezier(const DomainFunc<dim, range> &func)
{
  const auto *bzr = dynamic_cast<const BezierTP<dim, range> *>(&func);
  return bzr != nullptr;
}

}// namespace qugar::impl

#endif// QUGAR_IMPL_BEZIER_TP_HPP