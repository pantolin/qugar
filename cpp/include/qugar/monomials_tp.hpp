// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_MONOMIALS_TP_HPP
#define QUGAR_IMPL_MONOMIALS_TP_HPP

//! @file monomials_tp.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of tensor-product monomials class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/polynomial_tp.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>

#include <algoim/xarray.hpp>

#include <memory>
#include <vector>

namespace qugar::impl {

template<int dim, int range> class BezierTP;

//! @brief dim-dimensional tensor-product monomials function.
//!
//! @tparam dim Dimension of the parametric domain.
//! @tparam range Dimension of the image.
template<int dim, int range = 1> class MonomialsTP : public PolynomialTP<dim, range>
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
  explicit MonomialsTP(const TensorSizeTP<dim> &order);

  //! @brief Constructor.
  //!
  //! @param coefs Coefficients of the polynomial.
  //! The ordering is such that dimension 0 is inner-most, i.e.,
  //! iterates the fastest, while dimension dim-1 is outer-most and iterates the slowest.
  //! @param order Order of the polynomial along each parametric direction.
  MonomialsTP(const TensorSizeTP<dim> &order, const std::vector<CoefsType> &coefs);

  //! @brief Copy constructor.
  //!
  //! @param monomials Monomials to copy.
  MonomialsTP(const MonomialsTP<dim, range> &monomials);

  //! @brief Creates a derivative of the current MonomialsTP object in the specified direction.
  //!
  //! This function generates a new MonomialsTP object that represents the derivative
  //! of the current object in the direction specified by the parameter `dir`.
  //!
  //! @param dir The direction in which to take the derivative. This should be an integer
  //!            representing the axis or dimension along which the derivative is computed.
  //!
  //! @return A std::shared_ptr to a new MonomialsTP object that represents the derivative
  //!         of the current object in the specified direction.
  //!
  //! @note If the monomials has already order 1 (degree 0) in the specified direction,
  //! just a clone of the current object is returned.
  [[nodiscard]] std::shared_ptr<MonomialsTP<dim, range>> create_derivative(const int dir) const;

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
  //! @brief Extracts a facet from the monomials tensor product.
  //!
  //! This function extracts a facet of the monomials tensor product specified by the local facet ID.
  //!
  //! @tparam dim The dimension of the monomials tensor product.
  //! @tparam range The range type of the monomials tensor product.
  //! @param local_facet_id The ID of the local facet to extract.
  //! @return A shared pointer to the extracted facet.
  template<int dim_aux = dim>
    requires(dim == dim_aux && dim > 1)
  [[nodiscard]] std::shared_ptr<MonomialsTP<dim - 1, range>> extract_facet(const int local_facet_id) const;

  //! @brief Evaluates a monomials using Horner's method.
  //!
  //! For the method details check:
  //!  https://en.wikipedia.org/wiki/Horner%27s_method
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
  static Value<T> horner(const Point<dim, T> &point,
    typename std::vector<CoefsType>::const_iterator &coefs,
    const Vector<int, dim> &order);

  //! @brief Evaluates the gradient of the monomials using the Horner's method.
  //!
  //! For the method details check:
  //!  https://en.wikipedia.org/wiki/Horner%27s_method
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
  static Vector<Value<T>, dim + 1> horner_der(const Point<dim, T> &point,
    typename std::vector<CoefsType>::const_iterator &coefs,
    const Vector<int, dim> &order);

  //! @brief Transforms the coefficients of monomials to Bezier form.
  //!
  //! This function takes the coefficients of monomials and transforms them into
  //! the corresponding Bezier coefficients.
  //!
  //! @param monomials Monomials whose coefficients are transformed.
  //! @param bzr_coefs A vector containing the Bezier coefficients to be computed.
  static void transform_coefs_to_Bezier(const MonomialsTP<dim, range> &monomials, std::vector<CoefsType> &bzr_coefs);
};


}// namespace qugar::impl

#endif// QUGAR_IMPL_MONOMIALS_TP_HPP