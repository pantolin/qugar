// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_POLYNOMIAL_TP_HPP
#define QUGAR_IMPL_POLYNOMIAL_TP_HPP

//! @file polynomial_tp.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of tensor-product polynomial class.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/point.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/types.hpp>

#include <cassert>
#include <vector>

namespace qugar::impl {

//! @brief Base class for tensor-product polynomial functions.
//!
//! @tparam dim Dimension of the parametric domain.
//! @tparam range Dimension of the image.
template<int dim, int range> class PolynomialTP : public DomainFunc<dim, range>
{
public:
  //! Coefs type.
  using CoefsType = std::conditional_t<range == 1, real, Point<range>>;

  //! @brief Constructor.
  //!
  //! The coefficients vector is allocated by the constructor.
  //!
  //! @param order Order of the polynomial along each parametric direction.
  explicit PolynomialTP(const TensorSizeTP<dim> &order);

  //! @brief Constructs a constant value PolynomialTP object with the specified order.
  //!
  //! @param order The order of the polynomial tensor.
  //! @param value Constant value for all the coefficients.
  PolynomialTP(const TensorSizeTP<dim> &order, const CoefsType &value);

  //! @brief Constructor.
  //!
  //! @param coefs Coefficients of the polynomial.
  //! The ordering is such that dimension 0 is inner-most, i.e.,
  //! iterates the fastest, while dimension dim-1 is outer-most and iterates the slowest.
  //! @param order Order of the polynomial along each parametric direction.
  PolynomialTP(const TensorSizeTP<dim> &order, const std::vector<CoefsType> &coefs);

protected:
  //! Order of the polynomial along each parametric direction.
  TensorSizeTP<dim> order_;

  //! Coefficients of the polynomial.
  //! The ordering is such that dimension param_di-1 is inner-most, i.e.,
  //! iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
  std::vector<CoefsType> coefs_;

public:
  //! @brief Get the number of coeficients.
  //!
  //! @return Number of polynomial coefficients.
  [[nodiscard]] std::size_t get_num_coefs() const;

  //! @brief Gets the polynomial coefficients.
  //!
  //! @return Constant reference to the polynomial coefficients.
  [[nodiscard]] const std::vector<CoefsType> &get_coefs() const;

  //! @brief Gets the polynomial order along the parametric directions.
  //!
  //! @return Constant reference to polynomial order.
  [[nodiscard]] const TensorSizeTP<dim> &get_order() const;

  //! @brief Gets the polynomial order (degree + 1) along the direction @p dir.
  //!
  //! @param dir Direction along which the order (degree + 1) is retrieved.
  //! @return Polynomial order (degree + 1) along @p dir.
  [[nodiscard]] int get_order(int dir) const;

  //! @brief Retrieves the coefficients at the specified index.
  //!
  //! This function returns a constant reference to the coefficients of the polynomial
  //! at the given index. The coefficients are of type CoefsType.
  //!
  //! @param index The index of the coefficients to retrieve.
  //! @return A constant reference to the coefficients at the specified index.
  [[nodiscard]] const CoefsType &get_coef(int index) const;

  //! @brief Retrieves the coefficients at the specified index.
  //!
  //! This function returns a constant reference to the coefficients of the polynomial
  //! at the given index. The coefficients are of type CoefsType.
  //!
  //! @param index The index of the coefficients to retrieve.
  //! @return A non-constant reference to the coefficients at the specified index.
  [[nodiscard]] CoefsType &get_coef(int index);

  //! @brief Retrieves the coefficient associated with the given tensor index.
  //!
  //! This function returns a reference to the coefficient corresponding to the specified
  //! tensor index. The coefficient is part of the polynomial tensor product representation.
  //!
  //! @tparam dim The dimension of the tensor index.
  //! @param index The tensor index for which the coefficient is to be retrieved.
  //! @return A constant reference to the coefficient associated with the given tensor index.
  [[nodiscard]] const CoefsType &get_coef(const TensorIndexTP<dim> &index) const;

  //! @brief Retrieves the coefficient associated with the given tensor index.
  //!
  //! This function returns a reference to the coefficient corresponding to the specified
  //! tensor index. The coefficient is part of the polynomial tensor product representation.
  //!
  //! @tparam dim The dimension of the tensor index.
  //! @param index The tensor index for which the coefficient is to be retrieved.
  //! @return A non-constant reference to the coefficient associated with the given tensor index.
  [[nodiscard]] CoefsType &get_coef(const TensorIndexTP<dim> &index);

  //! @brief Gets the polynomial degree (order - 1) along the direction @p dir.
  //!
  //! @param dir Direction along which the order is retrieved.
  //! @return Polynomial degree (order - 1) along @p dir.
  [[nodiscard]] int get_degree(int dir) const;

  //! @brief Transforms (in place) the coefficients of the polynomial
  //! from @p old_domain to @p new_domain.
  //!
  //! @param old_domain Domain from which polynomial coefficients are transformed from.
  //! @param new_domain Domain to which polynomial coefficients are transformed to.
  void transform_image(const BoundBox<range> &old_domain, const BoundBox<range> &new_domain);

  //! @brief Applies a linear transformation to every coefficient.
  //!
  //! It modifies every coefficient as coef = scale * coef + shift.
  //! @param scale Scale of every coefficient.
  //! @param shift Shift of every coefficient.
  void coefs_linear_transform(const real scale, const CoefsType &shift);
};

//   /**
//    * @brief Iterator for edges of tensor product elements.
//    *
//    * @tparam N Dimension of the parametric domain.
//    */
//   template<int N> struct PolynomialTPEdgeIt
//   {
//     /**
//      * @brief Constructor.
//      *
//      * @param _order Order of the element along all the parametric directions.
//      * @param _edge_id Id of the element's edge (following lexicographical convention).
//      */
//     PolynomialTPEdgeIt(const Vector<int, N> &_order, const int _edge_id)
//       : act_dir(getActiveDir(_edge_id)), order(_order), i(), min(), max(), valid(true)
//     {
//       this->setMinMax(_order, _edge_id);
//     }

//     /**
//      * @brief Increments the iterator position.
//      */
//     PolynomialTPEdgeIt &operator++()
//     {
//       if (++i(act_dir) < max(act_dir))
//         return *this;
//       valid = false;
//       return *this;
//     }

//     /**
//      * @brief Returns a reference to the current tensor index.
//      *
//      * @return Current tensor index.
//      */
//     const Vector<int, N> &operator()() const { return i; }

//     /**
//      * @brief Gets the flat index of the current iterator's position.
//      * @return Flat index of the position.
//      */
//     int getFlatIndex() const
//     {
//       assert(valid && "Iterator is not in a valid state");
//       return util::toFlatIndex(this->order, i);
//     }

//     /**
//      * @brief Checks whether or not the iteration is valid
//      * of not (reached the end).
//      *
//      * @return True if the iterator is in a valid state, false otherwise.
//      */
//     bool operator~() const { return valid; }

//     /**
//      * @brief Resets the iterator to a valid state setting its tensor
//      * index to the beginning of the edge.
//      */
//     void reset()
//     {
//       this->i(act_dir) = 0;
//       this->valid = true;
//     }

//     /// Active parametric direction of the edge.
//     const int act_dir;

//   private:
//     /// Order of the element along all the parametric directions.
//     const Vector<int, N> order;
//     /// Tensor index of the iterator.
//     Vector<int, N> i;
//     /// Minimum and maximum tensor dimensions of the edge.
//     Vector<int, N> min, max;
//     /// Flag indicating if the iterator is in a valid state.
//     bool valid;


//     /**
//      * @brief Sets the minimum and maximum tensor dimensions of the edge.
//      *
//      * @param _order Polynomial order of the element along the parametric directions.
//      * @param _edge_id If of the edge.
//      */
//     void setMinMax(const Vector<int, N> &_order, const int _edge_id)
//     {
//       if constexpr (N == 2) {
//         if (_edge_id < 2) {
//           min(0) = _edge_id == 0 ? 0 : (_order(0) - 1);
//         } else {
//           min(1) = _edge_id == 2 ? 0 : (_order(1) - 1);
//         }
//       } else// if constexpr (N == 3)
//       {
//         if (_edge_id < 4) {
//           min(0) = (_edge_id % 2) == 0 ? 0 : (_order(0) - 1);
//           min(1) = (_edge_id / 2) == 0 ? 0 : (_order(1) - 1);
//         } else if (_edge_id < 8) {
//           min(0) = ((_edge_id - 4) % 2) == 0 ? 0 : (_order(0) - 1);
//           min(2) = ((_edge_id - 4) / 2) == 0 ? 0 : (_order(2) - 1);
//         } else {
//           min(1) = ((_edge_id - 8) % 2) == 0 ? 0 : (_order(1) - 1);
//           min(2) = ((_edge_id - 8) / 2) == 0 ? 0 : (_order(2) - 1);
//         }
//       }
//       this->max = this->min;

//       this->min(act_dir) = 0;
//       this->max(act_dir) = _order(act_dir);

//       this->i = this->min;
//     }

//     /**
//      * @brief Gets the number of edge in an element of dimension N.
//      * @return Number of edge.
//      */
//     static int getNumEdges()
//     {
//       static_assert(N == 2 || N == 3, "Not implemented.");
//       return N == 2 ? 4 : 12;
//     }

//     /**
//      * @brief Gets the active parametric direction of an edge.
//      *
//      * @param _edge_id If of the edge.
//      * @return Active parametric direction index.
//      */
//     static int getActiveDir(const int _edge_id)
//     {
//       static_assert(N == 2 || N == 3, "Not implemented.");
//       assert(0 <= _edge_id && _edge_id < getNumEdges());

//       int act_dir{ -1 };
//       if constexpr (N == 2) {
//         if (_edge_id < 2)
//           return 1;
//         else
//           return 0;
//       } else// if constexpr (N == 3)
//       {
//         if (_edge_id < 4)
//           return 2;
//         else if (_edge_id < 8)
//           return 1;
//         else
//           return 0;
//       }
//     }
//   };


}// namespace qugar::impl

#endif// QUGAR_IMPL_POLYNOMIAL_TP_HPP
