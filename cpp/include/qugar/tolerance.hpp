// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_LIBRARY_TOLERANCE_HPP

#define QUGAR_LIBRARY_TOLERANCE_HPP

//! @file tolerance.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Definition of tolerance related functionalities.
//! @version 0.0.2
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <qugar/point.hpp>
#include <qugar/types.hpp>

#include <cstdlib>
#include <vector>

namespace qugar {

//! @brief Class for tolerance related computations.
class Tolerance
{
public:
  //! @brief Constructor.
  //!
  //! @param value Tolerance value (it must be greater than zero).
  explicit Tolerance(const real value);

  //! @brief Constructor. Creates a new class instance
  //! using the largest of the two given tolerances.
  //!
  //! @param tol_0 First tolerance.
  //! @param tol_1 Second tolerance.
  explicit Tolerance(const real tol_0, const real tol_1);

  //! @brief Constructor. Creates a new class instance
  //! using the largest of the two given tolerances.
  //!
  //! @param tol_0 First tolerance.
  //! @param tol_1 Second tolerance.
  explicit Tolerance(const Tolerance &tol_0, const Tolerance &tol_1);

  //! @brief Default constructor.
  //! It initalizes the class instance with a near epsilon tolerance.
  explicit Tolerance();

  //! @brief Resets the tolerance value as the maximum between
  //! the current @ref tolerance_ and @p tol.
  //! @param tol Tolerance value to be reset.
  void update(const real tol);

  //! @brief Resets the tolerance value as the maximum between
  //! the current @ref tolerance_ and @p tol.
  //! @param tol Tolerance value to be reset.
  void update(const Tolerance &tol);

  //! @brief Returns the real tolerance value.
  //! @return Tolerance real value.
  [[nodiscard]] real value() const;


  //! @brief Check if @p val is zero up to tolerance.
  //!
  //! It checks \f$|val| \leq tolerance\f$
  //!
  //! @param val Value to be checked.
  //! @return Whether @p val is equal to zero up to tolerance.
  [[nodiscard]] bool is_zero(const real val) const;

  //! @brief Check if @p val is negative.
  //!
  //! It checks \f$val \leq -tolerance\f$
  //!
  //! @param val Value to be checked.
  //! @return Whether @p val is negative up to tolerance.
  [[nodiscard]] bool is_negative(const real val) const;

  //! @brief Check if @p val is positive.
  //!
  //! It checks \f$tolerance < val\f$
  //!
  //! @param val Value to be checked.
  //! @return Whether @p val is positive up to tolerance.
  [[nodiscard]] bool is_positive(const real val) const;

  //! @brief Compares if two values are equal up to tolerance.
  //!
  //! It checks \f$|lhs -rhs| \leq tolerance\f$
  //!
  //! @param lhs Left-hand-side value in the comparison.
  //! @param rhs Right-hand-side value in the comparison.
  //! @return Whether both arguments are equal up to tolerance.
  [[nodiscard]] bool equal(const real lhs, const real rhs) const;

  //! @brief Compares if two values are equal up to tolerance relative
  //! to the larger of the two arguments.
  //!
  //! It checks \f$|lhs -rhs| \leq tolerance \max (|lhs|, |rhs|) \f$
  //!
  //! @param lhs Left-hand-side value in the comparison.
  //! @param rhs Right-hand-side value in the comparison.
  //! @return Whether both arguments are relatively equal up to tolerance.
  [[nodiscard]] bool equal_rel(const real lhs, const real rhs) const;

  //! @brief Compares if two values are equal up to the different
  //! absolute and relative tolerances.
  //!
  //! It checks \f$|lhs -rhs| \leq (tolerance + rel_tolerance \max (|lhs|, |rhs|)) \f$
  //! where \f$tolerance\f$ is the current tolerance and \f$rel_tolerance\f$ is
  //! the one provided as input.
  //!
  //! @param lhs Left-hand-side value in the comparison.
  //! @param rhs Right-hand-side value in the comparison.
  //! @param rel_tolerance Relative tolerance.
  //! @return Whether both arguments are relatively equal up to tolerance.
  [[nodiscard]] bool equal_rel(const real lhs, const real rhs, const Tolerance &rel_tolerance) const;

  //! @brief Compares if a value is greater than other up to tolerance.
  //!
  //! It checks \f$(lhs - rhs) > tolerance\f$
  //!
  //! @param lhs Left-hand-side value in the comparison.
  //! @param rhs Right-hand-side value in the comparison.
  //! @return Whether @p lhs is greater than @p rhs up to tolerance.
  [[nodiscard]] bool greater_than(const real lhs, const real rhs) const;

  //! @brief Compares if a value is greater or equal than other up to tolerance.
  //!
  //! It checks \f$(lhs - rhs) > tolerance\f$ or \f$|rhs - lhs| < tolerance\f$.
  //!
  //! @param lhs Left-hand-side value in the comparison.
  //! @param rhs Right-hand-side value in the comparison.
  //! @return Whether @p lhs is greater or equal than @p rhs up to tolerance.
  [[nodiscard]] bool greater_equal_than(const real lhs, const real rhs) const;

  //! @brief Compares if a value is greater than other up to tolerance relative
  //! to the larger of the two argumetns
  //!
  //! It checks \f$(lhs - rhs) > tolerance \max (|lhs|, |rhs|)\f$
  //!
  //! @param lhs Left-hand-side value in the comparison.
  //! @param rhs Right-hand-side value in the comparison.
  //! @return Whether @p lhs is relatively greater than @p rhs up to tolerance.
  [[nodiscard]] bool greater_than_rel(const real lhs, const real rhs) const;

  //! @brief Compares if a value is smaller than other up to tolerance.
  //!
  //! It checks \f$(rhs - lhs) > tolerance\f$
  //!
  //! @param lhs Left-hand-side value in the comparison.
  //! @param rhs Right-hand-side value in the comparison.
  //! @return Whether @p lhs is smaller than @p rhs up to tolerance.
  [[nodiscard]] bool smaller_than(const real lhs, const real rhs) const;

  //! @brief Compares if a value is smaller or equal than other up to tolerance.
  //!
  //! It checks \f$(rhs - lhs) > tolerance\f$ or \f$|rhs - lhs| < tolerance\f$.
  //!
  //! @param lhs Left-hand-side value in the comparison.
  //! @param rhs Right-hand-side value in the comparison.
  //! @return Whether @p lhs is smaller or equal than @p rhs up to tolerance.
  [[nodiscard]] bool smaller_equal_than(const real lhs, const real rhs) const;

  //! @brief Compares if a value is smaller than other up to tolerance relative
  //! to the larger of the two argumetns
  //!
  //! It checks \f$(rhs - lhs) > tolerance \max (|lhs|, |rhs|)\f$
  //!
  //! @param lhs Left-hand-side value in the comparison.
  //! @param rhs Right-hand-side value in the comparison.
  //! @return Whether @p lhs is relatively smaller than @p rhs up to tolerance.
  [[nodiscard]] bool smaller_than_rel(const real lhs, const real rhs) const;

  //! @brief Makes the given @p values unique up to tolerance.
  //! @p values will be also sorted.
  //! @param values Vector of value of make unique.
  void unique(std::vector<real> &values) const;

  //! @brief Checks if two points are coincident.
  //!
  //! This function compares two points of the same dimension and type to determine if they are coincident.
  //! Two points are considered coincident if all their corresponding coordinates are equal up to tolerance.
  //!
  //! @tparam dim The dimension of the points.
  //! @tparam T The type of the coordinates of the points.
  //! @param pt_0 The first point to compare.
  //! @param pt_1 The second point to compare.
  //! @return true if the points are coincident, false otherwise.
  template<int dim, typename T>
  [[nodiscard]] bool coincident(const Point<dim, T> &pt_0, const Point<dim, T> &pt_1) const
  {
    for (int dir = 0; dir < dim; ++dir) {
      if (!this->equal(pt_0(dir), pt_1(dir))) {
        return false;
      }
    }
    return true;
  }


private:
  //! Tolerance value.
  real tolerance_;
};

}// namespace qugar

#endif// QUGAR_LIBRARY_TOLERANCE_HPP