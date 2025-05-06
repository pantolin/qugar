// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_LIBRARY_TENSOR_INDEX_TP_HPP
#define QUGAR_LIBRARY_TENSOR_INDEX_TP_HPP

//! @file tensor_index_tp.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of tensor-product index and size related classes.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <limits>
#include <qugar/vector.hpp>


#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace qugar {

template<int dim> class TensorIndexTP;

//! @brief Class representing a <tt>dim</tt>-dimensional tensor-product sizes container.
//!
//! @tparam dim Dimension of the container.
template<int dim> class TensorSizeTP : public Vector<int, dim>
{
public:
  //! @name Constructors

  //@{

  //! @brief Default constructor. Initializes indices to zero.
  TensorSizeTP();

  //! @brief Construct a new @ref TensorSizeTP object from size along directions.
  //! @param size Sizs along all the parametric directions.
  explicit TensorSizeTP(const int size);

  //! @brief Construct a new @ref TensorSizeTP object from sizes along directions.
  //! @param sizes Sizes along the parametric directions.
  explicit TensorSizeTP(const Vector<int, dim> &sizes);

  //! @brief Construct a new @ref TensorSizeTP object from sizes along directions.
  //! @param sizes Sizes along the parametric directions.
  explicit TensorSizeTP(const TensorIndexTP<dim> &sizes);

  //@}

  //! @brief Returns the tensor size casted as a Vector.
  //!
  //! @return A constant reference to a Vector.
  //! @note Constant version.
  const Vector<int, dim> &as_Vector() const;

  //! @brief Returns the tensor size casted as a Vector.
  //!
  //! @return A constant reference to a Vector.
  //! @note Non-constant version.
  Vector<int, dim> &as_Vector();

  //! @brief Gets the total number of entries (product of sizes along all directions).
  //! @return Total number of entries.
  [[nodiscard]] std::size_t size() const;

  //! @brief Checks if two tensor sizes are equal by comparing all its components. They are equal if all the
  //! components are equal.
  //! @return Whether they are equal or not.
  [[nodiscard]] bool operator==(const TensorSizeTP<dim> &rhs) const;

  //! @brief Adds two TensorSizeTP objects.
  //!
  //! This operator performs element-wise addition of two TensorSizeTP objects.
  //!
  //! @param rhs The right-hand side TensorSizeTP object to be added.
  //! @return A new TensorSizeTP object that is the result of the addition.
  [[nodiscard]] TensorSizeTP<dim> operator+(const TensorSizeTP<dim> &rhs) const;
};

//! @brief Class representing a <tt>dim</tt>-dimensional tensor-product indices.
//!
//! @tparam dim Dimension of the indices.
template<int dim> class TensorIndexTP : public Vector<int, dim>
{
public:
  //! @name Constructors
  //@{

  //! @brief Default constructor. Initializes indices to zero.
  template<int aux_dim = dim>
    requires(aux_dim == dim && dim > 0)
  TensorIndexTP() : TensorIndexTP(0)
  {}

  template<int aux_dim = dim>
    requires(aux_dim == dim && dim == 0)
  TensorIndexTP()
  {}

  //! @brief Constructs a new @ref TensorIndexTP object from an index.
  //! @param ind All the indices are set to this values.
  //! @note Not available for 0-dimensional case.
  template<int aux_dim = dim>
    requires(aux_dim == dim && dim > 0)
  explicit TensorIndexTP(int ind) : Vector<int, dim>(ind)
  {}

  //! @brief Constructs a new @ref TensorIndexTP object from its indices.
  //! @param indices Indices along the parametric directions.
  explicit TensorIndexTP(const Vector<int, dim> &indices);

  //! @brief Constructs a new @ref TensorIndexTP object from its indices.
  //! @param indices Indices along the parametric directions.
  explicit TensorIndexTP(const TensorSizeTP<dim> &indices);

  //! @brief Constructs a new @ref TensorIndexTP object from a flat index and
  //! an associated tensor size following the lexicographical ordering convention
  //! (i.e., lower indices run faster than higher ones).
  //! @param flat_index Flat index.
  //! @param size Tensor size the @p flat_index is associated to.
  //!
  //! @note Not available for 0-dimensional case.
  template<int aux_dim = dim>
    requires(aux_dim == dim && dim > 0)
  TensorIndexTP(std::int64_t flat_index, const TensorIndexTP<dim> &size)
  {
    auto total_size = static_cast<std::int64_t>(prod(size.as_Vector()));

    // NOLINTNEXTLINE (readability-simplify-boolean-expr)
    assert(0 <= flat_index && flat_index < total_size);

    for (int dir = dim - 1; dir >= 0; --dir) {
      total_size /= size(dir);
      const std::int64_t ind = flat_index / total_size;
      assert(ind < std::numeric_limits<int>::max());
      this->operator()(dir) = static_cast<int>(ind);
      flat_index -= ind * total_size;
    }
  }

  //! @brief Constructs a new @ref TensorIndexTP object from a flat index and
  //! an associated tensor size following the lexicographical ordering convention
  //! (i.e., lower indices run faster than higher ones).
  //! @param flat_index Flat index.
  //! @param size Tensor size the @p flat_index is associated to.
  //!
  //! @note Not available for 0-dimensional case.
  template<int aux_dim = dim>
    requires(aux_dim == dim && dim > 0)
  TensorIndexTP(std::int64_t flat_index, const TensorSizeTP<dim> &size) : TensorIndexTP(flat_index, TensorIndexTP(size))
  {}

  //! @brief Constructs the tensor index from an argument list.
  //! @param indices Input arguments for the indices.
  template<typename... T>
    requires(dim > 1 && sizeof...(T) == dim && std::conjunction_v<std::is_same<int, T>...>)
  explicit TensorIndexTP(T... indices) : TensorIndexTP(Vector<int, dim>{ indices... })
  {}

  //@}

  //! @brief Returns the tensor index casted as a Vector.
  //!
  //! @return A constant reference to a Vector.
  //! @note Constant version.
  const Vector<int, dim> &as_Vector() const
  {
    // For an unknown reason, the linker (llvm 17) doesn't find the symbol if the implementation is moved to the .cpp
    // file
    return dynamic_cast<const Vector<int, dim> &>(*this);
  }

  //! @brief Returns the tensor index casted as a Vector.
  //!
  //! @return A constant reference to a Vector.
  //! @note Non-constant version.
  Vector<int, dim> &as_Vector();

  //! @brief Gets the flat index associated to the current tensor index for a given
  //! tensor @p size.
  //! @param size Tensor size the current index is associated to.
  //! @return Computed flat index.
  //! @tparam S Size type.
  template<typename S> [[nodiscard]] std::int64_t flat(const S &size) const;

  //! @brief Checks if two tensor indices are equal by comparing all its components. They are equal if all the
  //! components are equal.
  //! @return Whether they are equal or not.
  [[nodiscard]] bool operator==(const TensorIndexTP<dim> &rhs) const;

  //! @brief Checks if two tensor indices are different by comparing all
  //! its components. They are different if at least one component is different.
  //! @return Whether they are different or not.
  [[nodiscard]] bool operator!=(const TensorIndexTP<dim> &rhs) const;

  //! @brief Checks if the current tensor index is smaller than @p rhs.
  //! To determine if one is smaller than the other, all the componets
  //! are compared starting from the last one.
  //! @return Whether the current tensor is smaller than @p rhs.
  [[nodiscard]] bool operator<(const TensorIndexTP<dim> &rhs) const;

  //! @brief Creates a new index by removing one of its component.
  //! @param comp Index of the component to be removed.
  //! @return New generated index.
  template<int aux_dim = dim>
    requires(aux_dim == dim && dim > 1)
  TensorIndexTP<dim - 1> remove_component(int comp) const;

  //! @brief Computes a hash value for the tensor index.
  //!
  //! This function generates a hash value that uniquely identifies the tensor index.
  //! The hash value can be used for efficient lookups and comparisons.
  //!
  //! @return A std::size_t value representing the hash of the tensor index.
  [[nodiscard]] std::size_t hash() const;
};

}// namespace qugar

namespace std {
template<int dim> struct hash<qugar::TensorIndexTP<dim>>
{
  size_t operator()(const qugar::TensorIndexTP<dim> &tid) const noexcept { return tid.hash(); }
};
}// namespace std

namespace qugar {

//! @brief Class representing a <tt>dim</tt>-dimensional range defined by lower and upper tensor bounds.
//!
//! @tparam dim Dimension of the indices.
template<int dim> class TensorIndexRangeTP
{
public:
  //! @brief Constructs a new @ref TensorIndexRangeTP object from its lower and upper bounds.
  //! @param lower_bound Lower bounds.
  //! @param upper_bound Upper bounds.
  TensorIndexRangeTP(const TensorIndexTP<dim> &lower_bound, const TensorIndexTP<dim> &upper_bound);

  //! @brief Constructs a new @ref TensorIndexRangeTP object from its upper bound. The lower bound
  //! is assumed to be zero.
  //! @param upper_bound Upper bounds.
  explicit TensorIndexRangeTP(const TensorIndexTP<dim> &upper_bound);

  //! @brief Constructs a new @ref TensorIndexRangeTP object from its upper bound. The lower bound
  //! is assumed to be zero.
  //! @param upper_bound Upper bounds.
  explicit TensorIndexRangeTP(const TensorSizeTP<dim> &upper_bound);

  //! @brief Constructs a new @ref TensorIndexRangeTP object from its upper bound. The lower bound
  //! is assumed to be zero.
  //! @param upper_bound Upper bounds. All components are set to the given value.
  explicit TensorIndexRangeTP(int upper_bound);

  //! @brief Gets the sizes along all the directions.
  //! @return Range sizes along all the directions.
  [[nodiscard]] TensorSizeTP<dim> get_sizes() const;

  //! @brief returns the number of entries in the range.
  //! @return Number of entries in the range.
  [[nodiscard]] std::size_t size() const;

  //! @brief Splits the current range along the direction
  //! with a largest number of indices.
  //!
  //! @return Two generated ranges wrapped in shared pointers.
  [[nodiscard]] std::array<TensorIndexRangeTP<dim>, 2> split() const;


  //! @brief Iterator class for tensor index ranges.
  class Iterator
  {
  public:
    //! @brief Constructs a new iterator object given and @p index and its lower and upper bounds.
    //! @param index Current index for the iterator.
    //! @param lower_bound Lower bounds.
    //! @param upper_bound Upper bounds.
    Iterator(const TensorIndexTP<dim> &index,
      const TensorIndexTP<dim> &lower_bound,
      const TensorIndexTP<dim> &upper_bound);

    //! @brief Dereference operator.
    //! @return Reference to the index pointed by the iterator.
    [[nodiscard]] const TensorIndexTP<dim> &operator*() const;

    //! @brief Indirection operator.
    //! @return Pointer to the index pointed by the iterator.
    [[nodiscard]] const TensorIndexTP<dim> *operator->() const;

    //! @brief Transforms the current tensor index into a flat one
    //! by considering the upper bound as associated size.
    //! @return Computed flat index.
    [[nodiscard]] std::int64_t flat() const;

    //! @brief Pre increments the iterator.
    //! @return Updated iterator.
    Iterator &operator++();

    //! @brief Post increments the iterator.
    //! @return Updated iterator.
    // NOLINTNEXTLINE (cert-dcl21-cpp) // Check deprecated.
    Iterator operator++(int);

    //! @brief Compares if two iterators are equal. They are considered to be equal
    //! if their indices and bounds are equal for all the components.
    //! @param rhs Iterator to compare to the current one.
    //! @return Whether both iterators are equal.
    [[nodiscard]] bool operator==(const Iterator &rhs) const;

    //! @brief Compares if two iterators are different. They are considered to be different
    //! if at least one of the components of their indices or bounds are different.
    //! @param rhs Iterator to compare to the current one.
    //! @return Whether both iterators are different.
    [[nodiscard]] bool operator!=(const Iterator &rhs) const;

  private:
    //! Iterator index.
    TensorIndexTP<dim> index_;
    //! Lower bound.
    TensorIndexTP<dim> lower_bound_;
    //! Upper bound.
    TensorIndexTP<dim> upper_bound_;
  };

  //! @brief Creates a begin iterator.
  //! @return Created iterator.
  //! @note Constant version.
  Iterator cbegin() const;

  //! @brief Creates a begin iterator.
  //! @return Created iterator.
  Iterator begin() const;

  //! @brief Creates an end iterator.
  //! @return Created iterator.
  //! @note Constant version.
  Iterator cend() const;

  //! @brief Creates an end iterator.
  //! @return Created iterator.
  Iterator end() const;

  //! @brief Gets the lower bound.
  //! @return Lower bound.
  [[nodiscard]] const TensorIndexTP<dim> &get_lower_bound() const;

  //! @brief Gets the upper bound.
  //! @return Upper bound.
  [[nodiscard]] const TensorIndexTP<dim> &get_upper_bound() const;

  //! @brief Checks that the given @p index is contained in the range.
  //! @param index Index to be checked.
  //! @return Whether the index is contained in the range or not.
  [[nodiscard]] bool is_in_range(const TensorIndexTP<dim> &index) const;

  //! @brief Checks that the given @p index is contained in the range.
  //! @param index Index to be checked.
  //! @param total_size Total size respect to which the tensor-index of @p index will be computed.
  //! @return Whether the index is contained in the range or not.
  [[nodiscard]] bool is_in_range(std::int64_t index, const TensorSizeTP<dim> &total_size) const;

private:
  //! Lower bounds.
  TensorIndexTP<dim> lower_bound_;
  //! Upper bounds.
  TensorIndexTP<dim> upper_bound_;
};


}// namespace qugar

#endif// QUGAR_LIBRARY_TENSOR_INDEX_TP_HPP