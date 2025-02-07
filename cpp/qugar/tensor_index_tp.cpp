// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file tensor_index_tp.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tensor-product index related classes.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/tensor_index_tp.hpp>

#include <qugar/vector.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <string_view>

namespace qugar {

template<int dim> TensorSizeTP<dim>::TensorSizeTP() : TensorSizeTP(Vector<int, dim>{}) {}

template<int dim> TensorSizeTP<dim>::TensorSizeTP(const int size) : Vector<int, dim>(size)
{
  assert(size >= 0);
}

template<int dim> TensorSizeTP<dim>::TensorSizeTP(const Vector<int, dim> &sizes) : Vector<int, dim>(sizes)
{
  assert(min(sizes) >= 0);
}

template<int dim> TensorSizeTP<dim>::TensorSizeTP(const TensorIndexTP<dim> &sizes) : TensorSizeTP(sizes.as_Vector()) {}


template<int dim> const Vector<int, dim> &TensorSizeTP<dim>::as_Vector() const
{
  return dynamic_cast<const Vector<int, dim> &>(*this);
}

template<int dim> Vector<int, dim> &TensorSizeTP<dim>::as_Vector()
{
  return dynamic_cast<Vector<int, dim> &>(*this);
}

template<int dim> int TensorSizeTP<dim>::size() const
{
  return prod(this->as_Vector());
}


template<int dim> bool TensorSizeTP<dim>::operator==(const TensorSizeTP<dim> &rhs) const
{
  return all(this->as_Vector() == rhs.as_Vector());
}

template<int dim> TensorSizeTP<dim> TensorSizeTP<dim>::operator+(const TensorSizeTP<dim> &rhs) const
{
  TensorSizeTP<dim> sum(*this);
  sum += rhs.as_Vector();
  return sum;
}


template<int dim> TensorIndexTP<dim>::TensorIndexTP(const Vector<int, dim> &indices) : Vector<int, dim>(indices)
{
  if constexpr (dim > 0) {
    assert(min(indices) >= 0);
  }
}

// Read comment in .hpp file.
// template<int dim> const Vector<int, dim> &TensorIndexTP<dim>::as_Vector() const
// {
//   return dynamic_cast<const Vector<int, dim> &>(*this);
// }

template<int dim>
TensorIndexTP<dim>::TensorIndexTP(const TensorSizeTP<dim> &indices) : TensorIndexTP(indices.as_Vector())
{}


template<int dim> Vector<int, dim> &TensorIndexTP<dim>::as_Vector()
{
  return dynamic_cast<Vector<int, dim> &>(*this);
}

template<int dim> bool TensorIndexTP<dim>::operator==(const TensorIndexTP<dim> &rhs) const
{
  if constexpr (dim > 0) {
    return all(this->as_Vector() == rhs.as_Vector());
  } else {
    return true;
  }
}

template<int dim> bool TensorIndexTP<dim>::operator!=(const TensorIndexTP<dim> &rhs) const
{
  return !(this->operator==(rhs));
}

template<int dim> bool TensorIndexTP<dim>::operator<(const TensorIndexTP<dim> &rhs) const
{
  if constexpr (dim == 0) {
    return false;
  } else {
    for (int dir = dim - 1; dir > 0; --dir) {
      if (rhs(dir) < this->operator()(dir)) {
        return false;
      }
    }
    return this->operator()(0) < rhs(0);
  }
};

template<int dim> template<typename S> int TensorIndexTP<dim>::flat(const S &size) const
{
  int accum_size{ 1 };

  int flat_index{ 0 };
  for (int dir = 0; dir < dim; ++dir) {
    const auto index_dir = this->operator()(dir);
    const auto size_dir = size(dir);
    assert(0 <= index_dir && index_dir < size_dir);

    flat_index += index_dir * accum_size;
    accum_size *= size_dir;
  }
  return flat_index;
}

template<int dim>
template<int aux_dim>
  requires(aux_dim == dim && dim > 1)
TensorIndexTP<dim - 1> TensorIndexTP<dim>::remove_component(const int comp) const
{
  TensorIndexTP<dim - 1> new_index;
  for (int dir = 0, local_dir = 0; dir < dim; ++dir) {
    if (dir != comp) {
      new_index[local_dir++] = this->operator[](dir);
    }
  }
  return new_index;
}

template<int dim> std::size_t TensorIndexTP<dim>::hash() const
{
  if constexpr (dim == 0) {
    return 0;
  } else {
    using View = std::basic_string_view<char32_t>;
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-reinterpret-cast)
    const auto *data = reinterpret_cast<const char32_t *>(this->as_Vector().data());
    return std::hash<View>{}(View(data, dim));
  }
}

template<int dim>
// NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
TensorIndexRangeTP<dim>::TensorIndexRangeTP(const TensorIndexTP<dim> &lower_bound,
  const TensorIndexTP<dim> &upper_bound)
  : lower_bound_(lower_bound), upper_bound_(upper_bound)
{
#ifndef NDEBUG
  for (int dir = 0; dir < dim; ++dir) {
    assert((this->lower_bound_(dir) <= this->upper_bound_(dir)));
  }
#endif// NDEBUG
}

template<int dim>
TensorIndexRangeTP<dim>::TensorIndexRangeTP(const TensorIndexTP<dim> &upper_bound)
  : TensorIndexRangeTP(TensorIndexTP<dim>(), upper_bound)
{}

template<int dim>
TensorIndexRangeTP<dim>::TensorIndexRangeTP(const TensorSizeTP<dim> &upper_bound)
  : TensorIndexRangeTP(TensorIndexTP<dim>(), TensorIndexTP<dim>(upper_bound))
{}

template<int dim>
TensorIndexRangeTP<dim>::TensorIndexRangeTP(int upper_bound) : TensorIndexRangeTP(TensorIndexTP<dim>(upper_bound))
{}


template<int dim> TensorSizeTP<dim> TensorIndexRangeTP<dim>::get_sizes() const
{
  return TensorSizeTP<dim>(this->upper_bound_.as_Vector() - this->lower_bound_.as_Vector());
}

template<int dim> int TensorIndexRangeTP<dim>::size() const
{
  return this->get_sizes().size();
}

template<int dim> auto TensorIndexRangeTP<dim>::cbegin() const -> Iterator
{
  return Iterator(this->lower_bound_, this->lower_bound_, this->upper_bound_);
}

template<int dim> auto TensorIndexRangeTP<dim>::begin() const -> Iterator
{
  return this->cbegin();
}

template<int dim> auto TensorIndexRangeTP<dim>::cend() const -> Iterator
{
  return Iterator(this->upper_bound_, this->lower_bound_, this->upper_bound_);
}

template<int dim> auto TensorIndexRangeTP<dim>::end() const -> Iterator
{
  return this->cend();
}

template<int dim> const TensorIndexTP<dim> &TensorIndexRangeTP<dim>::get_lower_bound() const
{
  return this->lower_bound_;
}

template<int dim> const TensorIndexTP<dim> &TensorIndexRangeTP<dim>::get_upper_bound() const
{
  return this->upper_bound_;
}

template<int dim> bool TensorIndexRangeTP<dim>::is_in_range(const TensorIndexTP<dim> &index) const
{
  return all(this->lower_bound_.as_Vector() <= index.as_Vector())
         && all(index.as_Vector() < this->upper_bound_.as_Vector());
}

template<int dim> bool TensorIndexRangeTP<dim>::is_in_range(const int index) const
{
  if (index >= TensorSizeTP<dim>(this->upper_bound_).size()) {
    return false;
  } else {
    return this->is_in_range(TensorIndexTP<dim>(index, this->upper_bound_));
  }
}


template<int dim> std::array<TensorIndexRangeTP<dim>, 2> TensorIndexRangeTP<dim>::split() const
{
  assert(this->size() > 1);

  const auto split_dir = argmax(this->get_sizes().as_Vector());
  const int split_id = (this->lower_bound_(split_dir) + this->upper_bound_(split_dir)) / 2;

  TensorIndexTP<dim> new_upper_bound = this->upper_bound_;
  new_upper_bound(split_dir) = split_id;

  TensorIndexTP<dim> new_lower_bound = this->lower_bound_;
  new_lower_bound(split_dir) = split_id;

  return { TensorIndexRangeTP<dim>{ this->lower_bound_, new_upper_bound },
    TensorIndexRangeTP<dim>{ new_lower_bound, this->upper_bound_ } };
}

template<int dim>
// NOLINTBEGIN (bugprone-easily-swappable-parameters)
TensorIndexRangeTP<dim>::Iterator::Iterator(const TensorIndexTP<dim> &index,
  const TensorIndexTP<dim> &lower_bound,
  const TensorIndexTP<dim> &upper_bound)
  : index_(index), lower_bound_(lower_bound), upper_bound_(upper_bound)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
#ifndef NDEBUG
  for (int dir = 0; dir < dim; ++dir) {
    assert((this->lower_bound_(dir) <= this->upper_bound_(dir)));
    assert((this->lower_bound_(dir) <= this->index_(dir) && this->index_(dir) <= this->upper_bound_(dir)));
  }
#endif// NDEBUG
}

template<int dim> const TensorIndexTP<dim> &TensorIndexRangeTP<dim>::Iterator::operator*() const
{
  return index_;
}

template<int dim> const TensorIndexTP<dim> *TensorIndexRangeTP<dim>::Iterator::operator->() const
{
  return &index_;
}

template<int dim> int TensorIndexRangeTP<dim>::Iterator::flat() const
{
  return this->index_.flat(this->upper_bound_);
}

template<int dim> auto TensorIndexRangeTP<dim>::Iterator::operator++() -> Iterator &
{
  for (int dir = 0; dir < dim; ++dir) {
    int &ind_dir = this->index_(dir);
    if (++ind_dir < this->upper_bound_(dir)) {
      return *this;
    }
    ind_dir = this->lower_bound_(dir);
  }

  // Setting the index to an invalid state.
  this->index_ = this->upper_bound_;

  return *this;
}

template<int dim> auto TensorIndexRangeTP<dim>::Iterator::operator++(int) -> Iterator
{
  Iterator tmp = *this;
  ++(*this);
  return tmp;
}

template<int dim>
bool TensorIndexRangeTP<dim>::Iterator::operator==(const typename TensorIndexRangeTP<dim>::Iterator &rhs) const
{
  assert(this->lower_bound_ == rhs.lower_bound_);
  assert(this->upper_bound_ == rhs.upper_bound_);
  return this->index_ == rhs.index_;
};

template<int dim>
bool TensorIndexRangeTP<dim>::Iterator::operator!=(const typename TensorIndexRangeTP<dim>::Iterator &rhs) const
{
  return !(*this == rhs);
};

// Instantiations
template class TensorSizeTP<1>;
template class TensorSizeTP<2>;
template class TensorSizeTP<3>;

template class TensorIndexTP<0>;
template class TensorIndexTP<1>;
template class TensorIndexTP<2>;
template class TensorIndexTP<3>;

template int TensorIndexTP<1>::flat<TensorIndexTP<1>>(const TensorIndexTP<1> &) const;
template int TensorIndexTP<1>::flat<TensorSizeTP<1>>(const TensorSizeTP<1> &) const;
template int TensorIndexTP<2>::flat<TensorIndexTP<2>>(const TensorIndexTP<2> &) const;
template int TensorIndexTP<2>::flat<TensorSizeTP<2>>(const TensorSizeTP<2> &) const;
template int TensorIndexTP<3>::flat<TensorIndexTP<3>>(const TensorIndexTP<3> &) const;
template int TensorIndexTP<3>::flat<TensorSizeTP<3>>(const TensorSizeTP<3> &) const;

template class TensorIndexRangeTP<1>;
template class TensorIndexRangeTP<2>;
template class TensorIndexRangeTP<3>;

}// namespace qugar
