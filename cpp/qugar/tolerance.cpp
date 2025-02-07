// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file tolerance.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tolerance related functionalities.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present


#include <qugar/tolerance.hpp>

#include <qugar/numbers.hpp>
#include <qugar/types.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>


namespace qugar {

Tolerance::Tolerance(const real value) : tolerance_(value)
{
  assert(0 <= this->tolerance_);
}

Tolerance::Tolerance(const real tol_0, const real tol_1) : Tolerance(tol_0)
{
  this->update(Tolerance(tol_1));
}

Tolerance::Tolerance(const Tolerance &tol_0, const Tolerance &tol_1) : Tolerance(tol_0)
{
  this->update(tol_1);
}

Tolerance::Tolerance() : Tolerance(numbers::near_eps) {}

void Tolerance::update(const real tol)
{
  this->update(Tolerance(tol));
}

void Tolerance::update(const Tolerance &tol)
{
  tolerance_ = std::max(tolerance_, tol.tolerance_);
}

real Tolerance::value() const
{
  return tolerance_;
}

bool Tolerance::is_zero(const real val) const
{
  return std::abs(val) <= this->tolerance_;
}

bool Tolerance::is_negative(const real val) const
{
  return val < (-this->tolerance_);
}

bool Tolerance::is_positive(const real val) const
{
  return this->tolerance_ < val;
}

bool Tolerance::equal(const real lhs, const real rhs) const
{
  return std::abs(lhs - rhs) <= this->tolerance_;
}

bool Tolerance::equal_rel(const real lhs, const real rhs) const
{
  return std::abs(lhs - rhs) <= ((std::abs(lhs) < std::abs(rhs) ? std::abs(rhs) : std::abs(lhs)) * this->tolerance_);
}

bool Tolerance::equal_rel(const real lhs, const real rhs, const Tolerance &rel_tolerance) const
{
  const real max_val = (std::abs(lhs) < std::abs(rhs) ? std::abs(rhs) : std::abs(lhs));
  return std::abs(lhs - rhs) <= (this->tolerance_ + rel_tolerance.tolerance_ * max_val);
}

bool Tolerance::greater_than(const real lhs, const real rhs) const
{
  return (lhs - rhs) > this->tolerance_;
}

bool Tolerance::greater_equal_than(const real lhs, const real rhs) const
{
  return this->equal(lhs, rhs) || this->greater_than(lhs, rhs);
}

bool Tolerance::greater_than_rel(const real lhs, const real rhs) const
{
  return (lhs - rhs) > ((std::abs(lhs) < std::abs(rhs) ? std::abs(rhs) : std::abs(lhs)) * this->tolerance_);
}

bool Tolerance::smaller_than(const real lhs, const real rhs) const
{
  // NOLINTNEXTLINE (readability-suspicious-call-argument)
  return this->greater_than(rhs, lhs);
}

bool Tolerance::smaller_equal_than(const real lhs, const real rhs) const
{
  // NOLINTNEXTLINE (readability-suspicious-call-argument)
  return this->greater_equal_than(rhs, lhs);
}

bool Tolerance::smaller_than_rel(const real lhs, const real rhs) const
{
  // NOLINTNEXTLINE (readability-suspicious-call-argument)
  return this->greater_than_rel(rhs, lhs);
}

void Tolerance::unique(std::vector<real> &values) const
{
  std::ranges::sort(values);

  const auto ret =
    std::ranges::unique(values, [this](const auto &lhs, const auto &rhs) { return this->equal(lhs, rhs); });
  values.erase(ret.begin(), ret.end());
}

}// namespace qugar