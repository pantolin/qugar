// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file qugar.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of nanobind wrappers for qugar.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/version.hpp>

#include <nanobind/nanobind.h>

#include <string>

namespace nb = nanobind;

namespace qugar::wrappers {
void common(nb::module_ &m);
void cut_quad(nb::module_ &m);
void quad(nb::module_ &m);
void unf_domain(nb::module_ &m);
void reparam(nb::module_ &m);
void impl_functions(nb::module_ &m);
}// namespace qugar::wrappers

NB_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "QUGaR Python interface";
  m.attr("__version__") = std::string(qugar::QUGAR_VERSION_STRING).c_str();
  m.attr("__git_commit_hash__") = std::string(qugar::QUGAR_VERSION_GIT).c_str();

#ifdef NDEBUG
  nanobind::set_leak_warnings(false);
#endif

  // Create common tools
  qugar::wrappers::common(m);

  // Quadrature data structures
  qugar::wrappers::quad(m);

  // Cut quadrature data structures
  qugar::wrappers::cut_quad(m);

  // Unfitted domains.
  qugar::wrappers::unf_domain(m);

  // Reparameterizations for general unfitted domains.
  qugar::wrappers::reparam(m);

  // Implicit functions library.
  qugar::wrappers::impl_functions(m);
}
