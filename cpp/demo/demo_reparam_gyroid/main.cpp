// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#include <qugar/bbox.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/impl_reparam.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/tpms_lib.hpp>
#include <qugar/types.hpp>


#include <array>
#include <memory>

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(/* int argc, const char **argv */)
{

  using namespace qugar;
  using namespace qugar::impl;

  static const int dim = 3;

  const auto gyroid = std::make_shared<tpms::Schoen<3>>(qugar::Vector<real, dim>(2., 2., 2.));

  const int n_elems_dir = 5;
  const int order{ 4 };

  const BoundBox<dim> domain_01;

  const auto grid = std::make_shared<CartGridTP<dim>>(
    domain_01, std::array<std::size_t, 3>({ { n_elems_dir, n_elems_dir, n_elems_dir } }));

  const qugar::impl::UnfittedImplDomain<dim> unf_domain(gyroid, grid);

  const auto reparam = create_reparameterization<dim, false>(unf_domain, order);
  reparam->write_VTK_file("gyroid");

  const auto reparam_levelset = create_reparameterization<dim, true>(unf_domain, order);
  reparam_levelset->write_VTK_file("gyroid_levelset");

  return 0;
}
