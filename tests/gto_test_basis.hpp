/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// A small hand-built Gaussian basis for tests that need second derivatives
// of the basis functions.
//
// The Slater paths stop at gradients -- STOPrimitiveEvaluator::max_deriv and
// STOContractedEvaluator::max_deriv are both BasisDeriv::Gradient, because
// higher derivatives of Slater functions are not derived in this library
// yet. So anything touching the collocation Laplacian or Hessian (the
// density Laplacian, Laplacian-dependent meta-GGAs) has to be tested with
// Gaussians, whose recursion runs to arbitrary order.
//
// This is not a production basis and is not meant to be. The tests using it
// check structural identities -- an integral against a matrix trace, one
// kernel family reducing to another -- which hold at any basis quality. A
// vendored basis-set file would need a parser, and there are none in the
// library by design.

#include <array>
#include <vector>

#include <Kokkos_Core.hpp>

#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_shell.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>

namespace Nukexc {
namespace test {

/**
 *  @brief A contracted Gaussian basis on a molecule's centres.
 *
 *  Two s shells per atom, a contracted one and a diffuse one so the density
 *  is not a single exponent, plus a p shell on every non-hydrogen. Exponents
 *  are STO-3G-like for the contracted s and chosen by hand elsewhere.
 */
inline Nukexc::GTOBasisSet<double> gto_basis(const Nukexc::Molecule &mol) {
  auto cen = Kokkos::create_mirror_view_and_copy(Nukexc::HostSpace{},
                                                 mol.atom_centers);
  auto Z = Kokkos::create_mirror_view_and_copy(Nukexc::HostSpace{}, mol.Z);

  std::vector<Nukexc::GTOShell<double>> sh;
  const auto t = Nukexc::ShellType::Spherical;
  for (size_t a = 0; a < mol.natoms; ++a) {
    const std::array<double, 3> o{cen(a)[0], cen(a)[1], cen(a)[2]};
    const bool heavy = Z(a) > 2;

    sh.emplace_back(0, t, o,
                    std::vector<double>{heavy ? 130.70932 : 3.42525091,
                                        heavy ? 23.808861 : 0.62391373,
                                        heavy ? 6.4436083 : 0.16885540},
                    std::vector<double>{0.15432897, 0.53532814, 0.44463454},
                    1);
    sh.emplace_back(0, t, o, std::vector<double>{heavy ? 0.38 : 0.12},
                    std::vector<double>{1.0}, 1);
    if (heavy)
      sh.emplace_back(1, t, o, std::vector<double>{5.0331513, 1.1695961,
                                                   0.3803890},
                      std::vector<double>{0.15591627, 0.60768372, 0.39195739},
                      1);
  }
  return Nukexc::GTOBasisSet<double>(std::move(sh));
}

} // namespace test
} // namespace Nukexc
