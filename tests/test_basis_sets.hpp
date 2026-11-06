/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// Gaussian basis sets for tests, built from a table rather than parsed.
//
// There are no basis-set parsers in the library by design, so tests that
// need a Gaussian basis construct one through the caller-array API. This
// header is the single place that happens: before it existed the same
// STO-3G shells were written out in three test files, with oxygen in one
// and carbon in another, which is the duplication the project's DRY policy
// is there to prevent -- and the kind that lets two copies drift apart
// without anything failing.
//
// STO-3G (Hehre, Stewart and Pople, J. Chem. Phys. 51 (1969) 2657), which
// is the "small basis" the screening tests want: minimal, so the shell
// count per atom is small and the cutoff radii modest, and contracted, so
// shells hold several primitives and the general-contraction path is
// exercised. Carbon and oxygen each contribute a p shell, which is what
// makes a block hold more than one basis function -- the case that the
// single-function Slater paths cannot represent.

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_shell.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>

namespace Nukexc {
namespace test {

/// One contracted shell of a tabulated basis.
struct TabulatedShell {
  int l;
  std::vector<double> zeta;
  std::vector<double> coeff;
};

/// STO-3G shells for one element, or an error for an element not tabulated.
inline std::vector<TabulatedShell> sto3g_shells(unsigned Z) {
  // Shared sp contraction coefficients.
  static const std::vector<double> c1s{0.1543289673, 0.5353281423,
                                       0.4446345422};
  static const std::vector<double> c2s{-0.09996722919, 0.3995128261,
                                       0.7001154689};
  static const std::vector<double> c2p{0.1559162750, 0.6076837186,
                                       0.3919573931};
  switch (Z) {
  case 1:
    return {{0, {3.425250914, 0.6239137298, 0.1688554040}, c1s}};
  case 6:
    return {{0, {71.61683735, 13.04509632, 3.530512160}, c1s},
            {0, {2.941249355, 0.6834830964, 0.2222899159}, c2s},
            {1, {2.941249355, 0.6834830964, 0.2222899159}, c2p}};
  case 7:
    return {{0, {99.10616896, 18.05231239, 4.885660238}, c1s},
            {0, {3.780455879, 0.8784966449, 0.2857143744}, c2s},
            {1, {3.780455879, 0.8784966449, 0.2857143744}, c2p}};
  case 8:
    return {{0, {130.7093214, 23.80886605, 6.443608313}, c1s},
            {0, {5.033151319, 1.169596125, 0.3803889600}, c2s},
            {1, {5.033151319, 1.169596125, 0.3803889600}, c2p}};
  default:
    throw std::invalid_argument(
        "sto3g_shells: no tabulated basis for Z = " + std::to_string(Z) +
        "; add it here rather than inlining shells at the call site");
  }
}

/// STO-3G for a molecule, as a GTOBasisSet.
inline GTOBasisSet<double> sto3g_basis(const Molecule &mol,
                                       ShellType type = ShellType::Spherical) {
  auto cen = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.atom_centers);
  auto Z = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.Z);

  std::vector<GTOShell<double>> sh;
  for (unsigned i = 0; i < mol.natoms; ++i) {
    const std::array<double, 3> O{cen(i)[0], cen(i)[1], cen(i)[2]};
    for (const auto &s : sto3g_shells(Z(i)))
      sh.emplace_back(s.l, type, O, s.zeta, s.coeff, 1);
  }
  return GTOBasisSet<double>(std::move(sh));
}

} // namespace test
} // namespace Nukexc
