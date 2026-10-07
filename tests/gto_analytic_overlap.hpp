/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// Closed-form overlap matrix of a Gaussian basis.
//
// An independent reference for anything that builds an overlap by
// quadrature. Extracted from test_gto_overlap.cxx so that the
// core-Hamiltonian tests can use the same closed form rather than a second
// implementation of it: two hand-written references that agree prove less
// than one reference used twice.
//
// Gaussians only, deliberately. The point of a closed form here is to be
// derived from a different route than the quadrature it checks, and the
// Gaussian overlap is the one that has an elementary one.
//
// water_basis also lives here, and is the Gaussian basis the tests share.
// It matters that it has a p shell: p, d and f shells are what make a
// block hold more than one function, and single-function blocks are
// exactly the case that hid the core-Hamiltonian defect this header's
// users now check for.

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_shell.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>

namespace Nukexc {
namespace test {

inline double binom(int n, int k) {
  if (k < 0 || k > n)
    return 0.0;
  double r = 1.0;
  for (int i = 0; i < k; ++i)
    r = r * (n - i) / (i + 1);
  return std::round(r);
}

inline double dfact_odd(int n) { // (n)!! for odd n, with (-1)!! = 1
  double r = 1.0;
  for (int k = n; k > 1; k -= 2)
    r *= k;
  return r;
}

/// f_j(la, lb, a, b) = sum_k C(la,k) C(lb,j-k) a^(la-k) b^(lb-j+k)
inline double overlap_f(int j, int la, int lb, double a, double b) {
  double acc = 0.0;
  for (int k = std::max(0, j - lb); k <= std::min(j, la); ++k)
    acc += binom(la, k) * binom(lb, j - k) * std::pow(a, la - k) *
           std::pow(b, lb - j + k);
  return acc;
}

/// One-dimensional primitive overlap factor.
inline double overlap_1d(int la, int lb, double pa, double pb, double gamma) {
  double acc = 0.0;
  for (int i = 0; i <= (la + lb) / 2; ++i)
    acc += overlap_f(2 * i, la, lb, pa, pb) * dfact_odd(2 * i - 1) /
           std::pow(2.0 * gamma, i);
  return acc * std::sqrt(M_PI / gamma);
}

/// Closed-form overlap matrix of a GTO basis, in the library's function order.
inline std::vector<double> analytic_overlap(const GTOBasisSet<double> &basis) {
  const int nbf = basis.nbf();
  std::vector<double> S(static_cast<size_t>(nbf) * nbf, 0.0);

  for (int is = 0; is < basis.nshells(); ++is) {
    const auto &A = basis.shell(is);
    const auto &cA = A.cart();
    const int la = A.l(), nca = static_cast<int>(cA.size());
    const auto TA = ylm_transmat(la);

    for (int js = 0; js < basis.nshells(); ++js) {
      const auto &B = basis.shell(js);
      const auto &cB = B.cart();
      const int lb = B.l(), ncb = static_cast<int>(cB.size());
      const auto TB = ylm_transmat(lb);

      const double dx = A.O()[0] - B.O()[0], dy = A.O()[1] - B.O()[1],
                   dz = A.O()[2] - B.O()[2];
      const double R2 = dx * dx + dy * dy + dz * dz;

      for (int ka = 0; ka < A.nctr(); ++ka)
        for (int kb = 0; kb < B.nctr(); ++kb) {

          // Cartesian block for this contraction pair.
          std::vector<double> blk(static_cast<size_t>(nca) * ncb, 0.0);
          for (int pa = 0; pa < A.nprim(); ++pa)
            for (int pb = 0; pb < B.nprim(); ++pb) {
              const double al = A.alpha()[pa], be = B.alpha()[pb];
              const double gamma = al + be;
              const double pref = std::exp(-al * be / gamma * R2) *
                                  A.coeff(pa, ka) * B.coeff(pb, kb);
              // Gaussian product center
              const double Px = (al * A.O()[0] + be * B.O()[0]) / gamma;
              const double Py = (al * A.O()[1] + be * B.O()[1]) / gamma;
              const double Pz = (al * A.O()[2] + be * B.O()[2]) / gamma;

              for (int ia = 0; ia < nca; ++ia)
                for (int ib = 0; ib < ncb; ++ib) {
                  const double sx =
                      overlap_1d(cA[ia].nx, cB[ib].nx, Px - A.O()[0],
                                 Px - B.O()[0], gamma);
                  const double sy =
                      overlap_1d(cA[ia].ny, cB[ib].ny, Py - A.O()[1],
                                 Py - B.O()[1], gamma);
                  const double sz =
                      overlap_1d(cA[ia].nz, cB[ib].nz, Pz - A.O()[2],
                                 Pz - B.O()[2], gamma);
                  blk[static_cast<size_t>(ia) * ncb + ib] +=
                      pref * sx * sy * sz;
                }
            }

          // relative normalizations
          for (int ia = 0; ia < nca; ++ia)
            for (int ib = 0; ib < ncb; ++ib)
              blk[static_cast<size_t>(ia) * ncb + ib] *=
                  cA[ia].relnorm * cB[ib].relnorm;

          // project to the angular representation of each shell
          const int na = A.is_spherical() ? 2 * la + 1 : nca;
          const int nb = B.is_spherical() ? 2 * lb + 1 : ncb;
          const int r0 = basis.shell_offset(is) + ka * na;
          const int c0 = basis.shell_offset(js) + kb * nb;

          for (int a = 0; a < na; ++a)
            for (int b = 0; b < nb; ++b) {
              double acc = 0.0;
              for (int ia = 0; ia < nca; ++ia) {
                const double ta =
                    A.is_spherical()
                        ? TA[static_cast<size_t>(a) * nca + ia]
                        : (ia == a ? 1.0 : 0.0);
                if (ta == 0.0)
                  continue;
                for (int ib = 0; ib < ncb; ++ib) {
                  const double tb =
                      B.is_spherical()
                          ? TB[static_cast<size_t>(b) * ncb + ib]
                          : (ib == b ? 1.0 : 0.0);
                  acc += ta * tb * blk[static_cast<size_t>(ia) * ncb + ib];
                }
              }
              S[static_cast<size_t>(r0 + a) * nbf + (c0 + b)] = acc;
            }
        }
    }
  }
  return S;
}

/// A small STO-3G-like water basis, built through the caller-array API.
inline GTOBasisSet<double> water_basis(const Molecule &mol,
                                      ShellType type) {
  auto cen = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.atom_centers);
  auto Z = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.Z);

  std::vector<GTOShell<double>> sh;
  for (size_t i = 0; i < mol.natoms; ++i) {
    const std::array<double, 3> O = {cen(i)[0], cen(i)[1], cen(i)[2]};
    if (Z(i) == 1) {
      sh.emplace_back(0, type, O, std::vector<double>{3.425250914, 0.6239137298,
                                                      0.1688554040},
                      std::vector<double>{0.1543289673, 0.5353281423,
                                          0.4446345422},
                      1);
    } else {
      // O: 1s, and a generally contracted 2s/2p sharing one exponent set
      sh.emplace_back(0, type, O,
                      std::vector<double>{130.7093214, 23.80886605,
                                          6.443608313},
                      std::vector<double>{0.1543289673, 0.5353281423,
                                          0.4446345422},
                      1);
      sh.emplace_back(0, type, O,
                      std::vector<double>{5.033151319, 1.169596125,
                                          0.3803889600},
                      std::vector<double>{-0.09996722919, 0.3995128261,
                                          0.7001154689},
                      1);
      sh.emplace_back(1, type, O,
                      std::vector<double>{5.033151319, 1.169596125,
                                          0.3803889600},
                      std::vector<double>{0.1559162750, 0.6076837186,
                                          0.3919573931},
                      1);
    }
  }
  return GTOBasisSet<double>(std::move(sh));
}

} // namespace test
} // namespace Nukexc
