/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 *
 *    Redistribution and use in source and binary forms, with or without
 *    modification, are permitted provided that the following conditions are
 *    met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

// ============================================================================
//  Spherical free-atom densities
// ============================================================================
//
// The density of a spherically averaged, neutral, free atom:
//
//     n_A(r) = (1/4pi) sum_i occ_i |R_i(r)|^2
//
// from the Roothaan-Hartree-Fock expansions of Koga, Kanayama, Watanabe and
// Thakkar, translated to C++ source by python/precompute_thakkar.py. The
// orbitals are contractions of Slater primitives whose principal quantum
// numbers vary WITHIN the contraction, which is why the primitive STO path
// cannot express them.
//
// Three separate things want this, which is why it is its own header rather
// than part of any one of them:
//
//   * Hirshfeld partitioning (hirshfeld.hpp), where the weight of a point
//     is the free atom's share of the superposed free-atom density;
//   * free-atom reference subtraction, where n - sum_A n_A is integrated
//     instead of n, so the quadrature only has to resolve the part of the
//     density that bonding actually changed;
//   * density-based adaptive grids, which need a cheap proxy for where the
//     density is before the density is known.
//
// The table is small -- 54 elements, 375 orbitals, 4302 primitives -- so it
// is uploaded whole. Do that once and keep the result; make_thakkar_density
// _table() allocates and copies, and is not something to call per grid.
//
// WHY THE COEFFICIENTS ARE PREMULTIPLIED
// --------------------------------------
// The published coefficients multiply NORMALIZED Slater primitives,
// chi_p(r) = N(n,zeta) r^(n-1) exp(-zeta r) with
// N = (2 zeta)^(n + 1/2) / sqrt((2n)!). Folding N into the coefficient at
// upload turns the inner loop into a multiply-add and keeps a factorial and
// a pow out of the hot path. The normalization is checked at generation
// time -- every orbital has unit self-overlap -- so there is no run-time
// cost to being sure of it either.

#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "nukexc_config.hpp"
#include "thakkar_data.hpp"

namespace Nukexc {

namespace detail {
/// x^n for small non-negative n. The principal quantum numbers here reach
/// 7, so a multiply chain beats a general pow() comfortably -- and this is
/// a hot path: Hirshfeld partitioning calls it once per atom per grid
/// point.
KOKKOS_INLINE_FUNCTION double ipow(double x, int n) {
  double v = 1.0;
  for (int i = 0; i < n; ++i)
    v *= x;
  return v;
}
} // namespace detail

/**
 *  @brief Device-resident table of spherical free-atom densities.
 *
 *  CSR twice over: orbitals are indexed by atomic number, primitives by
 *  orbital. An element with no data has an empty orbital range and its
 *  density is identically zero -- callers that cannot tolerate that should
 *  check has_element() rather than discover it as a silent zero.
 */
struct AtomicDensityTable {
  Kokkos::View<const int *> orb_offset;   ///< [max_Z + 2], indexed by Z
  Kokkos::View<const double *> orb_occ;   ///< [norb]
  Kokkos::View<const int *> prim_offset;  ///< [norb + 1]
  Kokkos::View<const int *> prim_n;       ///< [nprim] principal quantum number
  Kokkos::View<const double *> prim_zeta; ///< [nprim]
  Kokkos::View<const double *> prim_coeff; ///< [nprim], normalization folded in
  int max_Z = 0;

  /// Is this element tabulated?
  KOKKOS_INLINE_FUNCTION bool has_element(int Z) const {
    return Z >= 1 && Z <= max_Z && orb_offset(Z + 1) > orb_offset(Z);
  }

  /// Spherical free-atom density n_A(r). Zero for untabulated elements.
  KOKKOS_INLINE_FUNCTION double density(int Z, double r) const {
    if (Z < 1 || Z > max_Z)
      return 0.0;
    double acc = 0.0;
    for (int io = orb_offset(Z); io < orb_offset(Z + 1); ++io) {
      double R = 0.0;
      for (int ip = prim_offset(io); ip < prim_offset(io + 1); ++ip)
        R += prim_coeff(ip) * detail::ipow(r, prim_n(ip) - 1) *
             Kokkos::exp(-prim_zeta(ip) * r);
      acc += orb_occ(io) * R * R;
    }
    return acc * 0.25 * M_1_PI;
  }

  /**
   *  @brief n_A(r) and dn_A/dr together.
   *
   *  Sharing the exponential between the two is worth a factor of about
   *  two, and the gradient is never wanted without the value.
   */
  KOKKOS_INLINE_FUNCTION void density_and_deriv(int Z, double r, double &n,
                                                double &dndr) const {
    n = 0.0;
    dndr = 0.0;
    if (Z < 1 || Z > max_Z)
      return;
    for (int io = orb_offset(Z); io < orb_offset(Z + 1); ++io) {
      double R = 0.0, dR = 0.0;
      for (int ip = prim_offset(io); ip < prim_offset(io + 1); ++ip) {
        const int np = prim_n(ip);
        const double z = prim_zeta(ip);
        const double e = prim_coeff(ip) * Kokkos::exp(-z * r);
        const double rn1 = detail::ipow(r, np - 1);
        // r^(np-2) is singular at the origin for np = 1, but its
        // coefficient (np - 1) vanishes there; compute neither rather than
        // multiplying zero by an infinity.
        const double rn2 = (np >= 2) ? detail::ipow(r, np - 2) : 0.0;
        R += e * rn1;
        dR += e * ((np - 1) * rn2 - z * rn1);
      }
      n += orb_occ(io) * R * R;
      dndr += orb_occ(io) * 2.0 * R * dR;
    }
    n *= 0.25 * M_1_PI;
    dndr *= 0.25 * M_1_PI;
  }
};

/**
 *  @brief Upload the Thakkar free-atom densities.
 *
 *  Allocates and copies; call it once and hold the result.
 */
inline AtomicDensityTable make_thakkar_density_table() {
  using namespace thakkar;
  constexpr int noff = static_cast<int>(sizeof(orb_offset_) / sizeof(int));
  constexpr int norb = static_cast<int>(sizeof(orb_l_) / sizeof(int));
  constexpr int nprim = static_cast<int>(sizeof(prim_n_) / sizeof(int));

  // Fold the primitive normalization into the coefficient. (2n)! overflows
  // a double only well past n = 80, and the table stops at n = 9, but the
  // ratio is formed as a product of small factors anyway so that the
  // intermediate never gets large.
  std::vector<double> coeff(nprim);
  for (int i = 0; i < nprim; ++i) {
    const int n = prim_n_[i];
    const double z = prim_zeta_[i];
    // N = (2 zeta)^(n + 1/2) / sqrt((2n)!)
    double norm = Kokkos::sqrt(2.0 * z);
    for (int k = 0; k < n; ++k)
      norm *= 2.0 * z;
    double fact = 1.0;
    for (int k = 2; k <= 2 * n; ++k)
      fact *= static_cast<double>(k);
    coeff[i] = prim_coeff_[i] * norm / Kokkos::sqrt(fact);
  }

  auto up_i = [](const char *name, const int *src, int n) {
    Kokkos::View<int *> d(name, n);
    Kokkos::deep_copy(
        d, Kokkos::View<const int *, HostSpace,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(src, n));
    return Kokkos::View<const int *>(d);
  };
  auto up_d = [](const char *name, const double *src, int n) {
    Kokkos::View<double *> d(name, n);
    Kokkos::deep_copy(
        d, Kokkos::View<const double *, HostSpace,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(src, n));
    return Kokkos::View<const double *>(d);
  };

  AtomicDensityTable t;
  t.max_Z = max_Z;
  t.orb_offset = up_i("thakkar orb offset", orb_offset_, noff);
  t.orb_occ = up_d("thakkar orb occ", orb_occ_, norb);
  t.prim_offset = up_i("thakkar prim offset", prim_offset_, norb + 1);
  t.prim_n = up_i("thakkar prim n", prim_n_, nprim);
  t.prim_zeta = up_d("thakkar prim zeta", prim_zeta_, nprim);
  t.prim_coeff = up_d("thakkar prim coeff", coeff.data(), nprim);
  return t;
}

} // namespace Nukexc
