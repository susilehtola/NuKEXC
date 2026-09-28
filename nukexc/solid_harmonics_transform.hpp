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
//  Cartesian -> real solid harmonic transformation
// ============================================================================
//
// ylm_transmat(l) returns the (2l+1) x cart_size(l) matrix T that takes the
// Cartesian monomials x^nx y^ny z^nz of degree l to the normalized real solid
// harmonics r^l Y_lm, with m running -l .. +l:
//
//     Y_m = sum_c T(m + l, c) * x^nx(c) y^ny(c) z^nz(c)
//
// stored row-major, so element (m + l, c) sits at index (m + l) * ncart + c.
// The Cartesian index c is the ERKALE/getind() ordering documented in
// gto_shell.hpp.
//
// This is a direct port of ERKALE's calcYlm_coeff()/Ylm_transmat()
// (src/solidharmonics.cpp), so the phase and normalization conventions match
// ERKALE exactly: the m > 0 functions carry the cosine (A_m) part, m < 0 the
// sine (B_m) part, and the prefactor
//
//     sqrt((2l+1)/4pi) * 2^-l * [ sqrt(2 (l-|m|)! / (l+|m|)!) if m != 0 ]
//
// makes them *normalized* spherical harmonics (unit integral of |Y|^2 over the
// sphere) rather than the Racah-normalized variety.
//
// This is the same convention the STO path's generated harmonics in
// spherical_harmonics.hpp use (see python/precompute_solid_harmonics.py, which
// emits sqrt((2l+1)/4pi) for m = 0 and sqrt((2l+1)/2pi) otherwise), so the two
// basis families share one angular convention.
//
// The transform is built once per angular momentum when a basis set is
// constructed; it is not on any hot path.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace Nukexc {

namespace detail {

/// n! as a double. Exact for n <= 22, far beyond the angular momenta in use.
inline double sh_factorial(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i)
    r *= static_cast<double>(i);
  return r;
}

/// Binomial coefficient C(n, k) as a double.
inline double sh_choose(int n, int k) {
  if (k < 0 || k > n)
    return 0.0;
  // Multiplicative form, exact while the running value stays integral.
  double r = 1.0;
  const int kk = (k > n - k) ? n - k : k;
  for (int i = 0; i < kk; ++i)
    r = r * static_cast<double>(n - i) / static_cast<double>(i + 1);
  return std::round(r);
}

/// Flat Cartesian index, duplicated here so this header stands alone.
/// Identical to Nukexc::cart_index in gto_shell.hpp.
inline constexpr int sh_cart_index(int ny, int nz) {
  const int i = ny + nz;
  return i * (i + 1) / 2 + nz;
}

} // namespace detail

/**
 *  @brief Cartesian coefficients of the normalized real solid harmonic
 *         r^l Y_l,mval.
 *
 *  Port of ERKALE's calcYlm_coeff (src/solidharmonics.cpp).
 *
 *  @param l    angular momentum
 *  @param mval magnetic quantum number, -l .. +l
 *  @returns    cart_size(l) coefficients in ERKALE Cartesian order
 */
inline std::vector<double> calc_ylm_coeff(int l, int mval) {
  if (l < 0)
    throw std::invalid_argument("calc_ylm_coeff: negative l");
  if (mval < -l || mval > l)
    throw std::invalid_argument("calc_ylm_coeff: |m| > l");

  const int ncart = (l + 1) * (l + 2) / 2;
  std::vector<double> ret(static_cast<size_t>(ncart), 0.0);

  const int m = std::abs(mval);

  double prefac = std::sqrt((2 * l + 1) / (4.0 * M_PI)) * std::pow(2.0, -l);
  if (m != 0)
    prefac *= std::sqrt(detail::sh_factorial(l - m) * 2.0 /
                        detail::sh_factorial(l + m));

  // bar-Pi contribution
  for (int k = 0; k <= (l - m) / 2; ++k) {
    double ffac = ((k % 2) ? -1.0 : 1.0) * detail::sh_choose(l, k) *
                  detail::sh_choose(2 * (l - k), l);
    if (m != 0)
      ffac *= detail::sh_factorial(l - 2 * k) /
              detail::sh_factorial(l - 2 * k - m);
    ffac *= prefac;

    for (int a = 0; a <= k; ++a) {
      const double afac = detail::sh_choose(k, a) * ffac;

      for (int b = 0; b <= a; ++b) {
        const double fac = detail::sh_choose(a, b) * afac;

        const int zexp = 2 * b - 2 * k + l - m;
        const int yexp = 2 * (a - b);
        // the x exponent 2*(k-a) is implied by the degree, so the flat
        // Cartesian index needs only (yexp, zexp)

        if (mval > 0) {
          // A_m: the cos(m phi) part
          for (int p = 0; p <= m; ++p) {
            int cosfac;
            switch ((m - p) % 4) {
            case 0:
              cosfac = 1;
              break;
            case 2:
              cosfac = -1;
              break;
            default:
              cosfac = 0;
              break;
            }
            if (cosfac != 0)
              ret[static_cast<size_t>(
                  detail::sh_cart_index(yexp + m - p, zexp))] +=
                  cosfac * detail::sh_choose(m, p) * fac;
          }
        } else if (m == 0) {
          ret[static_cast<size_t>(detail::sh_cart_index(yexp, zexp))] += fac;
        } else {
          // B_m: the sin(|m| phi) part
          for (int p = 0; p <= m; ++p) {
            int sinfac;
            switch ((m - p) % 4) {
            case 1:
              sinfac = 1;
              break;
            case 3:
              sinfac = -1;
              break;
            default:
              sinfac = 0;
              break;
            }
            if (sinfac != 0)
              ret[static_cast<size_t>(
                  detail::sh_cart_index(yexp + m - p, zexp))] +=
                  sinfac * detail::sh_choose(m, p) * fac;
          }
        }
      }
    }
  }

  return ret;
}

/**
 *  @brief The (2l+1) x cart_size(l) Cartesian-to-spherical transformation,
 *         row-major, rows ordered m = -l .. +l.
 */
inline std::vector<double> ylm_transmat(int l) {
  const int ncart = (l + 1) * (l + 2) / 2;
  const int nlm = 2 * l + 1;
  std::vector<double> T(static_cast<size_t>(nlm) * ncart, 0.0);

  for (int m = -l; m <= l; ++m) {
    const std::vector<double> row = calc_ylm_coeff(l, m);
    for (int c = 0; c < ncart; ++c)
      T[static_cast<size_t>(m + l) * ncart + c] = row[static_cast<size_t>(c)];
  }
  return T;
}

} // namespace Nukexc
