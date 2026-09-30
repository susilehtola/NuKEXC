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
//  Gauss-Lobatto x uniform-phi spherical grid
// ============================================================================
//
// ERKALE's DFTLobatto angular grid (AngularGrid::lobatto_shell,
// src/dftgrid.cpp). A product rule: Gauss-Lobatto in cos(theta), uniform in
// phi, with the number of phi points reduced ring by ring as the rings
// shrink towards the poles.
//
// WHY A PRODUCT RULE AT ALL, GIVEN LEBEDEV EXISTS
// -----------------------------------------------
// Lebedev grids need roughly two thirds the points of a product rule at the
// same exactness degree, so for a FIXED grid they win. What they cannot do
// is nest: the 302-point rule does not contain the 194-point rule, so
// refining discards every point already evaluated. A product rule refines by
// doubling in each factor and keeps everything. That is the trade, and it is
// why this exists alongside the Lebedev path rather than replacing it.
//
// THE RING RULE
// -------------
// For a shell of angular order l:
//
//   * n_theta = (l+3)/2 Lobatto points in cos(theta);
//   * on each ring, n_phi is the smallest integer with
//     sin^{n_phi}(theta) < thr, capped at l+1, where thr is 1e-8 for
//     l <= 50 and 1e-9 above. Rings near the poles are short: at the poles
//     sin(theta) = 0 and a single point suffices;
//   * odd-numbered rings are rotated by pi/n_phi. Without the offset the
//     points line up along meridians and the grid acquires a spurious
//     n_phi-fold symmetry that the integrand may share;
//   * each point on a ring carries weight 2 pi w_lobatto / n_phi.
//
// Weights here are for the UNIT SPHERE and sum to 4 pi; ERKALE folds the
// radial weight in at this point, which is left to the caller.
//
// The Lobatto rule itself is IntegratorXX's GaussLobatto primitive, not a
// reimplementation. Note that Lobatto is the one primitive for which
// `bound_inclusive` should be true -- it places nodes at both endpoints by
// construction, which is exactly what puts a point at each pole -- and that
// is among the things reported in tests/reference/integratorxx-upstream.md.

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <integratorxx/quadratures/primitive/gausslobatto.hpp>

namespace Nukexc {

/// Unit-sphere quadrature: directions and weights summing to 4 pi.
struct SphericalGrid {
  std::vector<std::array<double, 3>> dir;
  std::vector<double> w;
  size_t size() const { return dir.size(); }
};

/// Number of phi points on a ring, per ERKALE's rule.
inline int lobatto_nphi(double sin_theta, int l) {
  const double thr = (l <= 50) ? 1e-8 : 1e-9;
  int nphi = 1;
  double t = sin_theta;
  while (t >= thr && nphi <= l + 1) {
    ++nphi;
    t *= sin_theta;
  }
  return nphi;
}

/**
 *  @brief ERKALE's Gauss-Lobatto angular grid for a shell of order l.
 *
 *  @param l  angular order; n_theta = (l+3)/2
 */
inline SphericalGrid lobatto_angular_grid(int l) {
  if (l < 0)
    throw std::invalid_argument("lobatto_angular_grid: negative order");

  const int nth = (l + 3) / 2;
  if (nth < 2)
    throw std::invalid_argument(
        "lobatto_angular_grid: order too small for a Lobatto rule "
        "(needs at least two theta points)");

  using Prim = IntegratorXX::GaussLobatto<double, double>;
  const auto [xl, wl] =
      IntegratorXX::quadrature_traits<Prim>::generate(nth);

  SphericalGrid g;
  for (int ith = 0; ith < nth; ++ith) {
    const double cth = xl[ith];
    // Clamp before the square root: at the endpoints cth is +-1 up to
    // round-off, and 1 - cth*cth can come out very slightly negative.
    const double s2 = 1.0 - cth * cth;
    const double sth = (s2 > 0.0) ? std::sqrt(s2) : 0.0;

    const int nphi = lobatto_nphi(sth, l);
    const double dphi = 2.0 * M_PI / nphi;
    // Stagger alternate rings so the points do not line up on meridians.
    const double phioff = (ith % 2) ? M_PI / nphi : 0.0;
    const double wpt = 2.0 * M_PI * wl[ith] / nphi;

    for (int ip = 0; ip < nphi; ++ip) {
      const double phi = ip * dphi + phioff;
      g.dir.push_back({sth * std::cos(phi), sth * std::sin(phi), cth});
      g.w.push_back(wpt);
    }
  }
  return g;
}

} // namespace Nukexc
