/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Bob Schreiner
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

#include "grid.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "stobasis.hpp"

namespace Nukexc {

KOKKOS_INLINE_FUNCTION
double I_tilde(const int n, const int l, const double r, const double zeta) {
  int a = n + l + 2;
  int b = n - l + 1;

  if (r < 1e-12) {
    // lower_gamma(a, 0)/r^(2l+1) -> 0 in the limit; only the upper_gamma term
    // survives
    double fact_b_minus_1 = 1.0;
    for (int i = 2; i < b; ++i)
      fact_b_minus_1 *= i;
    return fact_b_minus_1 / Kokkos::pow(zeta, b);
  }

  double zr = zeta * r;
  return lower_gamma(a, zr) /
             (Kokkos::pow(zeta, a) * Kokkos::pow(r, 2 * l + 1)) +
         upper_gamma(b, zr) / Kokkos::pow(zeta, b);
}

KOKKOS_INLINE_FUNCTION
double C_prefactor(const int n, const int l, const double zeta) {
  return 4 * M_PI * Kokkos::pow(2 * zeta, n + 0.5) /
         (Kokkos::sqrt(factorial(2 * n)) * (2 * l + 1));
}

// Potential — just three multiplications.
//
// Non-inlined on purpose: this pulls in the angular switch
// (real_solid_harmonic_cart_precomputed) AND the gamma-function series in
// I_tilde (lower_gamma / upper_gamma), which together are register-heavy.
// Keeping them in this callee's own frame stops them from widening the live
// register set of the integral kernels that fill a potential buffer (Coulomb
// Gram, aux overlap, exchange three-center).
NUKEXC_NOINLINE_FUNCTION
double sto_potential(const int n, const int l, const int m, const double x,
                     const double y, const double z, const double r,
                     const double zeta) {
  double val;
  real_solid_harmonic_cart_precomputed(l, m, x, y, z, val);
  double C = C_prefactor(n, l, zeta);
  double I = I_tilde(n, l, r, zeta);

  return C * val * I;
}

KOKKOS_INLINE_FUNCTION
double sto_potential(const STOBasisSet &basis, const int idx, const double x,
                     const double y, const double z, const double r) {
  const int n = basis.n(idx);
  const int l = basis.l(idx);
  const int m = basis.m(idx);
  const double zeta = basis.zeta(idx);

  return sto_potential(n, l, m, x, y, z, r, zeta);
}
void sto_potential_collocation_scaled(const ExecSpace space,
                                      const STOBasisSet basis,
                                      const FlatGrid grid,
                                      DeviceView2DLeft potential_collocation) {

  const int N_bf = basis.nbf();
  const int N_quad = grid.quad_points.extent(0);

  Kokkos::parallel_for(
      "Compute potentials",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0}, {N_bf, N_quad}),
      KOKKOS_LAMBDA(const int i, const int g) {
        const int n = basis.n(i);
        const int l = basis.l(i);
        const int m = basis.m(i);
        const double zeta = basis.zeta(i);
        const double x = grid.quad_points(g)[0] - basis.O(i)[0];
        const double y = grid.quad_points(g)[1] - basis.O(i)[1];
        const double z = grid.quad_points(g)[2] - basis.O(i)[2];
        const double r = dist(grid.quad_points(g), basis.O(i));
        potential_collocation(i, g) =
            sto_potential(n, l, m, x, y, z, r, zeta) * grid.weights(g);
      });
}
// ============================================================================
//  PotentialEvaluator adapter
// ============================================================================
//
// Lives here rather than in stobasis.hpp so that the XC path, which has no
// use for the Coulomb potential, does not have to pull in the gamma-function
// machinery. Inherits the basis-function side unchanged and adds the
// potential.

class STOPotentialEvaluator : public STOPrimitiveEvaluator {
public:
  using STOPrimitiveEvaluator::STOPrimitiveEvaluator;

  /// Electrostatic potential of the block's single basis function, at an
  /// ABSOLUTE point. The evaluator subtracts the center itself, matching
  /// eval_block, which is why callers no longer compute dx/dy/dz/r by hand.
  KOKKOS_INLINE_FUNCTION void
  eval_potential_block(const block_params &p, double x, double y, double z,
                       double *out, int stride) const {
    (void)stride;
    const double dx = x - p.O[0];
    const double dy = y - p.O[1];
    const double dz = z - p.O[2];
    const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
    out[0] = sto_potential(p.n, p.l, p.m, dx, dy, dz, r, p.zeta);
  }
};

static_assert(PotentialEvaluator<STOPotentialEvaluator>);

} // namespace Nukexc
