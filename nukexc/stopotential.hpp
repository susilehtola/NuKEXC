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

// int_pow throughout, and one shared exp(-zr).
//
// This is called once per (auxiliary function, grid point) on every Coulomb
// and exchange build, and it used to make five libm calls per evaluation:
// three pow() with integer exponents, which the compiler does not fold, plus
// an exp inside each of the two incomplete gammas at the same argument. What
// is left is one exp and one sqrt. Measured on water in TZ2P with its
// Cholesky fitting basis: the fitting metric and the Coulomb build (see the
// commit that introduced this).
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
    return fact_b_minus_1 / int_pow(zeta, b);
  }

  const double zr = zeta * r;
  const double exp_minus_zr = Kokkos::exp(-zr);
  return lower_gamma(a, zr, exp_minus_zr) /
             (int_pow(zeta, a) * int_pow(r, 2 * l + 1)) +
         upper_gamma(b, zr, exp_minus_zr) / int_pow(zeta, b);
}

/// (2 zeta)^(n + 1/2) is the one genuinely fractional power here, and it
/// factors: (2 zeta)^n sqrt(2 zeta). That leaves a hardware sqrt where there
/// was a libm pow.
KOKKOS_INLINE_FUNCTION
double C_prefactor(const int n, const int l, const double zeta) {
  const double two_zeta = 2 * zeta;
  return 4 * M_PI * int_pow(two_zeta, n) * Kokkos::sqrt(two_zeta) /
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

  /**
   *  @brief Electrostatic potential of every function in the shell, at an
   *         ABSOLUTE point. The evaluator subtracts the centre itself,
   *         matching eval_block.
   *
   *  The radial factor C(n, l, zeta) I(n, l, r, zeta) is the same for all
   *  2l+1 functions of the shell -- only the solid harmonic differs -- so
   *  it is computed ONCE here. That matters more than anywhere else in the
   *  library: I_tilde evaluates two incomplete gamma functions, this runs
   *  once per (auxiliary function, grid point) on every Coulomb and
   *  exchange build, and the per-function version computed it 2l+1 times
   *  over for a p or d shell.
   *
   *  One primitive per shell, which is what make_slater_radial produces
   *  from a primitive Slater basis and what every caller passes. A
   *  contracted fitting basis would need its coefficients threaded through
   *  here, and would have needed that before this too: C_prefactor carries
   *  the normalization the potential assumes, so the contraction cannot
   *  simply multiply it.
   */
  KOKKOS_INLINE_FUNCTION void
  eval_potential_block(const block_params &p, double x, double y, double z,
                       double *out, int stride) const {
    (void)stride;
    const double dx = x - p.O[0];
    const double dy = y - p.O[1];
    const double dz = z - p.O[2];
    const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

    const auto &rad = radial();
    const int n = rad.n(p.p0);
    const double zeta = rad.zeta(p.p0);
    const double radial_part =
        C_prefactor(n, p.l, zeta) * I_tilde(n, p.l, r, zeta);

    const int nfunc = 2 * p.l + 1;
    for (int i = 0; i < nfunc; ++i) {
      double S;
      real_solid_harmonic_cart_precomputed(p.l, i - p.l, dx, dy, dz, S);
      out[i] = radial_part * S;
    }
  }
};

static_assert(PotentialEvaluator<STOPotentialEvaluator>);

} // namespace Nukexc
