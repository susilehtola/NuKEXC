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
//  One shell = one radial function times its solid harmonics
// ============================================================================
//
//     chi_{lm}(r) = R(|r - O|) * S_lm(x - O)
//
// with S_lm = r^l Y_lm the real solid harmonic that the precomputed switches
// in spherical_harmonics.hpp return. Every basis family this library cares
// about is that product, and they differ in R alone:
//
//     primitive Slater     R = c r^(n-l-1) e^(-zeta r)
//     contracted Slater    R = sum_p c_p r^(n_p-l-1) e^(-zeta_p r)
//     primitive Gaussian   R = c e^(-alpha r^2)
//     contracted Gaussian  R = sum_p c_p e^(-alpha_p r^2)
//     numerical            R = u(r)/r^(l+1), u interpolated from a table
//
// The last line is not implemented here and does not need to be. The point
// of this header is that it COULD be, as one more policy, without touching
// the evaluator, the shell structure, the function ordering, the gradient
// chain rule, the screening radii or any of the integral code downstream.
//
// WHY TEMPLATE OVER THE RADIAL RATHER THAN WRITE AN EVALUATOR PER FAMILY
// ---------------------------------------------------------------------
// Because only the radial differs, and everything else is where the bugs
// are. The angular part is a switch over l and m that someone had to derive
// once; the gradient chain rule is
//
//     grad chi = dR/dr * rhat * S_lm + R * grad S_lm,
//
// which is three lines that are easy to get subtly wrong; the ordering of
// the 2l+1 functions within a shell has to agree with whatever the
// local-to-global map assumes; and the cutoff radius has to be honoured
// identically or screening means different things for different bases.
// Writing that four times is four chances to disagree, and the disagreement
// shows up as a wrong integral rather than as a compile error.
//
// A contracted radial with one primitive is a primitive radial, so there is
// no separate policy for the uncontracted case -- one Slater policy covers
// pSTO and cSTO, one Gaussian policy covers pGTO and cGTO.
//
// WHAT THIS IS NOT
// ----------------
// Not the Cartesian Gaussian path. nukexc/gto_eval.hpp evaluates Cartesian
// monomials x^i y^j z^k e^(-alpha r^2) and carries a derivative tower up to
// the gradient of the Laplacian; its angular factors are not solid
// harmonics and its blocks hold cart_size(l) functions rather than 2l+1.
// That is a different product and it stays where it is. This header is the
// solid-harmonic family.

#include <concepts>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basis_evaluator.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "spherical_harmonics.hpp"

namespace Nukexc {

/// Highest angular momentum the precomputed solid harmonics cover. Both the
/// value and the value-and-gradient switches run to l = 7.
inline constexpr int SOLID_HARMONIC_MAX_L = 7;

/**
 *  @brief Shell-level data every radial family needs, whatever its radial is.
 *
 *  Separated from the radial parameters so that a new family supplies only
 *  what is new. `prim_offset` is a prefix over the shells into whatever
 *  per-primitive arrays the policy keeps; a policy with no primitives (a
 *  tabulated radial, say) can ignore it.
 */
struct ShellTable {
  Kokkos::View<int *> l;          ///< angular momentum per shell
  Kokkos::View<int *> first_bf;   ///< (nshell+1) prefix of 2l+1
  Kokkos::View<int *> prim_offset; ///< (nshell+1) prefix into primitives
  Kokkos::View<Point *> center;
  Kokkos::View<double *> radius;  ///< screening cutoff per shell
  int nbf = 0;
  int nshell = 0;
};

/**
 *  @brief What a radial family has to provide.
 *
 *  `shell_params` is the per-shell handle the evaluator loads once and hands
 *  back for every point of that shell. It must carry the centre, the cutoff
 *  and l, because the generic evaluator needs those three; whatever else it
 *  carries is the policy's business.
 */
template <typename R>
concept RadialBasis =
    requires(const R rad, typename R::shell_params p, double &a, double &b) {
      typename R::shell_params;
      requires std::is_trivially_copyable_v<typename R::shell_params>;

      { std::integral_constant<int, R::max_l>{} };
      requires R::max_l >= 0 && R::max_l <= SOLID_HARMONIC_MAX_L;

      { rad.shells() } -> std::convertible_to<ShellTable>;
      { rad.load_shell(0) } -> std::same_as<typename R::shell_params>;

      // The radial itself. Two entry points so a value-only evaluation does
      // not pay for a derivative it will not use.
      { rad.value(p, 1.0) } -> std::convertible_to<double>;
      rad.value_and_derivative(p, 1.0, a, b);

      // The three fields the generic evaluator reads directly.
      { p.O } -> std::convertible_to<Point>;
      { p.r_cut } -> std::convertible_to<double>;
      { p.l } -> std::convertible_to<int>;
    };

/**
 *  @brief BasisEvaluator over any radial family.
 *
 *  A block is a shell: one radial times its 2l+1 angular components,
 *  ordered m = -l .. +l. That ordering is this header's convention and the
 *  only place it is written down.
 *
 *  Values and gradients. A Hessian would need the radial's second
 *  derivative, which is one more policy method and no change here, but it
 *  is not derived, and max_deriv says so rather than letting a caller find
 *  out from a static_assert.
 */
template <typename Radial>
  requires RadialBasis<Radial>
class SolidHarmonicEvaluator {
public:
  using block_params = typename Radial::shell_params;
  static constexpr int max_block_size = 2 * Radial::max_l + 1;
  static constexpr BasisDeriv max_deriv = BasisDeriv::Gradient;

  SolidHarmonicEvaluator() = default;
  explicit SolidHarmonicEvaluator(Radial rad)
      : rad_(rad), sh_(rad.shells()) {}

  int nbf() const { return sh_.nbf; }
  int nblock() const { return sh_.nshell; }
  KOKKOS_INLINE_FUNCTION int block_first_bf(int i) const {
    return sh_.first_bf(i);
  }
  KOKKOS_INLINE_FUNCTION int block_size(int i) const {
    return 2 * sh_.l(i) + 1;
  }
  Kokkos::View<Point *> centers() const { return sh_.center; }
  Kokkos::View<double *> radii() const { return sh_.radius; }
  const Radial &radial() const { return rad_; }

  KOKKOS_INLINE_FUNCTION block_params load_block(int i) const {
    return rad_.load_shell(i);
  }

  template <BasisDeriv D>
  KOKKOS_INLINE_FUNCTION void eval_block(const block_params &p, double x,
                                         double y, double z, double *out,
                                         int stride) const {
    static_assert(D == BasisDeriv::Value || D == BasisDeriv::Gradient,
                  "the solid-harmonic path implements values and gradients; "
                  "a Hessian needs the radial's second derivative");

    const int nfunc = 2 * p.l + 1;
    const double dx = x - p.O[0];
    const double dy = y - p.O[1];
    const double dz = z - p.O[2];
    const double r2 = dx * dx + dy * dy + dz * dz;

    // Beyond the cutoff the shell contributes nothing. Whether that is
    // exact or an approximation is the radial's business: a confined
    // numerical radial is identically zero out here, an analytic one is
    // merely small, and the cutoff radius is where the family's own
    // threshold put it.
    if (r2 >= p.r_cut * p.r_cut) {
      for (int i = 0; i < nfunc; ++i) {
        out[BF_VAL * stride + i] = 0.0;
        if constexpr (D == BasisDeriv::Gradient) {
          out[BF_DX * stride + i] = 0.0;
          out[BF_DY * stride + i] = 0.0;
          out[BF_DZ * stride + i] = 0.0;
        }
      }
      return;
    }

    // A point on the nucleus: the radial is evaluated at a floor, and the
    // angular part handles the origin itself -- every solid harmonic but
    // l = 0 vanishes there.
    const double r = Kokkos::max(Kokkos::sqrt(r2), 1e-300);

    if constexpr (D == BasisDeriv::Value) {
      const double R = rad_.value(p, r);
      for (int i = 0; i < nfunc; ++i) {
        double S;
        real_solid_harmonic_cart_precomputed(p.l, i - p.l, dx, dy, dz, S);
        out[BF_VAL * stride + i] = R * S;
      }
    } else {
      double R = 0.0, dRdr = 0.0;
      rad_.value_and_derivative(p, r, R, dRdr);
      // grad chi = dR/dr rhat S + R grad S, and rhat = (dx, dy, dz)/r.
      const double common = dRdr / r;
      for (int i = 0; i < nfunc; ++i) {
        double S, dSx, dSy, dSz;
        real_solid_harmonic_cart_and_grad_precomputed(p.l, i - p.l, dx, dy, dz,
                                                      S, dSx, dSy, dSz);
        out[BF_VAL * stride + i] = R * S;
        out[BF_DX * stride + i] = R * dSx + S * dx * common;
        out[BF_DY * stride + i] = R * dSy + S * dy * common;
        out[BF_DZ * stride + i] = R * dSz + S * dz * common;
      }
    }
  }

private:
  Radial rad_;
  ShellTable sh_;
};

// ============================================================================
//  The two analytic families
// ============================================================================

/**
 *  @brief R = sum_p c_p r^(n_p - l - 1) e^(-zeta_p r): Slater, contracted or
 *         not.
 *
 *  The exponent n_p - l - 1 is the power left over once the solid harmonic
 *  has taken r^l out of r^(n-1), which is the same split
 *  nukexc/stobasis.hpp makes.
 */
struct SlaterRadial {
  ShellTable sh;
  Kokkos::View<double *> zeta;  ///< per primitive
  Kokkos::View<double *> coeff; ///< per primitive, normalization folded in
  Kokkos::View<int *> n;        ///< principal quantum number per primitive

  struct shell_params {
    Point O;
    double r_cut;
    int l;
    int p0; ///< first primitive
    int np; ///< primitive count
  };
  static constexpr int max_l = SOLID_HARMONIC_MAX_L;

  ShellTable shells() const { return sh; }

  KOKKOS_INLINE_FUNCTION shell_params load_shell(int i) const {
    return shell_params{sh.center(i), sh.radius(i), sh.l(i),
                        sh.prim_offset(i),
                        sh.prim_offset(i + 1) - sh.prim_offset(i)};
  }

  KOKKOS_INLINE_FUNCTION double value(const shell_params &p,
                                      const double r) const {
    double R = 0.0;
    for (int ip = p.p0; ip < p.p0 + p.np; ++ip)
      R += coeff(ip) * int_pow(r, n(ip) - p.l - 1) *
           Kokkos::exp(-zeta(ip) * r);
    return R;
  }

  KOKKOS_INLINE_FUNCTION void value_and_derivative(const shell_params &p,
                                                   const double r, double &R,
                                                   double &dRdr) const {
    R = 0.0;
    dRdr = 0.0;
    for (int ip = p.p0; ip < p.p0 + p.np; ++ip) {
      const int k = n(ip) - p.l - 1;
      const double e = coeff(ip) * Kokkos::exp(-zeta(ip) * r);
      const double rk = int_pow(r, k);
      R += e * rk;
      // d/dr [r^k e^(-zeta r)] = (k r^(k-1) - zeta r^k) e^(-zeta r), and the
      // k = 0 term is absent rather than r^(-1) times zero.
      dRdr += e * ((k > 0 ? k * int_pow(r, k - 1) : 0.0) - zeta(ip) * rk);
    }
  }
};

/**
 *  @brief R = sum_p c_p e^(-alpha_p r^2): Gaussian, contracted or not.
 *
 *  The r^l of a spherical Gaussian lives in the solid harmonic, so the
 *  radial is the bare exponential -- which is why a Gaussian shell costs one
 *  exp per primitive per point however high its l.
 */
struct GaussianRadial {
  ShellTable sh;
  Kokkos::View<double *> alpha; ///< per primitive
  Kokkos::View<double *> coeff; ///< per primitive, normalization folded in

  struct shell_params {
    Point O;
    double r_cut;
    int l;
    int p0;
    int np;
  };
  static constexpr int max_l = SOLID_HARMONIC_MAX_L;

  ShellTable shells() const { return sh; }

  KOKKOS_INLINE_FUNCTION shell_params load_shell(int i) const {
    return shell_params{sh.center(i), sh.radius(i), sh.l(i),
                        sh.prim_offset(i),
                        sh.prim_offset(i + 1) - sh.prim_offset(i)};
  }

  KOKKOS_INLINE_FUNCTION double value(const shell_params &p,
                                      const double r) const {
    const double r2 = r * r;
    double R = 0.0;
    for (int ip = p.p0; ip < p.p0 + p.np; ++ip)
      R += coeff(ip) * Kokkos::exp(-alpha(ip) * r2);
    return R;
  }

  KOKKOS_INLINE_FUNCTION void value_and_derivative(const shell_params &p,
                                                   const double r, double &R,
                                                   double &dRdr) const {
    const double r2 = r * r;
    R = 0.0;
    dRdr = 0.0;
    for (int ip = p.p0; ip < p.p0 + p.np; ++ip) {
      const double e = coeff(ip) * Kokkos::exp(-alpha(ip) * r2);
      R += e;
      dRdr += -2.0 * alpha(ip) * r * e;
    }
  }
};

static_assert(RadialBasis<SlaterRadial>);
static_assert(RadialBasis<GaussianRadial>);
static_assert(BasisEvaluator<SolidHarmonicEvaluator<SlaterRadial>>);
static_assert(BasisEvaluator<SolidHarmonicEvaluator<GaussianRadial>>);
static_assert(BasisEvaluatorWith<SolidHarmonicEvaluator<SlaterRadial>,
                                 BasisDeriv::Gradient>);
static_assert(BasisEvaluatorWith<SolidHarmonicEvaluator<GaussianRadial>,
                                 BasisDeriv::Gradient>);

using SlaterShellEvaluator = SolidHarmonicEvaluator<SlaterRadial>;
using GaussianShellEvaluator = SolidHarmonicEvaluator<GaussianRadial>;

} // namespace Nukexc
