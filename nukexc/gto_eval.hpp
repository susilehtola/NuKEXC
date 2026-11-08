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
//  Device-side evaluation of Gaussian basis functions and their derivatives
// ============================================================================
//
// A port of ERKALE's GaussianShell::eval_bf_derivs (src/basis.cpp) to Kokkos,
// keeping its structure because that structure is the right one: build the
// power tables x^0..x^(l+order) once, evaluate each primitive's exp(-z r^2)
// once, and share both across every requested derivative output and every
// contraction of the shell.
//
// DERIVATIVE ORDER
// ----------------
// The order is a compile-time parameter, so a kernel only materializes the
// buffers it needs:
//
//   GTODeriv::Value        phi
//   GTODeriv::Gradient     phi, grad phi                     (GGA)
//   GTODeriv::Hessian      phi, grad phi, hess phi           (GGA forces,
//                                                             Laplacian mGGA)
//   GTODeriv::LaplGradient phi, grad, hess, grad(lapl phi)   (Laplacian mGGA
//                                                             forces; ERKALE's
//                                                             bf_lx/ly/lz)
//
// Phase 5's geometric kernels declare the order they need in their manifests,
// and adding one more is a matter of extending the enum and the _der<n>
// operators below -- the surrounding machinery does not change.
//
// THE ONE-DIMENSIONAL DERIVATIVE OPERATORS
// ----------------------------------------
// Everything reduces to derivatives of x^l exp(-z x^2), with the exponential
// factored out. Writing xr[k] = x^k:
//
//   d/dx    -> l xr[l-1] - 2z xr[l+1]
//   d2/dx2  -> l(l-1) xr[l-2] - 2z(2l+1) xr[l] + 4z^2 xr[l+2]
//   d3/dx3  -> l(l-1)(l-2) xr[l-3] - 6 z l^2 xr[l-1]
//              + 12 z^2 (l+1) xr[l+1] - 8 z^3 xr[l+3]
//
// which is why the power tables run to degree l + order.
//
// REGISTER PRESSURE
// -----------------
// A thread handles one (shell, contraction, point) and keeps a Cartesian
// accumulation buffer of cart_size(l) doubles per output component. For the
// common DFT case (l <= 4, so cart_size <= 15) a gradient evaluation needs
// 4 x 15 doubles, which is reasonable. High l combined with high derivative
// order is register-heavy; the standard remedy is per-l specialized kernels
// (what GauXC generates), which can be slotted in behind this same interface
// without changing callers.

#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "gto_basisset.hpp"
#include "gto_shell.hpp"
#include "basis_evaluator.hpp"
#include "nukexc_config.hpp"
#include "solid_harmonics_transform.hpp"
#include "spherical_harmonics.hpp"

namespace Nukexc {

/// Highest angular momentum the device evaluator supports. cart_size(7) = 36.
inline constexpr int GTO_MAX_L = 7;
/// Cartesian components at GTO_MAX_L, the bound on the per-thread buffers.
inline constexpr int GTO_MAX_CART = (GTO_MAX_L + 1) * (GTO_MAX_L + 2) / 2;
/// Power tables run to l + 3 for the highest supported derivative order.
inline constexpr int GTO_MAX_POW = GTO_MAX_L + 4;

/// Derivative order and component indices are the shared basis-evaluator
/// vocabulary; these aliases keep the GTO-specific spellings working.
using GTODeriv = BasisDeriv;

// ============================================================================
//  Device basis
// ============================================================================

/// Struct-of-Views mirror of GTOBasisSet, laid out for device access.
///
/// CSR throughout: shell `s` owns primitives [prim_offset(s), prim_offset(s+1)),
/// coefficients [coeff_offset(s), coeff_offset(s+1)) (contraction-major),
/// Cartesian normalizations [norm_offset(s), norm_offset(s+1)), and basis
/// functions [bf_offset(s), bf_offset(s+1)).
struct GTOBasisDevice {
  // per shell
  Kokkos::View<int *> l;
  Kokkos::View<int *> spherical; ///< 0 = Cartesian, 1 = spherical
  Kokkos::View<Point *> O;
  Kokkos::View<int *> nprim;
  Kokkos::View<int *> nctr;
  Kokkos::View<int *> prim_offset;
  Kokkos::View<int *> coeff_offset;
  Kokkos::View<int *> norm_offset;
  Kokkos::View<int *> bf_offset;
  Kokkos::View<double *> cutoff_radii;

  // flat
  Kokkos::View<double *> alpha;
  Kokkos::View<double *> coeff;
  Kokkos::View<double *> relnorm;

  /// Cartesian-to-spherical transforms, concatenated by angular momentum:
  /// the (2l+1) x cart_size(l) row-major block for l starts at tm_offset(l).
  Kokkos::View<double *> transmat;
  Kokkos::View<int *> tm_offset;

  /// Cartesian exponents, concatenated by angular momentum: the cart_size(l)
  /// triples for l start at cart_offset(l).
  Kokkos::View<int *> cart_nx;
  Kokkos::View<int *> cart_ny;
  Kokkos::View<int *> cart_nz;
  Kokkos::View<int *> cart_offset;

  /// Blocks: one per (shell, contraction) pair, precomputed at upload. The
  /// collocation path used to rebuild this list on every call, with a
  /// device-to-host copy and two host-to-device copies each time.
  Kokkos::View<int *> block_shell;
  Kokkos::View<int *> block_ctr;
  Kokkos::View<int *> block_first_bf;
  Kokkos::View<int *> block_nfunc;
  Kokkos::View<Point *> block_center;
  Kokkos::View<double *> block_radius;

  int nshell = 0;
  int nbf = 0;
  int max_l = 0;
  int nblock = 0;
};

/// Upload a host basis set to the device.
inline GTOBasisDevice make_gto_device_basis(const GTOBasisSet<double> &host) {
  const int nsh = host.nshells();
  if (nsh == 0)
    return GTOBasisDevice{};

  const int max_l = host.max_l();
  if (max_l > GTO_MAX_L)
    throw std::invalid_argument(
        "make_gto_device_basis: angular momentum " + std::to_string(max_l) +
        " exceeds GTO_MAX_L = " + std::to_string(GTO_MAX_L));

  std::vector<int> h_l(nsh), h_sph(nsh), h_np(nsh), h_nc(nsh);
  std::vector<int> h_po(nsh + 1, 0), h_co(nsh + 1, 0), h_no(nsh + 1, 0),
      h_bo(nsh + 1, 0);
  std::vector<Point> h_O(nsh);
  std::vector<double> h_cut(nsh), h_alpha, h_coeff, h_norm;

  for (int s = 0; s < nsh; ++s) {
    const auto &sh = host.shell(s);
    h_l[s] = sh.l();
    h_sph[s] = sh.is_spherical() ? 1 : 0;
    h_np[s] = sh.nprim();
    h_nc[s] = sh.nctr();
    h_O[s][0] = sh.O()[0];
    h_O[s][1] = sh.O()[1];
    h_O[s][2] = sh.O()[2];
    h_cut[s] = sh.cutoff_radius();

    h_alpha.insert(h_alpha.end(), sh.alpha().begin(), sh.alpha().end());
    h_coeff.insert(h_coeff.end(), sh.coeff().begin(), sh.coeff().end());
    for (const auto &c : sh.cart())
      h_norm.push_back(c.relnorm);

    h_po[s + 1] = static_cast<int>(h_alpha.size());
    h_co[s + 1] = static_cast<int>(h_coeff.size());
    h_no[s + 1] = static_cast<int>(h_norm.size());
    h_bo[s + 1] = h_bo[s] + sh.size();
  }

  // One transform and one Cartesian-exponent table per angular momentum.
  std::vector<double> h_tm;
  std::vector<int> h_tmo(max_l + 2, 0);
  std::vector<int> h_cnx, h_cny, h_cnz, h_co_l(max_l + 2, 0);
  for (int l = 0; l <= max_l; ++l) {
    const auto T = ylm_transmat(l);
    h_tm.insert(h_tm.end(), T.begin(), T.end());
    h_tmo[l + 1] = static_cast<int>(h_tm.size());

    for (const auto &c : cartesian_components(l)) {
      h_cnx.push_back(c.nx);
      h_cny.push_back(c.ny);
      h_cnz.push_back(c.nz);
    }
    h_co_l[l + 1] = static_cast<int>(h_cnx.size());
  }

  auto up_i = [](const char *name, const std::vector<int> &v) {
    Kokkos::View<int *> d(name, v.size());
    Kokkos::View<const int *, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h(v.data(), v.size());
    Kokkos::deep_copy(d, h);
    return d;
  };
  auto up_d = [](const char *name, const std::vector<double> &v) {
    Kokkos::View<double *> d(name, v.size());
    Kokkos::View<const double *, HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h(v.data(), v.size());
    Kokkos::deep_copy(d, h);
    return d;
  };

  GTOBasisDevice dev;
  dev.l = up_i("l", h_l);
  dev.spherical = up_i("spherical", h_sph);
  dev.nprim = up_i("nprim", h_np);
  dev.nctr = up_i("nctr", h_nc);
  dev.prim_offset = up_i("prim_offset", h_po);
  dev.coeff_offset = up_i("coeff_offset", h_co);
  dev.norm_offset = up_i("norm_offset", h_no);
  dev.bf_offset = up_i("bf_offset", h_bo);
  dev.alpha = up_d("alpha", h_alpha);
  dev.coeff = up_d("coeff", h_coeff);
  dev.relnorm = up_d("relnorm", h_norm);
  dev.cutoff_radii = up_d("cutoff_radii", h_cut);
  dev.transmat = up_d("transmat", h_tm);
  dev.tm_offset = up_i("tm_offset", h_tmo);
  dev.cart_nx = up_i("cart_nx", h_cnx);
  dev.cart_ny = up_i("cart_ny", h_cny);
  dev.cart_nz = up_i("cart_nz", h_cnz);
  dev.cart_offset = up_i("cart_offset", h_co_l);

  dev.O = Kokkos::View<Point *>("O", nsh);
  Kokkos::View<const Point *, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      hO(h_O.data(), h_O.size());
  Kokkos::deep_copy(dev.O, hO);

  // Block tables: one entry per (shell, contraction).
  {
    std::vector<int> bs, bc, bf, bn;
    std::vector<Point> bcen;
    std::vector<double> brad;
    for (int s = 0; s < nsh; ++s) {
      const auto &sh = host.shell(s);
      const int nfunc = sh.ncomp();
      for (int k = 0; k < sh.nctr(); ++k) {
        bs.push_back(s);
        bc.push_back(k);
        bf.push_back(h_bo[s] + k * nfunc);
        bn.push_back(nfunc);
        bcen.push_back(h_O[s]);
        brad.push_back(h_cut[s]);
      }
    }
    dev.block_shell = up_i("block_shell", bs);
    dev.block_ctr = up_i("block_ctr", bc);
    dev.block_first_bf = up_i("block_first_bf", bf);
    dev.block_nfunc = up_i("block_nfunc", bn);
    dev.block_radius = up_d("block_radius", brad);
    dev.block_center = Kokkos::View<Point *>("block_center", bcen.size());
    Kokkos::View<const Point *, HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        hbc(bcen.data(), bcen.size());
    Kokkos::deep_copy(dev.block_center, hbc);
    dev.nblock = static_cast<int>(bs.size());
  }

  dev.nshell = nsh;
  dev.nbf = host.nbf();
  dev.max_l = max_l;
  return dev;
}

// ============================================================================
//  One-dimensional derivative operators
// ============================================================================

KOKKOS_INLINE_FUNCTION
double gto_der1(const double *x, int l, double zeta) {
  double d = -2.0 * zeta * x[l + 1];
  if (l > 0)
    d += l * x[l - 1];
  return d;
}

KOKKOS_INLINE_FUNCTION
double gto_der2(const double *x, int l, double zeta) {
  double d = 4.0 * zeta * zeta * x[l + 2] - 2.0 * zeta * (2 * l + 1) * x[l];
  if (l > 1)
    d += l * (l - 1) * x[l - 2];
  return d;
}

KOKKOS_INLINE_FUNCTION
double gto_der3(const double *x, int l, double zeta) {
  double d = -8.0 * zeta * zeta * zeta * x[l + 3] +
             12.0 * zeta * zeta * (l + 1) * x[l + 1];
  if (l > 0)
    d -= 6.0 * zeta * l * l * x[l - 1];
  if (l > 2)
    d += l * (l - 1) * (l - 2) * x[l - 3];
  return d;
}

// ============================================================================
//  Shell evaluation
// ============================================================================

KOKKOS_INLINE_FUNCTION
constexpr int gto_ncomp(GTODeriv d) { return basis_ncomp(d); }

inline constexpr int GTO_VAL = BF_VAL;
inline constexpr int GTO_DX = BF_DX, GTO_DY = BF_DY, GTO_DZ = BF_DZ;
inline constexpr int GTO_DXX = BF_DXX, GTO_DXY = BF_DXY, GTO_DXZ = BF_DXZ;
inline constexpr int GTO_DYY = BF_DYY, GTO_DYZ = BF_DYZ, GTO_DZZ = BF_DZZ;
inline constexpr int GTO_LGX = BF_LGX, GTO_LGY = BF_LGY, GTO_LGZ = BF_LGZ;

/**
 *  @brief Evaluate one contraction of one shell at one point.
 *
 *  Writes gto_ncomp(Deriv) components for each of the shell's angular
 *  functions into `out`, laid out component-major:
 *
 *      out[icomp * out_stride + ifunc]
 *
 *  so a caller filling a (nbf, npts) collocation matrix passes
 *  out = &values(bf0, ipt) with out_stride = the distance between components.
 *
 *  @tparam Deriv which derivatives to produce
 */
template <GTODeriv Deriv>
KOKKOS_INLINE_FUNCTION void
gto_eval_contraction(const GTOBasisDevice &b, int ishell, int ictr, double px,
                     double py, double pz, double *out, int out_stride) {

  constexpr int NC = gto_ncomp(Deriv);
  constexpr bool do_grad = Deriv >= GTODeriv::Gradient;
  constexpr bool do_hess = Deriv >= GTODeriv::Hessian;
  constexpr bool do_lgrad = Deriv >= GTODeriv::LaplGradient;

  const int l = b.l(ishell);
  const int ncart = (l + 1) * (l + 2) / 2;
  const int np = b.nprim(ishell);
  const int p0 = b.prim_offset(ishell);
  const int c0 = b.coeff_offset(ishell) + ictr * np;
  const int n0 = b.norm_offset(ishell);
  const int k0 = b.cart_offset(l);

  const double x = px - b.O(ishell)[0];
  const double y = py - b.O(ishell)[1];
  const double z = pz - b.O(ishell)[2];
  const double r2 = x * x + y * y + z * z;

  // ------------------------------------------------------------------
  //  A spherical shell is R(r) S_lm. Build it that way.
  // ------------------------------------------------------------------
  //
  // The Cartesian route below builds cart_size(l) = (l+1)(l+2)/2 components
  // where the shell has 2l+1, renormalizes each, and then applies a
  // (2l+1) x cart_size(l) transform per derivative component per point. None
  // of that buys anything here: the separability the Cartesian form exists
  // for is an INTEGRAL property, and this is an evaluation at a point --
  // r2 is formed once and exp called once per primitive either way.
  //
  // It factors because a spherical shell's relnorm is one number for the
  // whole shell: gto_shell.hpp sets every Cartesian component's relative
  // normalization to unity over sqrt(S[0]) (see its normalization notes,
  // step 3), so
  //
  //     chi_m = sum_c T[m][c] rho (x^i y^j z^k) R(r) = rho R(r) S_lm,
  //
  // and solid_harmonics_transform.hpp's T emits exactly the normalized real
  // solid harmonic r^l Y_lm that spherical_harmonics.hpp's precomputed
  // switch returns. Measured on benzene, this is worth 1.3x to 2x on values
  // and 1.8x to 4.3x on gradients, growing with l -- see
  // benchmarking/benchmark_collocation.cxx, which also checks that the two
  // routes agree up to that single scalar.
  //
  // Cartesian shells cannot take this path: a Cartesian Gaussian is not one
  // R(r) S_lm. Neither can derivatives above the gradient, because the
  // precomputed harmonic switches stop at value-and-gradient.
  if constexpr (Deriv == GTODeriv::Value || Deriv == GTODeriv::Gradient) {
    if (b.spherical(ishell)) {
      const double rho = b.relnorm(n0);
      double R = 0.0, dR_over_r = 0.0;
      for (int ip = 0; ip < np; ++ip) {
        const double zeta = b.alpha(p0 + ip);
        const double e = b.coeff(c0 + ip) * Kokkos::exp(-zeta * r2);
        R += e;
        // dR/dr = -2 zeta r e, so dR/dr / r needs no square root at all --
        // and the gradient only ever wants it divided by r.
        dR_over_r += -2.0 * zeta * e;
      }
      R *= rho;
      dR_over_r *= rho;

      const int nlm = 2 * l + 1;
      if constexpr (Deriv == GTODeriv::Value) {
        for (int m = 0; m < nlm; ++m) {
          double S;
          real_solid_harmonic_cart_precomputed(l, m - l, x, y, z, S);
          out[BF_VAL * out_stride + m] = R * S;
        }
      } else {
        for (int m = 0; m < nlm; ++m) {
          double S, dSx, dSy, dSz;
          real_solid_harmonic_cart_and_grad_precomputed(l, m - l, x, y, z, S,
                                                        dSx, dSy, dSz);
          out[BF_VAL * out_stride + m] = R * S;
          out[BF_DX * out_stride + m] = R * dSx + S * x * dR_over_r;
          out[BF_DY * out_stride + m] = R * dSy + S * y * dR_over_r;
          out[BF_DZ * out_stride + m] = R * dSz + S * z * dR_over_r;
        }
      }
      return;
    }
  }

  // Power tables to degree l + order.
  const int pmax = l + static_cast<int>(Deriv);
  double xr[GTO_MAX_POW], yr[GTO_MAX_POW], zr[GTO_MAX_POW];
  xr[0] = yr[0] = zr[0] = 1.0;
  for (int i = 1; i <= pmax; ++i) {
    xr[i] = xr[i - 1] * x;
    yr[i] = yr[i - 1] * y;
    zr[i] = zr[i - 1] * z;
  }

  // Cartesian accumulation buffer, component-major.
  double buf[NC * GTO_MAX_CART];
  for (int i = 0; i < NC * ncart; ++i)
    buf[i] = 0.0;

  for (int ip = 0; ip < np; ++ip) {
    const double zeta = b.alpha(p0 + ip);
    const double w = b.coeff(c0 + ip);
    const double e = w * Kokkos::exp(-zeta * r2);

    for (int ic = 0; ic < ncart; ++ic) {
      const int nx = b.cart_nx(k0 + ic);
      const int ny = b.cart_ny(k0 + ic);
      const int nz = b.cart_nz(k0 + ic);
      const double xl = xr[nx], ym = yr[ny], zn = zr[nz];

      buf[GTO_VAL * ncart + ic] += xl * ym * zn * e;

      if constexpr (do_grad) {
        const double d1x = gto_der1(xr, nx, zeta);
        const double d1y = gto_der1(yr, ny, zeta);
        const double d1z = gto_der1(zr, nz, zeta);
        buf[GTO_DX * ncart + ic] += d1x * ym * zn * e;
        buf[GTO_DY * ncart + ic] += xl * d1y * zn * e;
        buf[GTO_DZ * ncart + ic] += xl * ym * d1z * e;

        if constexpr (do_hess) {
          const double d2x = gto_der2(xr, nx, zeta);
          const double d2y = gto_der2(yr, ny, zeta);
          const double d2z = gto_der2(zr, nz, zeta);
          buf[GTO_DXX * ncart + ic] += d2x * ym * zn * e;
          buf[GTO_DYY * ncart + ic] += xl * d2y * zn * e;
          buf[GTO_DZZ * ncart + ic] += xl * ym * d2z * e;
          buf[GTO_DXY * ncart + ic] += d1x * d1y * zn * e;
          buf[GTO_DXZ * ncart + ic] += d1x * ym * d1z * e;
          buf[GTO_DYZ * ncart + ic] += xl * d1y * d1z * e;

          if constexpr (do_lgrad) {
            const double d3x = gto_der3(xr, nx, zeta);
            const double d3y = gto_der3(yr, ny, zeta);
            const double d3z = gto_der3(zr, nz, zeta);
            buf[GTO_LGX * ncart + ic] +=
                (d3x * ym * zn + d1x * d2y * zn + d1x * ym * d2z) * e;
            buf[GTO_LGY * ncart + ic] +=
                (d2x * d1y * zn + xl * d3y * zn + xl * d1y * d2z) * e;
            buf[GTO_LGZ * ncart + ic] +=
                (d2x * ym * d1z + xl * d2y * d1z + xl * ym * d3z) * e;
          }
        }
      }
    }
  }

  // Per-Cartesian relative normalization, shared across components.
  for (int ic = 0; ic < ncart; ++ic) {
    const double rn = b.relnorm(n0 + ic);
    for (int k = 0; k < NC; ++k)
      buf[k * ncart + ic] *= rn;
  }

  if (b.spherical(ishell)) {
    const int nlm = 2 * l + 1;
    const int t0 = b.tm_offset(l);
    for (int m = 0; m < nlm; ++m)
      for (int k = 0; k < NC; ++k) {
        double acc = 0.0;
        for (int ic = 0; ic < ncart; ++ic)
          acc += b.transmat(t0 + m * ncart + ic) * buf[k * ncart + ic];
        out[k * out_stride + m] = acc;
      }
  } else {
    for (int ic = 0; ic < ncart; ++ic)
      for (int k = 0; k < NC; ++k)
        out[k * out_stride + ic] = buf[k * ncart + ic];
  }
}

// ============================================================================
//  Collocation
// ============================================================================

// ============================================================================
//  BasisEvaluator adapter
// ============================================================================
//
// One contraction of one shell per block, so all 2l+1 spherical components
// come out of a single pass over the primitives and one Cartesian-to-spherical
// transform. Splitting the contractions across blocks keeps the per-thread
// Cartesian buffer at cart_size(l) rather than cart_size(l) * nctr, at the
// cost of re-evaluating the shell's primitive exponentials once per
// contraction.

class GTOEvaluator {
public:
  struct block_params {
    int ishell;
    int ictr;
  };
  static constexpr int max_block_size = GTO_MAX_CART;

  GTOEvaluator() = default;
  explicit GTOEvaluator(GTOBasisDevice b) : b_(std::move(b)) {}

  int nbf() const { return b_.nbf; }
  int nblock() const { return b_.nblock; }
  KOKKOS_INLINE_FUNCTION int block_first_bf(int i) const {
    return b_.block_first_bf(i);
  }
  KOKKOS_INLINE_FUNCTION int block_size(int i) const {
    return b_.block_nfunc(i);
  }

  Kokkos::View<Point *> centers() const { return b_.block_center; }
  Kokkos::View<double *> radii() const { return b_.block_radius; }

  const GTOBasisDevice &basis() const { return b_; }

  KOKKOS_INLINE_FUNCTION block_params load_block(int i) const {
    return block_params{b_.block_shell(i), b_.block_ctr(i)};
  }

  /// Highest derivative order eval_block implements. The Gaussian path
  /// is derived to arbitrary order, so this is the highest the tower
  /// layout names rather than a limit of the recursion.
  static constexpr BasisDeriv max_deriv = BasisDeriv::LaplGradient;

  template <BasisDeriv D>
  KOKKOS_INLINE_FUNCTION void eval_block(const block_params &p, double x,
                                         double y, double z, double *out,
                                         int stride) const {
    gto_eval_contraction<D>(b_, p.ishell, p.ictr, x, y, z, out, stride);
  }

private:
  GTOBasisDevice b_;
};

static_assert(BasisEvaluator<GTOEvaluator>);

/// Convenience wrapper: collocation for a GTO basis through the generic path.
template <GTODeriv Deriv, typename PointsView, typename CompView>
void fill_collocation_gto(ExecSpace space, const GTOBasisDevice &b,
                          PointsView points, CompView comp) {
  fill_collocation<Deriv>(space, GTOEvaluator(b), points, comp);
}

} // namespace Nukexc
