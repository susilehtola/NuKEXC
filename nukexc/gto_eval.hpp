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
#include "nukexc_config.hpp"
#include "solid_harmonics_transform.hpp"

namespace Nukexc {

/// Highest angular momentum the device evaluator supports. cart_size(7) = 36.
inline constexpr int GTO_MAX_L = 7;
/// Cartesian components at GTO_MAX_L, the bound on the per-thread buffers.
inline constexpr int GTO_MAX_CART = (GTO_MAX_L + 1) * (GTO_MAX_L + 2) / 2;
/// Power tables run to l + 3 for the highest supported derivative order.
inline constexpr int GTO_MAX_POW = GTO_MAX_L + 4;

/// Which derivatives to evaluate.
enum class GTODeriv { Value = 0, Gradient = 1, Hessian = 2, LaplGradient = 3 };

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

  int nshell = 0;
  int nbf = 0;
  int max_l = 0;
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

/// Number of output components carried by a derivative order:
/// value (1), gradient (+3), Hessian (+6 unique), gradient of the
/// Laplacian (+3).
KOKKOS_INLINE_FUNCTION
constexpr int gto_ncomp(GTODeriv d) {
  return d == GTODeriv::Value          ? 1
         : d == GTODeriv::Gradient     ? 4
         : d == GTODeriv::Hessian      ? 10
                                       : 13;
}

/// Component indices within the output block.
enum : int {
  GTO_VAL = 0,
  GTO_DX = 1, GTO_DY = 2, GTO_DZ = 3,
  GTO_DXX = 4, GTO_DXY = 5, GTO_DXZ = 6, GTO_DYY = 7, GTO_DYZ = 8, GTO_DZZ = 9,
  GTO_LGX = 10, GTO_LGY = 11, GTO_LGZ = 12
};

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

/**
 *  @brief Fill collocation matrices for a whole basis over a batch of points.
 *
 *  `comp` must have extent (gto_ncomp(Deriv), nbf, npts): component, basis
 *  function, point. Component 0 is the value; see the GTO_* indices above.
 *
 *  One thread per (shell, contraction, point). Splitting the contractions
 *  across threads keeps the per-thread Cartesian buffer at cart_size(l)
 *  rather than cart_size(l) * nctr, at the cost of re-evaluating the shell's
 *  primitive exponentials once per contraction.
 */
template <GTODeriv Deriv, typename PointsView, typename CompView>
void fill_collocation_gto(ExecSpace space, const GTOBasisDevice &b,
                          PointsView points, CompView comp) {
  constexpr int NC = gto_ncomp(Deriv);
  const int npts = static_cast<int>(points.extent(0));
  if (b.nshell == 0 || npts == 0)
    return;

  if (static_cast<int>(comp.extent(0)) != NC ||
      static_cast<int>(comp.extent(1)) != b.nbf ||
      static_cast<int>(comp.extent(2)) != npts)
    throw std::invalid_argument(
        "fill_collocation_gto: output must be (ncomp, nbf, npts)");

  // Flatten (shell, contraction) so every thread has equal-shaped work.
  auto nctr = b.nctr;
  auto bf_offset = b.bf_offset;
  auto l_v = b.l;
  auto sph = b.spherical;

  const int nsh = b.nshell;
  Kokkos::View<int *> task_shell("task_shell", 0), task_ctr("task_ctr", 0);
  {
    auto nctr_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, nctr);
    int ntask = 0;
    for (int s = 0; s < nsh; ++s)
      ntask += nctr_h(s);
    std::vector<int> hs(ntask), hc(ntask);
    int t = 0;
    for (int s = 0; s < nsh; ++s)
      for (int k = 0; k < nctr_h(s); ++k, ++t) {
        hs[t] = s;
        hc[t] = k;
      }
    task_shell = Kokkos::View<int *>("task_shell", ntask);
    task_ctr = Kokkos::View<int *>("task_ctr", ntask);
    Kokkos::View<const int *, HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        vs(hs.data(), ntask), vc(hc.data(), ntask);
    Kokkos::deep_copy(task_shell, vs);
    Kokkos::deep_copy(task_ctr, vc);
  }
  const int ntask = static_cast<int>(task_shell.extent(0));

  Kokkos::parallel_for(
      "fill_collocation_gto",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0},
                                                        {ntask, npts}),
      KOKKOS_LAMBDA(const int t, const int g) {
        const int s = task_shell(t);
        const int k = task_ctr(t);
        const int l = l_v(s);
        const int ncomp_out = sph(s) ? (2 * l + 1) : ((l + 1) * (l + 2) / 2);
        const int bf0 = bf_offset(s) + k * ncomp_out;

        // Worst-case scratch for one contraction: every component of the
        // largest supported shell. The value-only case needs 36 doubles; the
        // Laplacian-gradient case needs 13x that, which is where per-l
        // specialization would pay off on a GPU.
        double local[NC * GTO_MAX_CART];
        gto_eval_contraction<Deriv>(b, s, k, points(g)[0], points(g)[1],
                                    points(g)[2], local, GTO_MAX_CART);

        for (int c = 0; c < NC; ++c)
          for (int i = 0; i < ncomp_out; ++i)
            comp(c, bf0 + i, g) = local[c * GTO_MAX_CART + i];
      });
}

} // namespace Nukexc
