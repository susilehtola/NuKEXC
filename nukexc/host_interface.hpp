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
//  Host-facing entry points
// ============================================================================
//
// What an SCF program calls. Density matrices in, exchange-correlation energy
// and Fock contribution out:
//
//     restricted    eval_vxc(P)      -> {Exc, Nel, Vxc}
//     unrestricted  eval_vxc(Pa, Pb) -> {Exc, Nel, Vxc_a, Vxc_b}
//
// These correspond to ERKALE's DFTGrid::eval_Fxc, so its call sites translate
// directly, but the NAME follows libxckernel and the wider literature rather
// than ERKALE: this is the order-1 quantity, the XC Fock matrix, which is
// `vxc`. ERKALE's capital-F `Fxc` reads as "Fock" and is unambiguous in
// context, but lowercase `fxc` means the SECOND derivative everywhere else --
// GauXC, PySCF, the response literature, and libxckernel's own order
// indexing. The name fxc is therefore reserved here for order 2, which
// arrives with the response kernels in Phase 5.
//
// The interface is plain host pointers, not Kokkos Views and not Armadillo
// matrices: a caller should not have to adopt NuKEXC's memory model, or
// ERKALE's, to use it. Matrix layout is explicit and checked rather than
// assumed -- a density matrix is symmetric, so row- and column-major differ
// only in the leading dimension, but the perturbed and complex matrices of
// later phases are not, and silently getting this wrong there would be
// expensive to debug.
//
// Nel is returned alongside Exc because it is the standard grid-quality
// diagnostic: the deviation of the integrated density from the known electron
// count is what tells an SCF whether its grid is adequate, and every code
// that integrates on a grid reports it.

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basis_evaluator.hpp"
#include "density.hpp"
#include "grid.hpp"
#include "nukexc_config.hpp"
#include "xc_integrals.hpp"

namespace Nukexc {

/// Storage order of a host matrix handed across the interface.
enum class MatrixLayout {
  RowMajor,   ///< element (i,j) at data[i*ld + j]
  ColumnMajor ///< element (i,j) at data[i + j*ld]
};

/// A host matrix the caller owns. Nothing is retained past the call.
struct HostMatrixView {
  const double *data = nullptr;
  int ld = 0;
  MatrixLayout layout = MatrixLayout::ColumnMajor;
};

/// A host matrix NuKEXC writes into. The caller owns the storage.
struct HostMatrixRef {
  double *data = nullptr;
  int ld = 0;
  MatrixLayout layout = MatrixLayout::ColumnMajor;
};

/// Thresholds, with ERKALE's defaults so a port starts from the same place.
struct XCThresholds {
  /// Below this density a point contributes nothing. ERKALE: DFTDensityThr.
  double density = 1e-10;
  /// Basis-function amplitude cutoff. ERKALE: DFTBasisThr.
  double basis = 1e-10;
  /// Grid points with a smaller weight are dropped.
  double weight = 1e-15;
};

/// What an XC evaluation reports back.
struct XCOutput {
  double Exc = 0.0;
  /// Integrated electron count. The deviation from the true count is the
  /// grid-quality diagnostic every SCF wants to see.
  double Nel = 0.0;
  /// Points at which libxc returned a non-finite value. ERKALE zeroes these
  /// silently with a warning; NuKEXC counts them so a caller can decide.
  long nan_points = 0;
};

namespace detail {

/// Copy a caller matrix into a device View, normalizing the layout.
inline DeviceView2DLeft upload_matrix(const HostMatrixView &m, int n,
                                      const char *what) {
  if (!m.data)
    throw std::invalid_argument(std::string("eval_vxc: ") + what +
                                " data pointer is null");
  const int ld = m.ld > 0 ? m.ld : n;
  if (ld < n)
    throw std::invalid_argument(std::string("eval_vxc: ") + what +
                                " leading dimension " + std::to_string(ld) +
                                " is smaller than the matrix order " +
                                std::to_string(n));

  DeviceView2DLeft d(what, n, n);
  auto h = Kokkos::create_mirror_view(d);
  const bool row = m.layout == MatrixLayout::RowMajor;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      h(i, j) = row ? m.data[static_cast<size_t>(i) * ld + j]
                    : m.data[static_cast<size_t>(i) + static_cast<size_t>(j) * ld];
  Kokkos::deep_copy(d, h);
  return d;
}

/// Accumulate a device matrix into a caller matrix, honouring its layout.
inline void accumulate_matrix(const DeviceView2DLeft &src,
                              const HostMatrixRef &dst, int n,
                              const char *what) {
  if (!dst.data)
    throw std::invalid_argument(std::string("eval_vxc: ") + what +
                                " output pointer is null");
  const int ld = dst.ld > 0 ? dst.ld : n;
  if (ld < n)
    throw std::invalid_argument(std::string("eval_vxc: ") + what +
                                " leading dimension is smaller than the "
                                "matrix order");

  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, src);
  const bool row = dst.layout == MatrixLayout::RowMajor;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const size_t idx = row ? static_cast<size_t>(i) * ld + j
                             : static_cast<size_t>(i) +
                                   static_cast<size_t>(j) * ld;
      dst.data[idx] += h(i, j);
    }
}

/// Integrate a grid quantity against the weights.
inline double integrate(const DeviceView1D &f, const DeviceView1D &w) {
  const int n = static_cast<int>(w.extent(0));
  double acc = 0.0;
  Kokkos::parallel_reduce(
      "integrate", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int g, double &a) { a += w(g) * f(g); }, acc);
  return acc;
}


// ---------------------------------------------------------------------------
//  Shared machinery for the restricted, unrestricted and diagonal paths
// ---------------------------------------------------------------------------

/// Collocation for an XC build: values always, gradients when the functional
/// needs them.
struct XCCollocation {
  DeviceView2DLeft chi;
  DeviceView2DLeft dchi_x, dchi_y, dchi_z;
  bool gga = false;
};

template <typename Evaluator>
XCCollocation build_collocation(const Evaluator &basis, const FlatGrid &grid,
                                bool gga) {
  const int n = basis.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));

  XCCollocation c;
  c.gga = gga;
  c.chi = DeviceView2DLeft("chi", n, npts);
  if (!gga) {
    fill_collocation_values(ExecSpace{}, basis, grid.quad_points, c.chi);
    Kokkos::fence();
    return c;
  }

  c.dchi_x = DeviceView2DLeft("dchi_x", n, npts);
  c.dchi_y = DeviceView2DLeft("dchi_y", n, npts);
  c.dchi_z = DeviceView2DLeft("dchi_z", n, npts);

  Kokkos::View<double ***> comp("comp", basis_ncomp(BasisDeriv::Gradient), n,
                                npts);
  fill_collocation<BasisDeriv::Gradient>(ExecSpace{}, basis, grid.quad_points,
                                         comp);
  auto chi = c.chi, gx = c.dchi_x, gy = c.dchi_y, gz = c.dchi_z;
  Kokkos::parallel_for(
      "split gradient collocation",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, npts}),
      KOKKOS_LAMBDA(const int i, const int g) {
        chi(i, g) = comp(BF_VAL, i, g);
        gx(i, g) = comp(BF_DX, i, g);
        gy(i, g) = comp(BF_DY, i, g);
        gz(i, g) = comp(BF_DZ, i, g);
      });
  Kokkos::fence();
  return c;
}

/// Which spin case a functional was initialized for, checked once.
inline bool require_family(const xc_func_type &func, int nspin,
                           const char *who) {
  const bool gga = func.info->family == XC_FAMILY_GGA ||
                   func.info->family == XC_FAMILY_HYB_GGA;
  if (func.info->family != XC_FAMILY_LDA && !gga)
    throw std::runtime_error(
        std::string(who) +
        ": only LDA and GGA families are wired up on the density-matrix path "
        "so far");
  if (func.nspin != nspin)
    throw std::runtime_error(
        std::string(who) + ": functional was initialized for " +
        (func.nspin == XC_POLARIZED ? "XC_POLARIZED" : "XC_UNPOLARIZED") +
        " but this entry point needs the other one");
  return gga;
}

/// Index of the points worth handing to libxc, i.e. those above the density
/// threshold. Skipping the rest is both faster and avoids asking for exc at
/// rho ~ 0.
inline std::vector<int> significant_points(const std::vector<double> &rho_tot,
                                           double thr) {
  std::vector<int> idx;
  idx.reserve(rho_tot.size());
  for (size_t g = 0; g < rho_tot.size(); ++g)
    if (rho_tot[g] > thr)
      idx.push_back(static_cast<int>(g));
  return idx;
}

/// Copy a host vector to the device.
inline DeviceView1D to_device(const std::vector<double> &v, const char *name) {
  DeviceView1D d(name, v.size());
  Kokkos::deep_copy(
      d, Kokkos::View<const double *, HostSpace,
                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v.data(),
                                                               v.size()));
  return d;
}

/// Copy a device vector to the host.
inline std::vector<double> to_host(const DeviceView1D &d) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, d);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}

/// V = Phi Z^T + (Phi Z^T)^T, the symmetrized Fock contribution.
inline DeviceView2DLeft symmetrized_gemm(const DeviceView2DLeft &chi,
                                         const DeviceView2DLeft &Z, int n) {
  DeviceView2DLeft V("V", n, n), Vsym("Vsym", n, n);
  KokkosBlas::gemm(ExecSpace{}, "N", "T", 1.0, chi, Z, 0.0, V);
  Kokkos::parallel_for(
      "symmetrize",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, n}),
      KOKKOS_LAMBDA(const int i, const int j) {
        Vsym(i, j) = V(i, j) + V(j, i);
      });
  Kokkos::fence();
  return Vsym;
}

/// diag(Phi Z^T + transpose)_i = 2 sum_g chi_i(g) Z_i(g), without forming
/// the full matrix -- the point of the diagonal-only entry point.
inline std::vector<double> symmetrized_diagonal(const DeviceView2DLeft &chi,
                                                const DeviceView2DLeft &Z,
                                                int n) {
  DeviceView1D d("diag", n);
  const int npts = static_cast<int>(chi.extent(1));
  Kokkos::parallel_for(
      "diagonal", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i) {
        double acc = 0.0;
        for (int g = 0; g < npts; ++g)
          acc += chi(i, g) * Z(i, g);
        d(i) = 2.0 * acc;
      });
  Kokkos::fence();
  return to_host(d);
}

} // namespace detail

/**
 *  @brief Integrated electron count for a density matrix.
 *
 *  ERKALE's DFTGrid::compute_Nel. Cheap relative to a full XC build, and the
 *  usual way to check a grid before trusting it.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value>
double compute_nel(const Evaluator &basis, const FlatGrid &grid,
                   const HostMatrixView &P) {
  const int n = basis.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));

  DeviceView2DLeft coll("collocation", n, npts);
  fill_collocation_values(ExecSpace{}, basis, grid.quad_points, coll);

  const auto Pd = detail::upload_matrix(P, n, "P");
  const DeviceView1D rho = compute_density_dm(coll, Pd);
  Kokkos::fence();
  return detail::integrate(rho, grid.weights);
}

/**
 *  @brief Restricted XC energy and Fock contribution from a density matrix.
 *
 *  Mirrors ERKALE's
 *      DFTGrid::eval_Fxc(int x, int c, const arma::mat& P, arma::mat& H,
 *                        double& Exc, double& Nel)
 *  with the functional supplied as an initialized libxc handle rather than
 *  two integer IDs, so the caller controls external parameters and the
 *  functional is initialized once per SCF rather than once per call.
 *
 *  @param Vxc accumulated into, matching ERKALE's convention that H is added
 *             to rather than overwritten.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
XCOutput eval_vxc(const Evaluator &basis, const FlatGrid &grid,
                  const HostMatrixView &P, const HostMatrixRef &Vxc,
                  const xc_func_type &func,
                  const XCThresholds &thr = XCThresholds{}) {
  const int n = basis.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  if (n == 0 || npts == 0)
    return XCOutput{};

  const bool gga = func.info->family == XC_FAMILY_GGA ||
                   func.info->family == XC_FAMILY_HYB_GGA;
  if (func.info->family != XC_FAMILY_LDA && !gga)
    throw std::runtime_error(
        "eval_vxc: only LDA and GGA families are wired up on the "
        "density-matrix path so far");
  if (func.nspin != XC_UNPOLARIZED)
    throw std::runtime_error(
        "eval_vxc(P) needs an XC_UNPOLARIZED functional; use the "
        "(Pa, Pb) overload for XC_POLARIZED");

  const auto Pd = detail::upload_matrix(P, n, "P");

  DeviceView2DLeft coll("collocation", n, npts);
  DeviceView2DLeft gx("dx", n, npts), gy("dy", n, npts), gz("dz", n, npts);
  if (gga) {
    Kokkos::View<double ***> comp("comp", basis_ncomp(BasisDeriv::Gradient), n,
                                  npts);
    fill_collocation<BasisDeriv::Gradient>(ExecSpace{}, basis,
                                           grid.quad_points, comp);
    Kokkos::parallel_for(
        "split gradient collocation",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, npts}),
        KOKKOS_LAMBDA(const int i, const int g) {
          coll(i, g) = comp(BF_VAL, i, g);
          gx(i, g) = comp(BF_DX, i, g);
          gy(i, g) = comp(BF_DY, i, g);
          gz(i, g) = comp(BF_DZ, i, g);
        });
  } else {
    fill_collocation_values(ExecSpace{}, basis, grid.quad_points, coll);
  }
  Kokkos::fence();

  // The existing XC kernels take MO coefficients and occupations. Rather than
  // duplicate them, feed the density matrix through as a single "orbital"
  // whose outer product is P: that is what the Cholesky-like factorization
  // below would cost, so instead compute rho directly and hand the kernels
  // the grid quantities they need.
  DeviceView1D rho("rho", npts), gxr("gxr", npts), gyr("gyr", npts),
      gzr("gzr", npts), sigma("sigma", npts);
  if (gga)
    compute_density_and_sigma_dm(coll, gx, gy, gz, Pd, rho, gxr, gyr, gzr,
                                 sigma);
  else
    rho = compute_density_dm(coll, Pd);
  Kokkos::fence();

  XCOutput out;
  out.Nel = detail::integrate(rho, grid.weights);

  // libxc on the host, for now. Phase 4 moves this to the device once libxc 8
  // exposes its GPU seam; the round trip is the single largest avoidable cost
  // in the current SCF loop.
  auto rho_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);
  auto sig_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma);
  std::vector<double> exc(npts, 0.0), vrho(npts, 0.0), vsigma(npts, 0.0);

  // Points below the density threshold are skipped rather than handed to
  // libxc, which is both faster and avoids asking for exc at rho ~ 0.
  std::vector<double> rho_in, sig_in;
  std::vector<int> idx;
  rho_in.reserve(npts);
  idx.reserve(npts);
  for (int g = 0; g < npts; ++g)
    if (rho_h(g) > thr.density) {
      idx.push_back(g);
      rho_in.push_back(rho_h(g));
      if (gga)
        sig_in.push_back(sig_h(g));
    }

  const int nin = static_cast<int>(idx.size());
  if (nin) {
    std::vector<double> e(nin), vr(nin), vs(gga ? nin : 0);
    if (gga)
      xc_gga_exc_vxc(&func, nin, rho_in.data(), sig_in.data(), e.data(),
                     vr.data(), vs.data());
    else
      xc_lda_exc_vxc(&func, nin, rho_in.data(), e.data(), vr.data());

    for (int k = 0; k < nin; ++k) {
      const bool ok = std::isfinite(e[k]) && std::isfinite(vr[k]) &&
                      (!gga || std::isfinite(vs[k]));
      if (!ok) {
        ++out.nan_points;
        continue;
      }
      exc[idx[k]] = e[k];
      vrho[idx[k]] = vr[k];
      if (gga)
        vsigma[idx[k]] = vs[k];
    }
  }

  DeviceView1D exc_d("exc", npts), vrho_d("vrho", npts),
      vsigma_d("vsigma", npts);
  Kokkos::deep_copy(exc_d, Kokkos::View<const double *, HostSpace,
                                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                               exc.data(), npts));
  Kokkos::deep_copy(vrho_d,
                    Kokkos::View<const double *, HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        vrho.data(), npts));
  Kokkos::deep_copy(vsigma_d,
                    Kokkos::View<const double *, HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        vsigma.data(), npts));

  // Exc = integral w rho exc
  {
    DeviceView1D integrand("rho exc", npts);
    Kokkos::parallel_for(
        "rho exc", Kokkos::RangePolicy<ExecSpace>(0, npts),
        KOKKOS_LAMBDA(const int g) { integrand(g) = rho(g) * exc_d(g); });
    out.Exc = detail::integrate(integrand, grid.weights);
  }

  // Z_u(g) = w [ 1/2 vrho phi_u + 2 vsigma (grad rho . grad phi_u) ],
  // then V = Phi Z^T + (Phi Z^T)^T.
  DeviceView2DLeft Z("Z", n, npts), V("V", n, n), Vsym("Vsym", n, n);
  auto w = grid.weights;
  Kokkos::parallel_for(
      "Z matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, npts}),
      KOKKOS_LAMBDA(const int u, const int g) {
        double z = 0.5 * coll(u, g) * vrho_d(g);
        if (gga)
          z += 2.0 * vsigma_d(g) *
               (gxr(g) * gx(u, g) + gyr(g) * gy(u, g) + gzr(g) * gz(u, g));
        Z(u, g) = w(g) * z;
      });
  KokkosBlas::gemm(ExecSpace{}, "N", "T", 1.0, coll, Z, 0.0, V);
  Kokkos::parallel_for(
      "symmetrize",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, n}),
      KOKKOS_LAMBDA(const int i, const int j) {
        Vsym(i, j) = V(i, j) + V(j, i);
      });
  Kokkos::fence();

  detail::accumulate_matrix(Vsym, Vxc, n, "Vxc");
  return out;
}

/// Unrestricted result: one Fock contribution per spin channel.
struct XCOutputPolarized {
  double Exc = 0.0;
  double Nel = 0.0;      ///< total, both channels
  double Nel_alpha = 0.0;
  double Nel_beta = 0.0;
  long nan_points = 0;
};

/**
 *  @brief Unrestricted XC energy and Fock contributions from spin densities.
 *
 *  ERKALE's
 *      DFTGrid::eval_Fxc(int x, int c, const arma::mat& Pa, const arma::mat& Pb,
 *                        arma::mat& Ha, arma::mat& Hb, double& Exc, double& Nel)
 *
 *  Both Vxc_a and Vxc_b are accumulated into, matching that convention.
 *
 *  libxc's polarized layout is interleaved per point: rho as (a, b), sigma as
 *  (aa, ab, bb), and likewise vrho and vsigma. The interleaving is done here
 *  rather than pushed onto the caller.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
XCOutputPolarized eval_vxc(const Evaluator &basis, const FlatGrid &grid,
                           const HostMatrixView &Pa, const HostMatrixView &Pb,
                           const HostMatrixRef &Vxc_a,
                           const HostMatrixRef &Vxc_b,
                           const xc_func_type &func,
                           const XCThresholds &thr = XCThresholds{}) {
  const int n = basis.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  if (n == 0 || npts == 0)
    return XCOutputPolarized{};

  const bool gga = detail::require_family(func, XC_POLARIZED, "eval_vxc(Pa,Pb)");

  const auto Pad = detail::upload_matrix(Pa, n, "Pa");
  const auto Pbd = detail::upload_matrix(Pb, n, "Pb");
  const auto c = detail::build_collocation(basis, grid, gga);

  DeviceView1D ra("rho_a", npts), rb("rho_b", npts);
  DeviceView1D gax("gax", npts), gay("gay", npts), gaz("gaz", npts);
  DeviceView1D gbx("gbx", npts), gby("gby", npts), gbz("gbz", npts);
  DeviceView1D saa("saa", npts), sbb("sbb", npts), sab("sab", npts);

  if (gga) {
    compute_density_and_sigma_dm(c.chi, c.dchi_x, c.dchi_y, c.dchi_z, Pad, ra,
                                 gax, gay, gaz, saa);
    compute_density_and_sigma_dm(c.chi, c.dchi_x, c.dchi_y, c.dchi_z, Pbd, rb,
                                 gbx, gby, gbz, sbb);
    Kokkos::parallel_for(
        "sigma_ab", Kokkos::RangePolicy<ExecSpace>(0, npts),
        KOKKOS_LAMBDA(const int g) {
          sab(g) = gax(g) * gbx(g) + gay(g) * gby(g) + gaz(g) * gbz(g);
        });
  } else {
    ra = compute_density_dm(c.chi, Pad);
    rb = compute_density_dm(c.chi, Pbd);
  }
  Kokkos::fence();

  XCOutputPolarized out;
  out.Nel_alpha = detail::integrate(ra, grid.weights);
  out.Nel_beta = detail::integrate(rb, grid.weights);
  out.Nel = out.Nel_alpha + out.Nel_beta;

  const auto ra_h = detail::to_host(ra), rb_h = detail::to_host(rb);
  const auto saa_h = detail::to_host(saa), sab_h = detail::to_host(sab),
             sbb_h = detail::to_host(sbb);

  std::vector<double> rho_tot(npts);
  for (int g = 0; g < npts; ++g)
    rho_tot[g] = ra_h[g] + rb_h[g];
  const auto idx = detail::significant_points(rho_tot, thr.density);
  const int nin = static_cast<int>(idx.size());

  std::vector<double> exc(npts, 0.0);
  std::vector<double> vra(npts, 0.0), vrb(npts, 0.0);
  std::vector<double> vsaa(npts, 0.0), vsab(npts, 0.0), vsbb(npts, 0.0);

  if (nin) {
    std::vector<double> rin(2 * nin), sin_(gga ? 3 * nin : 0);
    for (int k = 0; k < nin; ++k) {
      rin[2 * k] = ra_h[idx[k]];
      rin[2 * k + 1] = rb_h[idx[k]];
      if (gga) {
        sin_[3 * k] = saa_h[idx[k]];
        sin_[3 * k + 1] = sab_h[idx[k]];
        sin_[3 * k + 2] = sbb_h[idx[k]];
      }
    }
    std::vector<double> e(nin), vr(2 * nin), vs(gga ? 3 * nin : 0);
    if (gga)
      xc_gga_exc_vxc(&func, nin, rin.data(), sin_.data(), e.data(), vr.data(),
                     vs.data());
    else
      xc_lda_exc_vxc(&func, nin, rin.data(), e.data(), vr.data());

    for (int k = 0; k < nin; ++k) {
      bool ok = std::isfinite(e[k]) && std::isfinite(vr[2 * k]) &&
                std::isfinite(vr[2 * k + 1]);
      if (gga)
        ok = ok && std::isfinite(vs[3 * k]) && std::isfinite(vs[3 * k + 1]) &&
             std::isfinite(vs[3 * k + 2]);
      if (!ok) {
        ++out.nan_points;
        continue;
      }
      const int g = idx[k];
      exc[g] = e[k];
      vra[g] = vr[2 * k];
      vrb[g] = vr[2 * k + 1];
      if (gga) {
        vsaa[g] = vs[3 * k];
        vsab[g] = vs[3 * k + 1];
        vsbb[g] = vs[3 * k + 2];
      }
    }
  }

  const auto exc_d = detail::to_device(exc, "exc");
  {
    DeviceView1D integrand("rho exc", npts);
    Kokkos::parallel_for(
        "rho exc", Kokkos::RangePolicy<ExecSpace>(0, npts),
        KOKKOS_LAMBDA(const int g) {
          integrand(g) = (ra(g) + rb(g)) * exc_d(g);
        });
    out.Exc = detail::integrate(integrand, grid.weights);
  }

  // Z_a,i = w [ 1/2 vrho_a chi_i
  //             + (2 vsigma_aa grad rho_a + vsigma_ab grad rho_b) . grad chi_i ]
  // and the mirror for beta.
  auto build = [&](const DeviceView1D &vr_s, const DeviceView1D &vs_ss,
                   const DeviceView1D &vs_os, const DeviceView1D &gsx,
                   const DeviceView1D &gsy, const DeviceView1D &gsz,
                   const DeviceView1D &gox, const DeviceView1D &goy,
                   const DeviceView1D &goz) {
    DeviceView2DLeft Z("Z", n, npts);
    auto w = grid.weights;
    auto chi = c.chi, dx = c.dchi_x, dy = c.dchi_y, dz = c.dchi_z;
    const bool is_gga = gga;
    Kokkos::parallel_for(
        "Z matrix",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, npts}),
        KOKKOS_LAMBDA(const int u, const int g) {
          double z = 0.5 * chi(u, g) * vr_s(g);
          if (is_gga) {
            const double vx = 2.0 * vs_ss(g) * gsx(g) + vs_os(g) * gox(g);
            const double vy = 2.0 * vs_ss(g) * gsy(g) + vs_os(g) * goy(g);
            const double vz = 2.0 * vs_ss(g) * gsz(g) + vs_os(g) * goz(g);
            z += vx * dx(u, g) + vy * dy(u, g) + vz * dz(u, g);
          }
          Z(u, g) = w(g) * z;
        });
    return Z;
  };

  const auto vra_d = detail::to_device(vra, "vrho_a");
  const auto vrb_d = detail::to_device(vrb, "vrho_b");
  const auto vsaa_d = detail::to_device(vsaa, "vsigma_aa");
  const auto vsab_d = detail::to_device(vsab, "vsigma_ab");
  const auto vsbb_d = detail::to_device(vsbb, "vsigma_bb");

  const auto Za = build(vra_d, vsaa_d, vsab_d, gax, gay, gaz, gbx, gby, gbz);
  const auto Zb = build(vrb_d, vsbb_d, vsab_d, gbx, gby, gbz, gax, gay, gaz);

  detail::accumulate_matrix(detail::symmetrized_gemm(c.chi, Za, n), Vxc_a, n,
                            "Vxc_a");
  detail::accumulate_matrix(detail::symmetrized_gemm(c.chi, Zb, n), Vxc_b, n,
                            "Vxc_b");
  return out;
}

/**
 *  @brief Diagonal of the restricted XC Fock contribution.
 *
 *  ERKALE's DFTGrid::eval_diag_Fxc, which the adaptive grid uses: the
 *  convergence criterion looks at max_u |Delta F_uu|, so forming the full
 *  matrix would be wasted work. This skips the GEMM entirely and contracts
 *  only the diagonal, which is O(nbf * npts) rather than O(nbf^2 * npts).
 *
 *  @param diag accumulated into, length nbf.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
XCOutput eval_diag_vxc(const Evaluator &basis, const FlatGrid &grid,
                       const HostMatrixView &P, double *diag,
                       const xc_func_type &func,
                       const XCThresholds &thr = XCThresholds{}) {
  const int n = basis.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  if (!diag)
    throw std::invalid_argument("eval_diag_vxc: diagonal pointer is null");
  if (n == 0 || npts == 0)
    return XCOutput{};

  const bool gga = detail::require_family(func, XC_UNPOLARIZED,
                                          "eval_diag_vxc");
  const auto Pd = detail::upload_matrix(P, n, "P");
  const auto c = detail::build_collocation(basis, grid, gga);

  DeviceView1D rho("rho", npts), gx("gx", npts), gy("gy", npts),
      gz("gz", npts), sigma("sigma", npts);
  if (gga)
    compute_density_and_sigma_dm(c.chi, c.dchi_x, c.dchi_y, c.dchi_z, Pd, rho,
                                 gx, gy, gz, sigma);
  else
    rho = compute_density_dm(c.chi, Pd);
  Kokkos::fence();

  XCOutput out;
  out.Nel = detail::integrate(rho, grid.weights);

  const auto rho_h = detail::to_host(rho), sig_h = detail::to_host(sigma);
  const auto idx = detail::significant_points(rho_h, thr.density);
  const int nin = static_cast<int>(idx.size());

  std::vector<double> exc(npts, 0.0), vrho(npts, 0.0), vsigma(npts, 0.0);
  if (nin) {
    std::vector<double> rin(nin), sin_(gga ? nin : 0);
    for (int k = 0; k < nin; ++k) {
      rin[k] = rho_h[idx[k]];
      if (gga)
        sin_[k] = sig_h[idx[k]];
    }
    std::vector<double> e(nin), vr(nin), vs(gga ? nin : 0);
    if (gga)
      xc_gga_exc_vxc(&func, nin, rin.data(), sin_.data(), e.data(), vr.data(),
                     vs.data());
    else
      xc_lda_exc_vxc(&func, nin, rin.data(), e.data(), vr.data());
    for (int k = 0; k < nin; ++k) {
      const bool ok = std::isfinite(e[k]) && std::isfinite(vr[k]) &&
                      (!gga || std::isfinite(vs[k]));
      if (!ok) {
        ++out.nan_points;
        continue;
      }
      exc[idx[k]] = e[k];
      vrho[idx[k]] = vr[k];
      if (gga)
        vsigma[idx[k]] = vs[k];
    }
  }

  const auto exc_d = detail::to_device(exc, "exc");
  {
    DeviceView1D integrand("rho exc", npts);
    Kokkos::parallel_for(
        "rho exc", Kokkos::RangePolicy<ExecSpace>(0, npts),
        KOKKOS_LAMBDA(const int g) { integrand(g) = rho(g) * exc_d(g); });
    out.Exc = detail::integrate(integrand, grid.weights);
  }

  const auto vrho_d = detail::to_device(vrho, "vrho");
  const auto vsig_d = detail::to_device(vsigma, "vsigma");

  DeviceView2DLeft Z("Z", n, npts);
  {
    auto w = grid.weights;
    auto chi = c.chi, dx = c.dchi_x, dy = c.dchi_y, dz = c.dchi_z;
    const bool is_gga = gga;
    Kokkos::parallel_for(
        "Z matrix",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, npts}),
        KOKKOS_LAMBDA(const int u, const int g) {
          double z = 0.5 * chi(u, g) * vrho_d(g);
          if (is_gga)
            z += 2.0 * vsig_d(g) *
                 (gx(g) * dx(u, g) + gy(g) * dy(u, g) + gz(g) * dz(u, g));
          Z(u, g) = w(g) * z;
        });
  }

  const auto d = detail::symmetrized_diagonal(c.chi, Z, n);
  for (int i = 0; i < n; ++i)
    diag[i] += d[static_cast<size_t>(i)];
  return out;
}

/// Complex orbital coefficients in split real/imaginary storage.
///
/// C_im may be null for real orbitals, which is then handled without the
/// extra GEMMs rather than by multiplying zeros.
struct OrbitalsView {
  const double *C_re = nullptr;
  const double *C_im = nullptr; ///< null for real orbitals
  int ld = 0;
  MatrixLayout layout = MatrixLayout::ColumnMajor;
  int norb = 0;
};

/// Per-orbital XC results, one entry per orbital.
struct OrbitalXCOutput {
  std::vector<double> Exc;
  std::vector<double> Nel;
  long nan_points = 0;
};

/**
 *  @brief XC energy and Fock contribution for EACH orbital separately.
 *
 *  This is the Perdew-Zunger self-interaction correction path, corresponding
 *  to ERKALE's
 *      DFTGrid::eval_Fxc(int x, int c, const arma::cx_mat& C,
 *                        std::vector<arma::mat>& H,
 *                        std::vector<double>& Exc,
 *                        std::vector<double>& Nel, bool fock)
 *
 *  Each orbital is treated as its own spin-polarized system with
 *  rho_alpha = |psi_i|^2 and rho_beta = 0, which is what PZ-SIC prescribes,
 *  so the functional must be XC_POLARIZED.
 *
 *  The whole point of a dedicated entry point rather than a loop over
 *  eval_vxc is that the collocation is built ONCE and reused across all
 *  orbitals. That is the dominant cost, and PZ-SIC needs N_occ Fock builds.
 *
 *  No complex arithmetic appears: with real basis functions rho_i is real
 *  and the resulting potential is real symmetric, so split Re/Im storage and
 *  real GEMMs suffice. See the note in density.hpp.
 *
 *  @param Vxc one accumulated-into matrix per orbital, or empty to compute
 *             energies only (ERKALE's `fock` flag).
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
OrbitalXCOutput eval_vxc_orbitals(const Evaluator &basis, const FlatGrid &grid,
                                  const OrbitalsView &orbitals,
                                  const std::vector<HostMatrixRef> &Vxc,
                                  const xc_func_type &func,
                                  const XCThresholds &thr = XCThresholds{}) {
  const int n = basis.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  const int norb = orbitals.norb;
  if (n == 0 || npts == 0 || norb == 0)
    return OrbitalXCOutput{};
  if (!orbitals.C_re)
    throw std::invalid_argument("eval_vxc_orbitals: C_re is null");
  if (!Vxc.empty() && static_cast<int>(Vxc.size()) != norb)
    throw std::invalid_argument(
        "eval_vxc_orbitals: expected one output matrix per orbital, or none");

  const bool gga = detail::require_family(func, XC_POLARIZED,
                                          "eval_vxc_orbitals");

  // Upload coefficients, normalizing the layout.
  const int ld = orbitals.ld > 0 ? orbitals.ld : n;
  auto upload_coeff = [&](const double *src, const char *what) {
    DeviceView2DLeft d(what, n, norb);
    auto h = Kokkos::create_mirror_view(d);
    if (src) {
      const bool row = orbitals.layout == MatrixLayout::RowMajor;
      for (int u = 0; u < n; ++u)
        for (int i = 0; i < norb; ++i)
          h(u, i) = row ? src[static_cast<size_t>(u) * ld + i]
                        : src[static_cast<size_t>(u) +
                              static_cast<size_t>(i) * ld];
    } else {
      Kokkos::deep_copy(h, 0.0);
    }
    Kokkos::deep_copy(d, h);
    return d;
  };
  const auto Cre = upload_coeff(orbitals.C_re, "C_re");
  const auto Cim = upload_coeff(orbitals.C_im, "C_im");

  const auto c = detail::build_collocation(basis, grid, gga);

  DeviceView2D rho("rho_i", norb, npts), gx("gx_i", norb, npts),
      gy("gy_i", norb, npts), gz("gz_i", norb, npts),
      sigma("sigma_i", norb, npts);
  if (gga)
    compute_orbital_densities_and_sigma(c.chi, c.dchi_x, c.dchi_y, c.dchi_z,
                                        Cre, Cim, rho, gx, gy, gz, sigma);
  else
    rho = compute_orbital_densities(c.chi, Cre, Cim);
  Kokkos::fence();

  auto rho_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);
  auto sig_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma);

  OrbitalXCOutput out;
  out.Exc.assign(norb, 0.0);
  out.Nel.assign(norb, 0.0);

  auto w_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, grid.weights);

  for (int i = 0; i < norb; ++i) {
    // PZ-SIC: each orbital is its own fully polarized system,
    // rho_alpha = |psi_i|^2, rho_beta = 0.
    std::vector<double> ri(npts);
    for (int g = 0; g < npts; ++g)
      ri[g] = rho_h(i, g);

    double nel = 0.0;
    for (int g = 0; g < npts; ++g)
      nel += w_h(g) * ri[g];
    out.Nel[i] = nel;

    const auto idx = detail::significant_points(ri, thr.density);
    const int nin = static_cast<int>(idx.size());

    std::vector<double> exc(npts, 0.0), vra(npts, 0.0), vsaa(npts, 0.0);
    if (nin) {
      std::vector<double> rin(2 * nin, 0.0), sin_(gga ? 3 * nin : 0, 0.0);
      for (int k = 0; k < nin; ++k) {
        rin[2 * k] = ri[idx[k]]; // beta channel stays zero
        if (gga)
          sin_[3 * k] = sig_h(i, idx[k]); // sigma_ab and sigma_bb vanish
      }
      std::vector<double> e(nin), vr(2 * nin), vs(gga ? 3 * nin : 0);
      if (gga)
        xc_gga_exc_vxc(&func, nin, rin.data(), sin_.data(), e.data(),
                       vr.data(), vs.data());
      else
        xc_lda_exc_vxc(&func, nin, rin.data(), e.data(), vr.data());

      for (int k = 0; k < nin; ++k) {
        const bool ok = std::isfinite(e[k]) && std::isfinite(vr[2 * k]) &&
                        (!gga || std::isfinite(vs[3 * k]));
        if (!ok) {
          ++out.nan_points;
          continue;
        }
        exc[idx[k]] = e[k];
        vra[idx[k]] = vr[2 * k];
        if (gga)
          vsaa[idx[k]] = vs[3 * k];
      }
    }

    for (int g = 0; g < npts; ++g)
      out.Exc[i] += w_h(g) * ri[g] * exc[g];

    if (Vxc.empty())
      continue;

    const auto vra_d = detail::to_device(vra, "vrho_i");
    const auto vsaa_d = detail::to_device(vsaa, "vsigma_i");

    DeviceView2DLeft Z("Z", n, npts);
    auto wv = grid.weights;
    auto chi = c.chi, dx = c.dchi_x, dy = c.dchi_y, dz = c.dchi_z;
    const bool is_gga = gga;
    const int iorb = i;
    Kokkos::parallel_for(
        "Z matrix (orbital)",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, npts}),
        KOKKOS_LAMBDA(const int u, const int g) {
          double z = 0.5 * chi(u, g) * vra_d(g);
          if (is_gga)
            z += 2.0 * vsaa_d(g) *
                 (gx(iorb, g) * dx(u, g) + gy(iorb, g) * dy(u, g) +
                  gz(iorb, g) * dz(u, g));
          Z(u, g) = wv(g) * z;
        });

    detail::accumulate_matrix(detail::symmetrized_gemm(c.chi, Z, n), Vxc[i], n,
                              "Vxc[i]");
  }

  return out;
}

// ============================================================================
//  Functional evaluation as a seam
// ============================================================================
//
// Who evaluates the functional is deliberately not fixed.
//
//   * The HOST evaluates it and passes the derivative arrays in. No libxc in
//     NuKEXC's interface, no exposure to its API changing, and the host keeps
//     whatever functional handling it already has -- which for a code being
//     ported is all of it.
//
//   * NuKEXC evaluates it. This is what device residency wants: once libxc
//     exposes a GPU seam the arrays never leave the device, and the host
//     round trip in the convenience overloads below -- currently the largest
//     avoidable cost in the SCF loop -- disappears.
//
// Both drive the same assembly, so choosing one is not a commitment.
//
// The array route is inherently two-sided: the caller cannot produce vrho
// without rho, and rho needs the collocation. XCSession holds that state so
// the collocation and density are computed once rather than once per call.

/// Functional derivatives at the grid points, in libxc's layouts -- which is
/// what any caller already has, whoever computed them.
///
///   unpolarized   exc[npts], vrho[npts], vsigma[npts]
///   polarized     exc[npts], vrho[2*npts] as (a,b),
///                            vsigma[3*npts] as (aa,ab,bb)
///
/// vsigma is null for an LDA. exc is the energy density PER PARTICLE, so the
/// energy is integral w rho exc.
struct XCDerivativesView {
  const double *exc = nullptr;
  const double *vrho = nullptr;
  const double *vsigma = nullptr;
};

/**
 *  @brief Collocation and density held across a functional evaluation.
 *
 *  Usage, restricted:
 *
 *      XCSession s(basis, grid, needs_gradients);
 *      s.set_density(P);
 *      // caller evaluates the functional at s.rho() / s.sigma()
 *      auto out = s.accumulate_vxc(derivatives, Vxc);
 *
 *  The collocation is built once in the constructor and reused, so a caller
 *  sweeping several functionals over one density, or one functional over
 *  several densities, pays for it once.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
class XCSession {
public:
  XCSession(const Evaluator &basis, const FlatGrid &grid, bool need_gradients)
      : basis_(basis), grid_(grid), gga_(need_gradients),
        n_(basis.nbf()),
        npts_(static_cast<int>(grid.quad_points.extent(0))),
        coll_(detail::build_collocation(basis, grid, need_gradients)) {}

  int nbf() const { return n_; }
  int npoints() const { return npts_; }
  bool gradients() const { return gga_; }

  /// Contract a density matrix onto the grid. Must precede accumulate_vxc.
  void set_density(const HostMatrixView &P) {
    const auto Pd = detail::upload_matrix(P, n_, "P");
    rho_d_ = DeviceView1D("rho", npts_);
    gx_ = DeviceView1D("gx", npts_);
    gy_ = DeviceView1D("gy", npts_);
    gz_ = DeviceView1D("gz", npts_);
    sigma_d_ = DeviceView1D("sigma", npts_);
    if (gga_)
      compute_density_and_sigma_dm(coll_.chi, coll_.dchi_x, coll_.dchi_y,
                                   coll_.dchi_z, Pd, rho_d_, gx_, gy_, gz_,
                                   sigma_d_);
    else
      rho_d_ = compute_density_dm(coll_.chi, Pd);
    Kokkos::fence();
    rho_h_ = detail::to_host(rho_d_);
    sigma_h_ = detail::to_host(sigma_d_);
    has_density_ = true;
  }

  /// The ingredients the functional needs, on the host.
  const std::vector<double> &rho() const { return rho_h_; }
  const std::vector<double> &sigma() const { return sigma_h_; }
  /// Grid weights, so the caller can integrate its own quantities.
  const std::vector<double> &weights() const {
    if (w_h_.empty())
      w_h_ = detail::to_host(grid_.weights);
    return w_h_;
  }

  /// Integrated electron count for the density currently set.
  double nel() const {
    require_density("nel");
    return detail::integrate(rho_d_, grid_.weights);
  }

  /**
   *  @brief Assemble Exc and the Fock contribution from supplied derivatives.
   *
   *  @param d    functional derivatives, length npoints()
   *  @param Vxc  accumulated into; pass a null data pointer for energy only
   */
  XCOutput accumulate_vxc(const XCDerivativesView &d,
                          const HostMatrixRef &Vxc) {
    require_density("accumulate_vxc");
    if (!d.exc || !d.vrho)
      throw std::invalid_argument(
          "accumulate_vxc: exc and vrho are required");
    if (gga_ && !d.vsigma)
      throw std::invalid_argument(
          "accumulate_vxc: vsigma is required for a GGA session");

    XCOutput out;
    out.Nel = detail::integrate(rho_d_, grid_.weights);

    const std::vector<double> exc(d.exc, d.exc + npts_);
    const std::vector<double> vrho(d.vrho, d.vrho + npts_);
    const std::vector<double> vsigma =
        gga_ ? std::vector<double>(d.vsigma, d.vsigma + npts_)
             : std::vector<double>(npts_, 0.0);

    for (int g = 0; g < npts_; ++g)
      if (!std::isfinite(exc[g]) || !std::isfinite(vrho[g]) ||
          (gga_ && !std::isfinite(vsigma[g])))
        ++out.nan_points;

    const auto exc_d = detail::to_device(exc, "exc");
    {
      DeviceView1D integrand("rho exc", npts_);
      auto rho = rho_d_;
      Kokkos::parallel_for(
          "rho exc", Kokkos::RangePolicy<ExecSpace>(0, npts_),
          KOKKOS_LAMBDA(const int g) { integrand(g) = rho(g) * exc_d(g); });
      out.Exc = detail::integrate(integrand, grid_.weights);
    }

    if (!Vxc.data)
      return out;

    const auto vrho_d = detail::to_device(vrho, "vrho");
    const auto vsig_d = detail::to_device(vsigma, "vsigma");

    DeviceView2DLeft Z("Z", n_, npts_);
    {
      auto w = grid_.weights;
      auto chi = coll_.chi, dx = coll_.dchi_x, dy = coll_.dchi_y,
           dz = coll_.dchi_z;
      auto gx = gx_, gy = gy_, gz = gz_;
      const bool is_gga = gga_;
      Kokkos::parallel_for(
          "Z matrix",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                            {n_, npts_}),
          KOKKOS_LAMBDA(const int u, const int g) {
            double z = 0.5 * chi(u, g) * vrho_d(g);
            if (is_gga)
              z += 2.0 * vsig_d(g) *
                   (gx(g) * dx(u, g) + gy(g) * dy(u, g) + gz(g) * dz(u, g));
            Z(u, g) = w(g) * z;
          });
    }
    detail::accumulate_matrix(detail::symmetrized_gemm(coll_.chi, Z, n_), Vxc,
                              n_, "Vxc");
    return out;
  }

  /// Diagonal-only variant, for the adaptive grid.
  XCOutput accumulate_diag_vxc(const XCDerivativesView &d, double *diag) {
    require_density("accumulate_diag_vxc");
    if (!diag)
      throw std::invalid_argument("accumulate_diag_vxc: null diagonal");

    HostMatrixRef none{};
    XCOutput out = accumulate_vxc(d, none); // energy and Nel, no matrix

    const std::vector<double> vrho(d.vrho, d.vrho + npts_);
    const std::vector<double> vsigma =
        gga_ ? std::vector<double>(d.vsigma, d.vsigma + npts_)
             : std::vector<double>(npts_, 0.0);
    const auto vrho_d = detail::to_device(vrho, "vrho");
    const auto vsig_d = detail::to_device(vsigma, "vsigma");

    DeviceView2DLeft Z("Z", n_, npts_);
    auto w = grid_.weights;
    auto chi = coll_.chi, dx = coll_.dchi_x, dy = coll_.dchi_y,
         dz = coll_.dchi_z;
    auto gx = gx_, gy = gy_, gz = gz_;
    const bool is_gga = gga_;
    Kokkos::parallel_for(
        "Z matrix (diag)",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n_, npts_}),
        KOKKOS_LAMBDA(const int u, const int g) {
          double z = 0.5 * chi(u, g) * vrho_d(g);
          if (is_gga)
            z += 2.0 * vsig_d(g) *
                 (gx(g) * dx(u, g) + gy(g) * dy(u, g) + gz(g) * dz(u, g));
          Z(u, g) = w(g) * z;
        });

    const auto dg = detail::symmetrized_diagonal(coll_.chi, Z, n_);
    for (int i = 0; i < n_; ++i)
      diag[i] += dg[static_cast<size_t>(i)];
    return out;
  }

private:
  void require_density(const char *who) const {
    if (!has_density_)
      throw std::logic_error(std::string(who) +
                             ": set_density must be called first");
  }

  Evaluator basis_;
  FlatGrid grid_;
  bool gga_;
  int n_, npts_;
  detail::XCCollocation coll_;
  DeviceView1D rho_d_, gx_, gy_, gz_, sigma_d_;
  std::vector<double> rho_h_, sigma_h_;
  mutable std::vector<double> w_h_;
  bool has_density_ = false;
};

} // namespace Nukexc
