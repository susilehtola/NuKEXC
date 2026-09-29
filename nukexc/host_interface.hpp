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

} // namespace Nukexc
