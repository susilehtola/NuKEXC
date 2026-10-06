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
//  A kernel-backed XC session
// ============================================================================
//
// Same job as host_interface.hpp's XCSession -- hold the collocation and the
// density across a functional evaluation -- but the Fock assembly comes from
// libxckernel instead of being written out here, which is what lets it cover
// meta-GGAs at all.
//
// This is deliberately a SEPARATE class rather than a new mode of XCSession.
// XCSession's hand-written LDA and GGA assembly is validated against ERKALE,
// which makes it the only oracle available for the kernels (they agree to
// 1.2e-14 and 8.5e-15; see tests/reference/libxckernel-upstream.md). An
// oracle one has edited is not an oracle, so it is left untouched and
// bit-for-bit, and the two are compared in tests/test_xc_kernel_session.cxx.
// Once meta-GGA parity against ERKALE is established the two can be merged
// the other way round, with the kernels backing XCSession.
//
// The collocation tower is kept rather than split and discarded. XCSession
// builds the (ncomp, nbf, npts) tower, copies the components into separate
// LayoutLeft views for the density contraction, and lets the tower die; the
// kernels want exactly that tower, contiguous, component-major. So it is
// held here and the split views alias nothing -- they are a genuine second
// copy, because KokkosBlas::gemm wants LayoutLeft for the density
// contraction while the kernels index chi + k*nbf*npts, i.e. LayoutRight
// within each component. One copy of the tower is far cheaper than
// evaluating the basis twice, which is the alternative.

#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basis_evaluator.hpp"
#include "density.hpp"
#include "grid.hpp"
#include "host_interface.hpp"
#include "nukexc_config.hpp"
#include "xc_kernels.hpp"

namespace Nukexc {

/**
 *  @brief Collocation, density and meta ingredients for one functional.
 *
 *  Usage:
 *
 *      XCKernelSession s(basis, grid, XCFamily::MetaGGATau);
 *      s.set_density(P);
 *      // caller evaluates the functional at s.rho(), s.sigma(), s.tau()
 *      auto out = s.accumulate_fock(derivs, zk, Vxc);
 *
 *  The collocation is built once in the constructor and the kernel scratch
 *  once with it, so an SCF loop allocates nothing per iteration.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
class XCKernelSession {
public:
  XCKernelSession(const Evaluator &basis, const FlatGrid &grid, XCFamily family)
      : family_(family), deriv_(collocation_deriv(family)), n_(basis.nbf()),
        npts_(static_cast<int>(grid.quad_points.extent(0))), grid_(grid) {
    if (n_ <= 0)
      throw std::invalid_argument("XCKernelSession: empty basis");
    if (family_ == XCFamily::MetaGGALapl &&
        !BasisEvaluatorWith<Evaluator, BasisDeriv::Hessian>)
      throw std::invalid_argument(
          "XCKernelSession: a Laplacian meta-GGA needs an evaluator that "
          "provides second derivatives");

    build_tower(basis, grid);
    work_.resize(static_cast<size_t>(
        xc_fock_work_size(family_, npts_, n_)));
  }

  int nbf() const { return n_; }
  int npoints() const { return npts_; }
  XCFamily family() const { return family_; }
  /// Derivative order of the collocation tower this session holds.
  BasisDeriv collocation_order() const { return deriv_; }

  /// Contract a density matrix onto the grid. Must precede the accumulators.
  void set_density(const HostMatrixView &P) {
    const auto Pd = detail::upload_matrix(P, n_, "P");
    rho_d_ = DeviceView1D("rho", npts_);
    gx_ = DeviceView1D("rho_x", npts_);
    gy_ = DeviceView1D("rho_y", npts_);
    gz_ = DeviceView1D("rho_z", npts_);
    sigma_d_ = DeviceView1D("sigma", npts_);

    if (family_ == XCFamily::LDA) {
      rho_d_ = compute_density_dm(chi_, Pd);
    } else {
      compute_density_and_sigma_dm(chi_, gx_chi_, gy_chi_, gz_chi_, Pd, rho_d_,
                                   gx_, gy_, gz_, sigma_d_);
    }

    if (needs_tau()) {
      tau_d_ = DeviceView1D("tau", npts_);
      lapl_d_ = DeviceView1D("lapl_rho", npts_);
      // lapl_chi_ is empty unless the family needs the Laplacian, and
      // compute_meta_fields_dm then computes tau alone. Passing something
      // else of the right shape would give a wrong Laplacian instead of
      // none, which is the failure worth designing out.
      compute_meta_fields_dm(chi_, gx_chi_, gy_chi_, gz_chi_, lapl_chi_, Pd,
                             tau_d_, lapl_d_);
    }
    Kokkos::fence();

    rho_h_ = detail::to_host(rho_d_);
    if (family_ != XCFamily::LDA) {
      sigma_h_ = detail::to_host(sigma_d_);
      gx_h_ = detail::to_host(gx_);
      gy_h_ = detail::to_host(gy_);
      gz_h_ = detail::to_host(gz_);
    }
    if (needs_tau())
      tau_h_ = detail::to_host(tau_d_);
    if (family_ == XCFamily::MetaGGALapl)
      lapl_h_ = detail::to_host(lapl_d_);
    has_density_ = true;
  }

  /// Ingredients for the functional, on the host, in Libxc's layouts.
  /// Each throws rather than returning an empty vector if the density has
  /// not been set, or if this family does not have that ingredient: an
  /// empty array reaching Libxc would be read as npts zeros.
  const std::vector<double> &rho() const {
    need("rho");
    return rho_h_;
  }
  const std::vector<double> &sigma() const {
    need("sigma");
    have(sigma_h_, "sigma");
    return sigma_h_;
  }
  const std::vector<double> &tau() const {
    need("tau");
    have(tau_h_, "tau");
    return tau_h_;
  }
  const std::vector<double> &lapl() const {
    need("lapl");
    have(lapl_h_, "lapl");
    return lapl_h_;
  }
  const std::vector<double> &rho_x() const {
    need("rho_x");
    have(gx_h_, "rho_x");
    return gx_h_;
  }
  const std::vector<double> &rho_y() const {
    need("rho_y");
    have(gy_h_, "rho_y");
    return gy_h_;
  }
  const std::vector<double> &rho_z() const {
    need("rho_z");
    have(gz_h_, "rho_z");
    return gz_h_;
  }

  const std::vector<double> &weights() const {
    if (w_h_.empty())
      w_h_ = detail::to_host(grid_.weights);
    return w_h_;
  }

  /// Integrated electron count for the density currently set.
  double nel() const {
    need("nel");
    return detail::integrate(rho_d_, grid_.weights);
  }

  /// Exc = sum_g w_g rho_g zk_g, with zk Libxc's energy per particle.
  double energy(const double *zk) const {
    need("energy");
    return xc_energy_restricted(npts_, weights().data(), rho_h_.data(), zk);
  }

  /**
   *  @brief Accumulate Exc and the XC Fock matrix.
   *
   *  @param d    Libxc derivative arrays; which are required follows from
   *              the family and is checked
   *  @param zk   energy density per particle, or null for the matrix alone
   *  @param Vxc  accumulated into; a null data pointer gives energy only
   */
  XCOutput accumulate_fock(const XCKernelDerivs &d, const double *zk,
                           const HostMatrixRef &Vxc) {
    need("accumulate_fock");

    XCOutput out;
    out.Nel = detail::integrate(rho_d_, grid_.weights);
    if (zk)
      out.Exc = energy(zk);

    if (!Vxc.data)
      return out;

    // The kernel accumulates into a dense (nbf, nbf) block of its own. The
    // XC Fock matrix is symmetric, so which of the two orderings the kernel
    // writes does not matter here -- but that is a property worth asserting
    // rather than assuming, which test_xc_kernel_session.cxx does.
    std::vector<double> V(static_cast<size_t>(n_) * n_, 0.0);
    const XCKernelFields fl{.w = weights().data(),
                            .rho_x = gx_h_.empty() ? nullptr : gx_h_.data(),
                            .rho_y = gy_h_.empty() ? nullptr : gy_h_.data(),
                            .rho_z = gz_h_.empty() ? nullptr : gz_h_.data()};
    const int rc = xc_fock_restricted(family_, npts_, n_, tower_h_.data(), fl,
                                      d, V.data(), work_.data());
    if (rc != 0)
      throw std::runtime_error("xc kernel returned " + std::to_string(rc));

    const int ld = Vxc.ld > 0 ? Vxc.ld : n_;
    for (int i = 0; i < n_; ++i)
      for (int j = 0; j < n_; ++j) {
        const double v = V[static_cast<size_t>(i) * n_ + j];
        const size_t off = (Vxc.layout == MatrixLayout::ColumnMajor)
                               ? static_cast<size_t>(i) +
                                     static_cast<size_t>(j) * ld
                               : static_cast<size_t>(i) * ld +
                                     static_cast<size_t>(j);
        Vxc.data[off] += v;
      }
    return out;
  }

private:
  bool needs_tau() const {
    return family_ == XCFamily::MetaGGATau ||
           family_ == XCFamily::MetaGGALapl;
  }

  void need(const char *what) const {
    if (!has_density_)
      throw std::logic_error(std::string("XCKernelSession::") + what +
                             ": set_density has not been called");
  }

  void have(const std::vector<double> &v, const char *what) const {
    if (v.empty())
      throw std::logic_error(std::string("XCKernelSession::") + what +
                             ": not computed for a " +
                             xc_family_name(family_) + " functional");
  }

  void build_tower(const Evaluator &basis, const FlatGrid &grid) {
    const int nc = basis_ncomp(deriv_);
    tower_ = Kokkos::View<double ***>("chi tower", nc, n_, npts_);
    switch (deriv_) {
    case BasisDeriv::Value:
      fill_collocation<BasisDeriv::Value>(ExecSpace{}, basis, grid.quad_points,
                                          tower_);
      break;
    case BasisDeriv::Gradient:
      fill_collocation<BasisDeriv::Gradient>(ExecSpace{}, basis,
                                             grid.quad_points, tower_);
      break;
    case BasisDeriv::Hessian:
      if constexpr (BasisEvaluatorWith<Evaluator, BasisDeriv::Hessian>)
        fill_collocation<BasisDeriv::Hessian>(ExecSpace{}, basis,
                                              grid.quad_points, tower_);
      else
        throw std::invalid_argument(
            "XCKernelSession: evaluator provides no second derivatives");
      break;
    default:
      throw std::invalid_argument("XCKernelSession: unsupported order");
    }
    Kokkos::fence();

    // Split for the density contraction, which goes through
    // KokkosBlas::gemm and therefore wants LayoutLeft.
    chi_ = DeviceView2DLeft("chi", n_, npts_);
    auto tw = tower_;
    auto chi = chi_;
    if (deriv_ == BasisDeriv::Value) {
      Kokkos::parallel_for(
          "split value",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                            {n_, npts_}),
          KOKKOS_LAMBDA(const int u, const int g) {
            chi(u, g) = tw(BF_VAL, u, g);
          });
    } else {
      gx_chi_ = DeviceView2DLeft("dchi_x", n_, npts_);
      gy_chi_ = DeviceView2DLeft("dchi_y", n_, npts_);
      gz_chi_ = DeviceView2DLeft("dchi_z", n_, npts_);
      auto gxc = gx_chi_, gyc = gy_chi_, gzc = gz_chi_;
      const bool hess = (deriv_ == BasisDeriv::Hessian);
      if (hess)
        lapl_chi_ = DeviceView2DLeft("lapl_chi", n_, npts_);
      auto lp = lapl_chi_;
      Kokkos::parallel_for(
          "split tower",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                            {n_, npts_}),
          KOKKOS_LAMBDA(const int u, const int g) {
            chi(u, g) = tw(BF_VAL, u, g);
            gxc(u, g) = tw(BF_DX, u, g);
            gyc(u, g) = tw(BF_DY, u, g);
            gzc(u, g) = tw(BF_DZ, u, g);
            if (hess)
              lp(u, g) = tw(BF_DXX, u, g) + tw(BF_DYY, u, g) +
                         tw(BF_DZZ, u, g);
          });
    }
    Kokkos::fence();

    // The kernels take a raw pointer. On a host backend the tower's own
    // memory would serve, but mirroring keeps this correct when the
    // execution space is a device: the kernels are XCK_HD and could be
    // launched there, which is Step 2 of the integration, not this one.
    auto m = Kokkos::create_mirror_view_and_copy(HostSpace{}, tower_);
    tower_h_ = std::vector<double>(m.data(), m.data() + m.size());
  }

  XCFamily family_;
  BasisDeriv deriv_;
  int n_, npts_;
  FlatGrid grid_;

  Kokkos::View<double ***> tower_;
  std::vector<double> tower_h_;
  DeviceView2DLeft chi_, gx_chi_, gy_chi_, gz_chi_, lapl_chi_;

  DeviceView1D rho_d_, gx_, gy_, gz_, sigma_d_, tau_d_, lapl_d_;
  std::vector<double> rho_h_, sigma_h_, gx_h_, gy_h_, gz_h_, tau_h_, lapl_h_;
  mutable std::vector<double> w_h_;
  std::vector<double> work_;
  bool has_density_ = false;
};

} // namespace Nukexc
