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
// meta-GGAs at all, restricted and unrestricted alike. The hand-written
// paths in host_interface.hpp stop at GGA in both spin cases.
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
// DENSE AND SCREENED IN ONE CLASS
// -------------------------------
// Where the collocation lives is a policy -- xc_collocation_store.hpp --
// and everything else is this file:
//
//     XCKernelSession          dense, one (ncomp, nbf, npts) tower kept
//     XCScreenedKernelSession  packed per-box blocks, re-packed per build
//
// The two have the same interface down to the last accessor, which is what
// lets the dense one serve as the oracle for the packed one. Everything
// between the two -- Libxc's interleaved layouts, the energy expression,
// the spin bookkeeping, the operand plumbing -- is written once here, so a
// screened meta-GGA reaches exactly the kernel a dense one does.
//
// RESTRICTED AND UNRESTRICTED IN ONE SESSION
// ------------------------------------------
// Spin is a property of the density, not of the basis: the collocation is
// the same for both channels, and it is the expensive thing the session
// exists to hold. So there is one class with two set_density overloads
// rather than a restricted and an unrestricted class sharing a base, and an
// unrestricted evaluation builds the collocation exactly once. Which
// overload was called last is remembered, and the ingredient accessors of
// the other spin case then throw -- otherwise rho() after
// set_density(Pa, Pb) would hand back whatever the previous restricted call
// left behind.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basis_evaluator.hpp"
#include "density.hpp"
#include "grid.hpp"
#include "host_interface.hpp"
#include "nukexc_config.hpp"
#include "octree.hpp"
#include "xc_collocation_store.hpp"
#include "xc_kernels.hpp"

namespace Nukexc {

/**
 *  @brief Which kernel family a Libxc functional needs.
 *
 *  The mapping lives here because this is where Libxc and the kernels meet:
 *  xc_kernels.hpp deliberately knows nothing about Libxc (it takes derivative
 *  arrays, not functionals), while host_interface.hpp knows nothing about
 *  kernel families.
 *
 *  The meta-GGA split is on XC_FLAGS_NEEDS_LAPLACIAN rather than on the
 *  functional's name. A tau-only meta-GGA handed the Laplacian family would
 *  still give the right answer -- vlapl comes back zero -- but it would pay
 *  for a Hessian collocation tower it never reads, and asking the functional
 *  is both cheaper and not a list to maintain.
 *
 *  @throws std::invalid_argument for a family with no kernel (LCA, OEP)
 */
inline XCFamily xc_family_of(const xc_func_type &func) {
  switch (func.info->family) {
  case XC_FAMILY_LDA:
  case XC_FAMILY_HYB_LDA:
    return XCFamily::LDA;
  case XC_FAMILY_GGA:
  case XC_FAMILY_HYB_GGA:
    return XCFamily::GGA;
  case XC_FAMILY_MGGA:
  case XC_FAMILY_HYB_MGGA:
    return (func.info->flags & XC_FLAGS_NEEDS_LAPLACIAN)
               ? XCFamily::MetaGGALapl
               : XCFamily::MetaGGATau;
  default:
    break;
  }
  throw std::invalid_argument(
      std::string("xc_family_of: no kernel for Libxc family ") +
      std::to_string(func.info->family) + " (functional " +
      (func.info->name ? func.info->name : "?") + ")");
}

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
 *  The collocation store and the kernel scratch are built once in the
 *  constructor, so an SCF loop allocates nothing per iteration.
 *
 *  @tparam Store where the collocation lives; see xc_collocation_store.hpp
 */
template <typename Store>
  requires XCCollocationStore<Store>
class XCKernelSessionT {
public:
  XCKernelSessionT(Store store, const FlatGrid &grid, const char *name)
      : store_(std::move(store)), family_(store_.family()),
        deriv_(store_.order()), n_(store_.nbf()), npts_(store_.npoints()),
        grid_(grid), name_(name) {}

  int nbf() const { return n_; }
  int npoints() const { return npts_; }
  XCFamily family() const { return family_; }
  /// Derivative order of the collocation this session holds.
  BasisDeriv collocation_order() const { return deriv_; }
  /// Doubles the collocation store holds between builds. The whole reason
  /// there are two stores, so it is reportable rather than inferred.
  size_t collocation_doubles() const {
    return store_.collocation_doubles();
  }

  /// Contract a density matrix onto the grid. Must precede the accumulators.
  void set_density(const HostMatrixView &P) {
    const auto Pd = detail::upload_matrix(P, n_, "P");
    ing_.resize(family_, npts_);
    store_.contract(Pd, ing_);
    Kokkos::fence();

    rho_h_ = detail::to_host(ing_.rho);
    if (family_ != XCFamily::LDA) {
      sigma_h_ = detail::to_host(ing_.sigma);
      gx_h_ = detail::to_host(ing_.rho_x);
      gy_h_ = detail::to_host(ing_.rho_y);
      gz_h_ = detail::to_host(ing_.rho_z);
    }
    if (needs_tau())
      tau_h_ = detail::to_host(ing_.tau);
    if (family_ == XCFamily::MetaGGALapl)
      lapl_h_ = detail::to_host(ing_.lapl);
    spin_ = SpinState::Restricted;
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
    return detail::integrate(ing_.rho, grid_.weights);
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
    out.Nel = detail::integrate(ing_.rho, grid_.weights);
    if (zk)
      out.Exc = energy(zk);

    if (!Vxc.data)
      return out;

    // The store accumulates into a dense (nbf, nbf) block of its own. The
    // XC Fock matrix is symmetric, so which of the two orderings the kernel
    // writes does not matter here -- but that is a property worth asserting
    // rather than assuming, which test_xc_kernel_session.cxx does.
    std::vector<double> V(static_cast<size_t>(n_) * n_, 0.0);
    const XCKernelFields fl{.w = weights().data(),
                            .rho_x = gx_h_.empty() ? nullptr : gx_h_.data(),
                            .rho_y = gy_h_.empty() ? nullptr : gy_h_.data(),
                            .rho_z = gz_h_.empty() ? nullptr : gz_h_.data()};
    const int rc = store_.fock(fl, d, V.data());
    if (rc != 0)
      throw std::runtime_error("xc kernel returned " + std::to_string(rc));

    scatter(V, Vxc);
    return out;
  }

  // ==========================================================================
  //  Unrestricted (spin-polarized) path
  // ==========================================================================

  /**
   *  @brief Contract alpha and beta density matrices onto the grid.
   *
   *  Replaces whatever density was set before, restricted or not. The
   *  collocation is untouched, so switching spin case costs no basis
   *  evaluation.
   *
   *  sigma_ab = grad rho_a . grad rho_b is formed here: it is not something
   *  one density matrix can produce, and both spin channels' Fock matrices
   *  depend on it.
   */
  void set_density(const HostMatrixView &Pa, const HostMatrixView &Pb) {
    const auto Pad = detail::upload_matrix(Pa, n_, "Pa");
    const auto Pbd = detail::upload_matrix(Pb, n_, "Pb");

    ing_a_.resize(family_, npts_);
    ing_b_.resize(family_, npts_);
    store_.contract(Pad, ing_a_);
    store_.contract(Pbd, ing_b_);

    if (family_ != XCFamily::LDA) {
      if (static_cast<int>(sab_.extent(0)) != npts_)
        sab_ = DeviceView1D("sigma_ab", npts_);
      dot_gradients(ing_a_.rho_x, ing_a_.rho_y, ing_a_.rho_z, ing_b_.rho_x,
                    ing_b_.rho_y, ing_b_.rho_z, sab_);
    }
    Kokkos::fence();

    const auto ra_h = detail::to_host(ing_a_.rho);
    const auto rb_h = detail::to_host(ing_b_.rho);
    rho_tot_h_.resize(static_cast<size_t>(npts_));
    rho_u_h_.resize(2 * static_cast<size_t>(npts_));
    for (int g = 0; g < npts_; ++g) {
      rho_u_h_[2 * static_cast<size_t>(g)] = ra_h[g];
      rho_u_h_[2 * static_cast<size_t>(g) + 1] = rb_h[g];
      rho_tot_h_[static_cast<size_t>(g)] = ra_h[g] + rb_h[g];
    }

    if (family_ != XCFamily::LDA) {
      const auto aa = detail::to_host(ing_a_.sigma),
                 ab = detail::to_host(sab_),
                 bb = detail::to_host(ing_b_.sigma);
      sigma_u_h_.resize(3 * static_cast<size_t>(npts_));
      for (int g = 0; g < npts_; ++g) {
        sigma_u_h_[3 * static_cast<size_t>(g)] = aa[g];
        sigma_u_h_[3 * static_cast<size_t>(g) + 1] = ab[g];
        sigma_u_h_[3 * static_cast<size_t>(g) + 2] = bb[g];
      }
      gax_h_u_ = detail::to_host(ing_a_.rho_x);
      gay_h_u_ = detail::to_host(ing_a_.rho_y);
      gaz_h_u_ = detail::to_host(ing_a_.rho_z);
      gbx_h_u_ = detail::to_host(ing_b_.rho_x);
      gby_h_u_ = detail::to_host(ing_b_.rho_y);
      gbz_h_u_ = detail::to_host(ing_b_.rho_z);
    } else {
      sigma_u_h_.clear();
    }

    if (needs_tau()) {
      // Spin-resolved by construction: the kinetic energy density of
      // whichever density matrix was contracted, so P^alpha in gives
      // Libxc's tau_a rather than half the total.
      const auto ta = detail::to_host(ing_a_.tau),
                 tb = detail::to_host(ing_b_.tau);
      tau_u_h_.resize(2 * static_cast<size_t>(npts_));
      for (int g = 0; g < npts_; ++g) {
        tau_u_h_[2 * static_cast<size_t>(g)] = ta[g];
        tau_u_h_[2 * static_cast<size_t>(g) + 1] = tb[g];
      }
    } else {
      tau_u_h_.clear();
    }

    if (family_ == XCFamily::MetaGGALapl) {
      const auto la = detail::to_host(ing_a_.lapl),
                 lb = detail::to_host(ing_b_.lapl);
      lapl_u_h_.resize(2 * static_cast<size_t>(npts_));
      for (int g = 0; g < npts_; ++g) {
        lapl_u_h_[2 * static_cast<size_t>(g)] = la[g];
        lapl_u_h_[2 * static_cast<size_t>(g) + 1] = lb[g];
      }
    } else {
      lapl_u_h_.clear();
    }

    // Sized here rather than in the constructor, which does not know
    // whether this session will ever be used spin-polarized. A no-op from
    // the second call on, so an SCF loop still allocates nothing.
    unpacked_.resize(family_, npts_);

    spin_ = SpinState::Unrestricted;
  }

  /// Spin-resolved ingredients in Libxc's polarized layout, interleaved per
  /// point: rho as (a, b), sigma as (aa, ab, bb), tau and lapl as (a, b).
  /// Ready to hand straight to an XC_POLARIZED functional.
  const std::vector<double> &rho_spin() const {
    need_u("rho_spin");
    return rho_u_h_;
  }
  const std::vector<double> &sigma_spin() const {
    need_u("sigma_spin");
    have(sigma_u_h_, "sigma_spin");
    return sigma_u_h_;
  }
  const std::vector<double> &tau_spin() const {
    need_u("tau_spin");
    have(tau_u_h_, "tau_spin");
    return tau_u_h_;
  }
  const std::vector<double> &lapl_spin() const {
    need_u("lapl_spin");
    have(lapl_u_h_, "lapl_spin");
    return lapl_u_h_;
  }

  /// Electron count per channel, for the density currently set.
  double nel_alpha() const {
    need_u("nel_alpha");
    return detail::integrate(ing_a_.rho, grid_.weights);
  }
  double nel_beta() const {
    need_u("nel_beta");
    return detail::integrate(ing_b_.rho, grid_.weights);
  }

  /**
   *  @brief Accumulate Exc and both spin channels of the XC Fock matrix.
   *
   *  @param packed Libxc's polarized derivative arrays as Libxc writes them,
   *                interleaved per point; de-interleaved internally, so the
   *                caller never writes that loop
   *  @param zk     energy density per particle, or null for the matrices
   *                alone
   *  @param Vxc_a  accumulated into
   *  @param Vxc_b  accumulated into
   *
   *  Both matrices are needed or neither: two null data pointers give the
   *  energy and electron counts alone, and exactly one null is a mistake
   *  rather than a request, so it throws.
   */
  XCOutputPolarized accumulate_fock(const XCPackedDerivsU &packed,
                                    const double *zk,
                                    const HostMatrixRef &Vxc_a,
                                    const HostMatrixRef &Vxc_b) {
    need_u("accumulate_fock");

    XCOutputPolarized out;
    out.Nel_alpha = detail::integrate(ing_a_.rho, grid_.weights);
    out.Nel_beta = detail::integrate(ing_b_.rho, grid_.weights);
    out.Nel = out.Nel_alpha + out.Nel_beta;
    // The energy expression does not know about spin: it is the total
    // density against zk either way, so the restricted kernel is the right
    // one and there is no polarized counterpart to reach for.
    if (zk)
      out.Exc =
          xc_energy_restricted(npts_, weights().data(), rho_tot_h_.data(), zk);

    if (!Vxc_a.data && !Vxc_b.data)
      return out;
    if (!Vxc_a.data || !Vxc_b.data)
      throw std::invalid_argument(
          std::string(name_) +
          "::accumulate_fock: one of Vxc_a and Vxc_b is null. "
          "Pass both to assemble both channels, or neither for the energy "
          "alone -- a single channel of an unrestricted XC potential is not "
          "a meaningful quantity on its own");

    const XCKernelDerivsU d = unpacked_.unpack(family_, npts_, packed);
    const XCKernelFieldsU fl{
        .w = weights().data(),
        .rho_a_x = gax_h_u_.empty() ? nullptr : gax_h_u_.data(),
        .rho_a_y = gay_h_u_.empty() ? nullptr : gay_h_u_.data(),
        .rho_a_z = gaz_h_u_.empty() ? nullptr : gaz_h_u_.data(),
        .rho_b_x = gbx_h_u_.empty() ? nullptr : gbx_h_u_.data(),
        .rho_b_y = gby_h_u_.empty() ? nullptr : gby_h_u_.data(),
        .rho_b_z = gbz_h_u_.empty() ? nullptr : gbz_h_u_.data()};

    // One buffer, reused: the two channels differ only in which kernel runs
    // and where the result lands.
    std::vector<double> V(static_cast<size_t>(n_) * n_);
    const auto channel = [&](XCSpin sp, const HostMatrixRef &M) {
      std::fill(V.begin(), V.end(), 0.0);
      const int rc = store_.fock_spin(sp, fl, d, V.data());
      if (rc != 0)
        throw std::runtime_error(std::string("xc kernel returned ") +
                                 std::to_string(rc) + " for the " +
                                 xc_spin_name(sp) + " channel");
      scatter(V, M);
    };
    channel(XCSpin::Alpha, Vxc_a);
    channel(XCSpin::Beta, Vxc_b);
    return out;
  }

protected:
  /// The collocation store, for a derived class that wants to reach past
  /// the session -- the packed one's layout, for instance.
  const Store &store() const { return store_; }

private:
  bool needs_tau() const {
    return family_ == XCFamily::MetaGGATau ||
           family_ == XCFamily::MetaGGALapl;
  }

  void need(const char *what) const {
    if (spin_ == SpinState::None)
      throw std::logic_error(std::string(name_) + "::" + what +
                             ": set_density has not been called");
    if (spin_ != SpinState::Restricted)
      throw std::logic_error(
          std::string(name_) + "::" + what +
          ": the density currently set is spin-polarized; use the "
          "spin-resolved accessors and the (Vxc_a, Vxc_b) overload");
  }

  void need_u(const char *what) const {
    if (spin_ == SpinState::None)
      throw std::logic_error(std::string(name_) + "::" + what +
                             ": set_density has not been called");
    if (spin_ != SpinState::Unrestricted)
      throw std::logic_error(
          std::string(name_) + "::" + what +
          ": the density currently set is spin-restricted; use rho(), "
          "sigma() and the single-matrix overload");
  }

  /// Copy the store's dense (nbf, nbf) block into the caller's matrix,
  /// accumulating and honouring its layout and leading dimension. Shared by
  /// the restricted path and both spin channels.
  void scatter(const std::vector<double> &V, const HostMatrixRef &M) const {
    const int ld = M.ld > 0 ? M.ld : n_;
    for (int i = 0; i < n_; ++i)
      for (int j = 0; j < n_; ++j) {
        const double v = V[static_cast<size_t>(i) * n_ + j];
        const size_t off = (M.layout == MatrixLayout::ColumnMajor)
                               ? static_cast<size_t>(i) +
                                     static_cast<size_t>(j) * ld
                               : static_cast<size_t>(i) * ld +
                                     static_cast<size_t>(j);
        M.data[off] += v;
      }
  }

  void have(const std::vector<double> &v, const char *what) const {
    if (v.empty())
      throw std::logic_error(std::string(name_) + "::" + what +
                             ": not computed for a " +
                             xc_family_name(family_) + " functional");
  }

  Store store_;
  XCFamily family_;
  BasisDeriv deriv_;
  int n_, npts_;
  FlatGrid grid_;
  const char *name_;

  XCIngredients ing_, ing_a_, ing_b_;
  DeviceView1D sab_;
  std::vector<double> rho_h_, sigma_h_, gx_h_, gy_h_, gz_h_, tau_h_, lapl_h_;
  mutable std::vector<double> w_h_;

  /// Which set_density overload was called last.
  enum class SpinState { None, Restricted, Unrestricted };
  SpinState spin_ = SpinState::None;

  /// Interleaved for Libxc: (a, b) and (aa, ab, bb) per point.
  std::vector<double> rho_u_h_, sigma_u_h_, tau_u_h_, lapl_u_h_;
  /// Total density, for the energy and the electron count.
  std::vector<double> rho_tot_h_;
  /// De-interleaved for the kernels.
  std::vector<double> gax_h_u_, gay_h_u_, gaz_h_u_, gbx_h_u_, gby_h_u_,
      gbz_h_u_;
  XCUnpackedDerivsU unpacked_;
};

/**
 *  @brief The kernel-backed session on a dense collocation tower.
 *
 *  Memory is ncomp * nbf * npts doubles and the basis is evaluated once in
 *  the constructor. Where that fits it is the faster path, by one and a
 *  half to two -- one large GEMM beats thousands of small ones -- and it is
 *  also what a Laplacian meta-GGA needs, since a packed block stops at the
 *  gradient. xc_collocation_store.hpp has the table, and the build flag the
 *  comparison turns on.
 */
template <typename Evaluator>
class XCKernelSession
    : public XCKernelSessionT<DenseCollocationStore<Evaluator>> {
  using Base = XCKernelSessionT<DenseCollocationStore<Evaluator>>;

public:
  XCKernelSession(const Evaluator &basis, const FlatGrid &grid,
                  XCFamily family)
      : Base(DenseCollocationStore<Evaluator>(basis, grid, family), grid,
             "XCKernelSession") {}
};

/**
 *  @brief The kernel-backed session on packed per-box collocation blocks.
 *
 *  Memory is one chunk of boxes at a time instead of the whole tower, and
 *  the basis is evaluated once per pass rather than once per session. That
 *  is the same trade every other screened path in this library makes, and
 *  it costs the same thing: about one and a half to two times the time per
 *  build, measured in xc_collocation_store.hpp. What it buys is a footprint
 *  that does not grow with nbf * npts, which is what lets a molecule run
 *  whose tower would not fit at all.
 *
 *  Covers every family the dense session covers, and refuses a Laplacian
 *  meta-GGA in exactly the same case: an evaluator without second
 *  derivatives, which the Slater radials are and the Gaussians are not.
 */
template <typename Evaluator>
class XCScreenedKernelSession
    : public XCKernelSessionT<PackedCollocationStore<Evaluator>> {
  using Base = XCKernelSessionT<PackedCollocationStore<Evaluator>>;

public:
  XCScreenedKernelSession(const Evaluator &basis, const FlatGrid &grid,
                          const NeighborList &nl, XCFamily family,
                          const size_t budget_bytes = (size_t{64} << 20))
      : Base(PackedCollocationStore<Evaluator>(basis, grid, nl, family,
                                               budget_bytes),
             grid, "XCScreenedKernelSession") {}

  /// The packing layout, for a caller that wants the box statistics.
  const PackedBlockLayout &layout() const { return this->store().layout(); }
};

} // namespace Nukexc
