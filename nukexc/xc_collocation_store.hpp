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
//  Where the collocation lives during a kernel-backed XC evaluation
// ============================================================================
//
// A kernel-backed XC build needs the collocation tower twice: once to
// contract a density matrix into the grid ingredients, and once more to
// contract the functional derivatives back into a Fock matrix. There are
// two ways to supply it, and they trade the same way everything else in
// this library does:
//
//   DenseCollocationStore   one (ncomp, nbf, npts) tower, built once in the
//                           constructor and kept. Memory is ncomp * nbf *
//                           npts doubles whether or not the functions reach
//                           the points, and the kernel contracts over the
//                           WHOLE basis at every point.
//
//   PackedCollocationStore  per-box (nbe_b, npts_b) blocks, re-packed on
//                           every use. Memory is one chunk of boxes at a
//                           time; the basis is evaluated once per pass
//                           instead of once per session, and each kernel
//                           call contracts over that box's local functions
//                           only.
//
// Which one a session uses changes nothing a caller can see, which is the
// point: the dense store is validated against ERKALE and so serves as the
// oracle for the packed one (tests/test_xc_kernel_session.cxx).
//
// WHICH ONE TO USE, MEASURED
// --------------------------
// The same answer the hand-written paths give (packed_blocks.hpp has that
// table): where the dense store fits it is the faster one, and the packed
// store is for the molecule whose tower does not fit. Per Fock build on
// benzene, 150 functions, 35360 points, three runs back to back from
// benchmark_xc --sessions --nrad=35 --nang=15 --box-size=128:
//
//     family      dense               screened            MiB D / S
//     LDA         0.35 / 0.18 / 0.16  0.57 / 0.40 / 0.35   82.1 / 111.4
//     GGA         0.79 / 0.47 / 0.38  1.22 / 1.01 / 0.97  324.9 / 220.2
//     mGGA-tau    1.07 / 0.90 / 0.80  1.71 / 1.28 / 1.34  324.9 / 220.2
//
// Dense runs about one and a half to two times faster, because it hands the
// whole contraction to one large GEMM while the packed store hands BLAS
// thousands of small ones -- n of order 60, k of order 128 -- where there
// is nothing to amortize.
//
// THAT ANSWER DEPENDS ON THE KERNELS HAVING A BLAS, WHICH IS A BUILD FLAG
// ----------------------------------------------------------------------
// libxckernel's Stage B is its own loop nest unless XCKERNEL_USE_BLAS is
// defined, and the top-level CMakeLists defines it. Without it the table
// above reads the other way round -- dense GGA 3.6-6.6 s against screened
// 2.1-2.3 -- because the flop counts then decide and they are nbf^2 * npts
// against sum_b nbe_b^2 * npts_b. Four to thirteen times on the dense GGA
// is a large number to find behind a macro, and the reason it is that large
// is the shape of the generated kernels: xck_gga_r_o1 writes the potential
// as four rank updates where a hand-derived form that exploits the symmetry
// of the result needs one. There are four times the GEMM flops to hide, and
// only a real GEMM can hide them.
//
// So: measure this again if the kernels are regenerated or the flag moves.
// The conclusion is not a property of the layouts, it is a property of the
// layouts AND what contracts them.
//
// The memory column is a budget, not a measurement of need: the packed
// store's footprint is set by build_packed_block_layout's chunk budget
// (64 MB per unit buffer by default), which is why it can come out ABOVE
// the dense store on a small problem -- water, and benzene at LDA, where
// the whole tower is smaller than one chunk. What it does not do is grow
// with nbf * npts, which is the only thing that matters at a hundred atoms.
//
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basis_evaluator.hpp"
#include "density.hpp"
#include "grid.hpp"
#include "nukexc_config.hpp"
#include "octree.hpp"
#include "packed_blocks.hpp"
#include "xc_kernels.hpp"

namespace Nukexc {

/**
 *  @brief The grid ingredients one density matrix produces.
 *
 *  Which of them are populated follows from the family, and the ones it
 *  does not have stay empty rather than zero: an empty view reaching Libxc
 *  would be read as npts zeros, so the distinction has to survive.
 */
struct XCIngredients {
  DeviceView1D rho, rho_x, rho_y, rho_z, sigma, tau, lapl;

  /// Allocate what @p f needs and empty the rest. A view that is already
  /// the right length is kept, so an SCF loop allocates once.
  void resize(XCFamily f, int npts) {
    const bool grad = (f != XCFamily::LDA);
    const bool want_tau =
        (f == XCFamily::MetaGGATau || f == XCFamily::MetaGGALapl);
    want(rho, "rho", true, npts);
    want(rho_x, "rho_x", grad, npts);
    want(rho_y, "rho_y", grad, npts);
    want(rho_z, "rho_z", grad, npts);
    want(sigma, "sigma", grad, npts);
    want(tau, "tau", want_tau, npts);
    want(lapl, "lapl_rho", f == XCFamily::MetaGGALapl, npts);
  }

private:
  static void want(DeviceView1D &v, const char *name, bool needed,
                   int npts) {
    if (!needed) {
      v = DeviceView1D();
      return;
    }
    if (static_cast<int>(v.extent(0)) != npts)
      v = DeviceView1D(name, npts);
  }
};

/**
 *  @brief What a kernel-backed session needs of its collocation.
 *
 *  `fock` and `fock_spin` accumulate the kernel's dense (nbf, nbf) block
 *  into @p V, row-major, which the caller has zeroed; they return the
 *  kernel's status, nonzero only on a failure that cannot happen when the
 *  store supplies its own scratch, which both do.
 */
template <typename S>
concept XCCollocationStore =
    requires(S &s, const S &cs, const DeviceView2DLeft &P, XCIngredients &ing,
             const XCKernelFields &fl, const XCKernelDerivs &d,
             const XCKernelFieldsU &flu, const XCKernelDerivsU &du,
             double *V) {
      { cs.nbf() } -> std::convertible_to<int>;
      { cs.npoints() } -> std::convertible_to<int>;
      { cs.family() } -> std::convertible_to<XCFamily>;
      { cs.order() } -> std::convertible_to<BasisDeriv>;
      { cs.collocation_doubles() } -> std::convertible_to<size_t>;
      cs.contract(P, ing);
      { s.fock(fl, d, V) } -> std::convertible_to<int>;
      { s.fock_spin(XCSpin::Alpha, flu, du, V) } -> std::convertible_to<int>;
    };

// ============================================================================
//  Dense: one tower, built once
// ============================================================================

/**
 *  @brief The whole (ncomp, nbf, npts) collocation tower, kept.
 *
 *  Held rather than split and discarded: the kernels want exactly that
 *  tower, contiguous and component-major. The split (nbf, npts) views the
 *  density contraction uses alias nothing -- they are a genuine second
 *  copy, because KokkosBlas::gemm wants LayoutLeft while the kernels index
 *  chi + k*nbf*npts, i.e. LayoutRight within each component. One copy of
 *  the tower is far cheaper than evaluating the basis twice, which is the
 *  alternative.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
class DenseCollocationStore {
public:
  DenseCollocationStore(const Evaluator &basis, const FlatGrid &grid,
                        XCFamily family)
      : family_(family), deriv_(collocation_deriv(family)), n_(basis.nbf()),
        npts_(static_cast<int>(grid.quad_points.extent(0))) {
    if (n_ <= 0)
      throw std::invalid_argument("DenseCollocationStore: empty basis");
    if (family_ == XCFamily::MetaGGALapl &&
        !BasisEvaluatorWith<Evaluator, BasisDeriv::Hessian>)
      throw std::invalid_argument(
          "DenseCollocationStore: a Laplacian meta-GGA needs an evaluator "
          "that provides second derivatives");
    build_tower(basis, grid);
    work_.resize(static_cast<size_t>(xc_fock_work_size(family_, npts_, n_)));
  }

  int nbf() const { return n_; }
  int npoints() const { return npts_; }
  XCFamily family() const { return family_; }
  BasisDeriv order() const { return deriv_; }

  void contract(const DeviceView2DLeft &P, XCIngredients &out) const {
    if (family_ == XCFamily::LDA) {
      out.rho = compute_density_dm(chi_, P);
    } else {
      compute_density_and_sigma_dm(chi_, gx_chi_, gy_chi_, gz_chi_, P, out.rho,
                                   out.rho_x, out.rho_y, out.rho_z, out.sigma);
    }
    if (out.tau.extent(0) > 0) {
      // lapl_chi_ is empty unless the family needs the Laplacian, and
      // compute_meta_fields_dm then computes tau alone and leaves out.lapl
      // -- which is itself empty -- untouched. Passing something else of
      // the right shape would give a wrong Laplacian instead of none,
      // which is the failure worth designing out.
      compute_meta_fields_dm(chi_, gx_chi_, gy_chi_, gz_chi_, lapl_chi_, P,
                             out.tau, out.lapl);
    }
    Kokkos::fence();
  }

  /// Doubles held between builds, which is what the dense path trades
  /// away: the host tower the kernels read, the split views the density
  /// contraction reads, and the kernel scratch.
  size_t collocation_doubles() const {
    return tower_h_.size() + chi_.size() + gx_chi_.size() + gy_chi_.size() +
           gz_chi_.size() + lapl_chi_.size() + work_.size() + work_u_.size();
  }

  int fock(const XCKernelFields &fl, const XCKernelDerivs &d, double *V) {
    return xc_fock_restricted(family_, npts_, n_, tower_h_.data(), fl, d, V,
                              work_.data());
  }

  int fock_spin(XCSpin s, const XCKernelFieldsU &fl,
                const XCKernelDerivsU &d, double *V) {
    // Sized here rather than in the constructor, which does not know
    // whether this store will ever be used spin-polarized. A no-op from the
    // second call on, so an SCF loop still allocates nothing.
    work_u_.resize(static_cast<size_t>(
        xc_fock_work_size_unrestricted(family_, npts_, n_)));
    return xc_fock_unrestricted(family_, s, npts_, n_, tower_h_.data(), fl, d,
                                V, work_u_.data());
  }

private:
  void build_tower(const Evaluator &basis, const FlatGrid &grid) {
    const int nc = basis_ncomp(deriv_);
    Kokkos::View<double ***> tower("chi tower", nc, n_, npts_);
    switch (deriv_) {
    case BasisDeriv::Value:
      fill_collocation<BasisDeriv::Value>(ExecSpace{}, basis, grid.quad_points,
                                          tower);
      break;
    case BasisDeriv::Gradient:
      fill_collocation<BasisDeriv::Gradient>(ExecSpace{}, basis,
                                             grid.quad_points, tower);
      break;
    case BasisDeriv::Hessian:
      if constexpr (BasisEvaluatorWith<Evaluator, BasisDeriv::Hessian>)
        fill_collocation<BasisDeriv::Hessian>(ExecSpace{}, basis,
                                              grid.quad_points, tower);
      else
        throw std::invalid_argument(
            "DenseCollocationStore: evaluator provides no second "
            "derivatives");
      break;
    default:
      throw std::invalid_argument("DenseCollocationStore: unsupported order");
    }
    Kokkos::fence();

    // Split for the density contraction, which goes through
    // KokkosBlas::gemm and therefore wants LayoutLeft.
    chi_ = DeviceView2DLeft("chi", n_, npts_);
    auto tw = tower;
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

    auto m = Kokkos::create_mirror_view_and_copy(HostSpace{}, tower);
    tower_h_ = std::vector<double>(m.data(), m.data() + m.size());
  }

  XCFamily family_;
  BasisDeriv deriv_;
  int n_, npts_;
  std::vector<double> tower_h_;
  DeviceView2DLeft chi_, gx_chi_, gy_chi_, gz_chi_, lapl_chi_;
  std::vector<double> work_, work_u_;
};

// ============================================================================
//  Packed: per-box blocks, re-packed on every use
// ============================================================================

/**
 *  @brief The collocation as per-box packed blocks, never held whole.
 *
 *  One kernel call per spatial box, on that box's own points and its own
 *  local function list, scattered through the local-to-global map. The
 *  ingredients come from density.hpp's packed_density, which packs the same
 *  blocks for the density contraction.
 *
 *  A Laplacian meta-GGA is rejected: pack_collocation goes to the gradient,
 *  and extending it is a code-generation job in the solid-harmonic
 *  switches, not a change here. LDA, GGA and tau meta-GGA are covered,
 *  which is every family whose collocation stops at the gradient.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
class PackedCollocationStore {
public:
  PackedCollocationStore(const Evaluator &basis, const FlatGrid &grid,
                         const NeighborList &nl, XCFamily family,
                         const size_t budget_bytes = (size_t{64} << 20))
      : basis_(basis), grid_(grid), nl_(nl), family_(family),
        deriv_(collocation_deriv(family)), n_(basis.nbf()),
        npts_(static_cast<int>(grid.quad_points.extent(0))),
        mppb_(nl.max_points_per_box) {
    if (n_ <= 0)
      throw std::invalid_argument("PackedCollocationStore: empty basis");
    if (family_ == XCFamily::MetaGGALapl)
      throw std::invalid_argument(
          "PackedCollocationStore: a Laplacian meta-GGA needs the second "
          "derivatives of the basis, which a packed block does not carry; "
          "use the dense store for XCFamily::MetaGGALapl");

    L_ = build_packed_block_layout(basis_, nl_, budget_bytes);
    nbf_h_ = Kokkos::create_mirror_view_and_copy(HostSpace{}, L_.map.nbf);
    npts_h_ = Kokkos::create_mirror_view_and_copy(HostSpace{}, L_.npts);
    coll_off_h_ = Kokkos::create_mirror_view_and_copy(HostSpace{}, L_.coll_off);
    blk_off_h_ = Kokkos::create_mirror_view_and_copy(HostSpace{}, L_.blk_off);

    const int nc = basis_ncomp(deriv_);
    tower_ = Kokkos::View<double *>("packed tower", nc * L_.max_chunk_coll);
    tower_h_ = Kokkos::create_mirror_view(tower_);
    blocks_ = Kokkos::View<double *>("packed blocks", L_.max_chunk_blk);
    blocks_h_ = Kokkos::create_mirror_view(blocks_);
    V_ = DeviceView2DLeft("packed V", n_, n_);

    // Per-box kernel scratch. The exact per-box requirement is prefixed so
    // that the boxes of a chunk can run in parallel without a thread index:
    // box b writes [work_off(b) - work_off(b0), work_off(b+1) - work_off(b0)).
    work_off_ = work_prefix(false);
    work_off_u_ = work_prefix(true);
    work_h_ = Kokkos::View<double *, HostSpace>(
        "packed xc work", max_chunk_work(work_off_));
  }

  int nbf() const { return n_; }
  int npoints() const { return npts_; }
  XCFamily family() const { return family_; }
  BasisDeriv order() const { return deriv_; }
  const PackedBlockLayout &layout() const { return L_; }

  /// Doubles held between builds, counted the same way as the dense
  /// store's so the two are comparable: one chunk's tower and blocks, the
  /// kernel scratch, and the global matrix the boxes scatter into. The host
  /// mirror is counted only when it is a second allocation, which on a host
  /// backend it is not.
  size_t collocation_doubles() const {
    size_t n = tower_.size() + blocks_.size() + work_h_.size() +
               work_h_u_.size() + V_.size();
    if (tower_h_.data() != tower_.data())
      n += tower_h_.size() + blocks_h_.size();
    return n;
  }

  void contract(const DeviceView2DLeft &P, XCIngredients &out) const {
    if (family_ == XCFamily::LDA) {
      packed_density<BasisDeriv::Value>(basis_, grid_, nl_, L_, P, out.rho);
      return;
    }
    packed_density<BasisDeriv::Gradient>(basis_, grid_, nl_, L_, P, out.rho,
                                         out.rho_x, out.rho_y, out.rho_z,
                                         out.tau);
    dot_gradients(out.rho_x, out.rho_y, out.rho_z, out.rho_x, out.rho_y,
                  out.rho_z, out.sigma);
    Kokkos::fence();
  }

  int fock(const XCKernelFields &fl, const XCKernelDerivs &d, double *V) {
    detail::check_operands(family_, fl, d);
    const XCFamily f = family_;
    return assemble(
        [f, &fl, &d](const int np, const int n, const double *chi_b,
                     double *out_b, double *work_b, const int g0) {
          const XCKernelFields flb{.w = fl.w + g0,
                                   .rho_x = off(fl.rho_x, g0),
                                   .rho_y = off(fl.rho_y, g0),
                                   .rho_z = off(fl.rho_z, g0)};
          const XCKernelDerivs db{.vrho = off(d.vrho, g0),
                                  .vsigma = off(d.vsigma, g0),
                                  .vtau = off(d.vtau, g0),
                                  .vlapl = off(d.vlapl, g0)};
          return xc_fock_restricted_unchecked(f, np, n, chi_b, flb, db, out_b,
                                              work_b);
        },
        work_h_, work_off_, V);
  }

  int fock_spin(XCSpin s, const XCKernelFieldsU &fl,
                const XCKernelDerivsU &d, double *V) {
    detail::check_operands_u(family_, s, fl, d);
    if (work_h_u_.extent(0) == 0)
      work_h_u_ = Kokkos::View<double *, HostSpace>(
          "packed xc work u", max_chunk_work(work_off_u_));
    const XCFamily f = family_;
    return assemble(
        [f, s, &fl, &d](const int np, const int n, const double *chi_b,
                        double *out_b, double *work_b, const int g0) {
          const XCKernelFieldsU flb{.w = fl.w + g0,
                                    .rho_a_x = off(fl.rho_a_x, g0),
                                    .rho_a_y = off(fl.rho_a_y, g0),
                                    .rho_a_z = off(fl.rho_a_z, g0),
                                    .rho_b_x = off(fl.rho_b_x, g0),
                                    .rho_b_y = off(fl.rho_b_y, g0),
                                    .rho_b_z = off(fl.rho_b_z, g0)};
          const XCKernelDerivsU db{.vrho_a = off(d.vrho_a, g0),
                                   .vrho_b = off(d.vrho_b, g0),
                                   .vsigma_aa = off(d.vsigma_aa, g0),
                                   .vsigma_ab = off(d.vsigma_ab, g0),
                                   .vsigma_bb = off(d.vsigma_bb, g0),
                                   .vtau_a = off(d.vtau_a, g0),
                                   .vtau_b = off(d.vtau_b, g0),
                                   .vlapl_a = off(d.vlapl_a, g0),
                                   .vlapl_b = off(d.vlapl_b, g0)};
          return xc_fock_unrestricted_unchecked(f, s, np, n, chi_b, flb, db,
                                                out_b, work_b);
        },
        work_h_u_, work_off_u_, V);
  }

private:
  /// A per-point operand array sliced to a box's point range, or null if
  /// the family does not have that operand.
  static const double *off(const double *p, const int g0) {
    return p ? p + g0 : nullptr;
  }

  std::vector<size_t> work_prefix(const bool unrestricted) const {
    std::vector<size_t> pre(static_cast<size_t>(L_.num_boxes) + 1, 0);
    for (int b = 0; b < L_.num_boxes; ++b) {
      const int64_t np = npts_h_(b), nb = nbf_h_(b);
      const int64_t w = unrestricted
                            ? xc_fock_work_size_unrestricted(family_, np, nb)
                            : xc_fock_work_size(family_, np, nb);
      pre[static_cast<size_t>(b) + 1] =
          pre[static_cast<size_t>(b)] + static_cast<size_t>(w);
    }
    return pre;
  }

  size_t max_chunk_work(const std::vector<size_t> &pre) const {
    size_t m = 0;
    for (size_t c = 0; c + 1 < L_.chunk_begin.size(); ++c)
      m = std::max(m, pre[static_cast<size_t>(L_.chunk_begin[c + 1])] -
                          pre[static_cast<size_t>(L_.chunk_begin[c])]);
    return m;
  }

  /**
   *  @brief Pack, run @p kern on every box of every chunk, scatter.
   *
   *  Written once because the restricted assembly and the two spin channels
   *  differ in nothing but which kernel the box call reaches for.
   */
  template <typename BoxKernel>
  int assemble(const BoxKernel &kern,
               const Kokkos::View<double *, HostSpace> &work,
               const std::vector<size_t> &work_off, double *V) {
    const int nc = basis_ncomp(deriv_);
    Kokkos::deep_copy(V_, 0.0);
    int bad = 0;

    for (size_t c = 0; c + 1 < L_.chunk_begin.size(); ++c) {
      const int b0 = L_.chunk_begin[c], b1 = L_.chunk_begin[c + 1];

      if (deriv_ == BasisDeriv::Value)
        pack_collocation<BasisDeriv::Value, PackedLayout::Tower>(
            basis_, grid_, nl_, L_, b0, b1, tower_);
      else
        pack_collocation<BasisDeriv::Gradient, PackedLayout::Tower>(
            basis_, grid_, nl_, L_, b0, b1, tower_);
      Kokkos::fence();

      // On a host backend the mirror IS the tower, and there is nothing to
      // copy; this is the one line that would have to change for a device
      // kernel.
      if (tower_h_.data() != tower_.data())
        Kokkos::deep_copy(tower_h_, tower_);
      // The kernels accumulate into their output block.
      Kokkos::deep_copy(blocks_h_, 0.0);

      const size_t w0 = work_off[static_cast<size_t>(b0)];
      const size_t c0 = coll_off_h_(b0), k0 = blk_off_h_(b0);
      int chunk_bad = 0;
      Kokkos::parallel_reduce(
          "packed xc fock",
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(b0, b1),
          [&](const int b, int &acc) {
            const int np = npts_h_(b), n = nbf_h_(b);
            if (np == 0 || n == 0)
              return;
            const double *chi_b =
                tower_h_.data() + nc * (coll_off_h_(b) - c0);
            double *out_b = blocks_h_.data() + (blk_off_h_(b) - k0);
            double *work_b =
                work.data() + (work_off[static_cast<size_t>(b)] - w0);
            if (kern(np, n, chi_b, out_b, work_b, b * mppb_) != 0)
              ++acc;
          },
          chunk_bad);
      bad += chunk_bad;

      if (blocks_.data() != blocks_h_.data())
        Kokkos::deep_copy(blocks_, blocks_h_);
      scatter_packed_blocks(L_, b0, b1, blocks_, V_);
      Kokkos::fence();
    }

    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, V_);
    for (int i = 0; i < n_; ++i)
      for (int j = 0; j < n_; ++j)
        V[static_cast<size_t>(i) * n_ + j] += h(i, j);
    return bad;
  }

  Evaluator basis_;
  FlatGrid grid_;
  NeighborList nl_;
  XCFamily family_;
  BasisDeriv deriv_;
  int n_, npts_, mppb_;
  PackedBlockLayout L_;

  Kokkos::View<int *, HostSpace> nbf_h_, npts_h_;
  Kokkos::View<size_t *, HostSpace> coll_off_h_, blk_off_h_;

  Kokkos::View<double *> tower_, blocks_;
  /// Mirrors of the two buffers above. decltype rather than a HostMirror
  /// typedef, which this Kokkos does not spell that way.
  using HostMirror1D =
      decltype(Kokkos::create_mirror_view(Kokkos::View<double *>()));
  HostMirror1D tower_h_, blocks_h_;
  Kokkos::View<double *, HostSpace> work_h_, work_h_u_;
  std::vector<size_t> work_off_, work_off_u_;
  DeviceView2DLeft V_;
};

} // namespace Nukexc
