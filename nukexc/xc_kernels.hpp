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
//  XC energy and Fock matrices from libxckernel
// ============================================================================
//
// The exchange-correlation chain rule is not written here, and is not
// written anywhere else in NuKEXC either. Libxc owns the functional
// derivative tower; libxckernel owns the derivatives of the density
// ingredients with respect to the density matrix and emits the contraction;
// NuKEXC owns the ingredients and the quadrature. This header is only the
// seam between the last two.
//
// The vendored kernels live in external/xckernel (see external/README.md
// for the generating command and version). They are header-only: constexpr
// coefficient tables plus a templated entry point, so there is nothing to
// link and nothing that stops them being inlined.
//
// WHY THE OPERANDS GO THROUGH STRUCTS
// -----------------------------------
// Each kernel also takes its operands as two positional arrays, and for
// xck_mgga_r_o1 the functional-derivative order is
//
//     vlapl, vrho, vsigma, vtau
//
// which is deliberately NOT Libxc's declaration order (rho, sigma, lapl,
// tau). All four are const double* of length npts, so passing them in the
// wrong order is undetectable: no type error, no arity error, just wrong
// numbers. libxckernel 0.5.0 therefore also emits named operand structs,
// and this header uses only those -- the order becomes the compiler's
// problem. The static_asserts below additionally pin each kernel's arity,
// so a regeneration at a different libxckernel revision that added or
// removed an operand fails to build rather than reading past an array.
//
// WHAT THE CALLER STILL HAS TO GET RIGHT
// --------------------------------------
// The collocation layout. `chi` is the derivative tower as
// (ncomp, nbf, npts) with components ordered 1, x, y, z, xx, xy, xz, yy,
// yz, zz -- exactly NuKEXC's BF_VAL..BF_DZZ and exactly what
// fill_collocation<D> writes, so a Kokkos::View<double***> filled by it can
// be handed over by .data() with no repacking. The number of components
// each family needs is collocation_deriv() below; passing a shorter tower
// reads out of bounds, which is why it is a function here rather than a
// convention in a comment.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nukexc_config.hpp"

#include "xckernel/kernels/xck_gga_r_o0.hpp"
#include "xckernel/kernels/xck_gga_r_o1.hpp"
#include "xckernel/kernels/xck_gga_ua_o1.hpp"
#include "xckernel/kernels/xck_gga_ub_o1.hpp"
#include "xckernel/kernels/xck_lda_r_o0.hpp"
#include "xckernel/kernels/xck_lda_r_o1.hpp"
#include "xckernel/kernels/xck_lda_ua_o1.hpp"
#include "xckernel/kernels/xck_lda_ub_o1.hpp"
#include "xckernel/kernels/xck_mgga_r_o0.hpp"
#include "xckernel/kernels/xck_mgga_r_o1.hpp"
#include "xckernel/kernels/xck_mgga_tau_r_o0.hpp"
#include "xckernel/kernels/xck_mgga_tau_r_o1.hpp"
#include "xckernel/kernels/xck_mgga_tau_ua_o1.hpp"
#include "xckernel/kernels/xck_mgga_tau_ub_o1.hpp"
#include "xckernel/kernels/xck_mgga_ua_o1.hpp"
#include "xckernel/kernels/xck_mgga_ub_o1.hpp"

#include "basis_evaluator.hpp"

namespace Nukexc {

/// Which ingredients a functional depends on, i.e. which kernel applies.
enum class XCFamily {
  LDA,         ///< rho
  GGA,         ///< rho, sigma
  MetaGGATau,  ///< rho, sigma, tau
  MetaGGALapl, ///< rho, sigma, lapl, tau -- libxckernel's full "mgga"
};

// Arity, pinned against the vendored headers. If a regeneration changes an
// operand list these fail here rather than silently in the kernel.
static_assert(xckernel::xck_lda_r_o1_n_fields == 1);
static_assert(xckernel::xck_lda_r_o1_n_xc == 1);
static_assert(xckernel::xck_gga_r_o1_n_fields == 4);
static_assert(xckernel::xck_gga_r_o1_n_xc == 2);
static_assert(xckernel::xck_mgga_tau_r_o1_n_fields == 4);
static_assert(xckernel::xck_mgga_tau_r_o1_n_xc == 3);
static_assert(xckernel::xck_mgga_r_o1_n_fields == 4);
static_assert(xckernel::xck_mgga_r_o1_n_xc == 4);

/// Human-readable family name, for error messages.
inline const char *xc_family_name(XCFamily f) {
  switch (f) {
  case XCFamily::LDA:
    return "LDA";
  case XCFamily::GGA:
    return "GGA";
  case XCFamily::MetaGGATau:
    return "meta-GGA (tau)";
  case XCFamily::MetaGGALapl:
    return "meta-GGA (tau and Laplacian)";
  }
  return "unknown";
}

/**
 *  @brief Which collocation derivative order a family's kernel reads.
 *
 *  Not a convention to be remembered: the kernel indexes
 *  chi + k*nbf*npts for every component it uses, so a tower built to a
 *  lower order would be read out of bounds.
 */
inline BasisDeriv collocation_deriv(XCFamily f) {
  switch (f) {
  case XCFamily::LDA:
    return BasisDeriv::Value; // 1 component
  case XCFamily::GGA:
  case XCFamily::MetaGGATau:
    return BasisDeriv::Gradient; // 4
  case XCFamily::MetaGGALapl:
    return BasisDeriv::Hessian; // 10
  }
  throw std::invalid_argument("collocation_deriv: unknown family");
}

/// Per-point ingredients the kernels take, by name.
struct XCKernelFields {
  const double *w = nullptr;     ///< grid weights, always needed
  const double *rho_x = nullptr; ///< density gradient; null for an LDA
  const double *rho_y = nullptr;
  const double *rho_z = nullptr;
};

/**
 *  @brief Libxc derivative arrays, by name.
 *
 *  Whoever evaluated the functional supplies these. Which are required
 *  depends on the family and is checked, loudly: a null vtau reaching a
 *  meta-GGA kernel would otherwise be dereferenced inside it.
 */
struct XCKernelDerivs {
  const double *vrho = nullptr;
  const double *vsigma = nullptr; ///< GGA and above
  const double *vtau = nullptr;   ///< meta-GGA
  const double *vlapl = nullptr;  ///< MetaGGALapl only
};

/// Scratch the Fock kernel needs, in doubles. Allocate once and reuse.
inline int64_t xc_fock_work_size(XCFamily f, int64_t npts, int64_t nbf) {
  switch (f) {
  case XCFamily::LDA:
    return xckernel::xck_lda_r_o1_work(npts, nbf);
  case XCFamily::GGA:
    return xckernel::xck_gga_r_o1_work(npts, nbf);
  case XCFamily::MetaGGATau:
    return xckernel::xck_mgga_tau_r_o1_work(npts, nbf);
  case XCFamily::MetaGGALapl:
    return xckernel::xck_mgga_r_o1_work(npts, nbf);
  }
  throw std::invalid_argument("xc_fock_work_size: unknown family");
}

namespace detail {

inline void require(const double *p, const char *what, XCFamily f) {
  if (!p)
    throw std::invalid_argument(std::string("xc kernel: ") + what +
                                " is required for a " + xc_family_name(f) +
                                " functional");
}

inline void check_operands(XCFamily f, const XCKernelFields &fl,
                           const XCKernelDerivs &d) {
  require(fl.w, "w", f);
  require(d.vrho, "vrho", f);
  if (f != XCFamily::LDA) {
    require(fl.rho_x, "rho_x", f);
    require(fl.rho_y, "rho_y", f);
    require(fl.rho_z, "rho_z", f);
    require(d.vsigma, "vsigma", f);
  }
  if (f == XCFamily::MetaGGATau || f == XCFamily::MetaGGALapl)
    require(d.vtau, "vtau", f);
  if (f == XCFamily::MetaGGALapl)
    require(d.vlapl, "vlapl", f);
}

} // namespace detail

/**
 *  @brief Accumulate the restricted XC Fock matrix.
 *
 *  @param chi   collocation tower, (ncomp, nbf, npts), ncomp as
 *               collocation_deriv(f) requires
 *  @param out   (nbf, nbf), ACCUMULATED into
 *  @param work  at least xc_fock_work_size(f, npts, nbf) doubles; a null
 *               pointer makes the kernel allocate internally, which is
 *               what we are trying to avoid in an SCF loop
 *
 *  Returns the kernel's status, nonzero only on an internal allocation
 *  failure -- which cannot happen when `work` is supplied.
 */
inline int xc_fock_restricted(XCFamily f, int64_t npts, int64_t nbf,
                              const double *chi, const XCKernelFields &fl,
                              const XCKernelDerivs &d, double *out,
                              double *work) {
  detail::check_operands(f, fl, d);
  if (!chi || !out)
    throw std::invalid_argument("xc_fock_restricted: null chi or out");

  switch (f) {
  case XCFamily::LDA: {
    const xckernel::xck_lda_r_o1_fields<double> kf{.w = fl.w};
    const xckernel::xck_lda_r_o1_xc<double> kx{.vrho = d.vrho};
    return xckernel::xck_lda_r_o1_t<double>(npts, nbf, chi, kf, kx, out, work);
  }
  case XCFamily::GGA: {
    const xckernel::xck_gga_r_o1_fields<double> kf{
        .w = fl.w, .rho_x = fl.rho_x, .rho_y = fl.rho_y, .rho_z = fl.rho_z};
    const xckernel::xck_gga_r_o1_xc<double> kx{.vrho = d.vrho,
                                               .vsigma = d.vsigma};
    return xckernel::xck_gga_r_o1_t<double>(npts, nbf, chi, kf, kx, out, work);
  }
  case XCFamily::MetaGGATau: {
    const xckernel::xck_mgga_tau_r_o1_fields<double> kf{
        .w = fl.w, .rho_x = fl.rho_x, .rho_y = fl.rho_y, .rho_z = fl.rho_z};
    const xckernel::xck_mgga_tau_r_o1_xc<double> kx{
        .vrho = d.vrho, .vsigma = d.vsigma, .vtau = d.vtau};
    return xckernel::xck_mgga_tau_r_o1_t<double>(npts, nbf, chi, kf, kx, out,
                                                 work);
  }
  case XCFamily::MetaGGALapl: {
    const xckernel::xck_mgga_r_o1_fields<double> kf{
        .w = fl.w, .rho_x = fl.rho_x, .rho_y = fl.rho_y, .rho_z = fl.rho_z};
    // Named, so the vlapl-first order is the compiler's problem.
    const xckernel::xck_mgga_r_o1_xc<double> kx{.vlapl = d.vlapl,
                                                .vrho = d.vrho,
                                                .vsigma = d.vsigma,
                                                .vtau = d.vtau};
    return xckernel::xck_mgga_r_o1_t<double>(npts, nbf, chi, kf, kx, out,
                                             work);
  }
  }
  throw std::invalid_argument("xc_fock_restricted: unknown family");
}

/**
 *  @brief The XC energy, sum_g w_g rho_g zk_g.
 *
 *  zk is Libxc's energy density PER PARTICLE. Every family shares this
 *  form, so the kernel is the same for all of them; it is routed through
 *  libxckernel anyway rather than written here, so that there is exactly
 *  one place the convention lives.
 *
 *  Spin does not enter either: an unrestricted evaluation has the same
 *  Exc = sum_g w_g rho_g zk_g with rho the TOTAL density, which is why
 *  there is no unrestricted counterpart to this function and libxckernel
 *  emits no _ua_o0 or _ub_o0 kernel. Pass rho_a + rho_b.
 */
inline double xc_energy_restricted(int64_t npts, const double *w,
                                   const double *rho, const double *zk) {
  if (!w || !rho || !zk)
    throw std::invalid_argument("xc_energy_restricted: null operand");
  return xckernel::xck_lda_r_o0_t<double>(npts, w, rho, zk);
}

// ============================================================================
//  Unrestricted (spin-polarized) Fock assembly
// ============================================================================
//
// One kernel per spin channel, and each channel's kernel reads BOTH spin
// density gradients: sigma_ab = grad rho_a . grad rho_b couples the channels,
// so d Exc / d P^a depends on grad rho_b. That is why XCKernelFieldsU carries
// six gradient components rather than three, and why handing the alpha kernel
// only the alpha gradient would silently drop the cross term.
//
// THE TRAP THE NAMED STRUCTS EXIST FOR
// ------------------------------------
// The two channels do not take their sigma derivatives in the same
// positional order. libxckernel emits
//
//     xck_gga_ua_o1_xc_names = {"vrho_0", "vsigma_0", "vsigma_1"}
//     xck_gga_ub_o1_xc_names = {"vrho_1", "vsigma_1", "vsigma_2"}
//
// i.e. slot 1 is vsigma_aa for alpha but vsigma_ab for beta, and slot 2 is
// vsigma_ab for alpha but vsigma_bb for beta. Positionally the beta list is
// (opposite-spin, same-spin) where alpha's is (same-spin, opposite-spin).
// Both are const double*, so swapping them is undetectable: no type error,
// no arity error, just a wrong exchange-correlation potential that still
// looks plausible. Only the named structs are used below.

/// Which spin channel's Fock matrix a kernel call produces.
enum class XCSpin {
  Alpha, ///< d Exc / d P^alpha
  Beta,  ///< d Exc / d P^beta
};

/// Human-readable spin name, for error messages.
inline const char *xc_spin_name(XCSpin s) {
  return s == XCSpin::Alpha ? "alpha" : "beta";
}

// Arity, pinned against the vendored headers, as for the restricted set.
static_assert(xckernel::xck_lda_ua_o1_n_fields == 1);
static_assert(xckernel::xck_lda_ua_o1_n_xc == 1);
static_assert(xckernel::xck_lda_ub_o1_n_fields == 1);
static_assert(xckernel::xck_lda_ub_o1_n_xc == 1);
static_assert(xckernel::xck_gga_ua_o1_n_fields == 7);
static_assert(xckernel::xck_gga_ua_o1_n_xc == 3);
static_assert(xckernel::xck_gga_ub_o1_n_fields == 7);
static_assert(xckernel::xck_gga_ub_o1_n_xc == 3);
static_assert(xckernel::xck_mgga_tau_ua_o1_n_fields == 7);
static_assert(xckernel::xck_mgga_tau_ua_o1_n_xc == 4);
static_assert(xckernel::xck_mgga_tau_ub_o1_n_fields == 7);
static_assert(xckernel::xck_mgga_tau_ub_o1_n_xc == 4);
static_assert(xckernel::xck_mgga_ua_o1_n_fields == 7);
static_assert(xckernel::xck_mgga_ua_o1_n_xc == 5);
static_assert(xckernel::xck_mgga_ub_o1_n_fields == 7);
static_assert(xckernel::xck_mgga_ub_o1_n_xc == 5);

/// Per-point ingredients the unrestricted kernels take, by name. Both spin
/// gradients are needed by both channels; see above.
struct XCKernelFieldsU {
  const double *w = nullptr; ///< grid weights, always needed
  const double *rho_a_x = nullptr; ///< alpha gradient; null for an LDA
  const double *rho_a_y = nullptr;
  const double *rho_a_z = nullptr;
  const double *rho_b_x = nullptr; ///< beta gradient; null for an LDA
  const double *rho_b_y = nullptr;
  const double *rho_b_z = nullptr;
};

/**
 *  @brief Libxc polarized derivative arrays, one component per array.
 *
 *  This is the form the kernels index: each pointer is npts doubles with
 *  unit stride. Libxc does not write them this way -- it interleaves the
 *  components per point -- so XCPackedDerivsU and XCUnpackedDerivsU below
 *  exist to get from its layout to this one without the caller writing the
 *  loop.
 */
struct XCKernelDerivsU {
  const double *vrho_a = nullptr;
  const double *vrho_b = nullptr;
  const double *vsigma_aa = nullptr; ///< GGA and above
  const double *vsigma_ab = nullptr;
  const double *vsigma_bb = nullptr;
  const double *vtau_a = nullptr; ///< meta-GGA
  const double *vtau_b = nullptr;
  const double *vlapl_a = nullptr; ///< MetaGGALapl only
  const double *vlapl_b = nullptr;
};

/**
 *  @brief Libxc polarized derivative arrays exactly as Libxc writes them.
 *
 *  Interleaved per grid point: vrho as (a, b), vsigma as (aa, ab, bb), vtau
 *  as (a, b), vlapl as (a, b). So vrho is 2*npts doubles, vsigma is 3*npts,
 *  and vsigma[3*g + 1] is the cross term at point g.
 */
struct XCPackedDerivsU {
  const double *vrho = nullptr;   ///< 2*npts
  const double *vsigma = nullptr; ///< 3*npts, GGA and above
  const double *vtau = nullptr;   ///< 2*npts, meta-GGA
  const double *vlapl = nullptr;  ///< 2*npts, MetaGGALapl only
};

/**
 *  @brief De-interleaved copies of Libxc's polarized derivative arrays.
 *
 *  Owns the storage, so an SCF loop can keep one of these alive and unpack
 *  into it every iteration without allocating. Only the arrays the family
 *  needs are sized; the rest stay empty and unpack() leaves the matching
 *  pointers null, which is what makes the operand check below able to tell
 *  "not needed" from "forgotten".
 */
struct XCUnpackedDerivsU {
  std::vector<double> vrho_a, vrho_b;
  std::vector<double> vsigma_aa, vsigma_ab, vsigma_bb;
  std::vector<double> vtau_a, vtau_b;
  std::vector<double> vlapl_a, vlapl_b;

  /// Size the arrays this family needs, so unpack() allocates nothing.
  void resize(XCFamily f, int64_t npts) {
    const size_t n = static_cast<size_t>(npts);
    vrho_a.resize(n);
    vrho_b.resize(n);
    if (f != XCFamily::LDA) {
      vsigma_aa.resize(n);
      vsigma_ab.resize(n);
      vsigma_bb.resize(n);
    }
    if (f == XCFamily::MetaGGATau || f == XCFamily::MetaGGALapl) {
      vtau_a.resize(n);
      vtau_b.resize(n);
    }
    if (f == XCFamily::MetaGGALapl) {
      vlapl_a.resize(n);
      vlapl_b.resize(n);
    }
  }

  /**
   *  @brief De-interleave @p packed into this storage.
   *
   *  Resizes first, which is a no-op once it has been called for the same
   *  family and point count.
   *
   *  @return pointers into this object; it must outlive them
   *  @throws std::invalid_argument if an array the family needs is null
   */
  XCKernelDerivsU unpack(XCFamily f, int64_t npts,
                         const XCPackedDerivsU &packed);
};

namespace detail {

/// Copy component @p icomp of an @p ncomp-way interleaved Libxc array.
inline void unpack_component(const double *src, int ncomp, int icomp,
                             int64_t npts, std::vector<double> &dst) {
  for (int64_t g = 0; g < npts; ++g)
    dst[static_cast<size_t>(g)] = src[g * ncomp + icomp];
}

inline void check_operands_u(XCFamily f, XCSpin s, const XCKernelFieldsU &fl,
                             const XCKernelDerivsU &d) {
  require(fl.w, "w", f);
  // Both channels read both derivatives of rho and both gradients, so the
  // requirement does not depend on which spin is being assembled.
  require(d.vrho_a, "vrho_a", f);
  require(d.vrho_b, "vrho_b", f);
  if (f != XCFamily::LDA) {
    require(fl.rho_a_x, "rho_a_x", f);
    require(fl.rho_a_y, "rho_a_y", f);
    require(fl.rho_a_z, "rho_a_z", f);
    require(fl.rho_b_x, "rho_b_x", f);
    require(fl.rho_b_y, "rho_b_y", f);
    require(fl.rho_b_z, "rho_b_z", f);
    require(d.vsigma_ab, "vsigma_ab", f);
    // The same-spin sigma derivative is the one that differs by channel.
    if (s == XCSpin::Alpha)
      require(d.vsigma_aa, "vsigma_aa", f);
    else
      require(d.vsigma_bb, "vsigma_bb", f);
  }
  if (f == XCFamily::MetaGGATau || f == XCFamily::MetaGGALapl)
    require(s == XCSpin::Alpha ? d.vtau_a : d.vtau_b,
            s == XCSpin::Alpha ? "vtau_a" : "vtau_b", f);
  if (f == XCFamily::MetaGGALapl)
    require(s == XCSpin::Alpha ? d.vlapl_a : d.vlapl_b,
            s == XCSpin::Alpha ? "vlapl_a" : "vlapl_b", f);
}

} // namespace detail

inline XCKernelDerivsU
XCUnpackedDerivsU::unpack(XCFamily f, int64_t npts,
                          const XCPackedDerivsU &packed) {
  resize(f, npts);
  detail::require(packed.vrho, "packed vrho", f);
  detail::unpack_component(packed.vrho, 2, 0, npts, vrho_a);
  detail::unpack_component(packed.vrho, 2, 1, npts, vrho_b);

  XCKernelDerivsU d{.vrho_a = vrho_a.data(), .vrho_b = vrho_b.data()};

  if (f != XCFamily::LDA) {
    detail::require(packed.vsigma, "packed vsigma", f);
    detail::unpack_component(packed.vsigma, 3, 0, npts, vsigma_aa);
    detail::unpack_component(packed.vsigma, 3, 1, npts, vsigma_ab);
    detail::unpack_component(packed.vsigma, 3, 2, npts, vsigma_bb);
    d.vsigma_aa = vsigma_aa.data();
    d.vsigma_ab = vsigma_ab.data();
    d.vsigma_bb = vsigma_bb.data();
  }
  if (f == XCFamily::MetaGGATau || f == XCFamily::MetaGGALapl) {
    detail::require(packed.vtau, "packed vtau", f);
    detail::unpack_component(packed.vtau, 2, 0, npts, vtau_a);
    detail::unpack_component(packed.vtau, 2, 1, npts, vtau_b);
    d.vtau_a = vtau_a.data();
    d.vtau_b = vtau_b.data();
  }
  if (f == XCFamily::MetaGGALapl) {
    detail::require(packed.vlapl, "packed vlapl", f);
    detail::unpack_component(packed.vlapl, 2, 0, npts, vlapl_a);
    detail::unpack_component(packed.vlapl, 2, 1, npts, vlapl_b);
    d.vlapl_a = vlapl_a.data();
    d.vlapl_b = vlapl_b.data();
  }
  return d;
}

/**
 *  @brief Scratch the unrestricted Fock kernels need, in doubles.
 *
 *  The larger of the two channels' requirements, so one buffer serves both.
 *  They happen to be equal at libxckernel 0.5.0, but that is not a promise
 *  the generated headers make.
 */
inline int64_t xc_fock_work_size_unrestricted(XCFamily f, int64_t npts,
                                             int64_t nbf) {
  switch (f) {
  case XCFamily::LDA:
    return std::max(xckernel::xck_lda_ua_o1_work(npts, nbf),
                    xckernel::xck_lda_ub_o1_work(npts, nbf));
  case XCFamily::GGA:
    return std::max(xckernel::xck_gga_ua_o1_work(npts, nbf),
                    xckernel::xck_gga_ub_o1_work(npts, nbf));
  case XCFamily::MetaGGATau:
    return std::max(xckernel::xck_mgga_tau_ua_o1_work(npts, nbf),
                    xckernel::xck_mgga_tau_ub_o1_work(npts, nbf));
  case XCFamily::MetaGGALapl:
    return std::max(xckernel::xck_mgga_ua_o1_work(npts, nbf),
                    xckernel::xck_mgga_ub_o1_work(npts, nbf));
  }
  throw std::invalid_argument("xc_fock_work_size_unrestricted: unknown family");
}

/**
 *  @brief Accumulate one spin channel of the unrestricted XC Fock matrix.
 *
 *  Call it twice, once per XCSpin, with the same fields and derivatives:
 *  everything but @p s and @p out is shared between the channels.
 *
 *  @param s     which channel, i.e. whether the result is d Exc / d P^alpha
 *               or d Exc / d P^beta
 *  @param chi   collocation tower, (ncomp, nbf, npts), ncomp as
 *               collocation_deriv(f) requires -- the same tower for both
 *               channels, since the basis does not know about spin
 *  @param out   (nbf, nbf), ACCUMULATED into
 *  @param work  at least xc_fock_work_size_unrestricted(f, npts, nbf)
 *               doubles; null makes the kernel allocate internally
 *
 *  Returns the kernel's status, nonzero only on an internal allocation
 *  failure -- which cannot happen when `work` is supplied.
 */
inline int xc_fock_unrestricted(XCFamily f, XCSpin s, int64_t npts,
                                int64_t nbf, const double *chi,
                                const XCKernelFieldsU &fl,
                                const XCKernelDerivsU &d, double *out,
                                double *work) {
  detail::check_operands_u(f, s, fl, d);
  if (!chi || !out)
    throw std::invalid_argument("xc_fock_unrestricted: null chi or out");

  const bool alpha = (s == XCSpin::Alpha);

  switch (f) {
  case XCFamily::LDA: {
    if (alpha) {
      const xckernel::xck_lda_ua_o1_fields<double> kf{.w = fl.w};
      const xckernel::xck_lda_ua_o1_xc<double> kx{.vrho_0 = d.vrho_a};
      return xckernel::xck_lda_ua_o1_t<double>(npts, nbf, chi, kf, kx, out,
                                               work);
    }
    const xckernel::xck_lda_ub_o1_fields<double> kf{.w = fl.w};
    const xckernel::xck_lda_ub_o1_xc<double> kx{.vrho_1 = d.vrho_b};
    return xckernel::xck_lda_ub_o1_t<double>(npts, nbf, chi, kf, kx, out,
                                             work);
  }
  case XCFamily::GGA: {
    if (alpha) {
      const xckernel::xck_gga_ua_o1_fields<double> kf{
          .w = fl.w,
          .rho_a_x = fl.rho_a_x,
          .rho_a_y = fl.rho_a_y,
          .rho_a_z = fl.rho_a_z,
          .rho_b_x = fl.rho_b_x,
          .rho_b_y = fl.rho_b_y,
          .rho_b_z = fl.rho_b_z};
      // vsigma_0 is aa, vsigma_1 is ab.
      const xckernel::xck_gga_ua_o1_xc<double> kx{.vrho_0 = d.vrho_a,
                                                  .vsigma_0 = d.vsigma_aa,
                                                  .vsigma_1 = d.vsigma_ab};
      return xckernel::xck_gga_ua_o1_t<double>(npts, nbf, chi, kf, kx, out,
                                               work);
    }
    const xckernel::xck_gga_ub_o1_fields<double> kf{.w = fl.w,
                                                    .rho_a_x = fl.rho_a_x,
                                                    .rho_a_y = fl.rho_a_y,
                                                    .rho_a_z = fl.rho_a_z,
                                                    .rho_b_x = fl.rho_b_x,
                                                    .rho_b_y = fl.rho_b_y,
                                                    .rho_b_z = fl.rho_b_z};
    // vsigma_1 is ab, vsigma_2 is bb: the slots are not the alpha ones.
    const xckernel::xck_gga_ub_o1_xc<double> kx{.vrho_1 = d.vrho_b,
                                                .vsigma_1 = d.vsigma_ab,
                                                .vsigma_2 = d.vsigma_bb};
    return xckernel::xck_gga_ub_o1_t<double>(npts, nbf, chi, kf, kx, out,
                                             work);
  }
  case XCFamily::MetaGGATau: {
    if (alpha) {
      const xckernel::xck_mgga_tau_ua_o1_fields<double> kf{
          .w = fl.w,
          .rho_a_x = fl.rho_a_x,
          .rho_a_y = fl.rho_a_y,
          .rho_a_z = fl.rho_a_z,
          .rho_b_x = fl.rho_b_x,
          .rho_b_y = fl.rho_b_y,
          .rho_b_z = fl.rho_b_z};
      const xckernel::xck_mgga_tau_ua_o1_xc<double> kx{
          .vrho_0 = d.vrho_a,
          .vsigma_0 = d.vsigma_aa,
          .vsigma_1 = d.vsigma_ab,
          .vtau_0 = d.vtau_a};
      return xckernel::xck_mgga_tau_ua_o1_t<double>(npts, nbf, chi, kf, kx,
                                                    out, work);
    }
    const xckernel::xck_mgga_tau_ub_o1_fields<double> kf{
        .w = fl.w,
        .rho_a_x = fl.rho_a_x,
        .rho_a_y = fl.rho_a_y,
        .rho_a_z = fl.rho_a_z,
        .rho_b_x = fl.rho_b_x,
        .rho_b_y = fl.rho_b_y,
        .rho_b_z = fl.rho_b_z};
    const xckernel::xck_mgga_tau_ub_o1_xc<double> kx{.vrho_1 = d.vrho_b,
                                                     .vsigma_1 = d.vsigma_ab,
                                                     .vsigma_2 = d.vsigma_bb,
                                                     .vtau_1 = d.vtau_b};
    return xckernel::xck_mgga_tau_ub_o1_t<double>(npts, nbf, chi, kf, kx, out,
                                                  work);
  }
  case XCFamily::MetaGGALapl: {
    if (alpha) {
      const xckernel::xck_mgga_ua_o1_fields<double> kf{
          .w = fl.w,
          .rho_a_x = fl.rho_a_x,
          .rho_a_y = fl.rho_a_y,
          .rho_a_z = fl.rho_a_z,
          .rho_b_x = fl.rho_b_x,
          .rho_b_y = fl.rho_b_y,
          .rho_b_z = fl.rho_b_z};
      // Named, so the vlapl-first order is the compiler's problem.
      const xckernel::xck_mgga_ua_o1_xc<double> kx{.vlapl_0 = d.vlapl_a,
                                                   .vrho_0 = d.vrho_a,
                                                   .vsigma_0 = d.vsigma_aa,
                                                   .vsigma_1 = d.vsigma_ab,
                                                   .vtau_0 = d.vtau_a};
      return xckernel::xck_mgga_ua_o1_t<double>(npts, nbf, chi, kf, kx, out,
                                                work);
    }
    const xckernel::xck_mgga_ub_o1_fields<double> kf{.w = fl.w,
                                                     .rho_a_x = fl.rho_a_x,
                                                     .rho_a_y = fl.rho_a_y,
                                                     .rho_a_z = fl.rho_a_z,
                                                     .rho_b_x = fl.rho_b_x,
                                                     .rho_b_y = fl.rho_b_y,
                                                     .rho_b_z = fl.rho_b_z};
    const xckernel::xck_mgga_ub_o1_xc<double> kx{.vlapl_1 = d.vlapl_b,
                                                 .vrho_1 = d.vrho_b,
                                                 .vsigma_1 = d.vsigma_ab,
                                                 .vsigma_2 = d.vsigma_bb,
                                                 .vtau_1 = d.vtau_b};
    return xckernel::xck_mgga_ub_o1_t<double>(npts, nbf, chi, kf, kx, out,
                                              work);
  }
  }
  throw std::invalid_argument("xc_fock_unrestricted: unknown family");
}

} // namespace Nukexc
