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

#include <cstdint>
#include <stdexcept>
#include <string>

#include "nukexc_config.hpp"

#include "xckernel/kernels/xck_gga_r_o0.hpp"
#include "xckernel/kernels/xck_gga_r_o1.hpp"
#include "xckernel/kernels/xck_lda_r_o0.hpp"
#include "xckernel/kernels/xck_lda_r_o1.hpp"
#include "xckernel/kernels/xck_mgga_r_o0.hpp"
#include "xckernel/kernels/xck_mgga_r_o1.hpp"
#include "xckernel/kernels/xck_mgga_tau_r_o0.hpp"
#include "xckernel/kernels/xck_mgga_tau_r_o1.hpp"

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
 */
inline double xc_energy_restricted(int64_t npts, const double *w,
                                   const double *rho, const double *zk) {
  if (!w || !rho || !zk)
    throw std::invalid_argument("xc_energy_restricted: null operand");
  return xckernel::xck_lda_r_o0_t<double>(npts, w, rho, zk);
}

} // namespace Nukexc
