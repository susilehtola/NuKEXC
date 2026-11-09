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
//  The screened polarized XC quadrature
// ============================================================================
//
// Orbitals in, device matrices out: the shape xc_integrals.hpp's dense entry
// points have, so a caller can hold both and compare them -- which
// tests/test_xc_screened.cxx does. What this one does NOT have is a
// chain rule of its own. The potential comes from libxckernel by way of
// XCScreenedKernelSession, which is why ONE function covers LDA, GGA and
// tau meta-GGA where there used to be one per family, each carrying its own
// derivative algebra.
//
// What was deleted, and why it had to be
// --------------------------------------
// compute_gga_lsda_sparse built its potential from
//
//     vec_up = 2 vsigma_uu grad rho_a + vsigma_ud grad rho_b
//     Z      = w (1/2 chi vrho + vec . grad chi),  V = Z chi^T + its transpose
//
// which is the GGA functional derivative with respect to the density matrix
// written out by hand. NuKEXC's rule is that no XC chain-rule term is
// hand-derived here: Libxc owns the functional-derivative tower,
// libxckernel owns the derivatives of the ingredients with respect to the
// density matrix, and this library owns the ingredients and the quadrature.
// That rule exists because the term above is exactly the kind that stays
// plausible while being wrong -- a factor of two on the same-spin sigma, or
// the two spin gradients swapped, costs a few microhartree and nothing
// crashes. The generated kernel gets it from the same tables that produced
// the restricted and the unrestricted cases, and a regeneration that
// changed an operand list fails to build rather than to converge.
//
// The hand-written DENSE path in xc_integrals.hpp still has that algebra,
// deliberately: it is validated against ERKALE and is therefore the only
// oracle the kernels have. An oracle one has edited is not an oracle. So
// this is the screened half of the convergence, and the test that compares
// it against the dense hand-written path is now a parity test between two
// independent derivations rather than between two spellings of one.
//
// WHAT IT COST, MEASURED
// ----------------------
// Nothing, and both families got faster. Per polarized Fock build on
// benzene, 150 functions, 35360 points, three runs back to back from
// benchmark_xc --nrad=35 --nang=15 --box-size=128:
//
//     path                              LSDA                GGA
//     hand-written (deleted)      1.62 / 1.69 / 1.82  2.47 / 1.79 / 2.88
//     kernels, libxckernel 0.5.0  1.14 / 1.17 / 1.15  1.93 / 2.03 / 1.96
//     kernels, libxckernel 0.7.0  1.11 / 1.01 / 1.03  1.70 / 1.55 / 1.52
//
// The energies agree with the deleted path to all eight printed digits, and
// the whole water PBE SCF converges to the same -76.3833763115 Eh through
// this path and the dense hand-written one -- ten digits, not a tolerance.
//
// The GGA row is the one worth reading twice. At 0.5.0 it was a wash:
// xck_gga_ua_o1 wrote the potential as four rank updates where the
// hand-derived form above exploited the symmetry of the result and needed
// one, and four times the GEMM flops ate the whole saving. 0.7.0's shared
// product planner emits exactly that one product -- accumulate the half
// product, then add it to its own transpose -- so the generated kernel now
// does the same algebra the deleted code did, and is faster than it.
//
// What is left of the dependency is the BLAS flag: without
// XCKERNEL_USE_BLAS the 0.7.0 GGA build takes 2.1-2.3 s and the LSDA
// 1.5-1.7, i.e. back to roughly the path it replaced. The LSDA is the
// family libxckernel has always emitted as one product, so what the flag
// buys there is purely GEMM quality.
//
// WHY IT LIVES IN ITS OWN HEADER
// ------------------------------
// host_interface.hpp includes xc_integrals.hpp, so a function that needs
// both XC_result_polarized and the session cannot live in xc_integrals.hpp
// without closing the include cycle.

#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <xc.h>

#include "basis_evaluator.hpp"
#include "density.hpp"
#include "grid.hpp"
#include "nukexc_config.hpp"
#include "octree.hpp"
#include "xc_integrals.hpp"
#include "xc_kernel_session.hpp"

namespace Nukexc {

/**
 *  @brief Screened spin-polarized XC energy and potentials.
 *
 *  @param basis      shell evaluator; gradients are required, since every
 *                    family above an LDA needs them and the family is not
 *                    known until @p func is read
 *  @param nl         shell neighbour list over the boxed grid
 *  @param mo_alpha   (nbf, n_occ) orbitals, contracted with @p occ_alpha
 *  @param func       a functional initialized with XC_POLARIZED
 *
 *  Covers LDA, GGA and tau meta-GGA. A Laplacian meta-GGA throws: a packed
 *  collocation block stops at the gradient.
 *
 *  @return the energy and both spin channels' potentials, as device matrices
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
inline XC_result_polarized compute_xc_polarized_sparse(
    const Evaluator &basis, const FlatGrid &grid, const NeighborList &nl,
    const DeviceView2DLeft &mo_alpha, const DeviceView1D &occ_alpha,
    const DeviceView2DLeft &mo_beta, const DeviceView1D &occ_beta,
    const xc_func_type &func) {
  if (func.nspin != XC_POLARIZED)
    throw std::runtime_error(
        "compute_xc_polarized_sparse requires a functional initialized with "
        "XC_POLARIZED");

  const XCFamily family = xc_family_of(func);
  XCScreenedKernelSession<Evaluator> s(basis, grid, nl, family);
  const int nbf = s.nbf(), ng = s.npoints();

  // A LayoutLeft mirror is column-major and contiguous, so the mirror's own
  // buffer IS the column-major matrix the session's interface takes: the
  // round trip through the host costs nbf^2 doubles and no repacking.
  auto Pa = Kokkos::create_mirror_view_and_copy(
      HostSpace{}, density_matrix_from_orbitals(mo_alpha, occ_alpha));
  auto Pb = Kokkos::create_mirror_view_and_copy(
      HostSpace{}, density_matrix_from_orbitals(mo_beta, occ_beta));
  s.set_density(HostMatrixView{Pa.data(), nbf, MatrixLayout::ColumnMajor},
                HostMatrixView{Pb.data(), nbf, MatrixLayout::ColumnMajor});

  // One Libxc call, in Libxc's own interleaved spin layout -- which is what
  // the session's spin accessors hand back, so there is no packing here
  // either.
  std::vector<double> zk(static_cast<size_t>(ng)),
      vrho(2 * static_cast<size_t>(ng)), vsigma(3 * static_cast<size_t>(ng)),
      vtau(2 * static_cast<size_t>(ng)), vlapl(2 * static_cast<size_t>(ng));
  switch (family) {
  case XCFamily::LDA:
    xc_lda_exc_vxc(&func, ng, s.rho_spin().data(), zk.data(), vrho.data());
    break;
  case XCFamily::GGA:
    xc_gga_exc_vxc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
                   zk.data(), vrho.data(), vsigma.data());
    break;
  case XCFamily::MetaGGATau: {
    // A tau-only meta-GGA reads no Laplacian, but Libxc's interface takes
    // one anyway; zeros are what it is entitled to see, and vlapl comes
    // back unused.
    const std::vector<double> lapl_in(2 * static_cast<size_t>(ng), 0.0);
    xc_mgga_exc_vxc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
                    lapl_in.data(), s.tau_spin().data(), zk.data(),
                    vrho.data(), vsigma.data(), vlapl.data(), vtau.data());
    break;
  }
  default:
    throw std::invalid_argument(
        std::string("compute_xc_polarized_sparse: no screened path for a ") +
        xc_family_name(family) + " functional");
  }

  XCPackedDerivsU d{.vrho = vrho.data()};
  if (family != XCFamily::LDA)
    d.vsigma = vsigma.data();
  if (family == XCFamily::MetaGGATau)
    d.vtau = vtau.data();

  DeviceView2DLeft Va("V alpha", nbf, nbf), Vb("V beta", nbf, nbf);
  auto Vah = Kokkos::create_mirror_view(Va);
  auto Vbh = Kokkos::create_mirror_view(Vb);
  Kokkos::deep_copy(Vah, 0.0);
  Kokkos::deep_copy(Vbh, 0.0);
  const XCOutputPolarized out = s.accumulate_fock(
      d, zk.data(), HostMatrixRef{Vah.data(), nbf, MatrixLayout::ColumnMajor},
      HostMatrixRef{Vbh.data(), nbf, MatrixLayout::ColumnMajor});
  Kokkos::deep_copy(Va, Vah);
  Kokkos::deep_copy(Vb, Vbh);

  return XC_result_polarized{out.Exc, Va, Vb};
}

} // namespace Nukexc
