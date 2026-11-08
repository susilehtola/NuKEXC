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
//  The basis-evaluator concept
// ============================================================================
//
// Everything downstream of the basis -- densities, collocation, the integral
// kernels -- is written once against this concept and instantiated per basis
// type. Static polymorphism, so there is no dispatch at run time: the
// evaluator inlines into the kernel that uses it.
//
// (The cost that inlining *does* carry on a GPU is register pressure, not call
// overhead, and batching does not amortize it. An evaluator whose inner
// evaluation is register-heavy should mark it NUKEXC_NOINLINE_FUNCTION, which
// caps the caller's live-register set at max-over-call-graph instead of the
// inlined sum. See nukexc_config.hpp.)
//
// BLOCKS
// ------
// The unit of work is a *block*: the set of basis functions that share
// evaluation work and are therefore cheapest to produce together.
//
//   STO primitive    one function per block  (each is independent)
//   STO contracted   one function per block
//   GTO              one contraction of one shell per block, so all 2l+1
//                    spherical components come out of a single pass over the
//                    primitives and one Cartesian-to-spherical transform
//
// Writing the interface per function rather than per block would force the
// GTO path to redo a shell's primitive loop once per m, so the block is the
// granularity that keeps every basis type efficient.
//
// Blocks also carry the screening geometry: one center and one cutoff radius
// each, which is exactly what the ArborX sphere/box neighbour lists consume.
//
// WHAT AN EVALUATOR MUST PROVIDE
// ------------------------------
//   using block_params = <trivially copyable handle>
//
//   static constexpr int max_block_size    compile-time bound on block_size,
//                                          so per-thread output buffers are
//                                          sized to the basis type (1 for an
//                                          STO, cart_size(l_max) for a GTO)
//   int  nbf()                    total basis functions
//   int  nblock()                 number of blocks
//   int  block_first_bf(int b)    first basis function of block b
//   int  block_size(int b)        functions in block b
//   View<Point*>  centers()       per block, for screening
//   View<double*> radii()         per block, for screening
//
//   block_params load_block(int b) const                      (device)
//   template <BasisDeriv D>
//   void eval_block(block_params, x, y, z,
//                   double* out, int stride) const            (device)
//
// eval_block writes basis_ncomp(D) components for each function of the block,
// component-major: out[icomp * stride + ifunc].

#include <concepts>
#include <stdexcept>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "nukexc_config.hpp"

namespace Nukexc {

// NAMING
// ------
// NuKEXC follows libxckernel's vocabulary, not ERKALE's, because libxckernel
// is what supplies the XC kernels from Phase 4 onward and its operand names
// are the actual interface between the two.
//
// Derivative ORDER indexes the functional derivative, as in libxckernel's
// xck_<family>_<spin>_o<order>:
//
//     o0   XC energy
//     o1   XC Fock matrix          <- "vxc" in GauXC/PySCF and the literature
//     o2+  response contraction    <- "fxc" is order 2, and only order 2
//
// ERKALE spells the order-1 quantity `Fxc` (capital F for Fock). That reads
// unambiguously in ERKALE but collides badly in lowercase C++, where `fxc`
// means the second derivative everywhere else. So the order-1 host entry
// point here is eval_vxc, and the name fxc is reserved for order 2.
//
// The component layout below is likewise chosen to match libxckernel's
// operands: see collocation_operands().

/// Which derivatives an evaluation should produce.
enum class BasisDeriv { Value = 0, Gradient = 1, Hessian = 2, LaplGradient = 3 };

/// Components carried by a derivative order: value (1), gradient (+3),
/// Hessian (+6 unique), gradient of the Laplacian (+3).
KOKKOS_INLINE_FUNCTION
constexpr int basis_ncomp(BasisDeriv d) {
  return d == BasisDeriv::Value      ? 1
         : d == BasisDeriv::Gradient ? 4
         : d == BasisDeriv::Hessian  ? 10
                                     : 13;
}

/// Component indices within an eval_block output buffer.
///
/// The ordering is libxckernel's: components BF_DX..BF_DZ form its `dchi`
/// operand and BF_DXX..BF_DZZ its `hess_chi`, which it documents as packed
/// xx,xy,xz,yy,yz,zz. A rank-3 (ncomp, nbf, ng) collocation array therefore
/// yields every operand as a contiguous slice -- see collocation_operands().
/// Plain ints rather than an enum: Kokkos::subview rejects an unscoped enum
/// as an integral index, and these are used exactly that way to slice out the
/// libxckernel operands.
inline constexpr int BF_VAL = 0;
inline constexpr int BF_DX = 1;
inline constexpr int BF_DY = 2;
inline constexpr int BF_DZ = 3;
inline constexpr int BF_DXX = 4;
inline constexpr int BF_DXY = 5;
inline constexpr int BF_DXZ = 6;
inline constexpr int BF_DYY = 7;
inline constexpr int BF_DYZ = 8;
inline constexpr int BF_DZZ = 9;
inline constexpr int BF_LGX = 10;
inline constexpr int BF_LGY = 11;
inline constexpr int BF_LGZ = 12;

/**
 *  @brief What density, collocation and the integral kernels require of a
 *         basis set.
 *
 *  The templated eval_block is checked at BasisDeriv::Value only -- a concept
 *  cannot readily quantify over the template argument, and instantiating the
 *  higher orders would reject an evaluator that legitimately supports only
 *  values. The remaining orders are checked where they are used.
 */
template <typename E>
concept BasisEvaluator = requires(const E e, int b) {
  typename E::block_params;
  requires std::is_trivially_copyable_v<typename E::block_params>;

  { std::integral_constant<int, E::max_block_size>{} };
  requires E::max_block_size >= 1;

  { e.nbf() } -> std::convertible_to<int>;
  { e.nblock() } -> std::convertible_to<int>;
  { e.block_first_bf(b) } -> std::convertible_to<int>;
  { e.block_size(b) } -> std::convertible_to<int>;
  { e.centers() } -> std::convertible_to<Kokkos::View<Point *>>;
  { e.radii() } -> std::convertible_to<Kokkos::View<double *>>;
  { e.load_block(b) } -> std::same_as<typename E::block_params>;
};

/// An evaluator that can also produce the given derivative order.
template <typename E, BasisDeriv D>
concept BasisEvaluatorWith =
    BasisEvaluator<E> &&
    // An evaluator must ADVERTISE the order, not merely accept the call.
    // eval_block<D> is a template whose unsupported orders fail on a
    // static_assert in the body, so a plain `requires` that the call is
    // well-formed is satisfied by every D -- which made this concept claim
    // orders the primitive STO path cannot deliver, and pushed the failure
    // from the caller down into stobasis.hpp. max_deriv is monotone in
    // BasisDeriv: a tower of a given order contains every lower one.
    (static_cast<int>(E::max_deriv) >= static_cast<int>(D)) &&
    requires(const E e, typename E::block_params p, double *out) {
      e.template eval_block<D>(p, 0.0, 0.0, 0.0, out, 1);
    };

/**
 *  @brief An evaluator that can also produce the electrostatic potential each
 *         basis function generates when treated as a charge density.
 *
 *      V_i(r) = integral  chi_i(r') / |r - r'|  d3r'
 *
 *  This is what the density-fitted Coulomb and exchange builds contract
 *  against, and it is available in closed form for both basis families --
 *  incomplete gamma functions for a Slater function, and the
 *  McMurchie-Davidson R_tuv auxiliaries over Boys functions for a Cartesian
 *  Gaussian. Different special functions, same interface.
 *
 *  Kept separate from BasisEvaluator because the XC path has no use for it,
 *  and requiring it there would force every evaluator to carry the machinery.
 */
template <typename E>
concept PotentialEvaluator =
    BasisEvaluator<E> && requires(const E e, typename E::block_params p,
                                  double *out) {
      e.eval_potential_block(p, 0.0, 0.0, 0.0, out, 1);
    };

// The single-function shims that used to live here -- eval_aux_potential,
// SingleBfHandle, and the load_shell/basis_eval_fast pair over an evaluator
// -- are gone. They existed so that kernels indexing the basis by FUNCTION
// could keep their shape while going through the evaluator concept, and they
// static_asserted max_block_size == 1 to stop a Gaussian basis reaching
// them. Every such kernel has since been rewritten to walk shells (issue
// #6), so the shims had no callers left, and keeping them would invite the
// indexing they were built to tolerate. stobasis.hpp keeps its own
// ShellParams and basis_eval_fast, which the Slater evaluator uses
// internally for its single-function blocks.

// ============================================================================
//  Generic collocation
// ============================================================================

/**
 *  @brief Fill a collocation block for any basis over a batch of points.
 *
 *  @param comp (basis_ncomp(D), nbf, npts): component, basis function, point.
 *
 *  One thread per (block, point). Each thread writes only the functions of
 *  its own block, so there are no atomics and no cross-thread coordination.
 */
template <BasisDeriv D, typename Evaluator, typename PointsView,
          typename CompView>
  requires BasisEvaluatorWith<Evaluator, D>
void fill_collocation(ExecSpace space, const Evaluator &basis,
                      PointsView points, CompView comp) {
  constexpr int NC = basis_ncomp(D);
  const int npts = static_cast<int>(points.extent(0));
  const int nblk = basis.nblock();
  if (nblk == 0 || npts == 0)
    return;

  if (static_cast<int>(comp.extent(0)) != NC ||
      static_cast<int>(comp.extent(1)) != basis.nbf() ||
      static_cast<int>(comp.extent(2)) != npts)
    throw std::invalid_argument(
        "fill_collocation: output must be (basis_ncomp(D), nbf, npts)");

  constexpr int MBS = Evaluator::max_block_size;

  Kokkos::parallel_for(
      "fill_collocation",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0},
                                                        {nblk, npts}),
      KOKKOS_LAMBDA(const int ib, const int g) {
        const auto p = basis.load_block(ib);
        const int bf0 = basis.block_first_bf(ib);
        const int n = basis.block_size(ib);

        double local[NC * MBS];
        basis.template eval_block<D>(p, points(g)[0], points(g)[1],
                                     points(g)[2], local, MBS);

        for (int c = 0; c < NC; ++c)
          for (int i = 0; i < n; ++i)
            comp(c, bf0 + i, g) = local[c * MBS + i];
      });
}

/**
 *  @brief Value-only collocation into a 2D (nbf, npts) matrix.
 *
 *  The shape the dense GEMM paths want. Separate from the component-major
 *  fill_collocation so those callers neither allocate a rank-3 view nor copy
 *  a slice out of one.
 */
template <typename Evaluator, typename PointsView, typename ValuesView>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value>
void fill_collocation_values(ExecSpace space, const Evaluator &basis,
                             PointsView points, ValuesView values) {
  const int npts = static_cast<int>(points.extent(0));
  const int nblk = basis.nblock();
  if (nblk == 0 || npts == 0)
    return;

  constexpr int MBS = Evaluator::max_block_size;
  Kokkos::parallel_for(
      "fill_collocation_values",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0},
                                                        {nblk, npts}),
      KOKKOS_LAMBDA(const int ib, const int g) {
        double local[MBS];
        basis.template eval_block<BasisDeriv::Value>(
            basis.load_block(ib), points(g)[0], points(g)[1], points(g)[2],
            local, 1);
        const int bf0 = basis.block_first_bf(ib);
        const int n = basis.block_size(ib);
        for (int i = 0; i < n; ++i)
          values(bf0 + i, g) = local[i];
      });
}

/**
 *  @brief Values and Cartesian gradients into four separate (nbf, npts) views.
 *
 *  The shape the dense core-Hamiltonian GEMMs want, for the same reason
 *  fill_collocation_values exists: those paths build S, T and V_n as
 *  A A^T products over per-component matrices, so a component-major rank-3
 *  tower would have to be sliced and copied before any of it could be used.
 *
 *  Like every routine here it sees the basis only as SHELLS -- blocks of
 *  functions on a centre, with a range used for screening. Nothing about
 *  the radial form reaches this level, which is why the same quadrature
 *  serves Gaussians, Slater functions and numerical atomic orbitals alike.
 *
 *  One thread per (shell, point), so each thread writes only its own
 *  shell's functions: no atomics, and no assumption that a shell holds a
 *  single function.
 */
template <typename Evaluator, typename PointsView, typename ValuesView,
          typename GradView>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
void fill_collocation_value_grad(ExecSpace space, const Evaluator &basis,
                                 PointsView points, ValuesView values,
                                 GradView gx, GradView gy, GradView gz) {
  const int npts = static_cast<int>(points.extent(0));
  const int nblk = basis.nblock();
  if (nblk == 0 || npts == 0)
    return;

  const int nbf = basis.nbf();
  for (const auto &v : {std::pair<int, int>{static_cast<int>(values.extent(0)),
                                            static_cast<int>(values.extent(1))},
                        {static_cast<int>(gx.extent(0)),
                         static_cast<int>(gx.extent(1))},
                        {static_cast<int>(gy.extent(0)),
                         static_cast<int>(gy.extent(1))},
                        {static_cast<int>(gz.extent(0)),
                         static_cast<int>(gz.extent(1))}})
    if (v.first != nbf || v.second < npts)
      throw std::invalid_argument(
          "fill_collocation_value_grad: outputs must be (nbf, >= npts)");

  constexpr int NC = basis_ncomp(BasisDeriv::Gradient);
  constexpr int MBS = Evaluator::max_block_size;

  Kokkos::parallel_for(
      "fill_collocation_value_grad",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0},
                                                        {nblk, npts}),
      KOKKOS_LAMBDA(const int ib, const int g) {
        double local[NC * MBS];
        basis.template eval_block<BasisDeriv::Gradient>(
            basis.load_block(ib), points(g)[0], points(g)[1], points(g)[2],
            local, MBS);
        const int bf0 = basis.block_first_bf(ib);
        const int n = basis.block_size(ib);
        for (int i = 0; i < n; ++i) {
          values(bf0 + i, g) = local[BF_VAL * MBS + i];
          gx(bf0 + i, g) = local[BF_DX * MBS + i];
          gy(bf0 + i, g) = local[BF_DY * MBS + i];
          gz(bf0 + i, g) = local[BF_DZ * MBS + i];
        }
      });
}

/**
 *  @brief libxckernel's collocation operands, as views onto a (ncomp, nbf, ng)
 *         array produced by fill_collocation.
 *
 *  Named exactly as the generated kernels' manifests name them, so a call site
 *  reads the same as the manifest it is satisfying:
 *
 *      chi       (nbf, ng)
 *      dchi      (3, nbf, ng)     packed x, y, z
 *      hess_chi  (6, nbf, ng)     packed xx, xy, xz, yy, yz, zz
 *
 *  These are subviews, not copies: the layout was chosen so no repacking is
 *  needed between NuKEXC's evaluator and libxckernel's kernels.
 */
template <typename CompView> struct CollocationOperands {
  using value_type = typename CompView::non_const_value_type;
  using memory_space = typename CompView::memory_space;
  // Subviews of a rank-3 array are strided, not contiguous in the Kokkos
  // type sense, even though each operand's own data is.
  using chi_type =
      Kokkos::View<value_type **, Kokkos::LayoutStride, memory_space>;
  using block_type =
      Kokkos::View<value_type ***, Kokkos::LayoutStride, memory_space>;

  chi_type chi;
  block_type dchi;
  block_type hess_chi;
  bool has_dchi = false;
  bool has_hess_chi = false;
};

/// Build the operand views for a collocation array of the given order.
template <BasisDeriv D, typename CompView>
CollocationOperands<CompView> collocation_operands(const CompView &comp) {
  constexpr bool has_d = D >= BasisDeriv::Gradient;
  constexpr bool has_h = D >= BasisDeriv::Hessian;

  CollocationOperands<CompView> op;
  op.chi = Kokkos::subview(comp, BF_VAL, Kokkos::ALL(), Kokkos::ALL());
  // A degenerate range keeps the type uniform when an operand is absent; the
  // has_* flags say whether it means anything.
  op.dchi = Kokkos::subview(
      comp, Kokkos::pair<int, int>{BF_DX, has_d ? BF_DZ + 1 : BF_DX},
      Kokkos::ALL(), Kokkos::ALL());
  op.hess_chi = Kokkos::subview(
      comp, Kokkos::pair<int, int>{BF_DXX, has_h ? BF_DZZ + 1 : BF_DXX},
      Kokkos::ALL(), Kokkos::ALL());
  op.has_dchi = has_d;
  op.has_hess_chi = has_h;
  return op;
}

} // namespace Nukexc
