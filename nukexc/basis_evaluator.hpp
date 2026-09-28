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
enum : int {
  BF_VAL = 0,
  BF_DX = 1,
  BF_DY = 2,
  BF_DZ = 3,
  BF_DXX = 4,
  BF_DXY = 5,
  BF_DXZ = 6,
  BF_DYY = 7,
  BF_DYZ = 8,
  BF_DZZ = 9,
  BF_LGX = 10,
  BF_LGY = 11,
  BF_LGZ = 12
};

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
    BasisEvaluator<E> && requires(const E e, typename E::block_params p,
                                  double *out) {
      e.template eval_block<D>(p, 0.0, 0.0, 0.0, out, 1);
    };

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

} // namespace Nukexc
