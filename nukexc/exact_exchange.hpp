/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Bob Schreiner
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

#include "density.hpp"
#include "grid.hpp"
#include "nukexc/octree.hpp"
#include "aux_potential.hpp"
#include "basis_evaluator.hpp"
#include "nukexc_config.hpp"
#include "packed_blocks.hpp"
#include "stobasis.hpp"
#include "stopotential.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_gemv_impl.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_gemm_impl.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <impl/Kokkos_Profiling.hpp>

namespace Nukexc {

DeviceView2DLeft compute_exact_exchange(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const DeviceView2DLeft basis_collocation,
    const DeviceView2DLeft basis_aux_collocation,
    const DeviceView2DLeft potential_collocation_scaled,
    const DeviceView2DLeft half_inverse_X) {

  Kokkos::Profiling::pushRegion("Compute_Exact_Exchange_Integral");
  const int N_bf = basis_collocation.extent(0);
  const int N_bf_aux = basis_aux_collocation.extent(0);
  const int N_quad = basis_collocation.extent(1);
  const int N_occ = mo_coeff.extent(0);
  const int K = half_inverse_X.extent(1);

  DeviceView2DLeft basis_collocation_scaled("Basis collocation Scaled", N_bf,
                                            N_quad);
  DeviceView2DLeft expansion_coeff("Expansion coeff", N_occ, N_quad);

  DeviceView2DLeft three_center_integral("Three_center_integral", N_bf_aux,
                                         N_bf);
  DeviceView2DLeft three_center_integral_scaled("Three_center_integral_scaled",
                                                N_bf, K);
  DeviceView2DLeft result("Exchange matrix", N_bf, N_bf);

  auto mo_coeff_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mo_coeff);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
  // Loop over the occupied orbitals and compute contributions for each occupied
  // orbital

  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, basis_collocation, 0.0,
                   expansion_coeff);

  for (unsigned int i = 0; i < N_occ; ++i) {

    Kokkos::parallel_for(
        "Scale collocation exact exhange", policy,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int g = team_member.league_rank();
          const double expansion_coeff_g = expansion_coeff(i, g);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, N_bf), [=](const int mu) {
                basis_collocation_scaled(mu, g) =
                    basis_collocation(mu, g) * expansion_coeff_g;
              });
        });

    KokkosBlas::gemm(space, "N", "T", 1.0, potential_collocation_scaled,
                     basis_collocation_scaled, 0.0, three_center_integral);

    KokkosBlas::gemm(space, "T", "N", 1.0, three_center_integral,
                     half_inverse_X, 0.0, three_center_integral_scaled);

    KokkosBlas::gemm(space, "N", "T", mo_coeff_h(i),
                     three_center_integral_scaled, three_center_integral_scaled,
                     1.0, result);
  }
  Kokkos::fence();
  Kokkos::Profiling::popRegion();
  return result;
}

// ============================================================================
//  Screened RI exact exchange
// ============================================================================
//
//     K_uv = sum_i occ_i sum_AB (u i|A) (A|B)^-1 (B|v i)
//
// with the half-transformed three-centre integral built on the grid,
//
//     B_i(A, u) = sum_g w_g V_A(g) psi_i(g) chi_u(g),
//     psi_i(g)  = sum_u C_ui chi_u(g).
//
// Per box: pack the collocation once per (shell, point), evaluate the
// auxiliary potential once per (auxiliary shell, point), form psi for a
// batch of orbitals, and accumulate
//
//     B_i(A, u) += sum_{g in box} [w_g V_A(g)] chi_u(g) psi_i(g)
//
// one dot product per (A, local u), scattered into B through the box's
// local-to-global map. The auxiliary index is NOT screened and cannot be:
// the Coulomb potential has a 1/r tail, so every auxiliary function reaches
// every box.
//
// This is the memory-frugal route, not the fast one: the dense path stores
// the collocation and the auxiliary potential and is about twice as quick
// per build wherever those tables fit. packed_blocks.hpp has the numbers.
//
// WHAT THIS FIXES, AND WHAT IT DOES NOT
// -------------------------------------
// It does not change the formalism, so the occupied-orbital loop that issue
// #8 objects to is still here and the dominant term is still
// O(nocc * naux * nbe * npts) -- with the box's nbe where the dense path has
// nbf, which is the one power the screening buys.
//
// What it does fix is the evaluation count, which was the real cost. The
// kernels this replaces evaluated the primary basis inside the loop over
// auxiliary functions, so a shell was evaluated once per
// (auxiliary function, shell, point, orbital): naux * nbe * npts * nocc
// evaluations where nbe * npts suffices per orbital batch. They evaluated
// the auxiliary potential once per (auxiliary function, point, ORBITAL),
// paying a factor nocc for a quantity that does not depend on the orbital.
// And they formed psi_i by looping over all nbf functions of the basis --
// not the box's -- with one evaluation each, which is the N_occ redundancy
// of issue #12 in its most expensive form.
//
// They also indexed the neighbour list as though it held basis functions and
// the auxiliary basis by function, so neither could be instantiated for a
// Gaussian basis. Nothing here has that restriction.

namespace detail {

/// The maximum occupied-orbital batch, and so the size of the per-point
/// register accumulator in chunk_orbitals_on_grid.
constexpr int MAX_ORBITAL_BATCH = 32;

/**
 *  @brief A batch of occupied orbitals on the points of one chunk of boxes.
 *
 *     psi_i(g) = sum_{u in box} C(u, i) chi_u(g)
 *
 *  with u running over the BOX's functions and the local-to-global map
 *  applied to reach C -- which is the whole point of the screening, since
 *  the alternative is a sum over all nbf functions of the basis.
 *
 *  psi is laid out (batch, chunk points): the exchange build scales one
 *  orbital's points at a time and wants them contiguous.
 */
inline void chunk_orbitals_on_grid(const ExecSpace &space,
                                   const PackedBlockLayout &L, const int b0,
                                   const int b1, const int g0,
                                   const Kokkos::View<double *> &chi,
                                   const DeviceView2DLeft &C, const int i0,
                                   const int nb,
                                   const Kokkos::View<double *> &psi,
                                   const size_t psi_stride) {
  using policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename policy_type::member_type;
  const int nbox = b1 - b0;
  if (nbox <= 0 || nb <= 0)
    return;
  if (nb > MAX_ORBITAL_BATCH)
    throw std::invalid_argument(
        "chunk_orbitals_on_grid: batch exceeds MAX_ORBITAL_BATCH");

  auto nbf_v = L.map.nbf;
  auto npts_v = L.npts;
  auto coll_off = L.coll_off;
  auto bf_off = L.bf_off;
  auto l2g = L.loc2glob;
  const int mppb = L.mppb;

  Kokkos::parallel_for(
      "Occupied orbitals on the grid", policy_type(space, nbox, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const member_type &team) {
        const int b = b0 + team.league_rank();
        const int n = nbf_v(b);
        const int np = npts_v(b);
        if (np == 0)
          return;
        const double *chi_b = chi.data() + (coll_off(b) - coll_off(b0));
        const size_t base = bf_off(b);
        const int gloc = b * mppb - g0;
        double *psi_b = psi.data();

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, np), [&](const int g) {
              double acc[MAX_ORBITAL_BATCH];
              for (int j = 0; j < nb; ++j)
                acc[j] = 0.0;
              for (int i = 0; i < n; ++i) {
                const double ci = chi_b[static_cast<size_t>(i) * np + g];
                const int mu = l2g(base + i);
                for (int j = 0; j < nb; ++j)
                  acc[j] += C(mu, i0 + j) * ci;
              }
              for (int j = 0; j < nb; ++j)
                psi_b[j * psi_stride + static_cast<size_t>(gloc + g)] =
                    acc[j];
            });
      });
}

/**
 *  @brief max_g |psi_j(g)| over each box of a chunk, written at the ABSOLUTE
 *         box index.
 *
 *  One pass over psi -- nbox * np * nb reads, against naux GEMMs per
 *  orbital in the build that uses it -- so the build computes it whether or
 *  not it will screen on it. Boxes with no points are left untouched;
 *  every reader guards on the point count as well.
 */
inline void chunk_box_amplitudes(const ExecSpace &space,
                                 const PackedBlockLayout &L, const int b0,
                                 const int b1, const int g0,
                                 const Kokkos::View<double *> &psi,
                                 const size_t psi_stride, const int nb,
                                 const Kokkos::View<double **> &pmax) {
  using policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename policy_type::member_type;
  const int nbox = b1 - b0;
  if (nbox <= 0 || nb <= 0)
    return;
  auto npts_v = L.npts;
  const int mppb = L.mppb;

  Kokkos::parallel_for(
      "Orbital amplitude per box", policy_type(space, nbox, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const member_type &team) {
        const int b = b0 + team.league_rank();
        const int np = npts_v(b);
        if (np == 0)
          return;
        const int gloc = b * mppb - g0;
        for (int j = 0; j < nb; ++j) {
          double m = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamVectorRange(team, np),
              [&](const int g, double &acc) {
                const double v = Kokkos::abs(
                    psi(j * psi_stride + static_cast<size_t>(gloc + g)));
                if (v > acc)
                  acc = v;
              },
              Kokkos::Max<double>(m));
          Kokkos::single(Kokkos::PerTeam(team), [&]() { pmax(b, j) = m; });
        }
      });
}

} // namespace detail

/// How much the orbital screening actually skipped.
///
/// Counted over (box, orbital) pairs, which is the unit of work the
/// screening decides on: one scale, one GEMM and one scatter per pair.
struct ExchangeScreeningStats {
  long long pairs_total = 0; ///< pairs that carry work at all
  long long pairs_alive = 0; ///< pairs whose work was done

  double alive_fraction() const {
    return pairs_total > 0
               ? static_cast<double>(pairs_alive) /
                     static_cast<double>(pairs_total)
               : 1.0;
  }
};

template <typename Evaluator, typename PotEvaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value> &&
           PotentialEvaluator<PotEvaluator>
inline DeviceView2DLeft compute_exact_exchange_screened(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const Evaluator basis,
    const PotEvaluator basis_aux, const FlatGrid grid, const NeighborList nl,
    const DeviceView2DLeft half_inverse_X, const int orbital_batch = 8,
    const double orbital_tol = 0.0,
    ExchangeScreeningStats *stats = nullptr) {

  Kokkos::Profiling::pushRegion("Compute Exact Exchange Integral Screened");

  const int N_bf = basis.nbf();
  const int N_aux = basis_aux.nbf();
  const int N_quad = static_cast<int>(grid.quad_points.extent(0));
  const int N_occ = static_cast<int>(mo_orbitals.extent(1));
  const int K = static_cast<int>(half_inverse_X.extent(1));

  if (static_cast<int>(mo_coeff.extent(0)) < N_occ)
    throw std::invalid_argument("compute_exact_exchange_screened: fewer "
                                "occupations than occupied orbitals");

  DeviceView2DLeft result("Exchange matrix", N_bf, N_bf);
  if (N_occ == 0 || N_aux == 0)
    return result;

  // Orbitals are done in batches, and the batch size is the one structural
  // knob this loop has. Everything outside the orbital loop -- packing the
  // collocation and, far more expensively, evaluating the auxiliary
  // potential -- is paid once per BATCH; the three-centre GEMM is paid once
  // per orbital. So the cost is passes * P + N_occ * Q with
  // passes = ceil(N_occ / batch). benchmark_exchange sweeps it; benzene, 72
  // functions, 642 auxiliaries, 22 occupied, 14670 points, three runs back
  // to back:
  //
  //     batch      1     2     4     8    16    32
  //     passes    22    11     6     3     2     1
  //     run 1   8.89  6.15  5.42  4.59  5.19  4.91
  //     run 2   7.79  4.85  4.71  4.17  4.02  3.81
  //     run 3   6.55  5.02  5.27  4.21  3.85  4.01
  //
  // Batching at all is worth about 1.6 to 2, so this is not a
  // micro-optimization. Past four the curve is FLAT to within this
  // machine's spread -- the first run alone looked like a minimum at eight
  // and a penalty beyond it, and the next two did not reproduce either.
  // Eight is therefore the default because it is on the flat part with the
  // smaller working set, not because it is optimal; the three-centre tensor
  // is (naux, nbf, batch) and nothing is measurably gained by growing it.
  //
  // Do not replace this with a memory budget. A budget would pick the
  // largest batch that fits, which buys fewer auxiliary-potential passes
  // than eight already does and costs a working set several times larger,
  // for a difference this machine cannot resolve. That was tried first.
  //
  // The parameter exists so the sweep above can be re-measured rather than
  // believed. The cap is a compile-time constant because the per-point
  // accumulator lives in registers.
  //
  // The orbital loop STAYS outermost. Issue #8 proposes hoisting the
  // three-centre construction out of it -- building (Q|mu nu) per box once
  // and contracting with the coefficients afterwards -- and the layout says
  // not to. That ordering replaces naux * nbf_b * np_b per orbital with
  // naux * nbf_b^2 per orbital plus a grid term, so its floor relative to
  // this one is about nbf_b / np_b. With 128-point boxes nbf_b saturates at
  // the screening radius and reaches the point count, the floor passes one
  // around C24, and beyond that NO occupancy makes the hoist cheaper; the
  // gap widens with system size rather than closing. It becomes worth doing
  // only together with much larger boxes, and then only above roughly 75
  // occupied orbitals. benchmark_exchange --layout-only prints the numbers
  // for any system, and its header carries the tables.
  // ORBITAL SCREENING, the other half of issue #8
  //
  // Everything inside the orbital loop is box-local: the box's collocation
  // scaled by the orbital, one GEMM against the box's auxiliary potential,
  // and the scatter of the (naux, nbf_b) block. If the orbital has no
  // amplitude anywhere in the box, all three contribute nothing and can be
  // skipped. orbital_tol is the threshold on max_g |psi_i(g)| over the box,
  // and zero -- the default -- screens nothing.
  //
  // This is where issue #8's "screen on the density matrix" lands in THIS
  // formalism, and it needs the caller's cooperation. Nothing here looks at
  // |P| directly, because the loop does not index function pairs; it indexes
  // occupied orbitals. Canonical orbitals are delocalized and the screening
  // then does nothing at all -- every orbital is alive in every box. The
  // density matrix's decay is available only through a factorization of P
  // that inherits it, which is what localize.hpp provides and what the
  // caller should pass: K depends on the occupied set only through P, so
  // substituting the pivoted Cholesky vectors for the orbitals is exact
  // rather than an approximation.
  //
  // The threshold is absolute, on an orbital normalized against the overlap,
  // so |psi| is O(1) in atomic units and a tolerance reads as a relative
  // one.
  //
  // WHAT IT IS WORTH, MEASURED
  //
  // Aligned alkanes, DZ, 128-point boxes, SAP-guess occupied orbitals, the
  // work weighted by nbf_b * np_b because that is what a kept pair costs
  // (benchmark_exchange --screen-only):
  //
  //     work kept        1e-6    1e-5    1e-4
  //     C4H10   canon    100.0    100.0   92.9
  //             Cholesky 100.0    100.0  100.0
  //     C8H18   canon    100.0     98.2   86.3
  //             Cholesky 100.0    100.0   98.9
  //     C12H26  canon     99.8     96.2   85.4
  //             Cholesky  99.9     98.2   89.5
  //     C16H34  canon     98.8     93.5   83.2
  //             Cholesky  98.3     92.6   79.5
  //
  // So it is real but small: a fifth of the three-centre work at C16H34,
  // for about 1e-6 relative error in K, improving with chain length (100 ->
  // 79 per cent from C4 to C16) but nowhere near the change of scaling
  // issue #8 asks for. The default is therefore off.
  //
  // The localization is not what limits this. The Cholesky factors' spread
  // is SIZE-INDEPENDENT -- 5.9, 6.4, 6.9, 7.0 basis functions across those
  // four chains -- while the canonical spread tracks the fully delocalized
  // limit almost exactly (61.3 against 65.8 at C16H34). What limits it is
  // that a second moment of 7 says nothing about the TAILS, and the tails
  // are what screening needs: over twenty angstroms of alkane a localized
  // orbital's amplitude falls only about four orders of magnitude, so the
  // tolerance that skips appreciable work is the same tolerance that costs
  // appreciable accuracy. That the two factorizations differ so little --
  // and that at 1e-4 the CANONICAL set sometimes keeps less, by dilution
  // rather than by locality -- is the same statement.
  //
  // It is also not the basis screening. Loosening it from 1e-8 to 1e-4
  // moves nbf/box from 143.9 to 92.2 at C16H34 and leaves the work-kept
  // curve where it was, to a tenth of a per cent. max_g |psi| over a box is
  // robust against which marginal functions the layout happens to keep.
  //
  // Why this does not reproduce GauXC's result: exx_ek_screening screens
  // SHELL PAIRS for (mu nu|la si) on |P_mu nu| times an integral bound --
  // a product of two decaying quantities, over a list that is quadratic in
  // the basis, so there is a great deal to remove. This loop has no pair
  // list. It has one orbital index, and one decaying quantity.
  //
  // What this does NOT fix: Bten is (naux, nbf, batch) dense and the
  // per-orbital contraction through the fitting metric is naux * nbf * K,
  // with no grid factor and no screening. Once the grid term is screened
  // down to O(N) that contraction is what is left, and it is cubic. Its
  // operand is sparse for a localized orbital -- the same boxes that were
  // skipped leave zero columns in Bten -- so the same information would
  // screen it, but that is a separate restructure.
  constexpr int MAX_ORBITAL_BATCH = detail::MAX_ORBITAL_BATCH;
  if (orbital_batch < 1)
    throw std::invalid_argument(
        "compute_exact_exchange_screened: orbital_batch must be positive");
  const int batch = std::min({N_occ, MAX_ORBITAL_BATCH, orbital_batch});

  // The chunk's point count is capped as well as its collocation size: the
  // auxiliary potential block is (naux, chunk points).
  const PackedBlockLayout L = build_packed_block_layout(
      basis, nl, (size_t{64} << 20), aux_chunk_points(N_aux, N_quad));

  // LayoutRight: the inner loop walks the points of one auxiliary function.
  using AuxPotential = Kokkos::View<double **, Kokkos::LayoutRight, MemSpace>;
  AuxPotential Vaux("aux potential chunk", N_aux, L.max_chunk_npts);
  Kokkos::View<double *> chi("packed chi", L.max_chunk_coll);
  // psi is (batch, chunk points): the GEMM below scales a box's collocation
  // by one orbital at a time, so that orbital's points have to be
  // contiguous.
  const size_t psi_stride = static_cast<size_t>(L.max_chunk_npts);
  // Vaux is allocated (N_aux, max_chunk_npts) row-major, so this is its row
  // stride; the GEMM needs it to view one box's columns.
  const size_t aux_row_stride = static_cast<size_t>(L.max_chunk_npts);
  Kokkos::View<double *> psi("psi", static_cast<size_t>(batch) * psi_stride);
  // chi scaled by one orbital, and the (naux, nbf_b) blocks the GEMM writes.
  Kokkos::View<double *> X("chi times psi", L.max_chunk_coll);
  // max_g |psi_j| over each box, for the orbital screening. Indexed by the
  // ABSOLUTE box number so no chunk-local bookkeeping is needed; it is
  // num_boxes * batch doubles, megabytes at the sizes where the exchange
  // build itself needs gigabytes.
  Kokkos::View<double **> pmax("orbital amplitude per box", L.num_boxes,
                               batch);
  Kokkos::View<double *> Tblk("three center blocks",
                              static_cast<size_t>(N_aux) * L.max_chunk_nbf);
  Kokkos::View<double ***, Kokkos::LayoutLeft, MemSpace> Bten(
      "three center integral", N_aux, N_bf, batch);
  DeviceView2DLeft Cmat("three center integral scaled", N_bf, K);

  const auto occ_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mo_coeff);
  auto w = grid.weights;
  auto nbf_v = L.map.nbf;
  auto npts_v = L.npts;
  auto coll_off = L.coll_off;
  auto bf_off = L.bf_off;
  auto l2g = L.loc2glob;
  const int mppb = L.mppb;
  // Captured by value into the device lambdas; zero screens nothing, since
  // pmax is a maximum of absolute values and is never negative.
  const double tol = orbital_tol;

  using policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = policy_type::member_type;

  for (int i0 = 0; i0 < N_occ; i0 += batch) {
    const int nb = std::min(batch, N_occ - i0);
    Kokkos::deep_copy(Bten, 0.0);

    for (size_t c = 0; c + 1 < L.chunk_begin.size(); ++c) {
      const int b0 = L.chunk_begin[c], b1 = L.chunk_begin[c + 1];
      const int nbox = b1 - b0;
      if (nbox <= 0)
        continue;
      // The boxes partition the grid into consecutive runs, so the chunk is
      // one contiguous point range.
      const int g0 = b0 * mppb;
      const int ng = std::min(b1 * mppb, N_quad) - g0;
      if (ng <= 0)
        continue;

      pack_collocation<BasisDeriv::Value>(basis, grid, nl, L, b0, b1, chi);
      fill_aux_potential_chunk(basis_aux, grid, g0, ng, Vaux);
      space.fence();

      Kokkos::parallel_for(
          "Weight the auxiliary potential",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0},
                                                            {N_aux, ng}),
          KOKKOS_LAMBDA(const int a, const int g) {
            Vaux(a, g) *= w(g0 + g);
          });

      detail::chunk_orbitals_on_grid(space, L, b0, b1, g0, chi, mo_orbitals,
                                     i0, nb, psi, psi_stride);
      space.fence();

      // The orbital's amplitude in each box, for the screening below. The
      // same quantity orbital_box_amplitudes reports, and cheap enough to
      // compute whether or not it will be used.
      detail::chunk_box_amplitudes(space, L, b0, b1, g0, psi, psi_stride, nb,
                                   pmax);
      space.fence();

      if (stats) {
        // Both counts exclude the boxes that carry no work at all -- no
        // points, or no functions reaching them -- because the build skips
        // those whether or not it screens, and counting them as "skipped"
        // would credit the screening with work nobody was going to do.
        long long alive = 0, boxes = 0;
        Kokkos::parallel_reduce(
            "Count live box-orbital pairs",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {b0, 0},
                                                              {b1, nb}),
            KOKKOS_LAMBDA(const int b, const int j, long long &acc) {
              if (npts_v(b) > 0 && nbf_v(b) > 0 && pmax(b, j) >= tol)
                acc += 1;
            },
            alive);
        Kokkos::parallel_reduce(
            "Count boxes that carry work",
            Kokkos::RangePolicy<ExecSpace>(space, b0, b1),
            KOKKOS_LAMBDA(const int b, long long &acc) {
              if (npts_v(b) > 0 && nbf_v(b) > 0)
                acc += 1;
            },
            boxes);
        space.fence();
        stats->pairs_total += boxes * nb;
        stats->pairs_alive += alive;
      }

      // One orbital at a time: scale the packed collocation by psi_j, then
      // ONE GEMM per box against the box's auxiliary potential block, then
      // scatter the (naux, nbf_b) result.
      //
      // The hand-written dot product this replaces ran at about 0.5 GFLOP/s
      // against BLAS's 2.5 on the same product. A dot product per
      // (auxiliary function, local function) has no reuse to exploit,
      // whatever the loop order; a GEMM over the box's points does, and
      // KokkosBatched gives one per team without materializing anything
      // larger than the box's own block.
      for (int j = 0; j < nb; ++j) {
        Kokkos::parallel_for(
            "Scale collocation by one orbital",
            policy_type(nbox, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const member_type &team) {
              const int b = b0 + team.league_rank();
              const int n = nbf_v(b);
              const int np = npts_v(b);
              if (n == 0 || np == 0)
                return;
              if (pmax(b, j) < tol)
                return;
              const size_t cb = coll_off(b) - coll_off(b0);
              const double *chi_b = chi.data() + cb;
              double *X_b = X.data() + cb;
              const int gloc = b * mppb - g0;
              const double *p_row =
                  psi.data() + j * psi_stride + static_cast<size_t>(gloc);

              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team, n), [&](const int i) {
                    const double *src = chi_b + static_cast<size_t>(i) * np;
                    double *dst = X_b + static_cast<size_t>(i) * np;
                    for (int g = 0; g < np; ++g)
                      dst[g] = src[g] * p_row[g];
                  });
            });

        // One thread per box, each doing a register-blocked serial GEMM:
        // the boxes are the parallel dimension and there are far more of
        // them than threads, so a team-level GEMM buys nothing and
        // KokkosBatched only offers its unblocked kernel at team level.
        Kokkos::parallel_for(
            "Three center integral, screened",
            Kokkos::RangePolicy<ExecSpace>(b0, b1),
            KOKKOS_LAMBDA(const int b) {
              const int n = nbf_v(b);
              const int np = npts_v(b);
              if (n == 0 || np == 0)
                return;
              if (pmax(b, j) < tol)
                return;
              const size_t cb = coll_off(b) - coll_off(b0);
              const size_t tb =
                  static_cast<size_t>(N_aux) * (bf_off(b) - bf_off(b0));
              const int gloc = b * mppb - g0;

              // The box's columns of the auxiliary potential: contiguous in
              // the point index, strided by the chunk width in the function
              // index.
              const Kokkos::LayoutStride vl(N_aux, aux_row_stride, np, 1);
              Kokkos::View<double **, Kokkos::LayoutStride, MemSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                  A(&Vaux(0, gloc), vl);
              const PackedMatrix B(X.data() + cb, n, np);
              const PackedMatrix T(Tblk.data() + tb, N_aux, n);

              KokkosBatched::SerialGemm<
                  KokkosBatched::Trans::NoTranspose,
                  KokkosBatched::Trans::Transpose,
                  KokkosBatched::Algo::Gemm::Blocked>::invoke(1.0, A, B, 0.0,
                                                              T);
            });

        Kokkos::parallel_for(
            "Scatter three center blocks", policy_type(nbox, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const member_type &team) {
              const int b = b0 + team.league_rank();
              const int n = nbf_v(b);
              if (n == 0 || npts_v(b) == 0)
                return;
              if (pmax(b, j) < tol)
                return;
              const size_t tb =
                  static_cast<size_t>(N_aux) * (bf_off(b) - bf_off(b0));
              const double *T = Tblk.data() + tb;
              const size_t base = bf_off(b);

              Kokkos::parallel_for(
                  Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, member_type>(
                      team, N_aux, n),
                  [&](const int a, const int i) {
                    const double v = T[static_cast<size_t>(a) * n + i];
                    if (v != 0.0)
                      Kokkos::atomic_add(&Bten(a, l2g(base + i), j), v);
                  });
            });
      }

      space.fence();
    }

    // K += occ_i (B_i^T X)(B_i^T X)^T, one orbital at a time: the product
    // over the fitting metric does not factor over the batch.
    for (int j = 0; j < nb; ++j) {
      auto Bj = Kokkos::subview(Bten, Kokkos::ALL(), Kokkos::ALL(), j);
      KokkosBlas::gemm(space, "T", "N", 1.0, Bj, half_inverse_X, 0.0, Cmat);
      KokkosBlas::gemm(space, "N", "T", occ_h(i0 + j), Cmat, Cmat, 1.0,
                       result);
    }
    space.fence();
  }

  Kokkos::fence();
  Kokkos::Profiling::popRegion();
  return result;
}

/**
 *  @brief max_g |psi_i(g)| per (box, orbital), without the exchange build.
 *
 *  This is exactly the quantity compute_exact_exchange_screened screens on,
 *  and it is produced by the same two kernels -- but without the auxiliary
 *  potential, the three-centre GEMMs, the scatter or the fitting metric, so
 *  it costs a small fraction of one build. That is what makes choosing a
 *  tolerance, or asking whether a tolerance would skip anything at all,
 *  affordable on a system large enough for the answer to differ.
 *
 *  The chunking does not enter the result: a box's amplitude does not depend
 *  on which chunk the box was processed in, nor on the orbital batch, so
 *  max_chunk_points may be left at zero even though the build caps it to
 *  pair the chunk with an (naux, npts) potential block.
 *
 *  @param box_work  if non-null, filled with nbf_b * np_b per box -- what
 *                   one (box, orbital) pair costs, up to the naux factor
 *                   every pair shares. A count of skipped pairs is not a
 *                   speedup; a count weighted by this is.
 *
 *  @return (num_boxes, norb) on the host. A box with no points reads zero.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value>
inline Kokkos::View<double **, HostSpace>
orbital_box_amplitudes(const ExecSpace space, const Evaluator basis,
                       const FlatGrid grid, const NeighborList nl,
                       const DeviceView2DLeft C,
                       const int max_chunk_points = 0,
                       std::vector<double> *box_work = nullptr) {
  const int norb = static_cast<int>(C.extent(1));
  const int N_quad = static_cast<int>(grid.quad_points.extent(0));
  const PackedBlockLayout L = build_packed_block_layout(
      basis, nl, (size_t{64} << 20), max_chunk_points);

  Kokkos::View<double **, HostSpace> out("orbital box amplitudes",
                                         std::max(L.num_boxes, 0), norb);
  if (box_work) {
    box_work->assign(static_cast<size_t>(std::max(L.num_boxes, 0)), 0.0);
    const auto nbf_h =
        Kokkos::create_mirror_view_and_copy(HostSpace{}, L.map.nbf);
    const auto npts_h =
        Kokkos::create_mirror_view_and_copy(HostSpace{}, L.npts);
    for (int b = 0; b < L.num_boxes; ++b)
      (*box_work)[static_cast<size_t>(b)] =
          static_cast<double>(nbf_h(b)) * static_cast<double>(npts_h(b));
  }
  if (norb == 0 || L.num_boxes == 0)
    return out;

  const int batch = std::min(norb, detail::MAX_ORBITAL_BATCH);
  Kokkos::View<double *> chi("packed chi", L.max_chunk_coll);
  const size_t psi_stride = static_cast<size_t>(L.max_chunk_npts);
  Kokkos::View<double *> psi("psi", static_cast<size_t>(batch) * psi_stride);
  Kokkos::View<double **> pmax("orbital amplitude per box", L.num_boxes,
                               batch);
  auto pmax_h = Kokkos::create_mirror_view(pmax);
  const int mppb = L.mppb;

  for (int i0 = 0; i0 < norb; i0 += batch) {
    const int nb = std::min(batch, norb - i0);
    Kokkos::deep_copy(pmax, 0.0);

    for (size_t c = 0; c + 1 < L.chunk_begin.size(); ++c) {
      const int b0 = L.chunk_begin[c], b1 = L.chunk_begin[c + 1];
      if (b1 <= b0)
        continue;
      const int g0 = b0 * mppb;
      if (std::min(b1 * mppb, N_quad) - g0 <= 0)
        continue;
      pack_collocation<BasisDeriv::Value>(basis, grid, nl, L, b0, b1, chi);
      space.fence();
      detail::chunk_orbitals_on_grid(space, L, b0, b1, g0, chi, C, i0, nb,
                                     psi, psi_stride);
      space.fence();
      detail::chunk_box_amplitudes(space, L, b0, b1, g0, psi, psi_stride, nb,
                                   pmax);
      space.fence();
    }

    Kokkos::deep_copy(pmax_h, pmax);
    for (int b = 0; b < L.num_boxes; ++b)
      for (int j = 0; j < nb; ++j)
        out(b, i0 + j) = pmax_h(b, j);
  }
  return out;
}

} // namespace Nukexc
