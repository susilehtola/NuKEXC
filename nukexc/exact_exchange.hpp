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

template <typename Evaluator, typename PotEvaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value> &&
           PotentialEvaluator<PotEvaluator>
inline DeviceView2DLeft compute_exact_exchange_screened(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const Evaluator basis,
    const PotEvaluator basis_aux, const FlatGrid grid, const NeighborList nl,
    const DeviceView2DLeft half_inverse_X) {

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

  // Orbitals are done in batches: the three-centre block is (naux, nbf) per
  // orbital, and one dot product accumulates the whole batch, so the batch
  // amortizes the packing and the potential over several orbitals. The bound
  // is a compile-time constant because the accumulator lives in registers.
  constexpr int MAX_ORBITAL_BATCH = 8;
  const int batch = std::min(N_occ, MAX_ORBITAL_BATCH);

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

      Kokkos::parallel_for(
          "Occupied orbitals on the grid", policy_type(nbox, Kokkos::AUTO()),
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
                      acc[j] += mo_orbitals(mu, i0 + j) * ci;
                  }
                  for (int j = 0; j < nb; ++j)
                    psi_b[j * psi_stride + static_cast<size_t>(gloc + g)] =
                        acc[j];
                });
          });
      space.fence();

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

} // namespace Nukexc
