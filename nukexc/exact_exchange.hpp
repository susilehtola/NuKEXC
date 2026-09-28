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
#include "basis_evaluator.hpp"
#include "nukexc_config.hpp"
#include "stobasis.hpp"
#include "stopotential.hpp"

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

template <typename Evaluator, typename PotEvaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value> &&
           PotentialEvaluator<PotEvaluator>
DeviceView2DLeft compute_exact_exchange_sparse(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const Evaluator basis,
    const PotEvaluator basis_aux, const FlatGrid grid, const NeighborList nl,
    const DeviceView2DLeft half_inverse_X) {

  Kokkos::Profiling::pushRegion("Compute Exact Exchange Integral Sparse");
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);
  const int N_occ = mo_coeff.extent(0);
  const int K = half_inverse_X.extent(1);

  const int max_points_per_box = nl.max_points_per_box;
  const int num_boxes = nl.offsets.extent(0) - 1;

  DeviceView2DLeft three_center_integral("Three_center_integral", N_bf_aux,
                                         N_bf);
  DeviceView2DLeft three_center_integral_scaled("Three_center_integral_scaled",
                                                N_bf, K);

  DeviceView2DLeft result("Exchange matrix", N_bf, N_bf);

  auto mo_coeff_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mo_coeff);

  Kokkos::TeamPolicy<ExecSpace> policy_boxes(space, num_boxes, Kokkos::AUTO());

  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
  typedef ExecSpace::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;

  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  int scratch_size_team = shared_view_double::shmem_size(max_points_per_box) +
                          shared_view_points::shmem_size(max_points_per_box);

  int scratch_size_thread = shared_view_double::shmem_size(max_points_per_box);

  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_size_team),
                                Kokkos::PerThread(scratch_size_thread));

  // Loop over the occupied orbitals and compute contributions for each occupied
  // orbital

  for (unsigned int i = 0; i < N_occ; ++i) {
    // Reset the three center integral
    Kokkos::deep_copy(space, three_center_integral, 0);
    ;

    Kokkos::parallel_for(
        "Sparse_three_center_integral_exchange", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double potential_scratch_scaled(
              team_member.thread_scratch(0), num_points);

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          // Fill points and weights scratch
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team_member, num_points),
              [=](const int local_g) {
                const int global_g = start_points + local_g;
                weights_scratch(local_g) = grid.weights(global_g);
                points_scratch(local_g) = grid.quad_points(global_g);

                // Fold the i_th orbital into the weights
                double orbital_i_at_point = 0.0;
                for (int k = 0; k < N_bf; ++k) {
                  orbital_i_at_point +=
                      mo_orbitals(k, i) *
                      basis_eval_fast(load_shell(basis, k),
                                      points_scratch(local_g)[0],
                                      points_scratch(local_g)[1],
                                      points_scratch(local_g)[2]);
                }
                weights_scratch(local_g) *= orbital_i_at_point;
              });
          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, N_bf_aux),
              [=](const int global_alpha) {
                // Pcompute the potential once and store in scratch
                for (int local_g = 0; local_g < num_points; ++local_g) {
                  
                  potential_scratch_scaled(local_g) =
                      eval_aux_potential(basis_aux, global_alpha, points_scratch(local_g)) *
                      weights_scratch(local_g);
                }

                for (int local_j = 0; local_j < num_neighbors; ++local_j) {
                  const int global_j = nl.neighbors(start_neighbors + local_j);
                  auto shell_j = load_shell(basis, global_j);

                  double local_sum = 0;
                  for (int local_g = 0; local_g < num_points; ++local_g) {
                    local_sum +=
                        potential_scratch_scaled(local_g) *
                        basis_eval_fast(shell_j, points_scratch(local_g)[0],
                                        points_scratch(local_g)[1],
                                        points_scratch(local_g)[2]);
                  }
                  Kokkos::atomic_add(
                      &three_center_integral(global_alpha, global_j),
                      local_sum);
                }
              });
        });

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
//  Neighbor-tiled exact exchange
// ============================================================================
//
// Same result as compute_exact_exchange_sparse. The redundant work in the
// sparse kernel is that phi_j is re-evaluated inside the aux-function (alpha)
// loop -- i.e. every basis function's phi is recomputed N_bf_aux times per box.
// Here phi is evaluated once per neighbor tile into a bounded team-scratch cache
// and reused across all alpha. The orbital fold (which already evaluates each
// phi_k exactly once) is left as-is. Cache footprint is
//   tile_size * max_points_per_box * sizeof(double), independent of neighbors.
template <typename Evaluator, typename PotEvaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value> &&
           PotentialEvaluator<PotEvaluator>
DeviceView2DLeft compute_exact_exchange_tiled(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const Evaluator basis,
    const PotEvaluator basis_aux, const FlatGrid grid, const NeighborList nl,
    const DeviceView2DLeft half_inverse_X, const int tile_size = 32) {

  Kokkos::Profiling::pushRegion("Compute Exact Exchange Integral Tiled");
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);
  const int N_occ = mo_coeff.extent(0);
  const int K = half_inverse_X.extent(1);

  const int max_points_per_box = nl.max_points_per_box;
  const int num_boxes = nl.offsets.extent(0) - 1;

  DeviceView2DLeft three_center_integral("Three_center_integral", N_bf_aux,
                                         N_bf);
  DeviceView2DLeft three_center_integral_scaled("Three_center_integral_scaled",
                                                N_bf, K);
  DeviceView2DLeft result("Exchange matrix", N_bf, N_bf);

  auto mo_coeff_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mo_coeff);

  using Bounds = Kokkos::LaunchBounds<128, NUKEXC_TILED_MIN_BLOCKS>;
  Kokkos::TeamPolicy<ExecSpace, Bounds> policy_boxes(space, num_boxes,
                                                     Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
  typedef ExecSpace::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;
  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  // Option A: the per-alpha potential lives in a single PerTeam [num_points]
  // buffer (filled and consumed one alpha at a time) instead of a PerThread
  // buffer replicated across every team thread. That per-thread replication was
  // the dominant shared-memory consumer (threads * num_points) and the occupancy
  // limiter; a single team buffer frees it.
  const int scratch_team =
      shared_view_double::shmem_size(max_points_per_box) +            // weights
      shared_view_points::shmem_size(max_points_per_box) +            // points
      shared_view_double::shmem_size(tile_size * max_points_per_box) + // phi tile
      shared_view_double::shmem_size(max_points_per_box); // potential (team)

  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_team));

  for (unsigned int i = 0; i < N_occ; ++i) {
    Kokkos::deep_copy(space, three_center_integral, 0);

    Kokkos::parallel_for(
        "Exchange tiled three-center integral", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          const int start_neighbors = nl.offsets(box_idx);
          const int num_neighbors = nl.offsets(box_idx + 1) - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);
          shared_view_points points_scratch(team_member.team_scratch(0),
                                             num_points);
          shared_view_double phi_cache(team_member.team_scratch(0),
                                       tile_size * num_points);
          shared_view_double potential_scratch_scaled(team_member.team_scratch(0),
                                                      num_points);

          // Fill points/weights and fold the i-th orbital into the weights.
          // (Each phi_k is evaluated once here -- no redundancy to tile away.)
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team_member, num_points),
              [=](const int local_g) {
                const int global_g = start_points + local_g;
                weights_scratch(local_g) = grid.weights(global_g);
                points_scratch(local_g) = grid.quad_points(global_g);

                double orbital_i_at_point = 0.0;
                for (int k = 0; k < N_bf; ++k) {
                  orbital_i_at_point +=
                      mo_orbitals(k, i) *
                      basis_eval_fast(load_shell(basis, k),
                                      points_scratch(local_g)[0],
                                      points_scratch(local_g)[1],
                                      points_scratch(local_g)[2]);
                }
                weights_scratch(local_g) *= orbital_i_at_point;
              });
          team_member.team_barrier();

          const int num_tiles = (num_neighbors + tile_size - 1) / tile_size;
          for (int jt = 0; jt < num_tiles; ++jt) {
            const int j0 = jt * tile_size;
            const int jlen = Kokkos::min(tile_size, num_neighbors - j0);

            // Evaluate this neighbor tile's phi once into the cache.
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, jlen), [=](const int lj) {
                  const int gj = nl.neighbors(start_neighbors + j0 + lj);
                  const auto shell = load_shell(basis, gj);
                  const int base = lj * num_points;
                  for (int g = 0; g < num_points; ++g)
                    phi_cache(base + g) = basis_eval_fast(
                        shell, points_scratch(g)[0], points_scratch(g)[1],
                        points_scratch(g)[2]);
                });
            team_member.team_barrier();

            // One alpha at a time: the whole team fills the shared potential
            // buffer, then the whole team contracts it against the cached tile
            // of phi_j. phi_j is still evaluated only once per tile (read from
            // phi_cache); only the single team potential buffer is reused.
            for (int global_alpha = 0; global_alpha < N_bf_aux; ++global_alpha) {
              Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(team_member, num_points),
                  [=](const int g) {
                    potential_scratch_scaled(g) =
                        eval_aux_potential(basis_aux, global_alpha, points_scratch(g)) *
                        weights_scratch(g);
                  });
              team_member.team_barrier();

              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team_member, jlen), [=](const int lj) {
                    const int gj = nl.neighbors(start_neighbors + j0 + lj);
                    double local_sum = 0;
                    for (int g = 0; g < num_points; ++g)
                      local_sum += potential_scratch_scaled(g) *
                                   phi_cache(lj * num_points + g);
                    Kokkos::atomic_add(&three_center_integral(global_alpha, gj),
                                       local_sum);
                  });
              team_member.team_barrier();
            }
          }
        });

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
} // namespace Nukexc
