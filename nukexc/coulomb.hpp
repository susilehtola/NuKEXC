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

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <impl/Kokkos_Profiling.hpp>

namespace Nukexc {

DeviceView2DLeft
compute_coulomb(const ExecSpace space, const DeviceView2DLeft mo_orbitals,
                const DeviceView1D mo_coeff,
                const DeviceView2DLeft basis_collocation,
                const DeviceView2DLeft basis_aux_collocation,
                const DeviceView2DLeft potential_collocation_scaled,
                const DeviceView2DLeft half_inverse_X) {

  Kokkos::Profiling::pushRegion("Compute Coulomb Integral");
  const int N_bf = basis_collocation.extent(0);
  const int N_bf_aux = basis_aux_collocation.extent(0);
  const int N_quad = basis_collocation.extent(1);
  const int K = half_inverse_X.extent(1);

  DeviceView2DLeft basis_collocation_scaled("Basis collocation Scaled", N_bf,
                                            N_quad);
  DeviceView1DLeft expansion_coeff("Expansion coeff", N_bf_aux);
  DeviceView1DLeft potential_on_grid("Expansion coeff scaled", N_quad);
  DeviceView2DLeft result("Coulomb matrix", N_bf, N_bf);
  DeviceView1DLeft density =
      compute_density(basis_collocation, mo_orbitals, mo_coeff);

  DeviceView1DLeft scaling_factor("Scaling factor", K);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Compute expansion coefficients
  KokkosBlas::gemv(space, "N", 1.0, potential_collocation_scaled, density, 0.0,
                   expansion_coeff);

  // Apply (A|B)^{-1}
  KokkosBlas::gemv(space, "T", 1.0, half_inverse_X, expansion_coeff, 0.0,
                   scaling_factor);
  KokkosBlas::gemv(space, "N", 1.0, half_inverse_X, scaling_factor, 0.0,
                   expansion_coeff);

  // Compute potential on grid
  KokkosBlas::gemv(space, "T", 1.0, potential_collocation_scaled,
                   expansion_coeff, 0.0, potential_on_grid);

  // Compute (mu nu | A)
  Kokkos::parallel_for(
      "Scale basis by potential", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double pot_g = potential_on_grid(g);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
              basis_collocation_scaled(i, g) = basis_collocation(i, g) * pot_g;
            });
      });

  KokkosBlas::gemm(space, "N", "T", 1.0, basis_collocation,
                   basis_collocation_scaled, 0.0, result);

  Kokkos::fence();
  Kokkos::Profiling::popRegion();
  return result;
}

template <typename Evaluator, typename PotEvaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value> &&
           PotentialEvaluator<PotEvaluator>
DeviceView2DLeft compute_coulomb_sparse(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const Evaluator basis,
    const PotEvaluator basis_aux, const FlatGrid grid, const NeighborList nl,
    const DeviceView2DLeft half_inverse_X) {

  Kokkos::Profiling::pushRegion("Compute Coulomb Integral Sparse");
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);
  const int N_occ = mo_coeff.extent(0);
  const int K = half_inverse_X.extent(1);

  const int max_points_per_box = nl.max_points_per_box;
  const int num_boxes = nl.offsets.extent(0) - 1;

  DeviceView1DLeft expansion_coeff("Expansion coeff", N_bf_aux);
  DeviceView2DLeft density_matrix("Density matrix", N_bf, N_bf);
  DeviceView2DLeft result("Coulomb matrix", N_bf, N_bf);

  Kokkos::parallel_for(
      "Fill Density matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
      KOKKOS_LAMBDA(const int mu, const int nu) {
        for (unsigned int k = 0; k < N_occ; ++k)
          density_matrix(mu, nu) +=
              mo_coeff(k) * mo_orbitals(mu, k) * mo_orbitals(nu, k);
      });

  DeviceView1DLeft scaling_factor("Scaling factor", K);

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
                          shared_view_double::shmem_size(max_points_per_box) +
                          shared_view_points::shmem_size(max_points_per_box);

  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_size_team));

  Kokkos::parallel_for(
      "Sparse Kernel: coeff(alpha) = sum_{ml} (alpha |l m) D_lm", policy_boxes,
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

        shared_view_double weights_scratch(team_member.team_scratch(0),
                                           num_points);

        shared_view_points points_scratch(team_member.team_scratch(0),
                                          num_points);

        shared_view_double density_scratch_scaled(team_member.team_scratch(0),
                                                  num_points);

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                             [=](const int local_g) {
                               const int global_g = start_points + local_g;
                               weights_scratch(local_g) =
                                   grid.weights(global_g);
                               points_scratch(local_g) =
                                   grid.quad_points(global_g);
                             });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, num_points),
            [=](const int local_g) {
              double phi_i;
              double phi_j;
              density_scratch_scaled(local_g) = 0;
              for (int local_i = 0; local_i < num_neighbors; ++local_i) {

                const int global_i = nl.neighbors(start_neighbors + local_i);
                const auto shell_i = load_shell(basis, global_i);
                phi_i = basis_eval_fast(shell_i, points_scratch(local_g)[0],
                                        points_scratch(local_g)[1],
                                        points_scratch(local_g)[2]);

                for (unsigned int local_j = 0; local_j <= local_i; ++local_j) {

                  const int global_j = nl.neighbors(start_neighbors + local_j);

                  const auto shell_j = load_shell(basis, global_j);
                  phi_j = basis_eval_fast(shell_j, points_scratch(local_g)[0],
                                          points_scratch(local_g)[1],
                                          points_scratch(local_g)[2]);

                  density_scratch_scaled(local_g) +=
                      phi_i * phi_j * density_matrix(global_i, global_j) *
                      weights_scratch(local_g);
                  if (local_i != local_j)
                    density_scratch_scaled(local_g) +=
                        phi_i * phi_j * density_matrix(global_j, global_i) *
                        weights_scratch(local_g);
                }
              }
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf_aux),
            [=](const int global_alpha) {
              double local_sum_alpha = 0;

              for (int local_g = 0; local_g < num_points; ++local_g) {

                
                local_sum_alpha +=
                    density_scratch_scaled(local_g) *
                    eval_aux_potential(basis_aux, global_alpha, points_scratch(local_g));
              }
              Kokkos::atomic_add(&expansion_coeff(global_alpha),
                                 local_sum_alpha);
            });
      });

  space.fence();
  // Apply (A|B)^{-1}
  KokkosBlas::gemv(space, "T", 1.0, half_inverse_X, expansion_coeff, 0.0,
                   scaling_factor);
  KokkosBlas::gemv(space, "N", 1.0, half_inverse_X, scaling_factor, 0.0,
                   expansion_coeff);

  Kokkos::parallel_for(
      "Sparse Kernel: J_{mu,nu} = sum_{alpha} (mu nu|alpha)", policy_boxes,
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

        shared_view_double weights_scratch(team_member.team_scratch(0),
                                           num_points);

        shared_view_points points_scratch(team_member.team_scratch(0),
                                          num_points);

        shared_view_double potential_scratch_scaled(team_member.team_scratch(0),
                                                    num_points);

        // Fill points and weights scratch
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                             [=](const int local_g) {
                               const int global_g = start_points + local_g;
                               weights_scratch(local_g) =
                                   grid.weights(global_g);
                               points_scratch(local_g) =
                                   grid.quad_points(global_g);
                               potential_scratch_scaled(local_g) = 0;
                             });
        team_member.team_barrier();

        // Fill potential scratch
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf_aux),
            [=](const int global_alpha) {
              for (int local_g = 0; local_g < num_points; ++local_g) {
                
                double potential_alpha_scaled =
                    eval_aux_potential(basis_aux, global_alpha, points_scratch(local_g)) *
                    weights_scratch(local_g) * expansion_coeff(global_alpha);

                Kokkos::atomic_add(&potential_scratch_scaled(local_g),
                                   potential_alpha_scaled);
              }
            });

        team_member.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, num_neighbors),
            [=](const int local_i) {
              const int global_i = nl.neighbors(start_neighbors + local_i);
              const auto shell_i = load_shell(basis, global_i);

              double phi_i;
              for (int local_j = 0; local_j <= local_i; ++local_j) {

                const int global_j = nl.neighbors(start_neighbors + local_j);
                const auto shell_j = load_shell(basis, global_j);

                double phi_j;
                double local_result = 0;

                for (int local_g = 0; local_g < num_points; ++local_g) {
                  phi_i = basis_eval_fast(shell_i, points_scratch(local_g)[0],
                                          points_scratch(local_g)[1],
                                          points_scratch(local_g)[2]);
                  phi_j = basis_eval_fast(shell_j, points_scratch(local_g)[0],
                                          points_scratch(local_g)[1],
                                          points_scratch(local_g)[2]);

                  local_result +=
                      potential_scratch_scaled(local_g) * phi_i * phi_j;
                }

                Kokkos::atomic_add(&result(global_i, global_j), local_result);
                if (local_i != local_j)
                  Kokkos::atomic_add(&result(global_j, global_i), local_result);
              }
            });
      });

  space.fence();
  Kokkos::Profiling::popRegion();
  return result;
}

// ============================================================================
//  Neighbor-tiled Coulomb
// ============================================================================
//
// Same result as compute_coulomb_sparse, but the phi(neighbor, point)
// evaluations that the two kernels otherwise recompute O(neighbors^2 * points)
// times are staged through a bounded, fixed-size neighbor-tile cache in team
// scratch. Two tile buffers hold one tile of neighbor-phi each (phi_a = outer
// tile, phi_b = inner tile); the pairwise contractions are done as blocked,
// lower-triangular tiles. The shared-memory footprint is
//   2 * tile_size * max_points_per_box * sizeof(double)
// and is INDEPENDENT of the number of neighbors -- so unlike a full
// [neighbors x points] cache it does not overflow scratch on large molecules.
template <typename Evaluator, typename PotEvaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value> &&
           PotentialEvaluator<PotEvaluator>
DeviceView2DLeft compute_coulomb_tiled(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const Evaluator basis,
    const PotEvaluator basis_aux, const FlatGrid grid, const NeighborList nl,
    const DeviceView2DLeft half_inverse_X, const int tile_size = 32) {

  Kokkos::Profiling::pushRegion("Compute Coulomb Integral Tiled");
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);
  const int N_occ = mo_coeff.extent(0);
  const int K = half_inverse_X.extent(1);

  const int max_points_per_box = nl.max_points_per_box;
  const int num_boxes = nl.offsets.extent(0) - 1;

  DeviceView1DLeft expansion_coeff("Expansion coeff", N_bf_aux);
  DeviceView2DLeft density_matrix("Density matrix", N_bf, N_bf);
  DeviceView2DLeft result("Coulomb matrix", N_bf, N_bf);

  Kokkos::parallel_for(
      "Fill Density matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
      KOKKOS_LAMBDA(const int mu, const int nu) {
        for (unsigned int k = 0; k < N_occ; ++k)
          density_matrix(mu, nu) +=
              mo_coeff(k) * mo_orbitals(mu, k) * mo_orbitals(nu, k);
      });

  DeviceView1DLeft scaling_factor("Scaling factor", K);

  // Kernel 1 (density, ~66 registers) is close to the 64-register LaunchBounds
  // cap, so it spills little and the occupancy gain is a net win -> bounded.
  // Kernel 2 (the J Gram, ~86 registers) is far above the cap: forcing it there
  // spills heavily and profiled *slower*, so it runs on an unbounded policy.
  using Bounds = Kokkos::LaunchBounds<128, NUKEXC_TILED_MIN_BLOCKS>;
  Kokkos::TeamPolicy<ExecSpace, Bounds> policy_boxes(space, num_boxes,
                                                     Kokkos::AUTO());
  Kokkos::TeamPolicy<ExecSpace> policy_gram(space, num_boxes, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
  typedef ExecSpace::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;
  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  // ---- Kernel 1 (tiled): rho(g) = sum_{i,j} phi_i(g) phi_j(g) D_ij, then
  //      expansion_coeff(alpha) += sum_g rho(g) (alpha|g) ----
  const int scratch_k1 =
      shared_view_double::shmem_size(max_points_per_box) * 2 + // weights + rho
      shared_view_points::shmem_size(max_points_per_box) +     // points
      shared_view_double::shmem_size(tile_size * max_points_per_box) *
          2; // phi_a + phi_b
  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_k1));

  Kokkos::parallel_for(
      "Coulomb tiled: coeff(alpha) = sum_{ij} (alpha|ij) D_ij", policy_boxes,
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
        shared_view_double density_scratch_scaled(team_member.team_scratch(0),
                                                  num_points);
        shared_view_double phi_a(team_member.team_scratch(0),
                                 tile_size * num_points);
        shared_view_double phi_b(team_member.team_scratch(0),
                                 tile_size * num_points);

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                             [=](const int local_g) {
                               const int global_g = start_points + local_g;
                               weights_scratch(local_g) =
                                   grid.weights(global_g);
                               points_scratch(local_g) =
                                   grid.quad_points(global_g);
                               density_scratch_scaled(local_g) = 0;
                             });
        team_member.team_barrier();

        const int num_tiles = (num_neighbors + tile_size - 1) / tile_size;
        for (int ti = 0; ti < num_tiles; ++ti) {
          const int i0 = ti * tile_size;
          const int ilen = Kokkos::min(tile_size, num_neighbors - i0);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, ilen), [=](const int li) {
                const int gi = nl.neighbors(start_neighbors + i0 + li);
                const auto shell = load_shell(basis, gi);
                const int base = li * num_points;
                for (int g = 0; g < num_points; ++g)
                  phi_a(base + g) = basis_eval_fast(shell, points_scratch(g)[0],
                                                    points_scratch(g)[1],
                                                    points_scratch(g)[2]);
              });
          team_member.team_barrier();

          for (int tj = 0; tj <= ti; ++tj) {
            const int j0 = tj * tile_size;
            const int jlen = Kokkos::min(tile_size, num_neighbors - j0);
            const bool diag = (tj == ti);
            if (!diag) {
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team_member, jlen),
                  [=](const int lj) {
                    const int gj = nl.neighbors(start_neighbors + j0 + lj);
                    const auto shell = load_shell(basis, gj);
                    const int base = lj * num_points;
                    for (int g = 0; g < num_points; ++g)
                      phi_b(base + g) = basis_eval_fast(
                          shell, points_scratch(g)[0], points_scratch(g)[1],
                          points_scratch(g)[2]);
                  });
              team_member.team_barrier();
            }
            const double *pj = diag ? phi_a.data() : phi_b.data();

            // Accumulate the full ordered double sum for this tile block into
            // rho(g). Threads own points, so density_scratch_scaled(g) has no
            // race; the lower-triangular tile loop + symmetric D terms
            // replicate the sparse kernel's sum over all ordered (i,j).
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_points),
                [=](const int g) {
                  double acc = 0;
                  for (int li = 0; li < ilen; ++li) {
                    const int gi = nl.neighbors(start_neighbors + i0 + li);
                    const double phi_i = phi_a(li * num_points + g);
                    const int lj_hi = diag ? li : (jlen - 1);
                    for (int lj = 0; lj <= lj_hi; ++lj) {
                      const int gj = nl.neighbors(start_neighbors + j0 + lj);
                      const double phi_ij = phi_i * pj[lj * num_points + g];
                      acc += phi_ij * density_matrix(gi, gj);
                      if (!(diag && li == lj))
                        acc += phi_ij * density_matrix(gj, gi);
                    }
                  }
                  density_scratch_scaled(g) += acc * weights_scratch(g);
                });
            team_member.team_barrier();
          }
        }

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf_aux),
            [=](const int global_alpha) {
              double local_sum_alpha = 0;
              for (int g = 0; g < num_points; ++g) {
                local_sum_alpha +=
                    density_scratch_scaled(g) *
                    eval_aux_potential(basis_aux, global_alpha, points_scratch(g));
              }
              Kokkos::atomic_add(&expansion_coeff(global_alpha),
                                 local_sum_alpha);
            });
      });

  space.fence();

  // Apply (A|B)^{-1}
  KokkosBlas::gemv(space, "T", 1.0, half_inverse_X, expansion_coeff, 0.0,
                   scaling_factor);
  KokkosBlas::gemv(space, "N", 1.0, half_inverse_X, scaling_factor, 0.0,
                   expansion_coeff);

  // ---- Kernel 2 (tiled): J_{i,j} = sum_g V(g) phi_i(g) phi_j(g) ----
  const int scratch_k2 =
      shared_view_double::shmem_size(max_points_per_box) * 2 + // weights + pot
      shared_view_points::shmem_size(max_points_per_box) +     // points
      shared_view_double::shmem_size(tile_size * max_points_per_box) *
          2; // phi_a + phi_b
  policy_gram.set_scratch_size(0, Kokkos::PerTeam(scratch_k2));

  Kokkos::parallel_for(
      "Coulomb tiled: J_{mu,nu} = sum_{alpha} (mu nu|alpha)", policy_gram,
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
        shared_view_double potential_scratch_scaled(team_member.team_scratch(0),
                                                    num_points);
        shared_view_double phi_a(team_member.team_scratch(0),
                                 tile_size * num_points);
        shared_view_double phi_b(team_member.team_scratch(0),
                                 tile_size * num_points);

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                             [=](const int local_g) {
                               const int global_g = start_points + local_g;
                               weights_scratch(local_g) =
                                   grid.weights(global_g);
                               points_scratch(local_g) =
                                   grid.quad_points(global_g);
                               potential_scratch_scaled(local_g) = 0;
                             });
        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf_aux),
            [=](const int global_alpha) {
              for (int g = 0; g < num_points; ++g) {
                const double pot =
                    eval_aux_potential(basis_aux, global_alpha, points_scratch(g)) *
                    weights_scratch(g) * expansion_coeff(global_alpha);
                Kokkos::atomic_add(&potential_scratch_scaled(g), pot);
              }
            });
        team_member.team_barrier();

        const int num_tiles = (num_neighbors + tile_size - 1) / tile_size;
        for (int ti = 0; ti < num_tiles; ++ti) {
          const int i0 = ti * tile_size;
          const int ilen = Kokkos::min(tile_size, num_neighbors - i0);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, ilen), [=](const int li) {
                const int gi = nl.neighbors(start_neighbors + i0 + li);
                const auto shell = load_shell(basis, gi);
                const int base = li * num_points;
                for (int g = 0; g < num_points; ++g)
                  phi_a(base + g) = basis_eval_fast(shell, points_scratch(g)[0],
                                                    points_scratch(g)[1],
                                                    points_scratch(g)[2]);
              });
          team_member.team_barrier();

          for (int tj = 0; tj <= ti; ++tj) {
            const int j0 = tj * tile_size;
            const int jlen = Kokkos::min(tile_size, num_neighbors - j0);
            const bool diag = (tj == ti);
            if (!diag) {
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team_member, jlen),
                  [=](const int lj) {
                    const int gj = nl.neighbors(start_neighbors + j0 + lj);
                    const auto shell = load_shell(basis, gj);
                    const int base = lj * num_points;
                    for (int g = 0; g < num_points; ++g)
                      phi_b(base + g) = basis_eval_fast(
                          shell, points_scratch(g)[0], points_scratch(g)[1],
                          points_scratch(g)[2]);
                  });
              team_member.team_barrier();
            }
            const double *pj = diag ? phi_a.data() : phi_b.data();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange(team_member, ilen, jlen),
                [=](const int li, const int lj) {
                  if (diag && lj > li)
                    return; // lower-triangular block only
                  const int gi = nl.neighbors(start_neighbors + i0 + li);
                  const int gj = nl.neighbors(start_neighbors + j0 + lj);
                  double lr = 0;
                  for (int g = 0; g < num_points; ++g)
                    lr += potential_scratch_scaled(g) *
                          phi_a(li * num_points + g) * pj[lj * num_points + g];
                  Kokkos::atomic_add(&result(gi, gj), lr);
                  if (gi != gj)
                    Kokkos::atomic_add(&result(gj, gi), lr);
                });
            team_member.team_barrier();
          }
        }
      });

  space.fence();
  Kokkos::Profiling::popRegion();
  return result;
}

template <typename PotEvaluator>
  requires PotentialEvaluator<PotEvaluator>
DeviceView2DLeft coulomb_overlap_integral_sparse(const ExecSpace space,
                                                 const PotEvaluator &basis_aux,
                                                 const FlatGrid &grid,
                                                 const NeighborList nl) {

  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);

  const int max_points_per_box = nl.max_points_per_box;
  const int num_boxes = nl.offsets.extent(0) - 1;

  DeviceView2DLeft overlap_matrix("Overlap matrix", N_bf_aux, N_bf_aux);
  DeviceView2DLeft overlap_matrix_sym("Overlap matrix Symmetrized", N_bf_aux,
                                      N_bf_aux);

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

  int scratch_size_thread =
      shared_view_double::shmem_size((max_points_per_box));

  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_size_team),
                                Kokkos::PerThread(scratch_size_thread));

  Kokkos::parallel_for(
      "Sparse Kernel: Coulomb overlap matrix", policy_boxes,
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

        shared_view_double weights_scratch(team_member.team_scratch(0),
                                           num_points);

        shared_view_points points_scratch(team_member.team_scratch(0),
                                          num_points);

        shared_view_double potential_scratch_scaled(
            team_member.thread_scratch(0), num_points);

        // Fill points and weights scratch
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                             [=](const int local_g) {
                               const int global_g = start_points + local_g;
                               weights_scratch(local_g) =
                                   grid.weights(global_g);
                               points_scratch(local_g) =
                                   grid.quad_points(global_g);
                             });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf_aux),
            [=](const int global_alpha) {
              for (int local_g = 0; local_g < num_points; ++local_g) {
                
                double potential_alpha =
                    eval_aux_potential(basis_aux, global_alpha, points_scratch(local_g));

                potential_scratch_scaled(local_g) =
                    potential_alpha * weights_scratch(local_g);
              }

              for (int local_i = 0; local_i < num_neighbors; ++local_i) {

                const int global_i = nl.neighbors(start_neighbors + local_i);
                const auto shell_i = load_shell(basis_aux, global_i);

                double phi_i;
                double local_result = 0;
                for (int local_g = 0; local_g < num_points; ++local_g) {
                  phi_i = basis_eval_fast(shell_i, points_scratch(local_g)[0],
                                          points_scratch(local_g)[1],
                                          points_scratch(local_g)[2]);
                  local_result += potential_scratch_scaled(local_g) * phi_i;
                }

                Kokkos::atomic_add(&overlap_matrix(global_i, global_alpha),
                                   local_result);
              }
            });
      });

  Kokkos::parallel_for(
      "Symmetrize Coulomb Overlap Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf_aux, N_bf_aux}),
      KOKKOS_LAMBDA(int i, int j) {
        overlap_matrix_sym(i, j) =
            0.5 * (overlap_matrix(i, j) + overlap_matrix(j, i));
      });

  return overlap_matrix_sym;
}

// ============================================================================
//  Neighbor-tiled Coulomb overlap metric (A|B)
// ============================================================================
//
// Same result as coulomb_overlap_integral_sparse, which forms
//   (i|alpha) = sum_g w(g) V_alpha(g) phi_i(g)
// but avoids re-evaluating phi_i for every aux function alpha. phi is staged
// once per neighbor tile into a bounded team-scratch cache, and (as in
// compute_exact_exchange_tiled, Option A) the per-alpha potential lives in a
// single PerTeam [num_points] buffer rather than a per-thread one -- so the
// shared-memory footprint is independent of both num_neighbors and the team
// size.
template <typename PotEvaluator>
  requires PotentialEvaluator<PotEvaluator>
DeviceView2DLeft coulomb_overlap_integral_tiled(const ExecSpace space,
                                                const PotEvaluator &basis_aux,
                                                const FlatGrid &grid,
                                                const NeighborList nl,
                                                const int tile_size = 32) {

  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);

  const int max_points_per_box = nl.max_points_per_box;
  const int num_boxes = nl.offsets.extent(0) - 1;

  DeviceView2DLeft overlap_matrix("Overlap matrix", N_bf_aux, N_bf_aux);
  DeviceView2DLeft overlap_matrix_sym("Overlap matrix Symmetrized", N_bf_aux,
                                      N_bf_aux);

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

  const int scratch_team =
      shared_view_double::shmem_size(max_points_per_box) + // weights
      shared_view_points::shmem_size(max_points_per_box) + // points
      shared_view_double::shmem_size(tile_size *
                                     max_points_per_box) + // phi tile
      shared_view_double::shmem_size(max_points_per_box);  // potential (team)

  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_team));

  Kokkos::parallel_for(
      "Tiled Kernel: Coulomb overlap matrix", policy_boxes,
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

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                             [=](const int local_g) {
                               const int global_g = start_points + local_g;
                               weights_scratch(local_g) =
                                   grid.weights(global_g);
                               points_scratch(local_g) =
                                   grid.quad_points(global_g);
                             });
        team_member.team_barrier();

        const int num_tiles = (num_neighbors + tile_size - 1) / tile_size;
        for (int jt = 0; jt < num_tiles; ++jt) {
          const int j0 = jt * tile_size;
          const int jlen = Kokkos::min(tile_size, num_neighbors - j0);

          // Evaluate this neighbor tile's phi once into the cache.
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, jlen), [=](const int lj) {
                const int gi = nl.neighbors(start_neighbors + j0 + lj);
                const auto shell = load_shell(basis_aux, gi);
                const int base = lj * num_points;
                for (int g = 0; g < num_points; ++g)
                  phi_cache(base + g) = basis_eval_fast(
                      shell, points_scratch(g)[0], points_scratch(g)[1],
                      points_scratch(g)[2]);
              });
          team_member.team_barrier();

          // One alpha at a time: the team fills the shared potential buffer,
          // then contracts it against the cached tile of phi_i.
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
                  const int gi = nl.neighbors(start_neighbors + j0 + lj);
                  double local_result = 0;
                  for (int g = 0; g < num_points; ++g)
                    local_result += potential_scratch_scaled(g) *
                                    phi_cache(lj * num_points + g);
                  Kokkos::atomic_add(&overlap_matrix(gi, global_alpha),
                                     local_result);
                });
            team_member.team_barrier();
          }
        }
      });

  Kokkos::parallel_for(
      "Symmetrize Coulomb Overlap Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf_aux, N_bf_aux}),
      KOKKOS_LAMBDA(int i, int j) {
        overlap_matrix_sym(i, j) =
            0.5 * (overlap_matrix(i, j) + overlap_matrix(j, i));
      });

  return overlap_matrix_sym;
}

} // namespace Nukexc
