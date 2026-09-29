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
#include "nukexc/grid.hpp"
#include "basis_evaluator.hpp"
#include "nukexc_config.hpp"
#include "octree.hpp"
#include "stobasis.hpp"

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>
#include <KokkosBatched_Dot.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_Util.hpp>

#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Pair.hpp>

namespace Nukexc {

template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value>
DeviceView1D compute_density(const Evaluator basis, const FlatGrid grid,
                             const DeviceView2DLeft mo_orbitals,
                             const DeviceView1D mo_coeff) {

  ExecSpace space;
  const int N_bf = basis.nbf();
  const int N_quad = grid.quad_points.extent(0);
  const int N_occ = mo_orbitals.extent(1);

  DeviceView2DLeft collocation_values("Basis collocation", N_bf, N_quad);
  fill_collocation_values(space, basis, grid.quad_points, collocation_values);

  DeviceView1D density("density", N_quad);
  DeviceView2D intermediate_matrix("intermediate", N_occ, N_quad);

  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_values, 0.0,
                   intermediate_matrix);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Contract the basis in order to get the density
  Kokkos::parallel_for(
      "Contract Basis", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        auto intermediate_subview =
            Kokkos::subview(intermediate_matrix, Kokkos::ALL(), g);
        double sum = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, N_occ),
            [=](const int i, double &update_sum) {
              update_sum += mo_coeff(i) * intermediate_subview(i) *
                            intermediate_subview(i);
            },
            sum);
        team_member.team_barrier();
        density(g) = sum;
      });

  return density;
};

template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value>
DeviceView1D compute_density_sparse(const Evaluator basis,
                                    const FlatGrid grid, const NeighborList nl,
                                    const DeviceView2DLeft mo_orbitals,
                                    const DeviceView1D mo_coeff) {

  ExecSpace space;
  const int N_quad = grid.quad_points.extent(0);
  const int N_occ = mo_orbitals.extent(1);

  const int max_points_per_box = nl.max_points_per_box;
  const int num_boxes = nl.offsets.extent(0) - 1;

  DeviceView1D density("density", N_quad);

  Kokkos::TeamPolicy<ExecSpace> policy_boxes(space, num_boxes, Kokkos::AUTO());

  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
  // Contract the basis in order to get the density
  Kokkos::parallel_for(
      "Contract Basis", policy_boxes,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, N_quad);
        const int num_points = end_points - start_points;

        const int start_neighbors = nl.offsets(box_idx);
        const int num_neighbors = nl.offsets(box_idx + 1) - start_neighbors;

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, num_points),
            [=](const int local_g) {
              const int global_g = start_points + local_g;
              Point point_g = grid.quad_points(global_g);

              for (int global_i = 0; global_i < N_occ; ++global_i) {
                double psi_i = 0.0;
                for (int local_j = 0; local_j < num_neighbors; ++local_j) {
                  // Neighbour-list entries are BLOCKS. For an STO a block is
                  // one function, so this is the same work as before; for a
                  // GTO it is one contraction, whose functions all share the
                  // single evaluation below.
                  const int blk = nl.neighbors(start_neighbors + local_j);
                  double phi[Evaluator::max_block_size];
                  basis.template eval_block<BasisDeriv::Value>(
                      basis.load_block(blk), point_g[0], point_g[1],
                      point_g[2], phi, 1);
                  const int bf0 = basis.block_first_bf(blk);
                  const int nfun = basis.block_size(blk);
                  for (int t = 0; t < nfun; ++t)
                    psi_i += mo_orbitals(bf0 + t, global_i) * phi[t];
                }
                density(global_g) += mo_coeff(global_i) * psi_i * psi_i;
              }
            });
      });

  return density;
};

DeviceView1D compute_density(const DeviceView2DLeft collocation_values,
                             const DeviceView2DLeft mo_orbitals,
                             const DeviceView1D mo_coeff) {
  ExecSpace space;
  const int N_quad = collocation_values.extent(1);
  const int N_occ = mo_orbitals.extent(1);

  DeviceView1D density("density", N_quad);
  DeviceView2D intermediate_matrix("intermediate", N_occ, N_quad);

  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_values, 0.0,
                   intermediate_matrix);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Contract the basis in order to get the density
  Kokkos::parallel_for(
      "Contract Basis", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        double sum = 0;
        auto intermediate_subview =
            Kokkos::subview(intermediate_matrix, Kokkos::ALL(), g);

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, N_occ),
            [=](const int i, double &update_sum) {
              update_sum += mo_coeff(i) * intermediate_subview(i) *
                            intermediate_subview(i);
            },
            sum);
        team_member.team_barrier();
        density(g) = sum;
      });

  return density;
};

void compute_density_and_sigma(const DeviceView2DLeft collocation_values,
                               const DeviceView2DLeft collocation_gx,
                               const DeviceView2DLeft collocation_gy,
                               const DeviceView2DLeft collocation_gz,
                               const DeviceView2DLeft mo_orbitals,
                               const DeviceView1D mo_coeff, DeviceView1D rho,
                               DeviceView1D gx_rho, DeviceView1D gy_rho,
                               DeviceView1D gz_rho, DeviceView1D sigma) {
  ExecSpace space;
  const int N_quad = collocation_values.extent(1);
  const int N_occ = mo_orbitals.extent(1);

  DeviceView2D intermediate_values("intermediate values", N_occ, N_quad);
  DeviceView2D intermediate_gx("intermediate gx", N_occ, N_quad);
  DeviceView2D intermediate_gy("intermediate gy", N_occ, N_quad);
  DeviceView2D intermediate_gz("intermediate gz", N_occ, N_quad);

  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_values, 0.0,
                   intermediate_values);
  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_gx, 0.0,
                   intermediate_gx);
  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_gy, 0.0,
                   intermediate_gy);
  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_gz, 0.0,
                   intermediate_gz);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Contract the basis in order to get the density
  Kokkos::parallel_for(
      "Contract Basis", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        double sum_values = 0;
        double sum_gx = 0;
        double sum_gy = 0;
        double sum_gz = 0;
        auto intermediate_subview_values =
            Kokkos::subview(intermediate_values, Kokkos::ALL(), g);
        auto intermediate_subview_gx =
            Kokkos::subview(intermediate_gx, Kokkos::ALL(), g);
        auto intermediate_subview_gy =
            Kokkos::subview(intermediate_gy, Kokkos::ALL(), g);
        auto intermediate_subview_gz =
            Kokkos::subview(intermediate_gz, Kokkos::ALL(), g);

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, N_occ),
            [=](const int i, double &update_sum, double &update_sum_gx,
                double &update_sum_gy, double &update_sum_gz) {
              update_sum += mo_coeff(i) * intermediate_subview_values(i) *
                            intermediate_subview_values(i);
              update_sum_gx += mo_coeff(i) * intermediate_subview_gx(i) *
                               intermediate_subview_values(i);
              update_sum_gy += mo_coeff(i) * intermediate_subview_gy(i) *
                               intermediate_subview_values(i);
              update_sum_gz += mo_coeff(i) * intermediate_subview_gz(i) *
                               intermediate_subview_values(i);
            },
            sum_values, sum_gx, sum_gy, sum_gz);
        team_member.team_barrier();
        rho(g) = sum_values;
        gx_rho(g) = 2.0 * sum_gx;
        gy_rho(g) = 2.0 * sum_gy;
        gz_rho(g) = 2.0 * sum_gz;
        sigma(g) = 4.0 * (sum_gx * sum_gx + sum_gy * sum_gy + sum_gz * sum_gz);
      });
};

// Sparse counterpart of compute_density_and_sigma. rho, gx_rho, gy_rho and
// gz_rho are accumulated into, so they must be zero on entry; sigma is
// overwritten. Each grid point belongs to exactly one box, so a given
// global_g has a single writer thread.
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
void compute_density_and_sigma_sparse(
    const Evaluator basis, const FlatGrid grid, const NeighborList nl,
    const DeviceView2DLeft mo_orbitals, const DeviceView1D mo_coeff,
    DeviceView1D rho, DeviceView1D gx_rho, DeviceView1D gy_rho,
    DeviceView1D gz_rho, DeviceView1D sigma) {

  ExecSpace space;
  const int N_quad = grid.quad_points.extent(0);
  const int N_occ = mo_orbitals.extent(1);

  const int max_points_per_box = nl.max_points_per_box;
  const int num_boxes = nl.offsets.extent(0) - 1;

  Kokkos::TeamPolicy<ExecSpace> policy_boxes(space, num_boxes, Kokkos::AUTO());

  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
  // Contract the basis in order to get the density and its gradient
  Kokkos::parallel_for(
      "Contract Basis", policy_boxes,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, N_quad);
        const int num_points = end_points - start_points;

        const int start_neighbors = nl.offsets(box_idx);
        const int num_neighbors = nl.offsets(box_idx + 1) - start_neighbors;

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, num_points),
            [=](const int local_g) {
              const int global_g = start_points + local_g;
              Point point_g = grid.quad_points(global_g);

              for (int global_i = 0; global_i < N_occ; ++global_i) {
                double psi_i = 0.0;
                double psi_gx_i = 0.0;
                double psi_gy_i = 0.0;
                double psi_gz_i = 0.0;
                for (int local_j = 0; local_j < num_neighbors; ++local_j) {
                  const int blk = nl.neighbors(start_neighbors + local_j);
                  constexpr int MBS = Evaluator::max_block_size;
                  double buf[basis_ncomp(BasisDeriv::Gradient) * MBS];
                  basis.template eval_block<BasisDeriv::Gradient>(
                      basis.load_block(blk), point_g[0], point_g[1],
                      point_g[2], buf, MBS);
                  const int bf0 = basis.block_first_bf(blk);
                  const int nfun = basis.block_size(blk);
                  for (int t = 0; t < nfun; ++t) {
                    const double c_ji = mo_orbitals(bf0 + t, global_i);
                    psi_i += c_ji * buf[BF_VAL * MBS + t];
                    psi_gx_i += c_ji * buf[BF_DX * MBS + t];
                    psi_gy_i += c_ji * buf[BF_DY * MBS + t];
                    psi_gz_i += c_ji * buf[BF_DZ * MBS + t];
                  }
                }
                rho(global_g) += mo_coeff(global_i) * psi_i * psi_i;
                gx_rho(global_g) += 2.0 * mo_coeff(global_i) * psi_gx_i * psi_i;
                gy_rho(global_g) += 2.0 * mo_coeff(global_i) * psi_gy_i * psi_i;
                gz_rho(global_g) += 2.0 * mo_coeff(global_i) * psi_gz_i * psi_i;
              }
              // sigma is quadratic in the summed gradient, so it can only be
              // formed once the loop over occupied orbitals has finished.
              sigma(global_g) = gx_rho(global_g) * gx_rho(global_g) +
                                gy_rho(global_g) * gy_rho(global_g) +
                                gz_rho(global_g) * gz_rho(global_g);
            });
      });
};

// EXPERIMENTAL -- meta-GGA groundwork. Not on any production path and not
// exercised by any test; its only caller is compute_mgga() in xc_integrals.hpp,
// which is equally experimental. Kept as the starting point for whoever adds
// mGGA support next.
// TODO: This function has not been tested and probably contains bugs
void compute_density_and_sigma_and_tau(
    const DeviceView2DLeft collocation_values,
    const DeviceView2DLeft collocation_gx,
    const DeviceView2DLeft collocation_gy,
    const DeviceView2DLeft collocation_gz, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, DeviceView1D rho, DeviceView1D gx_rho,
    DeviceView1D gy_rho, DeviceView1D gz_rho, DeviceView1D sigma,
    DeviceView1D tau) {
  ExecSpace space;
  const int N_quad = collocation_values.extent(1);
  const int N_occ = mo_orbitals.extent(1);

  DeviceView2D intermediate_values("intermediate values", N_occ, N_quad);
  DeviceView2D intermediate_gx("intermediate gx", N_occ, N_quad);
  DeviceView2D intermediate_gy("intermediate gy", N_occ, N_quad);
  DeviceView2D intermediate_gz("intermediate gz", N_occ, N_quad);

  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_values, 0.0,
                   intermediate_values);
  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_gx, 0.0,
                   intermediate_gx);
  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_gy, 0.0,
                   intermediate_gy);
  KokkosBlas::gemm(space, "T", "N", 1.0, mo_orbitals, collocation_gz, 0.0,
                   intermediate_gz);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Contract the basis in order to get the density
  Kokkos::parallel_for(
      "Contract Basis", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        double sum_values = 0;
        double sum_gx = 0;
        double sum_gy = 0;
        double sum_gz = 0;
        double sum_tau = 0;

        auto intermediate_subview_values =
            Kokkos::subview(intermediate_values, Kokkos::ALL(), g);
        auto intermediate_subview_gx =
            Kokkos::subview(intermediate_gx, Kokkos::ALL(), g);
        auto intermediate_subview_gy =
            Kokkos::subview(intermediate_gy, Kokkos::ALL(), g);
        auto intermediate_subview_gz =
            Kokkos::subview(intermediate_gz, Kokkos::ALL(), g);

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, N_occ),
            [=](const int i, double &update_sum, double &update_sum_gx,
                double &update_sum_gy, double &update_sum_gz,
                double &update_tau) {
              update_sum += mo_coeff(i) * intermediate_subview_values(i) *
                            intermediate_subview_values(i);
              update_sum_gx += mo_coeff(i) * intermediate_subview_gx(i) *
                               intermediate_subview_values(i);
              update_sum_gy += mo_coeff(i) * intermediate_subview_gy(i) *
                               intermediate_subview_values(i);
              update_sum_gz += mo_coeff(i) * intermediate_subview_gz(i) *
                               intermediate_subview_values(i);

              update_tau +=
                  mo_coeff(i) *
                  (intermediate_subview_gx(i) * intermediate_subview_gx(i) +
                   intermediate_subview_gy(i) * intermediate_subview_gy(i) +
                   intermediate_subview_gz(i) * intermediate_subview_gz(i));
            },
            sum_values, sum_gx, sum_gy, sum_gz, sum_tau);
        team_member.team_barrier();
        rho(g) = sum_values;
        gx_rho(g) = 2.0 * sum_gx;
        gy_rho(g) = 2.0 * sum_gy;
        gz_rho(g) = 2.0 * sum_gz;
        sigma(g) = 4.0 * (sum_gx * sum_gx + sum_gy * sum_gy + sum_gz * sum_gz);
        tau(g) = 0.5 * sum_tau;
      });
};
}; // namespace Nukexc
