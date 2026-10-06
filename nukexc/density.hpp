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

// ============================================================================
//  Which contraction to use
// ============================================================================
//
// Both routes are kept, because neither dominates:
//
//   Orbital        O(N_occ * nbf * npts)
//   DensityMatrix  O(nbf^2 * npts)
//
// The orbital route wins whenever N_occ is small against the basis size,
// which for a decent basis is the common case. The density-matrix route is
// the one that admits fractional and ensemble occupations, supplies the |P|
// magnitudes that shell-pair screening needs, and -- per Blum et al.
// (Eqs. 27-28) -- wins asymptotically once batching is in place, because the
// right comparison there is N_occ against the BATCH's non-zero basis count
// n_be rather than the total nbf. n_be is bounded while N_occ grows with
// system size.
//
// Auto therefore compares against nbf today and should compare against n_be
// once the kernels are batch-local; the crossover it picks is deliberately
// conservative rather than tuned, since the real decision belongs per batch.
enum class DensityPath {
  Orbital,       ///< contract through the occupied coefficients
  DensityMatrix, ///< contract through P
  Auto           ///< pick from the occupied fraction
};

/// Resolve Auto. Exposed so a caller can see what would be chosen.
inline DensityPath resolve_density_path(DensityPath p, int n_occ, int nbf) {
  if (p != DensityPath::Auto)
    return p;
  // The orbital route moves N_occ columns, the matrix route nbf. Prefer the
  // orbital route with a margin, since it also avoids forming P.
  return (2 * n_occ < nbf) ? DensityPath::Orbital : DensityPath::DensityMatrix;
}

// ============================================================================
//  Density-matrix formulation
// ============================================================================
//
// rho(r) = sum_uv P_uv chi_u(r) chi_v(r)
//
// The MO route below costs O(N_occ N_bf N_pts); this one costs
// O(N_bf^2 N_pts). Which is cheaper depends on the occupied fraction, but
// asymptotics are not the reason to prefer this one. Blum et al. (Comput.
// Phys. Commun. 180 (2009) 2175, Eqs. 27-28) make the point that a batch's
// non-zero basis count is bounded while N_occ grows with system size, so the
// MO route is the one that degrades. More immediately for NuKEXC: the density
// matrix is what SCF hosts actually hand over, it is what fractional and
// ensemble occupations require, and |P| magnitudes are the standard input to
// shell-pair screening, which the MO route cannot provide at all.
//
// The contraction is one GEMM (X = P Phi) plus a column-wise dot product.

/// rho from a density matrix and a precomputed collocation block.
inline DeviceView1D compute_density_dm(const DeviceView2DLeft collocation_values,
                                       const DeviceView2DLeft P) {
  ExecSpace space;
  const int N_bf = collocation_values.extent(0);
  const int N_quad = collocation_values.extent(1);

  DeviceView1D density("density", N_quad);
  DeviceView2DLeft X("P phi", N_bf, N_quad);

  KokkosBlas::gemm(space, "N", "N", 1.0, P, collocation_values, 0.0, X);

  Kokkos::parallel_for(
      "Contract density matrix", Kokkos::RangePolicy<ExecSpace>(space, 0, N_quad),
      KOKKOS_LAMBDA(const int g) {
        double acc = 0.0;
        for (int mu = 0; mu < N_bf; ++mu)
          acc += collocation_values(mu, g) * X(mu, g);
        density(g) = acc;
      });

  return density;
}

/// rho, its gradient and sigma from a density matrix.
///
/// With P symmetric, grad rho = 2 sum_u grad chi_u (P chi)_u, so the same
/// X = P Phi serves all four outputs and only one GEMM is needed.
inline void compute_density_and_sigma_dm(
    const DeviceView2DLeft collocation_values,
    const DeviceView2DLeft collocation_gx, const DeviceView2DLeft collocation_gy,
    const DeviceView2DLeft collocation_gz, const DeviceView2DLeft P,
    DeviceView1D rho, DeviceView1D gx_rho, DeviceView1D gy_rho,
    DeviceView1D gz_rho, DeviceView1D sigma) {
  ExecSpace space;
  const int N_bf = collocation_values.extent(0);
  const int N_quad = collocation_values.extent(1);

  DeviceView2DLeft X("P phi", N_bf, N_quad);
  KokkosBlas::gemm(space, "N", "N", 1.0, P, collocation_values, 0.0, X);

  Kokkos::parallel_for(
      "Contract density matrix with gradient",
      Kokkos::RangePolicy<ExecSpace>(space, 0, N_quad),
      KOKKOS_LAMBDA(const int g) {
        double r = 0.0, dx = 0.0, dy = 0.0, dz = 0.0;
        for (int mu = 0; mu < N_bf; ++mu) {
          const double x = X(mu, g);
          r += collocation_values(mu, g) * x;
          dx += collocation_gx(mu, g) * x;
          dy += collocation_gy(mu, g) * x;
          dz += collocation_gz(mu, g) * x;
        }
        rho(g) = r;
        gx_rho(g) = 2.0 * dx;
        gy_rho(g) = 2.0 * dy;
        gz_rho(g) = 2.0 * dz;
        sigma(g) = 4.0 * (dx * dx + dy * dy + dz * dz);
      });
}

/**
 *  @brief The meta-GGA density ingredients, tau and the density Laplacian.
 *
 *  These are INGREDIENTS, not chain-rule terms. Libxc owns the derivative
 *  tower and libxckernel owns the derivatives of the ingredients with
 *  respect to the density matrix; what is left for the host is the
 *  ingredients themselves, which is this.
 *
 *      tau    = 1/2 sum_uv P_uv grad chi_u . grad chi_v
 *      lapl   = 2 sum_u lapl chi_u (P chi)_u + 4 tau
 *
 *  The Laplacian follows from differentiating rho = sum_uv P_uv chi_u chi_v
 *  twice and using the symmetry of P, the same assumption
 *  compute_density_and_sigma_dm already makes when it writes
 *  grad rho = 2 (P chi) . grad chi. Both terms are needed: the second is
 *  exactly twice the raw gradient sum, so it is reused rather than
 *  recomputed.
 *
 *  tau is the TOTAL kinetic energy density, matching the total rho that
 *  compute_density_dm returns, which is Libxc's unpolarized convention.
 *  Since the kinetic matrix is T_uv = 1/2 integral w grad chi_u . grad chi_v,
 *  quadrature of tau over the same grid must reproduce Tr(P T) exactly --
 *  not merely physically -- which is what tests/test_meta_fields.cxx
 *  asserts.
 *
 *  One nbf x npts temporary is reused across the three Cartesian passes
 *  rather than holding three at once: at 13k points and a few hundred
 *  functions the difference is tens of megabytes.
 *
 *  Pass an EMPTY collocation_lapl to compute tau alone and leave lapl_rho
 *  untouched. A tau-only meta-GGA needs no second derivatives, so it has
 *  no Laplacian collocation to give -- and the alternative of passing
 *  something else of the right shape would silently produce a wrong
 *  Laplacian rather than no Laplacian.
 */
inline void compute_meta_fields_dm(const DeviceView2DLeft collocation_values,
                                   const DeviceView2DLeft collocation_gx,
                                   const DeviceView2DLeft collocation_gy,
                                   const DeviceView2DLeft collocation_gz,
                                   const DeviceView2DLeft collocation_lapl,
                                   const DeviceView2DLeft P, DeviceView1D tau,
                                   DeviceView1D lapl_rho) {
  ExecSpace space;
  const int N_bf = collocation_values.extent(0);
  const int N_quad = collocation_values.extent(1);

  DeviceView2DLeft Y("P dchi", N_bf, N_quad);

  // Accumulate the raw sum_uv P_uv grad chi_u . grad chi_v into tau, one
  // Cartesian direction at a time; halve it at the end.
  Kokkos::deep_copy(tau, 0.0);
  for (int d = 0; d < 3; ++d) {
    const DeviceView2DLeft dchi =
        (d == 0) ? collocation_gx : (d == 1) ? collocation_gy : collocation_gz;
    KokkosBlas::gemm(space, "N", "N", 1.0, P, dchi, 0.0, Y);
    Kokkos::parallel_for(
        "accumulate tau", Kokkos::RangePolicy<ExecSpace>(space, 0, N_quad),
        KOKKOS_LAMBDA(const int g) {
          double t = 0.0;
          for (int mu = 0; mu < N_bf; ++mu)
            t += dchi(mu, g) * Y(mu, g);
          tau(g) += t;
        });
  }

  if (collocation_lapl.extent(0) == 0) {
    // tau only.
    Kokkos::parallel_for(
        "halve tau", Kokkos::RangePolicy<ExecSpace>(space, 0, N_quad),
        KOKKOS_LAMBDA(const int g) { tau(g) *= 0.5; });
    return;
  }

  if (static_cast<int>(collocation_lapl.extent(0)) != N_bf ||
      static_cast<int>(collocation_lapl.extent(1)) != N_quad)
    throw std::invalid_argument(
        "compute_meta_fields_dm: collocation_lapl must be (nbf, npts) or "
        "empty");

  KokkosBlas::gemm(space, "N", "N", 1.0, P, collocation_values, 0.0, Y);
  Kokkos::parallel_for(
      "meta fields", Kokkos::RangePolicy<ExecSpace>(space, 0, N_quad),
      KOKKOS_LAMBDA(const int g) {
        double l = 0.0;
        for (int mu = 0; mu < N_bf; ++mu)
          l += collocation_lapl(mu, g) * Y(mu, g);
        // tau still holds the raw gradient sum, which is 2 tau.
        lapl_rho(g) = 2.0 * l + 2.0 * tau(g);
        tau(g) *= 0.5;
      });
}

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
// ============================================================================
//  Complex orbitals, in split real/imaginary storage
// ============================================================================
//
// Needed by Perdew-Zunger self-interaction correction, which wants the
// density of each orbital separately.
//
// For REAL basis functions no complex arithmetic is required, even though the
// orbitals are complex. The per-orbital density matrix P^i_uv = conj(c_ui)
// c_vi is Hermitian, but rho_i is real, so only its real part contributes:
//
//   rho_i = sum_uv Re(conj(c_ui) c_vi) chi_u chi_v
//         = (Re psi_i)^2 + (Im psi_i)^2
//
// with Re psi_i = sum_u Re(c_ui) chi_u and likewise for the imaginary part.
// Two real GEMMs, and the resulting potential matrix is real symmetric. This
// is the split Re/Im layout libxckernel recommends, and it is why the PZ-SIC
// path does not require templating the library on a complex scalar --
// complex BASIS functions (GIAO, Bloch) and the paramagnetic current of the
// gauge-corrected meta-GGAs do, and those are separate work.

/// Per-orbital densities |psi_i|^2 for complex orbitals in split storage.
///
/// @param C_re, C_im  (nbf, norb) LayoutLeft coefficient blocks
/// @returns (norb, npts) densities, one row per orbital
inline DeviceView2D compute_orbital_densities(
    const DeviceView2DLeft collocation_values, const DeviceView2DLeft C_re,
    const DeviceView2DLeft C_im) {
  ExecSpace space;
  const int npts = collocation_values.extent(1);
  const int norb = C_re.extent(1);

  DeviceView2D psi_re("psi_re", norb, npts), psi_im("psi_im", norb, npts);
  KokkosBlas::gemm(space, "T", "N", 1.0, C_re, collocation_values, 0.0, psi_re);
  KokkosBlas::gemm(space, "T", "N", 1.0, C_im, collocation_values, 0.0, psi_im);

  DeviceView2D rho("orbital densities", norb, npts);
  Kokkos::parallel_for(
      "orbital densities",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0},
                                                        {norb, npts}),
      KOKKOS_LAMBDA(const int i, const int g) {
        rho(i, g) = psi_re(i, g) * psi_re(i, g) + psi_im(i, g) * psi_im(i, g);
      });
  return rho;
}

/// Per-orbital densities and their gradients.
///
/// grad rho_i = 2 [ Re psi_i grad(Re psi_i) + Im psi_i grad(Im psi_i) ]
inline void compute_orbital_densities_and_sigma(
    const DeviceView2DLeft collocation_values,
    const DeviceView2DLeft collocation_gx, const DeviceView2DLeft collocation_gy,
    const DeviceView2DLeft collocation_gz, const DeviceView2DLeft C_re,
    const DeviceView2DLeft C_im, DeviceView2D rho, DeviceView2D gx_rho,
    DeviceView2D gy_rho, DeviceView2D gz_rho, DeviceView2D sigma) {
  ExecSpace space;
  const int npts = collocation_values.extent(1);
  const int norb = C_re.extent(1);

  DeviceView2D pr("psi_re", norb, npts), pi_("psi_im", norb, npts);
  DeviceView2D prx("prx", norb, npts), pry("pry", norb, npts),
      prz("prz", norb, npts);
  DeviceView2D pix("pix", norb, npts), piy("piy", norb, npts),
      piz("piz", norb, npts);

  KokkosBlas::gemm(space, "T", "N", 1.0, C_re, collocation_values, 0.0, pr);
  KokkosBlas::gemm(space, "T", "N", 1.0, C_im, collocation_values, 0.0, pi_);
  KokkosBlas::gemm(space, "T", "N", 1.0, C_re, collocation_gx, 0.0, prx);
  KokkosBlas::gemm(space, "T", "N", 1.0, C_re, collocation_gy, 0.0, pry);
  KokkosBlas::gemm(space, "T", "N", 1.0, C_re, collocation_gz, 0.0, prz);
  KokkosBlas::gemm(space, "T", "N", 1.0, C_im, collocation_gx, 0.0, pix);
  KokkosBlas::gemm(space, "T", "N", 1.0, C_im, collocation_gy, 0.0, piy);
  KokkosBlas::gemm(space, "T", "N", 1.0, C_im, collocation_gz, 0.0, piz);

  Kokkos::parallel_for(
      "orbital densities and gradients",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0},
                                                        {norb, npts}),
      KOKKOS_LAMBDA(const int i, const int g) {
        const double a = pr(i, g), b = pi_(i, g);
        rho(i, g) = a * a + b * b;
        const double dx = 2.0 * (a * prx(i, g) + b * pix(i, g));
        const double dy = 2.0 * (a * pry(i, g) + b * piy(i, g));
        const double dz = 2.0 * (a * prz(i, g) + b * piz(i, g));
        gx_rho(i, g) = dx;
        gy_rho(i, g) = dy;
        gz_rho(i, g) = dz;
        sigma(i, g) = dx * dx + dy * dy + dz * dz;
      });
}

} // namespace Nukexc
