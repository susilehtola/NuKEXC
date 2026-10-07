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
#include "sap.hpp"
#include "stobasis.hpp"

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>
#include <KokkosBatched_Dot.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_Util.hpp>

#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Pair.hpp>
#include <stdexcept>

namespace Nukexc {
// A helper to determine batch size based on available memory or a fixed
// constant
const size_t CHUNK_SIZE = 50000;

template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value>
DeviceView2DLeft
overlap_integral(const Evaluator &basis, Kokkos::View<Point *> quadrature_points,
                 Kokkos::View<double *, ExecSpace> quadrature_weights) {

  int N = basis.nbf();
  int total_quad_points = quadrature_points.extent(0);

  DeviceView2DLeft overlap_matrix("Overlap matrix", N, N);
  // Pre-allocate ALL buffers — two sets, one per stream
  DeviceView2DLeft col_a("col_a", N, CHUNK_SIZE);
  DeviceView2DLeft col_b("col_b", N, CHUNK_SIZE);
  DeviceView2DLeft wt_a("wt_a", N, CHUNK_SIZE);
  DeviceView2DLeft wt_b("wt_b", N, CHUNK_SIZE);

  ExecSpace space_a, space_b;

  for (size_t start = 0; start < total_quad_points; start += CHUNK_SIZE) {
    int current_batch_size = std::min(CHUNK_SIZE, total_quad_points - start);
    bool even = (start / CHUNK_SIZE) % 2 == 0;

    auto &col_cur = even ? col_a : col_b;
    auto &wt_cur = even ? wt_a : wt_b;
    auto &space_cur = even ? space_a : space_b;
    auto &space_prev = even ? space_b : space_a;

    auto batch_pts = Kokkos::subview(
        quadrature_points, std::make_pair(start, start + current_batch_size));
    auto batch_wts = Kokkos::subview(
        quadrature_weights, std::make_pair(start, start + current_batch_size));
    auto col_view = Kokkos::subview(col_cur, Kokkos::ALL,
                                    std::make_pair(0, current_batch_size));
    auto wt_view = Kokkos::subview(wt_cur, Kokkos::ALL,
                                   std::make_pair(0, current_batch_size));

    // Collocation on space_cur — can overlap with GEMM still running
    // on space_prev from the last iteration
    fill_collocation_values(space_cur, basis, batch_pts, col_view);

    Kokkos::parallel_for(
        "Scale",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            space_cur, {0, 0}, {N, current_batch_size}),
        KOKKOS_LAMBDA(int i, int g) {
          wt_view(i, g) = batch_wts(g) * col_view(i, g);
        });

    // CRITICAL: wait for previous GEMM to finish before starting next one
    // They cannot run concurrently — both write to overlap_matrix
    space_prev.fence();

    KokkosBlas::gemm(space_cur, "N", "T", 1.0, wt_view, col_view, 1.0,
                     overlap_matrix);
  }

  space_a.fence();
  space_b.fence();
  return overlap_matrix;
}

struct CoreHamiltonianResult {
  DeviceView2DLeft overlap;
  DeviceView2DLeft kinetic;
  DeviceView2DLeft nuclear;
  DeviceView2DLeft hamiltonian; // T + V_n
};

// With use_SAP the nuclear term is built from the tabulated SAP screened
// charges instead of the bare Z, so the returned `hamiltonian` is T + V_SAP,
// i.e. the SAP guess Fock matrix. `overlap` and `kinetic` are unaffected.
// sap_dir overrides the potential directory; empty falls back to
// NUKEXC_SAP_DIR and then to the in-tree default.
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
CoreHamiltonianResult compute_core_hamiltonian(const Evaluator &basis,
                                               const FlatGrid &grid,
                                               const bool use_SAP = false,
                                               const std::string &sap_dir = "") {

  int N = basis.nbf();
  auto quadrature_points = grid.quad_points;
  auto quadrature_weights = grid.weights;
  auto Z = grid.Z;
  auto atom_centers = grid.atom_centers;

  int total_quad_points = quadrature_points.extent(0);

  CoreHamiltonianResult result;
  result.overlap = DeviceView2DLeft("Overlap matrix", N, N);
  result.kinetic = DeviceView2DLeft("Kinetic matrix", N, N);
  result.nuclear = DeviceView2DLeft("Nuclear potential matrix", N, N);
  result.hamiltonian = DeviceView2DLeft("Core Hamiltonian", N, N);

  // SAP guess: replace the bare nuclear charges by the tabulated screened
  // charges Z_eff^A(r). Read once per distinct element and uploaded as a
  // (n_atoms, n_r) table; the radial grid inverts analytically, so the lookup
  // in the kernel below costs a couple of flops and two loads.
  SapPotentials sap;
  if (use_SAP)
    sap = load_sap_potentials(
        grid.Z, sap_dir.empty()
                    ? sap_potential_dir("input/potentials/nr/LDAX/")
                    : sap_dir);

  // Single set of double buffers shared across all three integrals
  DeviceView2DLeft col_a("col_a", N, CHUNK_SIZE);
  DeviceView2DLeft col_b("col_b", N, CHUNK_SIZE);
  DeviceView2DLeft wt_overlap_a("wt_overlap_a", N, CHUNK_SIZE);
  DeviceView2DLeft wt_overlap_b("wt_overlap_b", N, CHUNK_SIZE);
  DeviceView2DLeft wt_nuclear_a("wt_nuclear_a", N, CHUNK_SIZE);
  DeviceView2DLeft wt_nuclear_b("wt_nuclear_b", N, CHUNK_SIZE);

  DeviceView2DLeft Gx_a("Gx_a", N, CHUNK_SIZE), Gx_b("Gx_b", N, CHUNK_SIZE);
  DeviceView2DLeft Gy_a("Gy_a", N, CHUNK_SIZE), Gy_b("Gy_b", N, CHUNK_SIZE);
  DeviceView2DLeft Gz_a("Gz_a", N, CHUNK_SIZE), Gz_b("Gz_b", N, CHUNK_SIZE);

  // sqrt of the nuclear attraction per point, one per double-buffer half.
  DeviceView1D vnuc_a("vnuc_a", CHUNK_SIZE), vnuc_b("vnuc_b", CHUNK_SIZE);

  ExecSpace space_a, space_b;

  for (size_t start = 0; start < total_quad_points; start += CHUNK_SIZE) {
    int current_batch_size = std::min(CHUNK_SIZE, total_quad_points - start);
    bool even = (start / CHUNK_SIZE) % 2 == 0;

    auto &wt_ov_cur = even ? wt_overlap_a : wt_overlap_b;
    auto &wt_nuc_cur = even ? wt_nuclear_a : wt_nuclear_b;
    auto &Gx_cur = even ? Gx_a : Gx_b;
    auto &Gy_cur = even ? Gy_a : Gy_b;
    auto &Gz_cur = even ? Gz_a : Gz_b;
    auto &space_cur = even ? space_a : space_b;
    auto &space_prev = even ? space_b : space_a;

    auto batch_pts = Kokkos::subview(
        quadrature_points, std::make_pair(start, start + current_batch_size));
    auto batch_wts = Kokkos::subview(
        quadrature_weights, std::make_pair(start, start + current_batch_size));

    auto overlap_view = Kokkos::subview(wt_ov_cur, Kokkos::ALL,
                                        std::make_pair(0, current_batch_size));
    auto nuclear_view = Kokkos::subview(wt_nuc_cur, Kokkos::ALL,
                                        std::make_pair(0, current_batch_size));
    auto Gx_view = Kokkos::subview(Gx_cur, Kokkos::ALL,
                                   std::make_pair(0, current_batch_size));
    auto Gy_view = Kokkos::subview(Gy_cur, Kokkos::ALL,
                                   std::make_pair(0, current_batch_size));
    auto Gz_view = Kokkos::subview(Gz_cur, Kokkos::ALL,
                                   std::make_pair(0, current_batch_size));

    // Basis evaluation goes through the shell abstraction: one thread per
    // (shell, point), each writing only its own shell's functions. The
    // previous version walked basis functions one at a time through
    // load_shell, which static_asserts a single function per block -- so
    // this routine, although templated over the evaluator concept, could
    // not be instantiated with any evaluator whose shells hold more than
    // one function. Every Gaussian or numerical-orbital basis is such an
    // evaluator, which made the whole core-Hamiltonian family Slater-only
    // in practice.
    fill_collocation_value_grad(space_cur, basis, batch_pts, overlap_view,
                                Gx_view, Gy_view, Gz_view);

    // sqrt of the nuclear attraction, per point. Hoisted out of the scaling
    // pass below because it depends only on the point: the fused version
    // recomputed the atom sum once per basis function.
    auto vnuc_cur = Kokkos::subview(even ? vnuc_a : vnuc_b,
                                    std::make_pair(0, current_batch_size));
    Kokkos::parallel_for(
        "Nuclear attraction",
        Kokkos::RangePolicy<ExecSpace>(space_cur, 0, current_batch_size),
        KOKKOS_LAMBDA(const int g) {
          const Point local_pt = batch_pts(g);
          double v = 0.0;
          // Z_eff(r) -> Z as r -> 0, so the SAP branch reduces to the bare
          // nuclear attraction at the cusp and shares the epsilon_shift
          // guard.
          for (size_t k = 0; k < atom_centers.extent(0); ++k) {
            const double r = dist(local_pt, atom_centers(k)) + epsilon_shift;
            v += (use_SAP ? sap_zeff(sap, static_cast<int>(k), r)
                          : double(Z(k))) /
                 r;
          }
          vnuc_cur(g) = Kokkos::sqrt(v);
        });

    Kokkos::parallel_for(
        "Scale by the quadrature weight",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            space_cur, {0, 0}, {N, current_batch_size}),
        KOKKOS_LAMBDA(const int i, const int g) {
          const double wf = Kokkos::sqrt(batch_wts(g));
          const double ov_wf = overlap_view(i, g) * wf;
          overlap_view(i, g) = ov_wf;
          nuclear_view(i, g) = ov_wf * vnuc_cur(g);
          Gx_view(i, g) *= wf;
          Gy_view(i, g) *= wf;
          Gz_view(i, g) *= wf;
        });

    // Serialise GEMMs against previous iteration
    space_prev.fence();

    // Overlap: S += wt_ov * col^T
    KokkosBlas::gemm(space_cur, "N", "T", 1.0, overlap_view, overlap_view, 1.0,
                     result.overlap);

    // Nuclear: V += wt_nuc * col^T
    KokkosBlas::gemm(space_cur, "N", "T", -1.0, nuclear_view, nuclear_view, 1.0,
                     result.nuclear);

    // Kinetic: T += 0.5 * G{xyz} * G{xyz}^T
    KokkosBlas::gemm(space_cur, "N", "T", 0.5, Gx_view, Gx_view, 1.0,
                     result.kinetic);
    KokkosBlas::gemm(space_cur, "N", "T", 0.5, Gy_view, Gy_view, 1.0,
                     result.kinetic);
    KokkosBlas::gemm(space_cur, "N", "T", 0.5, Gz_view, Gz_view, 1.0,
                     result.kinetic);
  }

  space_a.fence();
  space_b.fence();

  // Form H = T + V_n
  auto T = result.kinetic;
  auto V = result.nuclear;
  auto H = result.hamiltonian;
  Kokkos::parallel_for(
      "Form core Hamiltonian",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j) { H(i, j) = T(i, j) + V(i, j); });

  return result;
}

struct CoreHamiltonianReducer {
  double s = 0.0, v = 0.0, t = 0.0;

  KOKKOS_INLINE_FUNCTION
  CoreHamiltonianReducer &operator+=(const CoreHamiltonianReducer &rhs) {
    s += rhs.s;
    v += rhs.v;
    t += rhs.t;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  CoreHamiltonianReducer operator+(const CoreHamiltonianReducer &rhs) const {
    return {s + rhs.s, v + rhs.v, t + rhs.t};
  }
  KOKKOS_INLINE_FUNCTION
  void join(CoreHamiltonianReducer &dst, const CoreHamiltonianReducer &src) {
    dst.s += src.s;
    dst.v += src.v;
    dst.t += src.t;
  }

  KOKKOS_INLINE_FUNCTION static void init(CoreHamiltonianReducer &val) {
    val.s = 0.0;
    val.v = 0.0;
    val.t = 0.0;
  }
};

template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
CoreHamiltonianResult compute_core_hamiltonian_screened(
    const Evaluator &basis, const FlatGrid &grid, const NeighborList &nl) {

  int N = basis.nbf();
  auto Z = grid.Z;
  auto atom_centers = grid.atom_centers;

  const int max_points_per_box = nl.max_points_per_box;
  const int total_points = nl.total_points;
  const int num_boxes = nl.offsets.extent(0) - 1;

  CoreHamiltonianResult result;
  result.overlap = DeviceView2DLeft("Overlap matrix", N, N);
  result.kinetic = DeviceView2DLeft("Kinetic matrix", N, N);
  result.nuclear = DeviceView2DLeft("Nuclear potential matrix", N, N);
  result.hamiltonian = DeviceView2DLeft("Core Hamiltonian", N, N);

  Kokkos::TeamPolicy<ExecSpace> policy(num_boxes, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Compute Core Hamiltonian Screened", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();

        // Compute number of points per box
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, total_points);
        const int num_points = end_points - start_points;

        // Compute number of neighbors per box
        const int start_neighbors = nl.offsets(box_idx);
        const int end_neighbors = nl.offsets(box_idx + 1);
        const int num_neighbors = end_neighbors - start_neighbors;

        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                      num_neighbors),
            [=](const int local_i, const int local_j) {
              const int global_i = nl.neighbors(start_neighbors + local_i);
              const int global_j = nl.neighbors(start_neighbors + local_j);

              if (global_i < global_j)
                return;

              const auto local_basis_i = load_shell(basis, global_i);

              const auto local_basis_j = load_shell(basis, global_j);

              double total_s = 0.0;
              double total_t = 0.0;
              double total_v = 0.0;

              for (int local_g = 0; local_g < num_points; ++local_g) {

                const int global_g = max_points_per_box * box_idx + local_g;
                double w_g = grid.weights(global_g);
                Point quad_point_g = grid.quad_points(global_g);
                double v_g = 0.0;
                for (int k = 0; k < grid.atom_centers.extent(0); ++k) {
                  double r = dist(grid.atom_centers(k), quad_point_g);
                  v_g -= grid.Z(k) / r;
                }

                double basis_val_i;
                double basis_gx_i;
                double basis_gy_i;
                double basis_gz_i;

                double basis_val_j;
                double basis_gx_j;
                double basis_gy_j;
                double basis_gz_j;

                basis_eval_with_grad(local_basis_j, quad_point_g, basis_val_j,
                                     basis_gx_j, basis_gy_j, basis_gz_j);
                basis_eval_with_grad(local_basis_i, quad_point_g, basis_val_i,
                                     basis_gx_i, basis_gy_i, basis_gz_i);

                const double local_s = w_g * basis_val_i * basis_val_j;
                total_s += local_s;
                total_t += 0.5 * w_g *
                           (basis_gx_i * basis_gx_j + basis_gy_i * basis_gy_j +
                            basis_gz_i * basis_gz_j);
                total_v += v_g * local_s;
              }

              Kokkos::atomic_add(&result.overlap(global_i, global_j), total_s);
              Kokkos::atomic_add(&result.kinetic(global_i, global_j), total_t);
              Kokkos::atomic_add(&result.nuclear(global_i, global_j), total_v);
              if (global_i != global_j) {
                Kokkos::atomic_add(&result.overlap(global_j, global_i),
                                   total_s);
                Kokkos::atomic_add(&result.kinetic(global_j, global_i),
                                   total_t);
                Kokkos::atomic_add(&result.nuclear(global_j, global_i),
                                   total_v);
              }
            });
      });

  ExecSpace().fence();
  Kokkos::parallel_for(
      "Compute Core Hamiltonian Matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        result.hamiltonian(i, j) = result.kinetic(i, j) + result.nuclear(i, j);
      });

  Kokkos::fence();
  return result;
}

template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
CoreHamiltonianResult compute_core_hamiltonian_screened_scratch(
    const Evaluator &basis, const FlatGrid &grid, const NeighborList &nl) {
  int N = basis.nbf();
  auto Z = grid.Z;
  auto atom_centers = grid.atom_centers;

  const int max_points_per_box = nl.max_points_per_box;
  const int total_points = nl.total_points;
  const int num_boxes = nl.offsets.extent(0) - 1;

  CoreHamiltonianResult result;
  result.overlap = DeviceView2DLeft("Overlap matrix", N, N);
  result.kinetic = DeviceView2DLeft("Kinetic matrix", N, N);
  result.nuclear = DeviceView2DLeft("Nuclear potential matrix", N, N);
  result.hamiltonian = DeviceView2DLeft("Core Hamiltonian", N, N);

  // Define helpers for scratch space access
  typedef ExecSpace::scratch_memory_space ScratchSpace;

  typedef Kokkos::View<SingleBfHandle<Evaluator> *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_basis;

  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;

  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  using Bounds = Kokkos::LaunchBounds<256, 4>;
  Kokkos::TeamPolicy<ExecSpace, Bounds> policy(num_boxes, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  int scratch_size = shared_view_double::shmem_size(max_points_per_box) +
                     shared_view_double::shmem_size(max_points_per_box) +
                     shared_view_points::shmem_size(max_points_per_box);

  policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

  Kokkos::parallel_for(
      "Compute Core Hamiltonian Screened", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();

        // Compute number of points per box
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, total_points);
        const int num_points = end_points - start_points;

        // Compute number of neighbors per box
        const int start_neighbors = nl.offsets(box_idx);
        const int end_neighbors = nl.offsets(box_idx + 1);
        const int num_neighbors = end_neighbors - start_neighbors;

        shared_view_double weights_scratch(team_member.team_scratch(0),
                                           num_points);
        shared_view_double v_scratch(team_member.team_scratch(0), num_points);

        shared_view_points points_scratch(team_member.team_scratch(0),
                                          num_points);

        // Fill the scratch
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, num_points),
            [=](const int local_g) {
              const int global_g = start_points + local_g;
              weights_scratch(local_g) = grid.weights(global_g);
              points_scratch(local_g) = grid.quad_points(global_g);
              v_scratch(local_g) = 0.0;
              for (unsigned k = 0; k < atom_centers.extent(0); ++k) {
                double r = dist(points_scratch(local_g), atom_centers(k)) +
                           epsilon_shift;
                v_scratch(local_g) -= double(Z(k)) / r;
              }
            });

        team_member.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                      num_neighbors),
            [=](const int local_i, const int local_j) {
              const int global_i = nl.neighbors(start_neighbors + local_i);
              const int global_j = nl.neighbors(start_neighbors + local_j);

              if (global_i < global_j)
                return;

              const auto local_basis_i = load_shell(basis, global_i);

              const auto local_basis_j = load_shell(basis, global_j);

              double total_s = 0.0;
              double total_t = 0.0;
              double total_v = 0.0;

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, num_points),
                  [=](const int local_g, double &update_s, double &update_t,
                      double &update_v) {
                    double basis_val_i;
                    double basis_gx_i;
                    double basis_gy_i;
                    double basis_gz_i;

                    double basis_val_j;
                    double basis_gx_j;
                    double basis_gy_j;
                    double basis_gz_j;

                    basis_eval_with_grad(local_basis_j, points_scratch(local_g),
                                         basis_val_j, basis_gx_j, basis_gy_j,
                                         basis_gz_j);
                    basis_eval_with_grad(local_basis_i, points_scratch(local_g),
                                         basis_val_i, basis_gx_i, basis_gy_i,
                                         basis_gz_i);

                    const double local_s =
                        weights_scratch(local_g) * basis_val_i * basis_val_j;

                    update_s += local_s;
                    update_t +=
                        0.5 * weights_scratch(local_g) *
                        (basis_gx_i * basis_gx_j + basis_gy_i * basis_gy_j +
                         basis_gz_i * basis_gz_j);
                    update_v += v_scratch(local_g) * local_s;
                  },
                  total_s, total_t, total_v);

              Kokkos::atomic_add(&result.overlap(global_i, global_j), total_s);
              Kokkos::atomic_add(&result.kinetic(global_i, global_j), total_t);
              Kokkos::atomic_add(&result.nuclear(global_i, global_j), total_v);
              if (global_i != global_j) {
                Kokkos::atomic_add(&result.overlap(global_j, global_i),
                                   total_s);
                Kokkos::atomic_add(&result.kinetic(global_j, global_i),
                                   total_t);
                Kokkos::atomic_add(&result.nuclear(global_j, global_i),
                                   total_v);
              }
            });
      });

  ExecSpace().fence();
  Kokkos::parallel_for(
      "Compute Core Hamiltonian Matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        result.hamiltonian(i, j) = result.kinetic(i, j) + result.nuclear(i, j);
      });

  Kokkos::fence();
  return result;
}

template <typename Evaluator, int MAX_NEIGHBORS_TILE = 8,
          int MAX_POINTS_PER_TILE = 8>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
CoreHamiltonianResult compute_core_hamiltonian_screened_tiled(
    const Evaluator &basis, const FlatGrid &grid, const NeighborList &nl) {

  int N = basis.nbf();
  auto Z = grid.Z;
  auto atom_centers = grid.atom_centers;

  const int max_points_per_box = nl.max_points_per_box;
  const int total_points = nl.total_points;
  const int num_boxes = nl.offsets.extent(0) - 1;

  CoreHamiltonianResult result;
  result.overlap = DeviceView2DLeft("Overlap matrix", N, N);
  result.kinetic = DeviceView2DLeft("Kinetic matrix", N, N);
  result.nuclear = DeviceView2DLeft("Nuclear potential matrix", N, N);
  result.hamiltonian = DeviceView2DLeft("Core Hamiltonian", N, N);

  // Define helpers for scratch space access
  typedef ExecSpace::scratch_memory_space ScratchSpace;

  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;

  typedef Kokkos::View<double **, Kokkos::LayoutRight, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view2d_double;

  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  using Bounds = Kokkos::LaunchBounds<128, 2>;
  int fixed_team_size = 1;
  int fixed_vector_length = 1;
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
  fixed_team_size = 8;
  fixed_vector_length = 1;
#endif
  Kokkos::TeamPolicy<ExecSpace, Bounds> policy(num_boxes, fixed_team_size,
                                               fixed_vector_length);
  using member_type = Kokkos::TeamPolicy<ExecSpace, Bounds>::member_type;

  int scratch_size = 2 * shared_view_double::shmem_size(MAX_POINTS_PER_TILE) +
                     shared_view_points::shmem_size(MAX_POINTS_PER_TILE) +
                     9 * shared_view2d_double::shmem_size(MAX_NEIGHBORS_TILE,
                                                          MAX_POINTS_PER_TILE) +
                     3 * shared_view2d_double::shmem_size(MAX_NEIGHBORS_TILE,
                                                          MAX_NEIGHBORS_TILE);
  int scratch_size_one =
      2 * shared_view_double::shmem_size(max_points_per_box) +
      shared_view_points::shmem_size(max_points_per_box);

  policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
  policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_one));

  std::cout << "------------Allocated "
               "Memory-------------"
            << std::endl;
  std::cout << "Available L0 scratch : " << policy.scratch_size_max(0)
            << std::endl;
  std::cout << "Allocated L0 scratch : " << scratch_size << std::endl;
  std::cout << "------------------------"
               "-----------------"
            << std::endl;

  Kokkos::parallel_for(
      "Compute Core Hamiltonian "
      "Screened",
      policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();

        // Compute number of points
        // per box
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, total_points);
        const int num_points = end_points - start_points;

        // Compute number of
        // neighbors per box
        const int start_neighbors = nl.offsets(box_idx);
        const int end_neighbors = nl.offsets(box_idx + 1);
        const int num_neighbors = end_neighbors - start_neighbors;

        // Compute number of tiles
        const int num_tiles =
            (num_neighbors + MAX_NEIGHBORS_TILE - 1) / MAX_NEIGHBORS_TILE;

        shared_view_double weights_scratch(team_member.team_scratch(0),
                                           MAX_POINTS_PER_TILE);

        shared_view_double v_scratch(team_member.team_scratch(0),
                                     MAX_POINTS_PER_TILE);

        shared_view_points points_scratch(team_member.team_scratch(0),
                                          MAX_POINTS_PER_TILE);

        shared_view2d_double tile_val_i(team_member.team_scratch(0),
                                        MAX_NEIGHBORS_TILE,
                                        MAX_POINTS_PER_TILE);
        shared_view2d_double tile_nuc_i(team_member.team_scratch(0),
                                        MAX_NEIGHBORS_TILE,
                                        MAX_POINTS_PER_TILE);
        shared_view2d_double tile_val_j(team_member.team_scratch(0),
                                        MAX_NEIGHBORS_TILE,
                                        MAX_POINTS_PER_TILE);

        shared_view2d_double tile_gx_i(team_member.team_scratch(0),
                                       MAX_NEIGHBORS_TILE, MAX_POINTS_PER_TILE);

        shared_view2d_double tile_gy_i(team_member.team_scratch(0),
                                       MAX_NEIGHBORS_TILE, MAX_POINTS_PER_TILE);

        shared_view2d_double tile_gz_i(team_member.team_scratch(0),
                                       MAX_NEIGHBORS_TILE, MAX_POINTS_PER_TILE);

        shared_view2d_double tile_gx_j(team_member.team_scratch(0),
                                       MAX_NEIGHBORS_TILE, MAX_POINTS_PER_TILE);

        shared_view2d_double tile_gy_j(team_member.team_scratch(0),
                                       MAX_NEIGHBORS_TILE, MAX_POINTS_PER_TILE);

        shared_view2d_double tile_gz_j(team_member.team_scratch(0),
                                       MAX_NEIGHBORS_TILE, MAX_POINTS_PER_TILE);

        shared_view2d_double tile_overlap(team_member.team_scratch(0),
                                          MAX_NEIGHBORS_TILE,
                                          MAX_NEIGHBORS_TILE);
        shared_view2d_double tile_kinetic(team_member.team_scratch(0),
                                          MAX_NEIGHBORS_TILE,
                                          MAX_NEIGHBORS_TILE);
        shared_view2d_double tile_nuclear(team_member.team_scratch(0),
                                          MAX_NEIGHBORS_TILE,
                                          MAX_NEIGHBORS_TILE);

        // Use L1 scratch for pre-computed v across all points
        // Allocate full-box v/weights/points at L1 scratch level
        shared_view_double weights_full(team_member.team_scratch(1),
                                        num_points);
        shared_view_double v_full(team_member.team_scratch(1), num_points);
        shared_view_points points_full(team_member.team_scratch(1), num_points);

        // Fill once — expensive atom loop runs ONCE per box
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, num_points),
            [=](const int local_g) {
              const int global_g = start_points + local_g;
              weights_full(local_g) = grid.weights(global_g);
              points_full(local_g) = grid.quad_points(global_g);
              v_full(local_g) = 0.0;
              for (unsigned k = 0; k < atom_centers.extent(0); ++k) {
                double r =
                    dist(points_full(local_g), atom_centers(k)) + epsilon_shift;
                v_full(local_g) -= double(Z(k)) / r;
              }
            });
        team_member.team_barrier();

        for (int tile_i = 0; tile_i < num_tiles; ++tile_i) {
          int num_neighbors_tile_i = Kokkos::min(
              MAX_NEIGHBORS_TILE, num_neighbors - tile_i * MAX_NEIGHBORS_TILE);

          for (int tile_j = 0; tile_j <= tile_i; ++tile_j) {
            int num_neighbors_tile_j =
                Kokkos::min(MAX_NEIGHBORS_TILE,
                            num_neighbors - tile_j * MAX_NEIGHBORS_TILE);

            // Zero output tiles once before point tiling
            Kokkos::parallel_for(Kokkos::TeamThreadMDRange(team_member,
                                                           MAX_NEIGHBORS_TILE,
                                                           MAX_NEIGHBORS_TILE),
                                 [=](int i, int j) {
                                   tile_overlap(i, j) = 0.0;
                                   tile_kinetic(i, j) = 0.0;
                                   tile_nuclear(i, j) = 0.0;
                                 });
            team_member.team_barrier();

            const int num_pt_tiles =
                (num_points + MAX_POINTS_PER_TILE - 1) / MAX_POINTS_PER_TILE;

            for (int pt_tile = 0; pt_tile < num_pt_tiles; ++pt_tile) {
              const int pt_start = pt_tile * MAX_POINTS_PER_TILE;
              const int pt_end =
                  Kokkos::min(pt_start + MAX_POINTS_PER_TILE, num_points);
              const int num_pts = pt_end - pt_start;

              // Copy slice from full arrays into tile scratch
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team_member, num_pts),
                  [=](const int local_g) {
                    weights_scratch(local_g) = weights_full(pt_start + local_g);
                    points_scratch(local_g) = points_full(pt_start + local_g);
                    v_scratch(local_g) = v_full(pt_start + local_g);
                  });
              team_member.team_barrier();
              // Fill tile_i for this point tile
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team_member, num_neighbors_tile_i),
                  [=](const int local_i) {
                    const int global_i =
                        nl.neighbors(start_neighbors +
                                     tile_i * MAX_NEIGHBORS_TILE + local_i);

                    const auto p = load_shell(basis, global_i);
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorRange(team_member, num_pts),
                        [=](const int local_g) {
                          basis_eval_with_grad(p, points_scratch(local_g),
                                               tile_val_i(local_i, local_g),
                                               tile_gx_i(local_i, local_g),
                                               tile_gy_i(local_i, local_g),
                                               tile_gz_i(local_i, local_g));
                          tile_val_i(local_i, local_g) *=
                              weights_scratch(local_g);
                          tile_nuc_i(local_i, local_g) =
                              tile_val_i(local_i, local_g) * v_scratch(local_g);
                          tile_gx_i(local_i, local_g) *=
                              weights_scratch(local_g);
                          tile_gy_i(local_i, local_g) *=
                              weights_scratch(local_g);
                          tile_gz_i(local_i, local_g) *=
                              weights_scratch(local_g);
                        });
                  });

              // Fill tile_j for this point tile
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team_member, num_neighbors_tile_j),
                  [=](const int local_j) {
                    const int global_j =
                        nl.neighbors(start_neighbors +
                                     tile_j * MAX_NEIGHBORS_TILE + local_j);
                    const auto p = load_shell(basis, global_j);
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorRange(team_member, num_pts),
                        [=](const int local_g) {
                          basis_eval_with_grad(p, points_scratch(local_g),
                                               tile_val_j(local_j, local_g),
                                               tile_gx_j(local_j, local_g),
                                               tile_gy_j(local_j, local_g),
                                               tile_gz_j(local_j, local_g));
                        });
                  });

              team_member.team_barrier();

              auto val_i = Kokkos::subview(tile_val_i, Kokkos::ALL,
                                           Kokkos::make_pair(0, num_pts));
              auto nuc_i = Kokkos::subview(tile_nuc_i, Kokkos::ALL,
                                           Kokkos::make_pair(0, num_pts));
              auto val_j = Kokkos::subview(tile_val_j, Kokkos::ALL,
                                           Kokkos::make_pair(0, num_pts));
              auto gx_i = Kokkos::subview(tile_gx_i, Kokkos::ALL,
                                          Kokkos::make_pair(0, num_pts));
              auto gy_i = Kokkos::subview(tile_gy_i, Kokkos::ALL,
                                          Kokkos::make_pair(0, num_pts));
              auto gz_i = Kokkos::subview(tile_gz_i, Kokkos::ALL,
                                          Kokkos::make_pair(0, num_pts));
              auto gx_j = Kokkos::subview(tile_gx_j, Kokkos::ALL,
                                          Kokkos::make_pair(0, num_pts));
              auto gy_j = Kokkos::subview(tile_gy_j, Kokkos::ALL,
                                          Kokkos::make_pair(0, num_pts));
              auto gz_j = Kokkos::subview(tile_gz_j, Kokkos::ALL,
                                          Kokkos::make_pair(0, num_pts));
              // Accumulate into output tiles across pt_tiles — beta=1.0 always
              // since we zeroed output tiles before the pt_tile loop
              KokkosBatched::TeamGemm<
                  member_type, KokkosBatched::Trans::NoTranspose,
                  KokkosBatched::Trans::Transpose,
                  KokkosBatched::Algo::Gemm::Unblocked>::invoke(team_member,
                                                                1.0, val_i,
                                                                val_j, 1.0,
                                                                tile_overlap);

              KokkosBatched::TeamGemm<
                  member_type, KokkosBatched::Trans::NoTranspose,
                  KokkosBatched::Trans::Transpose,
                  KokkosBatched::Algo::Gemm::Unblocked>::invoke(team_member,
                                                                1.0, nuc_i,
                                                                val_j, 1.0,
                                                                tile_nuclear);

              KokkosBatched::TeamGemm<
                  member_type, KokkosBatched::Trans::NoTranspose,
                  KokkosBatched::Trans::Transpose,
                  KokkosBatched::Algo::Gemm::Unblocked>::invoke(team_member,
                                                                0.5, gx_i, gx_j,
                                                                1.0,
                                                                tile_kinetic);
              KokkosBatched::TeamGemm<
                  member_type, KokkosBatched::Trans::NoTranspose,
                  KokkosBatched::Trans::Transpose,
                  KokkosBatched::Algo::Gemm::Unblocked>::invoke(team_member,
                                                                0.5, gy_i, gy_j,
                                                                1.0,
                                                                tile_kinetic);
              KokkosBatched::TeamGemm<
                  member_type, KokkosBatched::Trans::NoTranspose,
                  KokkosBatched::Trans::Transpose,
                  KokkosBatched::Algo::Gemm::Unblocked>::invoke(team_member,
                                                                0.5, gz_i, gz_j,
                                                                1.0,
                                                                tile_kinetic);

              team_member.team_barrier();
            } // pt_tile loop

            // Single scatter after all point tiles
            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange(team_member, num_neighbors_tile_i,
                                          num_neighbors_tile_j),
                [=](const int local_i, const int local_j) {
                  const int global_i = nl.neighbors(
                      start_neighbors + tile_i * MAX_NEIGHBORS_TILE + local_i);
                  const int global_j = nl.neighbors(
                      start_neighbors + tile_j * MAX_NEIGHBORS_TILE + local_j);
                  Kokkos::atomic_add(&result.overlap(global_i, global_j),
                                     tile_overlap(local_i, local_j));
                  Kokkos::atomic_add(&result.kinetic(global_i, global_j),
                                     tile_kinetic(local_i, local_j));
                  Kokkos::atomic_add(&result.nuclear(global_i, global_j),
                                     tile_nuclear(local_i, local_j));
                  if (tile_i != tile_j) {
                    Kokkos::atomic_add(&result.overlap(global_j, global_i),
                                       tile_overlap(local_i, local_j));
                    Kokkos::atomic_add(&result.kinetic(global_j, global_i),
                                       tile_kinetic(local_i, local_j));
                    Kokkos::atomic_add(&result.nuclear(global_j, global_i),
                                       tile_nuclear(local_i, local_j));
                  }
                });
          } // tile_j
        } // tile_i
      });

  ExecSpace().fence();
  Kokkos::parallel_for(
      "Compute Core Hamiltonian "
      "Matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        result.hamiltonian(i, j) = result.kinetic(i, j) + result.nuclear(i, j);
      });

  Kokkos::fence();
  return result;
}

template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
CoreHamiltonianResult compute_core_hamiltonian_screened_sparse(
    const Evaluator &basis, const FlatGrid &grid, const NeighborList &nl) {
  int N = basis.nbf();
  auto Z = grid.Z;
  auto atom_centers = grid.atom_centers;

  const int max_points_per_box = nl.max_points_per_box;
  const int total_points = nl.total_points;
  const int num_boxes = nl.offsets.extent(0) - 1;

  CoreHamiltonianResult result;
  result.overlap = DeviceView2DLeft("Overlap matrix", N, N);
  result.kinetic = DeviceView2DLeft("Kinetic matrix", N, N);
  result.nuclear = DeviceView2DLeft("Nuclear potential matrix", N, N);
  result.hamiltonian = DeviceView2DLeft("Core Hamiltonian", N, N);

  const int num_neighbors = nl.neighbors.extent(0);

  // We want Right layout on CPU and Left Layout on GPU, Kokkos should do this
  // natively
  Kokkos::View<double **, ExecSpace> sparse_basis_val(
      "Basis Values", num_neighbors, max_points_per_box);

  Kokkos::View<double **, ExecSpace> sparse_basis_gx(
      "Basis  Grad x", num_neighbors, max_points_per_box);
  Kokkos::View<double **, ExecSpace> sparse_basis_gy(
      "Basis  Grad y", num_neighbors, max_points_per_box);
  Kokkos::View<double **, ExecSpace> sparse_basis_gz(
      "Basis  Grad z", num_neighbors, max_points_per_box);

  // Define helpers for scratch space access
  typedef ExecSpace::scratch_memory_space ScratchSpace;

  typedef Kokkos::View<SingleBfHandle<Evaluator> *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_basis;

  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;

  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  Kokkos::TeamPolicy<ExecSpace> policy(num_boxes, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  int scratch_size = shared_view_double::shmem_size(max_points_per_box) +
                     shared_view_double::shmem_size(max_points_per_box) +
                     shared_view_points::shmem_size(max_points_per_box);

  policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

  Kokkos::parallel_for(
      "Compute Core Hamiltonian Screened Sparse", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();

        // Compute number of points per box
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, total_points);
        const int num_points = end_points - start_points;

        // Compute number of neighbors per box
        const int start_neighbors = nl.offsets(box_idx);
        const int end_neighbors = nl.offsets(box_idx + 1);
        const int num_neighbors = end_neighbors - start_neighbors;

        shared_view_double weights_scratch(team_member.team_scratch(0),
                                           num_points);
        shared_view_double v_scratch(team_member.team_scratch(0), num_points);

        shared_view_points points_scratch(team_member.team_scratch(0),
                                          num_points);

        auto subview_sparse_basis_val = Kokkos::subview(
            sparse_basis_val,
            Kokkos::make_pair(start_neighbors, start_neighbors + num_neighbors),
            Kokkos::ALL());

        auto subview_sparse_basis_gx = Kokkos::subview(
            sparse_basis_gx,
            Kokkos::make_pair(start_neighbors, start_neighbors + num_neighbors),
            Kokkos::ALL());
        auto subview_sparse_basis_gy = Kokkos::subview(
            sparse_basis_gy,
            Kokkos::make_pair(start_neighbors, start_neighbors + num_neighbors),
            Kokkos::ALL());
        auto subview_sparse_basis_gz = Kokkos::subview(
            sparse_basis_gz,
            Kokkos::make_pair(start_neighbors, start_neighbors + num_neighbors),
            Kokkos::ALL());

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, num_points),
            [=](const int local_g) {
              const int global_g = start_points + local_g;
              weights_scratch(local_g) = grid.weights(global_g);
              points_scratch(local_g) = grid.quad_points(global_g);
              v_scratch(local_g) = 0.0;
              for (unsigned k = 0; k < atom_centers.extent(0); ++k) {
                double r = dist(points_scratch(local_g), atom_centers(k)) +
                           epsilon_shift;
                v_scratch(local_g) -= double(Z(k)) / r;
              }
            });

        team_member.team_barrier();
        // Compute all basis functions
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, num_neighbors),
            [=](const int local_i) {
              const int global_i = nl.neighbors(start_neighbors + local_i);
              const auto local_basis_i = load_shell(basis, global_i);

              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, num_points),
                  [=](const int local_g) {
                    basis_eval_with_grad(
                        local_basis_i, points_scratch(local_g),
                        subview_sparse_basis_val(local_i, local_g),
                        subview_sparse_basis_gx(local_i, local_g),
                        subview_sparse_basis_gy(local_i, local_g),
                        subview_sparse_basis_gz(local_i, local_g));
                  });
            });

        team_member.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                      num_neighbors),
            [=](const int local_i, const int local_j) {
              const int global_i = nl.neighbors(start_neighbors + local_i);
              const int global_j = nl.neighbors(start_neighbors + local_j);

              if (global_i < global_j)
                return;

              double total_s = 0.0;
              double total_t = 0.0;
              double total_v = 0.0;

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, num_points),
                  [=](const int local_g, double &update_s, double &update_t,
                      double &update_v) {
                    const double local_s =
                        weights_scratch(local_g) *
                        subview_sparse_basis_val(local_i, local_g) *
                        subview_sparse_basis_val(local_j, local_g);

                    update_s += local_s;
                    update_t += 0.5 * weights_scratch(local_g) *
                                (subview_sparse_basis_gx(local_i, local_g) *
                                     subview_sparse_basis_gx(local_j, local_g) +
                                 subview_sparse_basis_gy(local_i, local_g) *
                                     subview_sparse_basis_gy(local_j, local_g) +
                                 subview_sparse_basis_gz(local_i, local_g) *
                                     subview_sparse_basis_gz(local_j, local_g));
                    update_v += v_scratch(local_g) * local_s;
                  },
                  total_s, total_t, total_v);

              Kokkos::atomic_add(&result.overlap(global_i, global_j), total_s);
              Kokkos::atomic_add(&result.kinetic(global_i, global_j), total_t);
              Kokkos::atomic_add(&result.nuclear(global_i, global_j), total_v);
              if (global_i != global_j) {
                Kokkos::atomic_add(&result.overlap(global_j, global_i),
                                   total_s);
                Kokkos::atomic_add(&result.kinetic(global_j, global_i),
                                   total_t);
                Kokkos::atomic_add(&result.nuclear(global_j, global_i),
                                   total_v);
              }
            });
      });

  ExecSpace().fence();
  Kokkos::parallel_for(
      "Compute Core Hamiltonian Matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        result.hamiltonian(i, j) = result.kinetic(i, j) + result.nuclear(i, j);
      });

  Kokkos::fence();
  return result;
}
} // namespace Nukexc
  //
  //
