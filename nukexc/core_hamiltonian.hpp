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
#include "packed_blocks.hpp"
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

/**
 *  @brief Screened core Hamiltonian: pack, GEMM, scatter.
 *
 *  Per box, the shell neighbour list gives the nbf_local functions that are
 *  non-negligible on at least one of its points. The collocation restricted
 *  to the box is then a dense (nbf_local, npts) block, and
 *
 *      S_local = (w chi) chi^T,
 *      V_local = (w v_nuc chi) chi^T,
 *      T_local = 1/2 (w grad chi) (grad chi)^T
 *
 *  are three GEMMs of that block against itself, scattered into the global
 *  matrices through the box's local-to-global function map. Blum et al.,
 *  Comput. Phys. Commun. 180 (2009) 2175, section 4.1 steps 2-5.
 *
 *  The gradient block is packed as (nbf_local, 3 * npts) with the Cartesian
 *  component as the slow index of the point axis, so the kinetic energy is
 *  ONE GEMM over 3 * npts columns rather than three over npts. The identity
 *  is just that summing three outer products over disjoint column ranges is
 *  one outer product over their concatenation.
 *
 *  WHAT THIS REPLACED, AND WHY IT HAD TO
 *  -------------------------------------
 *  The previous version gave each thread one pair of basis FUNCTIONS and
 *  walked the box's points, calling eval_block for both functions' shells at
 *  every point. That is O(nbf_local^2 * npts) shell evaluations where
 *  O(nshell_local * npts) suffices. For single-function shells -- every
 *  Slater basis -- the two counts coincide, which is why the old code looked
 *  correct and fast on the only basis it was ever measured with, and did
 *  O(max_block_size) redundant work on contracted Gaussians. Indexing by
 *  shell instead is what makes one implementation serve primitive Gaussians,
 *  contracted Gaussians, Slater functions and numerical atomic orbitals.
 *
 *  Three further variants of the old kernel -- _scratch, _tiled and _sparse
 *  -- differed only in how they arranged the same function-pair work in team
 *  scratch, so all three inherited the same redundancy. They are gone rather
 *  than ported, per the project's rule that a redundant implementation is
 *  deleted instead of fixed in parallel.
 *
 *  MEASURED
 *  --------
 *  benchmark_coreH --alg=screened, ADF TZP, 50 radial shells, angular order
 *  20, 32 points per box, OpenMP on three threads, H_el only:
 *
 *      water     2.121 s -> 0.186 s    11x
 *      benzene 526.661 s -> 6.304 s    84x
 *
 *  TZP is a Slater basis, so every shell holds one function and none of that
 *  comes from shell blocking: it is all the evaluation count. The old kernel
 *  evaluated two shells per function pair per point, 2 nbf_b^2 npts, where
 *  packing needs nshell_b npts. Contracted Gaussians gain the shell blocking
 *  on top of this.
 *
 *  The benzene/water cost ratio falls from 248 to 34 as well, because the
 *  quadratic factor that was being paid per point is gone.
 *
 *  No SAP: unlike compute_core_hamiltonian this path always uses the bare
 *  nuclear charges, so the returned `hamiltonian` is T + V_n.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
CoreHamiltonianResult compute_core_hamiltonian_screened(
    const Evaluator &basis, const FlatGrid &grid, const NeighborList &nl) {

  const int N = basis.nbf();

  CoreHamiltonianResult result;
  result.overlap = DeviceView2DLeft("Overlap matrix", N, N);
  result.kinetic = DeviceView2DLeft("Kinetic matrix", N, N);
  result.nuclear = DeviceView2DLeft("Nuclear potential matrix", N, N);
  result.hamiltonian = DeviceView2DLeft("Core Hamiltonian", N, N);

  const PackedBlockLayout L = build_packed_block_layout(basis, nl);
  const auto vnuc = nuclear_potential_on_grid(grid);

  // Allocated once at the largest chunk and reused for every chunk, so the
  // loop below allocates nothing.
  Kokkos::View<double *> chi("packed chi", L.max_chunk_coll);
  Kokkos::View<double *> grad("packed grad", 3 * L.max_chunk_coll);
  Kokkos::View<double *> wchi("weighted chi", L.max_chunk_coll);
  Kokkos::View<double *> wgrad("weighted grad", 3 * L.max_chunk_coll);
  Kokkos::View<double *> Sblk("S blocks", L.max_chunk_blk);
  Kokkos::View<double *> Tblk("T blocks", L.max_chunk_blk);
  Kokkos::View<double *> Vblk("V blocks", L.max_chunk_blk);

  using policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = policy_type::member_type;

  // The three products are fused into ONE team kernel rather than going
  // through packed_blocks.hpp's block_gemm_nt, because the weighted copies
  // are built here and the nuclear term reuses the overlap's buffer: the
  // team barriers that makes possible are exactly what a sequence of
  // separate GEMM launches cannot express. The GEMM itself is the shared
  // PackedGemmNT.

  auto nbf_v = L.map.nbf;
  auto npts_v = L.npts;
  auto coll_off = L.coll_off;
  auto blk_off = L.blk_off;
  auto w = grid.weights;
  const int mppb = nl.max_points_per_box;

  for (size_t c = 0; c + 1 < L.chunk_begin.size(); ++c) {
    const int box_begin = L.chunk_begin[c];
    const int box_end = L.chunk_begin[c + 1];
    const int nbox = box_end - box_begin;
    if (nbox <= 0)
      continue;

    pack_collocation<BasisDeriv::Gradient>(basis, grid, nl, L, box_begin,
                                           box_end, chi, grad);
    ExecSpace().fence();

    Kokkos::parallel_for(
        "core Hamiltonian local blocks", policy_type(nbox, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const member_type &team) {
          const int b = box_begin + team.league_rank();
          const int n = nbf_v(b);
          const int np = npts_v(b);
          if (n == 0 || np == 0)
            return;

          const size_t cb = coll_off(b) - coll_off(box_begin);
          const size_t bb = blk_off(b) - blk_off(box_begin);
          const int g0 = b * mppb;

          const PackedMatrix A(chi.data() + cb, n, np);
          const PackedMatrix G(grad.data() + 3 * cb, n, 3 * np);
          const PackedMatrix WA(wchi.data() + cb, n, np);
          const PackedMatrix WG(wgrad.data() + 3 * cb, n, 3 * np);
          const PackedMatrix S(Sblk.data() + bb, n, n);
          const PackedMatrix T(Tblk.data() + bb, n, n);
          const PackedMatrix V(Vblk.data() + bb, n, n);

          // Overlap.
          Kokkos::parallel_for(Kokkos::TeamThreadMDRange(team, n, np),
                               [&](const int i, const int g) {
                                 WA(i, g) = w(g0 + g) * A(i, g);
                               });
          team.team_barrier();
          PackedGemmNT::invoke(team, 1.0, WA, A, 0.0, S);
          team.team_barrier();

          // Nuclear attraction. WA is reused, hence the barrier above: the
          // overlap GEMM must have finished reading it.
          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team, n, np),
              [&](const int i, const int g) {
                WA(i, g) = w(g0 + g) * vnuc(g0 + g) * A(i, g);
              });
          team.team_barrier();
          PackedGemmNT::invoke(team, 1.0, WA, A, 0.0, V);

          // Kinetic energy, all three components in one product.
          Kokkos::parallel_for(Kokkos::TeamThreadMDRange(team, n, 3 * np),
                               [&](const int i, const int cg) {
                                 const int g = cg - (cg / np) * np;
                                 WG(i, cg) = 0.5 * w(g0 + g) * G(i, cg);
                               });
          team.team_barrier();
          PackedGemmNT::invoke(team, 1.0, WG, G, 0.0, T);
        });
    ExecSpace().fence();

    scatter_packed_blocks(L, box_begin, box_end, Sblk, result.overlap);
    scatter_packed_blocks(L, box_begin, box_end, Tblk, result.kinetic);
    scatter_packed_blocks(L, box_begin, box_end, Vblk, result.nuclear);
    ExecSpace().fence();
  }

  auto T = result.kinetic;
  auto V = result.nuclear;
  auto H = result.hamiltonian;
  Kokkos::parallel_for(
      "Compute Core Hamiltonian Matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        H(i, j) = T(i, j) + V(i, j);
      });

  Kokkos::fence();
  return result;
}

} // namespace Nukexc
