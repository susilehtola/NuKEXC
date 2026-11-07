/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// ============================================================================
//  Per-box packed collocation: pack -> GEMM -> scatter
// ============================================================================
//
// A screened integral over a batch of grid points is a small dense matrix
// product. The batch has nbe non-negligible shells on it, so the collocation
// restricted to the batch is a dense (nbe, npts) block, and
//
//     S_local = (w chi) chi^T,   V_local = (w v chi) chi^T,
//     T_local = 1/2 (w grad chi) (grad chi)^T
//
// are three GEMMs of that block against itself. The result is a dense
// (nbe, nbe) block that scatters into the global matrix through the box's
// local-to-global function map. Blum et al., Comput. Phys. Commun. 180
// (2009) 2175, section 4.1 steps 2-5; the same structure as GauXC's
// AoSScheme1.
//
// WHY A SEPARATE HEADER
// ---------------------
// Every screened path wants exactly this -- the core Hamiltonian, the Coulomb
// matrix, exact exchange and the XC quadrature -- and the project's DRY rule
// says the shared piece lives in a common header from the start rather than
// being written four times with four sets of indexing bugs.
//
// EVERYTHING HERE IS INDEXED BY SHELL
// -----------------------------------
// The neighbour list is over shells, the packing walks shells, and a thread
// owns one (shell, point) pair and writes every function of its shell. That
// is deliberate and it is the whole point: a shell is one radial function
// times its angular components, which is the structure primitive Gaussians,
// contracted Gaussians, Slater functions and numerical atomic orbitals all
// share. A kernel that gives a thread one basis-function PAIR instead --
// which is what compute_core_hamiltonian_screened used to do -- re-evaluates
// a shell once per function pair rather than once per point. For
// single-function shells that costs nothing, which is why such code looks
// correct on any Slater basis and silently does O(max_block_size) redundant
// work on anything else.
//
// MEMORY, AND WHY THE BOXES COME IN CHUNKS
// ----------------------------------------
// The packed collocation for the whole grid is sum_b nbe_b * npts_b doubles,
// times eight for the value block, the three gradient components and the
// weighted copies of both. That is gigabytes on a real molecule, so the
// boxes are processed in chunks sized to a byte budget, with the buffers
// allocated once at the largest chunk and reused: no allocation happens
// inside the loop over chunks.
//
// The offsets are built once for the whole grid; a chunk starting at box b0
// uses coll_off(b) - coll_off(b0), so there is no per-chunk bookkeeping to
// allocate either.

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_TeamVector_Impl.hpp>
#include <Kokkos_Core.hpp>

#include "basis_evaluator.hpp"
#include "nukexc_config.hpp"
#include "octree.hpp"

namespace Nukexc {

/**
 *  @brief Per-box local basis-function numbering for a shell neighbour list.
 *
 *  The neighbour list is over SHELLS, as octree.hpp says, since a shell is
 *  what carries a centre and a screening radius. Anything that walks basis
 *  functions needs each box's shells' local function offsets: shell k of box
 *  b starts at local function bf_offset(offsets(b) + k), and the box has
 *  nbf(b) local functions.
 *
 *  Computed outside the kernels because it is a prefix sum over each box's
 *  shells and does not depend on the grid.
 */
struct LocalBasisMap {
  Kokkos::View<int *> bf_offset; ///< parallel to nl.neighbors
  Kokkos::View<int *> nbf;       ///< per box
};

template <typename Evaluator>
  requires BasisEvaluator<Evaluator>
inline LocalBasisMap build_local_basis_map(const Evaluator &basis,
                                          const NeighborList &nl) {
  const int num_boxes = static_cast<int>(nl.offsets.extent(0)) - 1;
  LocalBasisMap m;
  m.bf_offset = Kokkos::View<int *>("local bf offset", nl.neighbors.extent(0));
  m.nbf = Kokkos::View<int *>("local nbf", num_boxes);

  auto off = nl.offsets;
  auto nb = nl.neighbors;
  auto bf_off = m.bf_offset;
  auto nbf = m.nbf;
  Kokkos::parallel_for(
      "local basis map", Kokkos::RangePolicy<ExecSpace>(0, num_boxes),
      KOKKOS_LAMBDA(const int b) {
        int acc = 0;
        for (int k = off(b); k < off(b + 1); ++k) {
          bf_off(k) = acc;
          acc += basis.block_size(nb(k));
        }
        nbf(b) = acc;
      });
  Kokkos::fence();
  return m;
}

/// Unmanaged row-major view over a slice of a packed buffer.
using PackedMatrix =
    Kokkos::View<double **, Kokkos::LayoutRight, MemSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/**
 *  @brief Packing offsets for the per-box collocation and matrix blocks.
 *
 *  All offsets are over the whole grid; a chunk subtracts its first box's
 *  offset. `chunk_begin` holds the chunk boundaries in box index, with a
 *  trailing entry equal to the box count, so chunk c is
 *  [chunk_begin[c], chunk_begin[c+1]).
 */
struct PackedBlockLayout {
  LocalBasisMap map;
  Kokkos::View<int *> npts;        ///< (nbox) points in each box
  Kokkos::View<size_t *> coll_off; ///< (nbox+1) prefix of nbf(b) * npts(b)
  Kokkos::View<size_t *> blk_off;  ///< (nbox+1) prefix of nbf(b)^2
  Kokkos::View<size_t *> bf_off;   ///< (nbox+1) prefix of nbf(b)
  Kokkos::View<int *> loc2glob;    ///< local function index -> global, packed
  std::vector<int> chunk_begin;    ///< host-side chunk boundaries
  size_t max_chunk_coll = 0;       ///< doubles per unit buffer, largest chunk
  size_t max_chunk_blk = 0;        ///< doubles per block buffer, largest chunk
  int num_boxes = 0;
  int max_nbf = 0;  ///< largest nbf(b), for team sizing
  int max_npts = 0; ///< largest npts(b)
};

/**
 *  @brief Build the packing layout and the chunking for a neighbour list.
 *
 *  @param basis         the evaluator, for shell sizes
 *  @param nl            the shell neighbour list
 *  @param budget_bytes  how much device memory ONE packed unit buffer may
 *                       use in a chunk. Eight such buffers are allocated by
 *                       the caller, so the real footprint is eight times
 *                       this plus the blocks. A chunk always holds at least
 *                       one box, so a single box larger than the budget is
 *                       processed anyway rather than silently skipped.
 */
template <typename Evaluator>
  requires BasisEvaluator<Evaluator>
inline PackedBlockLayout
build_packed_block_layout(const Evaluator &basis, const NeighborList &nl,
                          const size_t budget_bytes = (size_t{64} << 20)) {
  const int num_boxes = static_cast<int>(nl.offsets.extent(0)) - 1;
  if (num_boxes <= 0)
    throw std::invalid_argument(
        "build_packed_block_layout: the neighbour list has no boxes");

  PackedBlockLayout L;
  L.num_boxes = num_boxes;
  L.map = build_local_basis_map(basis, nl);

  // Points per box. The last box is short unless the point count divides
  // evenly, and every consumer addresses box b as
  // [b*mppb, min((b+1)*mppb, total)).
  L.npts = Kokkos::View<int *>("packed npts", num_boxes);
  {
    const int mppb = nl.max_points_per_box;
    const int total = nl.total_points;
    auto npts = L.npts;
    Kokkos::parallel_for(
        "packed npts", Kokkos::RangePolicy<ExecSpace>(0, num_boxes),
        KOKKOS_LAMBDA(const int b) {
          const int first = b * mppb;
          npts(b) = Kokkos::min(mppb, total - first);
        });
    Kokkos::fence();
  }

  // The prefix sums are built on the host: they are O(num_boxes) scalar work
  // and the chunk boundaries have to be decided there anyway.
  auto nbf_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, L.map.nbf);
  auto npts_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, L.npts);

  auto coll_off_h = Kokkos::View<size_t *, HostSpace>("coll_off_h",
                                                      num_boxes + 1);
  auto blk_off_h = Kokkos::View<size_t *, HostSpace>("blk_off_h",
                                                     num_boxes + 1);
  auto bf_off_h = Kokkos::View<size_t *, HostSpace>("bf_off_h", num_boxes + 1);

  coll_off_h(0) = blk_off_h(0) = bf_off_h(0) = 0;
  for (int b = 0; b < num_boxes; ++b) {
    const size_t n = static_cast<size_t>(nbf_h(b));
    const size_t p = static_cast<size_t>(npts_h(b));
    coll_off_h(b + 1) = coll_off_h(b) + n * p;
    blk_off_h(b + 1) = blk_off_h(b) + n * n;
    bf_off_h(b + 1) = bf_off_h(b) + n;
    L.max_nbf = std::max(L.max_nbf, nbf_h(b));
    L.max_npts = std::max(L.max_npts, npts_h(b));
  }

  // Greedy chunking under the budget, at least one box per chunk.
  const size_t budget_doubles = std::max<size_t>(1, budget_bytes / sizeof(double));
  L.chunk_begin.push_back(0);
  int b0 = 0;
  for (int b = 0; b < num_boxes; ++b) {
    const size_t here = coll_off_h(b + 1) - coll_off_h(b0);
    const bool alone = (b == b0);
    if (!alone && here > budget_doubles) {
      L.chunk_begin.push_back(b);
      b0 = b;
    }
  }
  L.chunk_begin.push_back(num_boxes);

  for (size_t c = 0; c + 1 < L.chunk_begin.size(); ++c) {
    const int lo = L.chunk_begin[c], hi = L.chunk_begin[c + 1];
    L.max_chunk_coll =
        std::max(L.max_chunk_coll, coll_off_h(hi) - coll_off_h(lo));
    L.max_chunk_blk = std::max(L.max_chunk_blk, blk_off_h(hi) - blk_off_h(lo));
  }

  L.coll_off = Kokkos::View<size_t *>("coll_off", num_boxes + 1);
  L.blk_off = Kokkos::View<size_t *>("blk_off", num_boxes + 1);
  L.bf_off = Kokkos::View<size_t *>("bf_off", num_boxes + 1);
  Kokkos::deep_copy(L.coll_off, coll_off_h);
  Kokkos::deep_copy(L.blk_off, blk_off_h);
  Kokkos::deep_copy(L.bf_off, bf_off_h);

  // Local-to-global function map, built by walking shells. This replaces the
  // binary search over per-box prefix sums the pair kernel used to do twice
  // per function pair.
  L.loc2glob = Kokkos::View<int *>("loc2glob", bf_off_h(num_boxes));
  {
    auto off = nl.offsets;
    auto nb = nl.neighbors;
    auto bf_offset = L.map.bf_offset;
    auto bf_box = L.bf_off;
    auto l2g = L.loc2glob;
    Kokkos::parallel_for(
        "loc2glob", Kokkos::RangePolicy<ExecSpace>(0, num_boxes),
        KOKKOS_LAMBDA(const int b) {
          const size_t base = bf_box(b);
          for (int k = off(b); k < off(b + 1); ++k) {
            const int blk = nb(k);
            const int g0 = basis.block_first_bf(blk);
            const int n = basis.block_size(blk);
            const int l0 = bf_offset(k);
            for (int i = 0; i < n; ++i)
              l2g(base + static_cast<size_t>(l0 + i)) = g0 + i;
          }
        });
    Kokkos::fence();
  }

  return L;
}

/**
 *  @brief Nuclear attraction -Z/r summed over centres, once per grid point.
 *
 *  Independent of the basis, so it is computed once for the whole grid rather
 *  than inside a per-box kernel. epsilon_shift guards a point landing exactly
 *  on a nucleus; it is not a physical regularization.
 */
inline Kokkos::View<double *> nuclear_potential_on_grid(const FlatGrid &grid) {
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  Kokkos::View<double *> v("vnuc", npts);
  auto pts = grid.quad_points;
  auto Z = grid.Z;
  auto centers = grid.atom_centers;
  const size_t natom = centers.extent(0);
  Kokkos::parallel_for(
      "nuclear potential on grid", Kokkos::RangePolicy<ExecSpace>(0, npts),
      KOKKOS_LAMBDA(const int g) {
        double acc = 0.0;
        const Point p = pts(g);
        for (size_t k = 0; k < natom; ++k)
          acc -= Z(k) / (dist(centers(k), p) + epsilon_shift);
        v(g) = acc;
      });
  Kokkos::fence();
  return v;
}

/**
 *  @brief Pack value and gradient collocation for one chunk of boxes.
 *
 *  One thread per (box, shell, point): the shell is evaluated once and all
 *  of its functions are written, so the evaluation count is
 *  O(sum_b nshell_b * npts_b) rather than the O(nbf_b^2 * npts_b) a
 *  function-pair kernel pays.
 *
 *  @param chi   packed (nbf_b, npts_b) blocks, row-major per box
 *  @param grad  packed (nbf_b, 3 * npts_b) blocks, component-major within
 *               the point axis: component c of point g at column
 *               c * npts_b + g. That layout makes the kinetic energy ONE
 *               GEMM over 3 * npts columns instead of three over npts.
 */
template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
inline void pack_collocation_value_grad(const Evaluator &basis,
                                        const FlatGrid &grid,
                                        const NeighborList &nl,
                                        const PackedBlockLayout &L,
                                        const int box_begin,
                                        const int box_end,
                                        Kokkos::View<double *> chi,
                                        Kokkos::View<double *> grad) {
  using policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = policy_type::member_type;

  const int nbox = box_end - box_begin;
  if (nbox <= 0)
    return;

  auto off = nl.offsets;
  auto nb = nl.neighbors;
  auto bf_offset = L.map.bf_offset;
  auto nbf_v = L.map.nbf;
  auto npts_v = L.npts;
  auto coll_off = L.coll_off;
  auto pts = grid.quad_points;
  const int mppb = nl.max_points_per_box;

  constexpr int NC = basis_ncomp(BasisDeriv::Gradient);
  constexpr int MBS = Evaluator::max_block_size;

  Kokkos::parallel_for(
      "pack collocation value grad", policy_type(nbox, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const member_type &team) {
        const int b = box_begin + team.league_rank();
        const int nshell = off(b + 1) - off(b);
        const int np = npts_v(b);
        const int nbf_b = nbf_v(b);
        if (nshell == 0 || np == 0 || nbf_b == 0)
          return;

        const size_t cb = coll_off(b) - coll_off(box_begin);
        double *chi_b = chi.data() + cb;
        double *grad_b = grad.data() + 3 * cb;
        const int g0 = b * mppb;

        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, member_type>(
                team, nshell, np),
            [&](const int k, const int g) {
              const int blk = nb(off(b) + k);
              const int n = basis.block_size(blk);
              const int l0 = bf_offset(off(b) + k);
              const Point p = pts(g0 + g);

              double local[NC * MBS];
              basis.template eval_block<BasisDeriv::Gradient>(
                  basis.load_block(blk), p[0], p[1], p[2], local, MBS);

              for (int i = 0; i < n; ++i) {
                const size_t row = static_cast<size_t>(l0 + i);
                chi_b[row * np + g] = local[BF_VAL * MBS + i];
                double *gr = grad_b + row * 3 * static_cast<size_t>(np);
                gr[0 * np + g] = local[BF_DX * MBS + i];
                gr[1 * np + g] = local[BF_DY * MBS + i];
                gr[2 * np + g] = local[BF_DZ * MBS + i];
              }
            });
      });
}

/**
 *  @brief Scatter packed (nbf_b, nbf_b) blocks into a global matrix.
 *
 *  Every box contributes the integral over its own points, so the global
 *  element (i, j) is the sum over the boxes holding both shells and no box
 *  double counts. The full square is scattered rather than a triangle: the
 *  blocks are symmetric, but halving the work here would cost a branch per
 *  element and the scatter is not where the time goes.
 */
inline void scatter_packed_blocks(const PackedBlockLayout &L,
                                  const int box_begin, const int box_end,
                                  const Kokkos::View<double *> &blocks,
                                  const DeviceView2DLeft &global) {
  using policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = policy_type::member_type;

  const int nbox = box_end - box_begin;
  if (nbox <= 0)
    return;

  auto nbf_v = L.map.nbf;
  auto blk_off = L.blk_off;
  auto bf_off = L.bf_off;
  auto l2g = L.loc2glob;

  Kokkos::parallel_for(
      "scatter packed blocks", policy_type(nbox, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const member_type &team) {
        const int b = box_begin + team.league_rank();
        const int n = nbf_v(b);
        if (n == 0)
          return;
        const double *blk = blocks.data() + (blk_off(b) - blk_off(box_begin));
        const size_t base = bf_off(b);

        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, member_type>(team, n, n),
            [&](const int i, const int j) {
              const double v = blk[static_cast<size_t>(i) * n + j];
              if (v != 0.0)
                Kokkos::atomic_add(&global(l2g(base + i), l2g(base + j)), v);
            });
      });
}

} // namespace Nukexc
