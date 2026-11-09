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
// WHAT THE SCREENED PATH IS FOR, MEASURED
// ---------------------------------------
// Memory, not time, for the routines in THIS library's own hand: wherever
// the dense path's tables fit it is the faster one, because it evaluates
// the collocation and the auxiliary potential ONCE into (nbf, npts) and
// (naux, npts) arrays and then does BLAS on them, while every routine built
// on this header re-evaluates both on every build.
//
// That reasoning turns on the dense side having BLAS, so it does not carry
// over to a contraction that has none. The kernel-backed XC path is exactly
// that case -- libxckernel's Stage B is its own loop nest -- and there the
// packed store is both smaller AND faster; see xc_collocation_store.hpp for
// the measurement. The question is always which side's flop count the
// library it contracts through can hide, not which layout is denser.
//
// C6H14, nrad 25, lmax 10, LDA, seconds per Fock build, screened against
// dense:
//
//     basis             nbf   naux        J              XC
//     zorabasis/DZ       88    954   2.73 / 0.34    0.99 / 0.55
//     DZ.cholesky        88    288   0.96 / 0.25    0.94 / 0.49
//     zorabasis/TZ2P    310    954   5.33 / 2.16   10.66 / 4.79
//     TZ2P.cholesky     310   1884   7.23 / 1.56    8.83 / 4.47
//
// The XC ratio is about two throughout, and the XC does not touch the
// auxiliary basis at all: that gap is the collocation, re-packed per build
// against a table the dense path keeps. The Coulomb ratio runs from 2.5 to 8
// and tracks naux/nbf, the auxiliary potential being what it re-evaluates
// twice per build.
//
// So the only question is ever whether the tables fit -- (nbf + naux) * npts
// doubles. On C10H22 in TZ2P.cholesky that is 1.1 GB, dense runs and wins by
// a factor of two per build; at a hundred atoms it is tens of gigabytes and
// dense cannot run at all. Caching the tables HERE was measured, costed and
// declined: it would turn this path into the dense one, which already
// exists. Do not re-derive this -- measure it again if the machine or the
// basis changes, but the shape of the answer is not in doubt.
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
  int max_nbf = 0;        ///< largest nbf(b), for team sizing
  int max_npts = 0;       ///< largest npts(b)
  int max_chunk_npts = 0; ///< largest point count in one chunk
  size_t max_chunk_nbf = 0; ///< largest sum of nbf(b) over one chunk

  /// First grid point of box b. The boxes partition the grid into
  /// consecutive runs, so a chunk of boxes is a contiguous point range --
  /// which is what lets a caller pair a chunk with a per-point block of
  /// something else, an auxiliary potential for instance.
  int mppb = 0;
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
 *  @param max_chunk_points  cap on the number of GRID POINTS in a chunk, or
 *                       zero for no cap. A caller that pairs the chunk with
 *                       a dense per-point block of its own -- RI exchange
 *                       pairs it with an (naux, npts) potential -- needs the
 *                       point count bounded as well as the collocation size.
 */
template <typename Evaluator>
  requires BasisEvaluator<Evaluator>
inline PackedBlockLayout
build_packed_block_layout(const Evaluator &basis, const NeighborList &nl,
                          const size_t budget_bytes = (size_t{64} << 20),
                          const int max_chunk_points = 0) {
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
  L.mppb = nl.max_points_per_box;
  L.chunk_begin.push_back(0);
  int b0 = 0;
  int pts_here = 0;
  for (int b = 0; b < num_boxes; ++b) {
    const size_t here = coll_off_h(b + 1) - coll_off_h(b0);
    const bool alone = (b == b0);
    const bool over_coll = here > budget_doubles;
    const bool over_pts = max_chunk_points > 0 &&
                          pts_here + npts_h(b) > max_chunk_points;
    if (!alone && (over_coll || over_pts)) {
      L.chunk_begin.push_back(b);
      b0 = b;
      pts_here = 0;
    }
    pts_here += npts_h(b);
  }
  L.chunk_begin.push_back(num_boxes);

  for (size_t c = 0; c + 1 < L.chunk_begin.size(); ++c) {
    const int lo = L.chunk_begin[c], hi = L.chunk_begin[c + 1];
    L.max_chunk_coll =
        std::max(L.max_chunk_coll, coll_off_h(hi) - coll_off_h(lo));
    L.max_chunk_blk = std::max(L.max_chunk_blk, blk_off_h(hi) - blk_off_h(lo));
    int pts = 0;
    for (int b = lo; b < hi; ++b)
      pts += npts_h(b);
    L.max_chunk_npts = std::max(L.max_chunk_npts, pts);
    L.max_chunk_nbf =
        std::max(L.max_chunk_nbf, bf_off_h(hi) - bf_off_h(lo));
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
 *  @brief Where pack_collocation puts a box's derivative components.
 *
 *  Two consumers want the same numbers in two different orders, and the
 *  shell evaluation is far too expensive to do twice, so the order is a
 *  parameter of the packing rather than a transpose afterwards.
 */
enum class PackedLayout {
  /// chi as (nbf_b, npts_b) and grad as (nbf_b, 3 * npts_b), the Cartesian
  /// components along each row's point axis. The block GEMMs contract over
  /// all three at once, which is why the components sit inside the row.
  Blocks,
  /// One (ncomp, nbf_b, npts_b) tower per box, component-major, which is
  /// what libxckernel indexes as chi + k*nbf*npts + u*npts + g. Everything
  /// goes in the `chi` buffer, which must therefore be ncomp times the size
  /// a Blocks packing needs; `grad` is unused.
  Tower,
};

/**
 *  @brief Pack collocation for one chunk of boxes.
 *
 *  One thread per (box, shell, point): the shell is evaluated once and all
 *  of its functions are written, so the evaluation count is
 *  O(sum_b nshell_b * npts_b) rather than the O(nbf_b^2 * npts_b) a
 *  function-pair kernel pays.
 *
 *  @tparam Deriv  Value packs only chi; Gradient also packs grad.
 *  @tparam Layout which of the two orders above to write.
 *  @param chi   packed (nbf_b, npts_b) blocks, row-major per box -- or the
 *               whole (ncomp, nbf_b, npts_b) tower under PackedLayout::Tower
 *  @param grad  packed (nbf_b, 3 * npts_b) blocks; ignored, and may be a
 *               default-constructed View, when Deriv is Value or the layout
 *               is Tower
 */
template <BasisDeriv Deriv, PackedLayout Layout = PackedLayout::Blocks,
          typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, Deriv>
inline void pack_collocation(const Evaluator &basis, const FlatGrid &grid,
                             const NeighborList &nl,
                             const PackedBlockLayout &L, const int box_begin,
                             const int box_end, Kokkos::View<double *> chi,
                             Kokkos::View<double *> grad =
                                 Kokkos::View<double *>()) {
  static_assert(Deriv == BasisDeriv::Value || Deriv == BasisDeriv::Gradient,
                "pack_collocation packs values or values and gradients");
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

  constexpr int NC = basis_ncomp(Deriv);
  constexpr int MBS = Evaluator::max_block_size;

  Kokkos::parallel_for(
      "pack collocation", policy_type(nbox, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const member_type &team) {
        const int b = box_begin + team.league_rank();
        const int nshell = off(b + 1) - off(b);
        const int np = npts_v(b);
        const int nbf_b = nbf_v(b);
        if (nshell == 0 || np == 0 || nbf_b == 0)
          return;

        const size_t cb = coll_off(b) - coll_off(box_begin);
        constexpr bool tower = (Layout == PackedLayout::Tower);
        // Under Tower every component lives in `chi`, so the box's base
        // offset carries the component factor and the component stride is
        // one whole (nbf_b, npts_b) block.
        double *chi_b = chi.data() + (tower ? NC * cb : cb);
        double *grad_b = (Deriv == BasisDeriv::Gradient && !tower)
                             ? grad.data() + 3 * cb
                             : nullptr;
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
              basis.template eval_block<Deriv>(basis.load_block(blk), p[0],
                                               p[1], p[2], local, MBS);

              for (int i = 0; i < n; ++i) {
                const size_t row = static_cast<size_t>(l0 + i);
                chi_b[row * np + g] = local[BF_VAL * MBS + i];
                if constexpr (Deriv == BasisDeriv::Gradient) {
                  if constexpr (Layout == PackedLayout::Tower) {
                    // One whole (nbf_b, npts_b) block between components.
                    const size_t comp_stride =
                        static_cast<size_t>(nbf_b) * np;
                    double *t = chi_b + row * np + g;
                    t[1 * comp_stride] = local[BF_DX * MBS + i];
                    t[2 * comp_stride] = local[BF_DY * MBS + i];
                    t[3 * comp_stride] = local[BF_DZ * MBS + i];
                  } else {
                    double *gr = grad_b + row * 3 * static_cast<size_t>(np);
                    gr[0 * np + g] = local[BF_DX * MBS + i];
                    gr[1 * np + g] = local[BF_DY * MBS + i];
                    gr[2 * np + g] = local[BF_DZ * MBS + i];
                  }
                }
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

/// Team-vector GEMM of one box's blocks, A B^T.
using PackedGemmNT = KokkosBatched::TeamVectorGemm<
    Kokkos::TeamPolicy<ExecSpace>::member_type,
    KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::Transpose,
    KokkosBatched::Algo::Gemm::Unblocked>;

/// Team-vector GEMM of one box's blocks, A B.
using PackedGemmNN = KokkosBatched::TeamVectorGemm<
    Kokkos::TeamPolicy<ExecSpace>::member_type,
    KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
    KokkosBatched::Algo::Gemm::Unblocked>;

/**
 *  @brief Per box: blocks_b = alpha * A_b B_b^T.
 *
 *  A_b and B_b are packed (nbf_b, ncol_mult * npts_b) collocation-shaped
 *  buffers and blocks_b is the (nbf_b, nbf_b) matrix block. One team per
 *  box, so the whole chunk is one kernel launch.
 *
 *  ncol_mult is 1 for a value block and 3 for a gradient block, where the
 *  three Cartesian components are laid out along the point axis and the
 *  contraction runs over all of them at once.
 */
inline void block_gemm_nt(const PackedBlockLayout &L, const int box_begin,
                          const int box_end, const int ncol_mult,
                          const double alpha, const Kokkos::View<double *> &A,
                          const Kokkos::View<double *> &B,
                          const Kokkos::View<double *> &blocks) {
  using policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = policy_type::member_type;

  const int nbox = box_end - box_begin;
  if (nbox <= 0)
    return;

  auto nbf_v = L.map.nbf;
  auto npts_v = L.npts;
  auto coll_off = L.coll_off;
  auto blk_off = L.blk_off;

  // One THREAD per box, with a register-blocked serial GEMM, rather than one
  // team per box with a team-level one. The boxes are the parallel dimension
  // and there are far more of them than threads, so the team level buys
  // nothing here -- and KokkosBatched implements Algo::Gemm::Blocked only for
  // its serial kernels, which measured 1.3x faster than the team-level
  // unblocked one on the same products.
  Kokkos::parallel_for(
      "block gemm A B^T", Kokkos::RangePolicy<ExecSpace>(box_begin, box_end),
      KOKKOS_LAMBDA(const int b) {
        const int n = nbf_v(b);
        const int np = npts_v(b);
        if (n == 0 || np == 0)
          return;
        const size_t cb = ncol_mult * (coll_off(b) - coll_off(box_begin));
        const size_t bb = blk_off(b) - blk_off(box_begin);
        const int ncol = ncol_mult * np;

        PackedMatrix Am(A.data() + cb, n, ncol);
        PackedMatrix Bm(B.data() + cb, n, ncol);
        PackedMatrix Cm(blocks.data() + bb, n, n);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::Transpose,
            KokkosBatched::Algo::Gemm::Blocked>::invoke(alpha, Am, Bm, 0.0,
                                                        Cm);
      });
}

/**
 *  @brief Per box: out_b = A_b B_b, with A_b an (nbf_b, nbf_b) matrix block.
 *
 *  The shape the density takes: a gathered density-matrix block times the
 *  box's collocation, giving a collocation-shaped result.
 */
inline void block_gemm_matrix_times_collocation(
    const PackedBlockLayout &L, const int box_begin, const int box_end,
    const int ncol_mult, const Kokkos::View<double *> &A,
    const Kokkos::View<double *> &B, const Kokkos::View<double *> &out) {
  using policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = policy_type::member_type;

  const int nbox = box_end - box_begin;
  if (nbox <= 0)
    return;

  auto nbf_v = L.map.nbf;
  auto npts_v = L.npts;
  auto coll_off = L.coll_off;
  auto blk_off = L.blk_off;

  Kokkos::parallel_for(
      "block gemm A B", Kokkos::RangePolicy<ExecSpace>(box_begin, box_end),
      KOKKOS_LAMBDA(const int b) {
        const int n = nbf_v(b);
        const int np = npts_v(b);
        if (n == 0 || np == 0)
          return;
        const size_t cb = ncol_mult * (coll_off(b) - coll_off(box_begin));
        const size_t bb = blk_off(b) - blk_off(box_begin);
        const int ncol = ncol_mult * np;

        PackedMatrix Am(A.data() + bb, n, n);
        PackedMatrix Bm(B.data() + cb, n, ncol);
        PackedMatrix Cm(out.data() + cb, n, ncol);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Blocked>::invoke(1.0, Am, Bm, 0.0, Cm);
      });
}

/**
 *  @brief Gather a global matrix into packed (nbf_b, nbf_b) blocks.
 *
 *  The inverse of scatter_packed_blocks, through the same local-to-global
 *  map: blocks_b(i, j) = global(l2g(i), l2g(j)). A box's shells appear once
 *  in its neighbour list, so no element is read twice into one block.
 */
inline void gather_packed_blocks(const PackedBlockLayout &L,
                                 const int box_begin, const int box_end,
                                 const DeviceView2DLeft &global,
                                 const Kokkos::View<double *> &blocks) {
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
      "gather packed blocks", policy_type(nbox, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const member_type &team) {
        const int b = box_begin + team.league_rank();
        const int n = nbf_v(b);
        if (n == 0)
          return;
        double *blk = blocks.data() + (blk_off(b) - blk_off(box_begin));
        const size_t base = bf_off(b);

        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, member_type>(team, n, n),
            [&](const int i, const int j) {
              blk[static_cast<size_t>(i) * n + j] =
                  global(l2g(base + i), l2g(base + j));
            });
      });
}

/**
 *  @brief out += Z chi^T over the whole grid, with Z built pointwise.
 *
 *  Every screened one-electron quadrature has this shape: a per-point
 *  operand times the collocation on one index, contracted against the bare
 *  collocation on the other. The Coulomb matrix takes Z = w v_J chi, the
 *  LDA potential Z = w v_rho chi, the GGA potential adds the gradient term,
 *  and the overlap is the Z = w chi case.
 *
 *  `zf` is called as zf(g, chi, dx, dy, dz) with g the GLOBAL point index
 *  and the four collocation values of one local function at that point. It
 *  must carry the quadrature weight itself, since only the caller knows
 *  whether the weight belongs on this factor or the other one. With
 *  Deriv == Value the three derivative arguments are zero and must not be
 *  used.
 *
 *  The result is NOT symmetrized: Z chi^T is symmetric only when Z is a
 *  multiple of chi, so the GGA caller has to add the transpose. The caller
 *  also zeroes `out`, because a caller summing several contributions wants
 *  them accumulated.
 */
template <BasisDeriv Deriv, typename Evaluator, typename ZFunctor>
  requires BasisEvaluatorWith<Evaluator, Deriv>
inline void accumulate_collocation_product(const Evaluator &basis,
                                           const FlatGrid &grid,
                                           const NeighborList &nl,
                                           const PackedBlockLayout &L,
                                           const ZFunctor zf,
                                           const DeviceView2DLeft &out) {
  using policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = policy_type::member_type;
  constexpr bool want_grad = (Deriv == BasisDeriv::Gradient);

  // Allocated once at the largest chunk and reused, so the loop over chunks
  // does not allocate.
  Kokkos::View<double *> chi("packed chi", L.max_chunk_coll);
  Kokkos::View<double *> grad("packed grad",
                              want_grad ? 3 * L.max_chunk_coll : 0);
  Kokkos::View<double *> Z("packed Z", L.max_chunk_coll);
  Kokkos::View<double *> blocks("packed blocks", L.max_chunk_blk);

  auto nbf_v = L.map.nbf;
  auto npts_v = L.npts;
  auto coll_off = L.coll_off;
  const int mppb = nl.max_points_per_box;

  for (size_t c = 0; c + 1 < L.chunk_begin.size(); ++c) {
    const int b0 = L.chunk_begin[c], b1 = L.chunk_begin[c + 1];
    const int nbox = b1 - b0;

    pack_collocation<Deriv>(basis, grid, nl, L, b0, b1, chi, grad);

    Kokkos::parallel_for(
        "build Z", policy_type(nbox, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const member_type &team) {
          const int b = b0 + team.league_rank();
          const int n = nbf_v(b);
          const int np = npts_v(b);
          if (n == 0 || np == 0)
            return;
          const size_t cb = coll_off(b) - coll_off(b0);
          const double *chi_b = chi.data() + cb;
          const double *grad_b = want_grad ? grad.data() + 3 * cb : nullptr;
          double *Z_b = Z.data() + cb;
          const int g0 = b * mppb;

          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, member_type>(team, n,
                                                                      np),
              [&](const int i, const int g) {
                const size_t row = static_cast<size_t>(i);
                double dx = 0.0, dy = 0.0, dz = 0.0;
                if constexpr (want_grad) {
                  const double *gr =
                      grad_b + row * 3 * static_cast<size_t>(np);
                  dx = gr[0 * np + g];
                  dy = gr[1 * np + g];
                  dz = gr[2 * np + g];
                }
                Z_b[row * np + g] =
                    zf(g0 + g, chi_b[row * np + g], dx, dy, dz);
              });
        });

    block_gemm_nt(L, b0, b1, 1, 1.0, Z, chi, blocks);
    scatter_packed_blocks(L, b0, b1, blocks, out);
  }
  Kokkos::fence();
}

/**
 *  @brief Z = w(g) v(g) chi: the pointwise weight an LDA potential, a
 *         Coulomb matrix or an overlap takes in
 *         accumulate_collocation_product.
 *
 *  A named functor rather than a lambda because a device lambda defined
 *  inside another lambda is not portable, and because this shape is wanted
 *  by more than one caller.
 */
template <typename WView, typename VView> struct WeightedCollocation {
  WView w;
  VView v;
  KOKKOS_INLINE_FUNCTION double operator()(const int g, const double chi,
                                           const double, const double,
                                           const double) const {
    return w(g) * v(g) * chi;
  }
};

} // namespace Nukexc
