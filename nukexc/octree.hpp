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

#include <ArborX.hpp>

#include "grid.hpp"
#include "nukexc/basis_evaluator.hpp"
#include "nukexc/stobasis.hpp"
#include "nukexc_config.hpp"

#include <detail/ArborX_Predicates.hpp>
#include <detail/ArborX_SpaceFillingCurves.hpp>
#include <detail/ArborX_TreeVisualization.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>
#include <sorting/Kokkos_SortByKeyPublicAPI.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nukexc/sym3x3.hpp"

namespace Nukexc {

using Point = ArborX::Point<3, double>;
using Box = ArborX::Box<3, double>;

/// How the grid points are grouped into the batches that become boxes.
enum class BatchScheme {
  /// Fixed-size tiles of a Morton (space-filling-curve) ordering.
  ///
  /// Cheap and fully parallel, but ArborX normalizes Morton coordinates to
  /// the grid's bounding box and then subdivides every axis into the same
  /// number of cells regardless of that axis's extent. An anisotropic
  /// molecule therefore gets anisotropic cells -- for a C16 chain along x
  /// the bounding box is 38.1 x 5.6 x 3.4 bohr, so cells are eleven times
  /// longer along the chain than across it, the tiles are elongated, and
  /// their bounding boxes overlap more shells than they need to. Retained
  /// for comparison, and because it is what every published NuKEXC number
  /// so far was measured with.
  MortonTiles,
  /// Recursive cut-plane bisection; the default.
  ///
  /// Blum et al., Comput. Phys. Commun. 180 (2009) 2175, step (1), after
  /// Kahan: split the point set along the plane through its centre of
  /// gravity that maximizes the sum of squared point-plane distances, and
  /// recurse. That plane's normal is the principal axis of the scatter
  /// matrix, so the cut direction is chosen from the data rather than from
  /// the coordinate axes: the boxes rotate with the molecule, and a chain
  /// along (1,1,1) is cut across the chain just as an axis-aligned one is.
  CutPlane
};

namespace detail {

/**
 *  @brief Reorder @p idx so each mppb-sized chunk is a compact batch.
 *
 *  Recursion as in BatchScheme::CutPlane. The cut is placed at a multiple
 *  of @p mppb rather than exactly at the median, which is what lets the
 *  caller keep a fixed-size tiling: every leaf then holds exactly @p mppb
 *  points except the last, so chunk boundaries and leaf boundaries coincide
 *  and every consumer that addresses box b as [b*mppb, (b+1)*mppb) stays
 *  correct. The cut *direction* is still Blum's; only its position along
 *  that direction is snapped, by at most half a box.
 *
 *  @param idx  indices into the point array, reordered in place
 *  @param n    how many
 *  @param xyz  3*num_points coordinates, indexed by the values in @p idx
 *  @param mppb target batch size
 */
inline void cut_plane_order(int *idx, const int n, const double *xyz,
                            const int mppb) {
  if (n <= mppb)
    return;

  // nbox >= 2 here, so 0 < cut < n and both halves shrink: the recursion
  // terminates, and at depth no worse than log2(n/mppb) + 1.
  const int nbox = (n + mppb - 1) / mppb;
  const int cut = (nbox / 2) * mppb;

  double c[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d)
      c[d] += xyz[3 * idx[i] + d];
  for (int d = 0; d < 3; ++d)
    c[d] /= n;

  // Scatter matrix about the centroid. sum_i (nrm . (p_i - c))^2 is
  // nrm^T S nrm, so maximizing it over unit nrm is an eigenproblem.
  double S[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  for (int i = 0; i < n; ++i) {
    const double *q = xyz + 3 * idx[i];
    const double p[3] = {q[0] - c[0], q[1] - c[1], q[2] - c[2]};
    for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b)
        S[a][b] += p[a] * p[b];
  }
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < a; ++b)
      S[a][b] = S[b][a];

  const auto nrm = principal_axis(S);

  // The centroid offset is common to every point, so it cannot change the
  // ordering and is left out of the projection.
  const auto proj = [&](int i) {
    const double *q = xyz + 3 * i;
    return nrm[0] * q[0] + nrm[1] * q[1] + nrm[2] * q[2];
  };
  std::nth_element(idx, idx + cut, idx + n,
                   [&](int a, int b) { return proj(a) < proj(b); });

  cut_plane_order(idx, cut, xyz, mppb);
  cut_plane_order(idx + cut, n - cut, xyz, mppb);
}

} // namespace detail

// WARNING: this function MUTATES the grid — grid.quad_points, grid.weights
// and grid.point_owner are permuted in place into batch order. Any data
// derived from the grid's point ordering (collocation matrices, cached
// densities, ...) computed BEFORE this call becomes invalid afterwards.
// Call this before filling any collocations.
inline Kokkos::View<Box *, ExecSpace>
create_bounding_boxes(FlatGrid &grid, const int max_points_per_bb,
                      const BatchScheme scheme = BatchScheme::CutPlane) {

  if (max_points_per_bb <= 0)
    throw std::invalid_argument(
        "create_bounding_boxes: max_points_per_bb must be positive, got " +
        std::to_string(max_points_per_bb));

  const int num_points = grid.quad_points.extent(0);
  if (num_points == 0)
    throw std::runtime_error(
        "create_bounding_boxes: the grid has no points; nothing downstream "
        "can recover from an empty box list, so this is an error here");
  if (grid.weights.extent(0) != static_cast<size_t>(num_points) ||
      grid.point_owner.extent(0) != static_cast<size_t>(num_points))
    throw std::runtime_error("create_bounding_boxes: grid arrays disagree on "
                             "the number of points");

  if (scheme == BatchScheme::MortonTiles) {
    // ── Copy grid coordinates into ArborX points ───────────────────────
    // ── Wrap each point as a degenerate box for BVH construction ───────
    Kokkos::View<Box *, ExecSpace> boxes("boxes", num_points);
    Kokkos::parallel_for(
        "copy_points_to_boxes", Kokkos::RangePolicy<ExecSpace>(0, num_points),
        KOKKOS_LAMBDA(int i) {
          boxes(i) = {grid.quad_points(i), grid.quad_points(i)};
        });

    // ── Build BVH over per-point boxes ─────────────────────────────────
    ArborX::BoundingVolumeHierarchy bvh{
        ExecSpace{}, ArborX::Experimental::attach_indices(boxes)};

    // ── Compute Morton-order permutation of the grid points ─────────────
    // We construct Nearest queries only to give ArborX a view with the
    // right geometry type; the queries are never actually executed.
    // n_neighbors = 1 is sufficient — only the query geometry is used.
    const int n_queries = bvh.size();
    Kokkos::View<ArborX::Nearest<Point> *, ExecSpace> queries("queries",
                                                              n_queries);
    Kokkos::parallel_for(
        "initialize_queries", Kokkos::RangePolicy(ExecSpace{}, 0, n_queries),
        KOKKOS_LAMBDA(int i) {
          queries(i) = ArborX::nearest(grid.quad_points(i), 1);
        });

    auto permute = ArborX::Details::computeSpaceFillingCurvePermutation(
        ExecSpace{},
        ArborX::Details::PredicateIndexables<decltype(queries)>{queries},
        ArborX::Experimental::Morton64{}, bvh.bounds());

    // ── Apply permutation to all grid arrays in lock-step ──────────────
    ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.quad_points);
    ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.weights);
    ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.point_owner);
  } else {
    // ── Recursive cut-plane permutation ────────────────────────────────
    // Built on the host: the recursion is a tree whose every node needs the
    // scatter matrix of its own point set, so there is no parallelism to
    // exploit at the top and the whole thing costs O(n log(n/mppb)) at grid
    // setup, far below the integration it feeds.
    auto qp =
        Kokkos::create_mirror_view_and_copy(HostSpace{}, grid.quad_points);
    std::vector<double> xyz(3 * static_cast<size_t>(num_points));
    double rmax = 0.0;
    for (int i = 0; i < num_points; ++i)
      for (int d = 0; d < 3; ++d) {
        const double c = qp(i)[d];
        // A non-finite coordinate poisons the centroid of every batch that
        // contains it, and from there the whole recursion. It is also
        // meaningless as a quadrature node. The Morton path tolerates it
        // silently -- producing a box with an infinite corner, which then
        // intersects every shell and screens nothing -- so check here and
        // say which point it was.
        if (!std::isfinite(c))
          throw std::runtime_error(
              "create_bounding_boxes: grid point " + std::to_string(i) +
              " has a non-finite coordinate (" + std::to_string(qp(i)[0]) +
              ", " + std::to_string(qp(i)[1]) + ", " +
              std::to_string(qp(i)[2]) +
              "). The radial quadrature has produced a node at infinity; it "
              "cannot be integrated on and it cannot be screened.");
        xyz[3 * static_cast<size_t>(i) + d] = c;
        rmax = std::max(rmax, std::abs(c));
      }
    if (!(rmax < 1e150))
      throw std::runtime_error(
          "create_bounding_boxes: grid extends to " + std::to_string(rmax) +
          " bohr; squaring that overflows when forming a scatter matrix");

    std::vector<int> idx(static_cast<size_t>(num_points));
    std::iota(idx.begin(), idx.end(), 0);
    detail::cut_plane_order(idx.data(), num_points, xyz.data(),
                            max_points_per_bb);

    Kokkos::View<int *, ExecSpace> permute("cut_plane_permutation",
                                           num_points);
    auto permute_h = Kokkos::create_mirror_view(permute);
    for (int i = 0; i < num_points; ++i)
      permute_h(i) = idx[static_cast<size_t>(i)];
    Kokkos::deep_copy(permute, permute_h);

    // applyPermutation gathers: view(i) = old(permute(i)), which is exactly
    // what cut_plane_order leaves in idx.
    ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.quad_points);
    ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.weights);
    ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.point_owner);
  }

  // ── Build tiling: group the reordered points into blocks of N ────────
  // Both schemes leave the points in an order whose consecutive runs are
  // spatially compact, so the tiling itself is common to them. For
  // CutPlane the chunk boundaries coincide with the recursion's leaves.
  const int n_bounding_boxes =
      (num_points + max_points_per_bb - 1) / max_points_per_bb;

  Kokkos::View<Box *, ExecSpace> bounding_boxes("bounding_boxes",
                                                n_bounding_boxes);

  Kokkos::parallel_for(
      "compute_bounding_boxes",
      Kokkos::RangePolicy<ExecSpace>(0, n_bounding_boxes),
      KOKKOS_LAMBDA(int i) {
        const int start = i * max_points_per_bb;

        const int count = Kokkos::min(max_points_per_bb, num_points - start);

        Box tile_box{grid.quad_points(start),
                     grid.quad_points(start)}; // init: min==max
        for (int k = 1; k < count; ++k) {
          const auto &p = grid.quad_points(start + k);
          for (int d = 0; d < 3; ++d) {
            tile_box.minCorner()[d] =
                Kokkos::min(tile_box.minCorner()[d], p[d]);
            tile_box.maxCorner()[d] =
                Kokkos::max(tile_box.maxCorner()[d], p[d]);
          }
        }
        bounding_boxes(i) = tile_box;
      });
  ExecSpace{}.fence();
  return bounding_boxes;
}

// Prints an ASCII histogram of the given values and writes pgfplots output
// (<output_stem>.dat and a self-contained <output_stem>.tex snippet).
inline void create_histogram(std::vector<double> percentages, int num_bins,
                             const std::string &output_stem = "histogram") {
  if (percentages.empty() || num_bins <= 0)
    return;

  // 1. Find data boundaries
  auto [min_it, max_it] =
      std::minmax_element(percentages.begin(), percentages.end());
  double min_val = *min_it;
  double max_val = *max_it;
  if (std::abs(max_val - min_val) < 1e-9)
    max_val += 1.0;

  // 2. Count frequencies into bins
  double bin_width = (max_val - min_val) / num_bins;
  std::vector<int> bins(num_bins, 0);
  for (double val : percentages) {
    int bin_idx = static_cast<int>((val - min_val) / bin_width);
    if (bin_idx >= num_bins)
      bin_idx = num_bins - 1;
    if (bin_idx < 0)
      bin_idx = 0;
    bins[bin_idx]++;
  }

  // 3. ASCII output (unchanged)
  int max_freq = *std::max_element(bins.begin(), bins.end());
  const int max_bar_width = 40;
  std::cout << "\n=== Histogram ===\n";
  for (int i = 0; i < num_bins; ++i) {
    double lo = min_val + i * bin_width;
    double hi = lo + bin_width;
    std::printf("[%6.2f%% - %6.2f%%]: ", lo, hi);
    int bar_length = (max_freq > max_bar_width)
                         ? (bins[i] * max_bar_width) / max_freq
                         : bins[i];
    std::cout << std::string(bar_length, '#') << " (" << bins[i] << ")\n";
  }
  std::cout << "=================\n";

  // 4. Write .dat file  (bin_center  count)
  //    pgfplots reads this with \addplot table
  const std::string dat_path = output_stem + ".dat";
  std::ofstream dat(dat_path);
  if (!dat) {
    std::cerr << "Could not open " << dat_path << "\n";
    return;
  }

  dat << "% bin_start bin_end bin_center count\n";
  for (int i = 0; i < num_bins; ++i) {
    double lo = min_val + i * bin_width;
    double hi = lo + bin_width;
    double mid = (lo + hi) / 2.0;
    dat << std::fixed << std::setprecision(6) << lo << " " << hi << " " << mid
        << " " << bins[i] << "\n";
  }
  dat.close();

  // 5. Write self-contained .tex snippet (no external .dat needed)
  const std::string tex_path = output_stem + ".tex";
  std::ofstream tex(tex_path);
  if (!tex) {
    std::cerr << "Could not open " << tex_path << "\n";
    return;
  }

  // Compute a clean tick spacing: aim for ~8 ticks across the range
  // Round up to a "nice" number (1, 2, 2.5, 5, 10, 20, ...)
  auto nice_step = [](double raw) -> double {
    double exp = std::pow(10.0, std::floor(std::log10(raw)));
    double f = raw / exp;
    if (f <= 1.0)
      return 1.0 * exp;
    else if (f <= 2.0)
      return 2.0 * exp;
    else if (f <= 2.5)
      return 2.5 * exp;
    else if (f <= 5.0)
      return 5.0 * exp;
    else
      return 10.0 * exp;
  };

  double tick_step = nice_step((max_val - min_val) / 8.0);
  // First tick: round min_val up to nearest multiple of tick_step
  double first_tick = std::ceil(min_val / tick_step) * tick_step;

  // Build the explicit tick list
  std::ostringstream tick_list;
  for (double t = first_tick; t <= max_val + 1e-9; t += tick_step)
    tick_list << std::fixed << std::setprecision(1) << t << ",";
  std::string ticks = tick_list.str();
  if (!ticks.empty())
    ticks.pop_back(); // remove trailing comma

  const double bar_width_frac = 0.92;
  tex << std::fixed << std::setprecision(4);
  tex << "% Auto-generated by create_histogram() -- do not edit by hand\n"
      << "\\begin{tikzpicture}\n"
      << "\\begin{axis}[\n"
      << "  ybar,\n"
      << "  bar width=" << bin_width * bar_width_frac << "pt,\n"
      << "  xlabel={Value (\\%)},\n"
      << "  ylabel={Count},\n"
      << "  xtick={" << ticks << "},\n" // <-- sparse, clean ticks
      << "  xticklabel={$\\pgfmathprintnumber"
         "[fixed,precision=1]{\\tick}\\%$},\n" // <-- math mode, no rotation
      << "  ymin=0,\n"
      << "  enlarge x limits=0.05,\n"
      << "  grid=major,\n"
      << "  grid style={dashed, gray!40},\n"
      << "  width=0.85\\linewidth,\n"
      << "  height=6cm,\n"
      << "]\n"
      << "\\addplot[fill=blue!40, draw=blue!70] coordinates {\n";

  for (int i = 0; i < num_bins; ++i) {
    double lo = min_val + i * bin_width;
    double mid = lo + bin_width / 2.0;
    tex << "  (" << mid << ", " << bins[i] << ")\n";
  }

  tex << "};\n"
      << "\\end{axis}\n"
      << "\\end{tikzpicture}\n";
  tex.close();

  std::cout << "LaTeX snippet written: " << tex_path << "\n";
  std::cout << "LaTeX files written: " << dat_path << " and " << tex_path
            << "\n";
}

// Data structure that will later be used to compute batched Dgemm and scalings
struct NeighborList {
  Kokkos::View<int *> neighbors; // flat list of basis function indices
  Kokkos::View<int *>
      offsets; // offsets(i)..offsets(i+1) gives neighbors of box i
  int max_points_per_box;
  int total_points;
};

// Histogram of the per-box neighbor counts, expressed as a percentage of the
// basis size. Prints to stdout and writes pgfplots files (see
// create_histogram).
inline void print_neighbor_list_histogram(
    const NeighborList &neighbor_list, const int num_basis_functions,
    const int num_bins = 50,
    const std::string &output_stem = "neighbor_histogram") {
  auto offsets_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       neighbor_list.offsets);
  const int num_boxes = static_cast<int>(offsets_h.extent(0)) - 1;
  if (num_boxes <= 0 || num_basis_functions <= 0)
    return;

  std::vector<double> percentages;
  percentages.reserve(num_boxes);
  for (int b = 0; b < num_boxes; ++b)
    percentages.push_back((offsets_h(b + 1) - offsets_h(b)) /
                          static_cast<double>(num_basis_functions) * 100.0);

  create_histogram(percentages, num_bins, output_stem);
}

struct NeighborInsertCallback {
  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &, Value const &value,
                                  Output const &out) const {
    out(value.index);
  }
};

// The neighbour list is over BLOCKS, not individual basis functions: a block
// is the evaluator's unit of work (one function for an STO, one contraction of
// one shell for a GTO) and carries a single center and cutoff radius. Screening
// per block rather than per function is both cheaper and the right granularity,
// since every function of a GTO shell shares the same support.
//
// Two stages. The bounding-volume query finds candidates; an exact test then
// discards the ones that only overlap the box and not the points in it. The
// second stage is not an optimization of the first, it is the criterion: a
// block belongs in box b's list iff at least one point of box b lies within
// that block's cutoff -- which is step (2) of Blum et al., Comput. Phys.
// Commun. 180 (2009) 2175, "the N_nz basis functions [that] exceed a given
// threshold on at least one point in the batch".
//
// Doing it exactly matters for more than a few percent of work, because a
// bounding box is not rotation invariant and the points in it are. A batch of
// grid points through a molecule lying along (1,1,1) has the same shape as one
// through a molecule lying along x, but its axis-aligned bounding box is
// larger -- measured on a C24 chain, 1.8 times larger in volume -- so a
// box-only screening makes the diagonal orientation cost more than the aligned
// one for no physical reason. Testing the points removes that dependence
// instead of trying to orient the boxes to reduce it.
template <typename Evaluator>
  requires BasisEvaluator<Evaluator>
void build_neighbor_list(const Evaluator &basis,
                         const Kokkos::View<Box *, ExecSpace> &bounding_boxes,
                         const int max_points_per_box,
                         const Kokkos::View<Point *> &quad_points,
                         NeighborList &neighbor_list,
                         const bool print_histogram = false,
                         const std::string &histogram_stem =
                             "neighbor_histogram") {

  auto basis_origins = basis.centers();
  auto cutoff_radii = basis.radii();

  const int N = basis_origins.extent(0);
  const int num_boxes = bounding_boxes.extent(0);
  const int total_points = static_cast<int>(quad_points.extent(0));

  // Every consumer addresses box b as points [b*mppb, (b+1)*mppb), and the
  // exact test below relies on the same mapping. If the box list did not come
  // from create_bounding_boxes with this mppb, that mapping is wrong and
  // every screened kernel would be reading the wrong points.
  if (max_points_per_box <= 0)
    throw std::invalid_argument("build_neighbor_list: max_points_per_box must "
                                "be positive");
  {
    const int expected =
        (total_points + max_points_per_box - 1) / max_points_per_box;
    if (num_boxes != expected)
      throw std::runtime_error(
          "build_neighbor_list: " + std::to_string(num_boxes) + " boxes for " +
          std::to_string(total_points) + " points at " +
          std::to_string(max_points_per_box) +
          " points per box; expected " + std::to_string(expected) +
          ". The box list and the grid do not belong to each other.");
  }

  // ── Build a sphere for each basis function ────────────────────────────
  // The sphere has the center at the basis origin and radius = cutoff radius.
  // A grid tile needs this basis function iff its bounding box intersects
  // the sphere.
  using Sphere = ArborX::Sphere<3, double>;
  Kokkos::View<Sphere *, ExecSpace> spheres("spheres", N);
  Kokkos::parallel_for(
      "build_basis_spheres", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(int i) {
        spheres(i) = Sphere{basis_origins(i), cutoff_radii(i)};
      });

  // ── BVH over bounding boxes ──────────────────────────────────
  // attach_indices so the query results carry the original bounding_box index.
  ArborX::BoundingVolumeHierarchy bvh{
      ExecSpace{}, ArborX::Experimental::attach_indices(spheres)};

  // ── One intersects(sphere) query per basis function ───────────────────
  // ArborX will return all spheres whose bounding volume overlaps the box.
  Kokkos::View<ArborX::Intersects<Box> *, ExecSpace> queries("queries",
                                                             num_boxes);
  Kokkos::parallel_for(
      "build_box_queries", Kokkos::RangePolicy<ExecSpace>(0, num_boxes),
      KOKKOS_LAMBDA(int i) {
        queries(i) = ArborX::intersects(bounding_boxes(i));
      });

  // ── Execute queries ───────────────────────────────────────────────────
  // offsets is length num_boxes+1 (CSR row pointers).
  // values contains PairValueIndex{sphere_value, basis_index}.

  Kokkos::View<int *, ExecSpace> offsets("offsets", 0);
  Kokkos::View<int *, ExecSpace> neighbors("neighbors", 0);

  // Use a custom callback to extract the index from PairValueIndex
  bvh.query(ExecSpace{}, queries, NeighborInsertCallback{}, neighbors, offsets);

  using policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // ── Exact stage: keep only blocks that reach a point of the box ───────
  // One team per box, one thread per candidate, scanning the box's own
  // points and stopping at the first hit. A block that really does overlap
  // -- the common case -- exits almost immediately, so the cost is carried
  // by the blocks being discarded, which is the work this buys back many
  // times over in every screened kernel downstream.
  const int ncand = static_cast<int>(neighbors.extent(0));
  Kokkos::View<int *, ExecSpace> keep("neighbor_keep", ncand);
  {
    auto pts = quad_points;
    const int mppb = max_points_per_box;
    const int npts = total_points;
    Kokkos::parallel_for(
        "Screen neighborlist exactly", policy(num_boxes, Kokkos::AUTO()),
        KOKKOS_LAMBDA(member_type team) {
          const int b = team.league_rank();
          const int first = offsets(b);
          const int last = offsets(b + 1);
          const int p0 = b * mppb;
          const int p1 = Kokkos::min(p0 + mppb, npts);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, first, last), [&](const int e) {
                const int j = neighbors(e);
                const auto o = basis_origins(j);
                const double r = cutoff_radii(j);
                const double r2 = r * r;
                int hit = 0;
                for (int p = p0; p < p1; ++p) {
                  const auto q = pts(p);
                  const double dx = q[0] - o[0];
                  const double dy = q[1] - o[1];
                  const double dz = q[2] - o[2];
                  if (dx * dx + dy * dy + dz * dz <= r2) {
                    hit = 1;
                    break;
                  }
                }
                keep(e) = hit;
              });
        });
  }

  // ── Compact the CSR ──────────────────────────────────────────────────
  Kokkos::View<int *, ExecSpace> kept_offsets("offsets", num_boxes + 1);
  Kokkos::View<int *, ExecSpace> counts("neighbor_counts", num_boxes);
  Kokkos::parallel_for(
      "Count kept neighbors", Kokkos::RangePolicy<ExecSpace>(0, num_boxes),
      KOKKOS_LAMBDA(const int b) {
        int n = 0;
        for (int e = offsets(b); e < offsets(b + 1); ++e)
          n += keep(e);
        counts(b) = n;
      });
  Kokkos::parallel_scan(
      "Scan neighbor counts", Kokkos::RangePolicy<ExecSpace>(0, num_boxes),
      KOKKOS_LAMBDA(const int b, int &acc, const bool final) {
        if (final)
          kept_offsets(b) = acc;
        acc += counts(b);
        if (final && b + 1 == num_boxes)
          kept_offsets(num_boxes) = acc;
      });
  ExecSpace{}.fence();

  int total_kept = 0;
  Kokkos::deep_copy(total_kept, Kokkos::subview(kept_offsets, num_boxes));

  Kokkos::View<int *, ExecSpace> kept("neighbors", total_kept);
  Kokkos::parallel_for(
      "Compact neighborlist", Kokkos::RangePolicy<ExecSpace>(0, num_boxes),
      KOKKOS_LAMBDA(const int b) {
        int w = kept_offsets(b);
        for (int e = offsets(b); e < offsets(b + 1); ++e)
          if (keep(e))
            kept(w++) = neighbors(e);
      });

  Kokkos::parallel_for(
      "Sort neighborlist", policy(num_boxes, Kokkos::AUTO()),
      KOKKOS_LAMBDA(member_type team_member) {
        const int box_idx = team_member.league_rank();
        const int start = kept_offsets(box_idx);
        const int end = kept_offsets(box_idx + 1);
        auto segment = Kokkos::subview(kept, Kokkos::make_pair(start, end));
        Kokkos::Experimental::sort_team(team_member, segment);
      });

  neighbor_list.neighbors = kept;
  neighbor_list.offsets = kept_offsets;
  neighbor_list.max_points_per_box = max_points_per_box;
  neighbor_list.total_points = total_points;

  if (print_histogram)
    print_neighbor_list_histogram(neighbor_list, N, 50, histogram_stem);
}

} // namespace Nukexc
