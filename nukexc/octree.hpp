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
#include <sstream>
#include <string>
#include <vector>

namespace Nukexc {

using Point = ArborX::Point<3, double>;
using Box = ArborX::Box<3, double>;

// WARNING: this function MUTATES the grid — grid.quad_points and grid.weights
// are permuted in place into Morton (space-filling-curve) order. Any data
// derived from the grid's point ordering (collocation matrices, cached
// densities, ...) computed BEFORE this call becomes invalid afterwards.
// Call this before filling any collocations.
Kokkos::View<Box *, ExecSpace>
create_bounding_boxes(FlatGrid &grid, const int max_points_per_bb) {

  const int num_points = grid.quad_points.extent(0);

  // ── Copy grid coordinates into ArborX points ─────────────────────────
  // ── Wrap each point as a degenerate box for BVH construction ─────────
  Kokkos::View<Box *, ExecSpace> boxes("boxes", num_points);
  Kokkos::parallel_for(
      "copy_points_to_boxes", Kokkos::RangePolicy<ExecSpace>(0, num_points),
      KOKKOS_LAMBDA(int i) {
        boxes(i) = {grid.quad_points(i), grid.quad_points(i)};
      });

  // ── Build BVH over per-point boxes ───────────────────────────────────
  ArborX::BoundingVolumeHierarchy bvh{
      ExecSpace{}, ArborX::Experimental::attach_indices(boxes)};

  // ── Compute Morton-order permutation of the grid points ───────────────
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

  // ── Apply permutation to all grid arrays in lock-step ────────────────
  ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.quad_points);
  ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.weights);
  ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.point_owner);

  // ── Build tiling: group the Morton-sorted points into blocks of N ────
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
template <typename Evaluator>
  requires BasisEvaluator<Evaluator>
void build_neighbor_list(const Evaluator &basis,
                         const Kokkos::View<Box *, ExecSpace> &bounding_boxes,
                         const int max_points_per_box, const int total_points,
                         NeighborList &neighbor_list,
                         const bool print_histogram = false,
                         const std::string &histogram_stem =
                             "neighbor_histogram") {

  auto basis_origins = basis.centers();
  auto cutoff_radii = basis.radii();

  const int N = basis_origins.extent(0);
  const int num_boxes = bounding_boxes.extent(0);

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

  Kokkos::parallel_for(
      "Sort neighborlist", policy(num_boxes, Kokkos::AUTO()),
      KOKKOS_LAMBDA(member_type team_member) {
        const int box_idx = team_member.league_rank();
        const int start = offsets(box_idx);
        const int end = offsets(box_idx + 1);
        auto segment =
            Kokkos::subview(neighbors, Kokkos::make_pair(start, end));
        Kokkos::Experimental::sort_team(team_member, segment);
      });

  neighbor_list.neighbors = neighbors;
  neighbor_list.offsets = offsets;
  neighbor_list.max_points_per_box = max_points_per_box;
  neighbor_list.total_points = total_points;

  if (print_histogram)
    print_neighbor_list_histogram(neighbor_list, N, 50, histogram_stem);
}

} // namespace Nukexc
