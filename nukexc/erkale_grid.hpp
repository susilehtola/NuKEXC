/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
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

// ============================================================================
//  Fixed molecular grids in ERKALE's sense
// ============================================================================
//
// A fixed grid is the outer product of a radial rule and an angular one,
// per atom, partitioned. There is no machinery here beyond that -- the
// radial rule is koster_radial.hpp, the angular rule is lobatto_angular.hpp
// or IntegratorXX's Lebedev, the partitioning is partition_schemes.hpp, and
// this just composes them into the FlatGrid the rest of the library
// consumes.
//
// ERKALE spells such a grid "75 -302": 75 radial shells, and a negative
// second field meaning a Lebedev rule of that many points (positive would
// mean a Lobatto order). Nothing about that notation needs reproducing; the
// arguments here are the numbers themselves.
//
// The one thing that is easy to get wrong: the radial weights from
// koster_chebyshev_radial EXCLUDE the r^2 volume element, matching both
// IntegratorXX's convention and ERKALE's radial_chebyshev. The r^2 is
// applied here, once, where the shells are assembled.

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "grid.hpp"
#include "koster_radial.hpp"
#include "lobatto_angular.hpp"
#include "molecule.hpp"
#include "nukexc_config.hpp"
#include "partition_schemes.hpp"

namespace Nukexc {

/// Which angular rule a fixed grid uses.
enum class AngularScheme {
  Lobatto, ///< Gauss-Lobatto x uniform phi, nestable under doubling
  Lebedev  ///< Lebedev-Laikov, fewer points at equal exactness, not nestable
};

/**
 *  @brief Drop points whose weight has fallen below a threshold.
 *
 *  Uses an exclusive prefix sum rather than an atomic counter, so the
 *  surviving points keep their original relative order. make_flat_grid's
 *  compaction uses Kokkos::atomic_fetch_add, which makes the output order
 *  depend on thread scheduling: the grid is still correct, but two runs on
 *  the same input can sum the same numbers in a different order and differ
 *  in the last bits. Worth avoiding in a library whose whole job is
 *  numerical integration.
 */
inline FlatGrid prune_grid(const FlatGrid &in, double threshold) {
  const int n = static_cast<int>(in.quad_points.extent(0));
  if (n == 0)
    return in;

  auto w = in.weights;
  Kokkos::View<int *> keep("keep", n + 1);
  Kokkos::parallel_scan(
      "prune scan", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i, int &acc, const bool final) {
        const bool k = w(i) > threshold;
        if (final)
          keep(i) = acc;
        if (k)
          ++acc;
        if (final && i == n - 1)
          keep(n) = acc;
      });
  Kokkos::fence();

  auto keep_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, keep);
  const int nkeep = keep_h(n);
  if (nkeep == n)
    return in;

  FlatGrid out;
  out.atom_centers = in.atom_centers;
  out.Z = in.Z;
  out.quad_points = Kokkos::View<Point *>("quad points", nkeep);
  out.weights = Kokkos::View<double *>("weights", nkeep);
  out.point_owner = Kokkos::View<int *>("point owner", nkeep);

  auto qp_in = in.quad_points, qp_out = out.quad_points;
  auto w_out = out.weights;
  auto own_in = in.point_owner, own_out = out.point_owner;
  Kokkos::parallel_for(
      "prune compact", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i) {
        if (w(i) > threshold) {
          const int d = keep(i);
          qp_out(d) = qp_in(i);
          w_out(d) = w(i);
          own_out(d) = own_in(i);
        }
      });
  Kokkos::fence();
  return out;
}

/// A fixed grid specification: the two numbers plus the scheme choices.
struct FixedGridSpec {
  int nrad = 75;
  int lmax = 29;
  AngularScheme angular = AngularScheme::Lobatto;
  PartitionScheme partition = PartitionScheme::ERKALE;
  /// Points below this weight are discarded after partitioning.
  double weight_threshold = 1e-15;
};

/**
 *  @brief Build a fixed molecular grid.
 *
 *  Every atom gets the same radial and angular rule, which is what "fixed"
 *  means and what preserves error cancellation between related systems --
 *  the property an adaptive grid puts at risk.
 */
inline FlatGrid make_fixed_grid(const Molecule &mol, const FixedGridSpec &spec) {
  if (spec.nrad < 1)
    throw std::invalid_argument("make_fixed_grid: nrad must be positive");
  if (spec.lmax < 0)
    throw std::invalid_argument("make_fixed_grid: negative lmax");
  if (!(spec.weight_threshold >= 0.0))
    throw std::invalid_argument("make_fixed_grid: bad weight threshold");
  if (spec.angular != AngularScheme::Lobatto)
    throw std::invalid_argument(
        "make_fixed_grid: only the Lobatto angular scheme is wired up here; "
        "use make_flat_grid for the IntegratorXX Lebedev path");

  const auto rad = koster_chebyshev_radial(static_cast<size_t>(spec.nrad));
  const auto ang = lobatto_angular_grid(spec.lmax);

  auto centers_h =
      Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.atom_centers);

  std::vector<Point> pts;
  std::vector<double> wts;
  std::vector<int> owner;
  const size_t per_atom = rad.size() * ang.size();
  pts.reserve(mol.natoms * per_atom);
  wts.reserve(mol.natoms * per_atom);
  owner.reserve(mol.natoms * per_atom);

  for (size_t ia = 0; ia < mol.natoms; ++ia)
    for (size_t ir = 0; ir < rad.size(); ++ir) {
      const double r = rad.r[ir];
      // r^2 belongs here: the radial weights exclude it by convention.
      const double wr = rad.w[ir] * r * r;
      for (size_t ip = 0; ip < ang.size(); ++ip) {
        Point p;
        p[0] = centers_h(ia)[0] + r * ang.dir[ip][0];
        p[1] = centers_h(ia)[1] + r * ang.dir[ip][1];
        p[2] = centers_h(ia)[2] + r * ang.dir[ip][2];
        pts.push_back(p);
        wts.push_back(wr * ang.w[ip]);
        owner.push_back(static_cast<int>(ia));
      }
    }

  const size_t n = pts.size();
  FlatGrid grid;
  grid.quad_points = Kokkos::View<Point *>("quad points", n);
  grid.weights = Kokkos::View<double *>("weights", n);
  grid.point_owner = Kokkos::View<int *>("point owner", n);
  grid.atom_centers = Kokkos::View<Point *>("atom centers", mol.natoms);
  grid.Z = Kokkos::View<unsigned *>("Z", mol.natoms);
  Kokkos::deep_copy(grid.atom_centers, mol.atom_centers);
  Kokkos::deep_copy(grid.Z, mol.Z);

  Kokkos::deep_copy(grid.quad_points,
                    Kokkos::View<Point *, HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        pts.data(), n));
  Kokkos::deep_copy(grid.weights,
                    Kokkos::View<double *, HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        wts.data(), n));
  Kokkos::deep_copy(grid.point_owner,
                    Kokkos::View<int *, HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        owner.data(), n));

  switch (spec.partition) {
  case PartitionScheme::ERKALE:
    partition_erkale(grid.atom_centers, grid.quad_points, grid.point_owner,
                     grid.weights);
    break;
  case PartitionScheme::Becke:
    partition_becke_exact(grid.atom_centers, grid.quad_points,
                          grid.point_owner, grid.weights);
    break;
  case PartitionScheme::LKO:
    partition_becke_team(grid.atom_centers, grid.quad_points,
                         grid.point_owner, grid.weights);
    break;
  }

  if (spec.weight_threshold > 0.0)
    grid = prune_grid(grid, spec.weight_threshold);
  return grid;
}

} // namespace Nukexc
