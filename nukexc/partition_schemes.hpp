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
//  Atomic partitioning schemes
// ============================================================================
//
// All of these turn a molecular integral into a sum of atom-centred ones by
// a partition of unity, and they differ only in the cell function. The
// choice matters for parity: two codes using "Becke weights" can disagree
// at the third digit if one clips the argument and the other does not.
//
//   Becke        the original, s(mu) = 1/2 (1 - p(p(p(mu)))) with
//                p(x) = 3x/2 - x^3/2. Smooth everywhere, never exactly 0 or
//                1, so every atom contributes at every point.
//
//   ERKALE       s(mu) = 1/2 (1 - p(p(q(mu, a)))), q clipping mu/a to
//                [-1, 1] with a = 0.7. TWO iterations of p rather than
//                three, because q has already done the work of the first.
//                The clip makes the cell function exactly 0 or 1 outside
//                |mu| > a, which is what allows Stratmann's shortcut below.
//
//   LKO          Becke's cell function with the internuclear distance capped
//                at R_cutoff in the mu denominator, plus the accompanying
//                neighbour truncation (Laqua, Kussmann, Ochsenfeld,
//                J. Chem. Phys. 149 (2018) 204111).
//
// A NOTE ON THE EXISTING partition_becke_team
// -------------------------------------------
// That routine caps the internuclear distance at 5 bohr,
// R_ij = min(dist, R_cutoff), inside the mu denominator. That cap is
// genuinely LKO's -- GauXC's LKO path does exactly the same -- but it is
// applied there WITHOUT any of LKO's screening, so it pays the full
// O(N_atoms^2) per point while returning LKO's approximate answer rather
// than Becke's exact one. It is a scheme, not a tuning knob, and calling it
// "Becke" is what makes it a trap. The schemes here are named for what they
// compute.

#include <Kokkos_Core.hpp>

#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"

namespace Nukexc {

/// Which cell function to use.
enum class PartitionScheme {
  Becke,  ///< original Becke, three iterations of p
  ERKALE, ///< clipped variant with a = 0.7, ERKALE's becke_weights
  LKO,    ///< Becke with the internuclear distance capped (see note above)
  /// Stockholder weights from superposed free-atom densities. Not a cell
  /// function at all: it needs the free-atom densities, so it lives in
  /// hirshfeld.hpp and takes an extra argument the others do not.
  Hirshfeld
};

namespace detail {

/// Becke's iteration polynomial p(x) = 3x/2 - x^3/2.
KOKKOS_INLINE_FUNCTION
double becke_p(double x) { return 1.5 * x - 0.5 * x * x * x; }

/// ERKALE's q: clip mu/a to [-1, 1], then one iteration of p.
/// Outside |mu| > a this saturates exactly, which is what makes the
/// Stratmann prescreen exact rather than approximate.
KOKKOS_INLINE_FUNCTION
double erkale_q(double mu, double a) {
  if (mu < -a)
    return -1.0;
  if (mu < a)
    return becke_p(mu / a);
  return 1.0;
}

/// ERKALE's cell function s(mu) = 1/2 (1 - p(p(q(mu, a)))).
KOKKOS_INLINE_FUNCTION
double erkale_s(double mu, double a) {
  return 0.5 * (1.0 - becke_p(becke_p(erkale_q(mu, a))));
}

/// Becke's cell function s(mu) = 1/2 (1 - p(p(p(mu)))).
KOKKOS_INLINE_FUNCTION
double becke_s(double mu) {
  return 0.5 * (1.0 - becke_p(becke_p(becke_p(mu))));
}

} // namespace detail

/**
 *  @brief ERKALE-compatible Becke partitioning.
 *
 *  Reproduces AngularGrid::becke_weights (src/dftgrid.cpp), including:
 *
 *   * the clipped cell function with a = 0.7 and two iterations of p;
 *   * UNCAPPED internuclear distances in mu -- unlike the LKO variant;
 *   * Stratmann's prescreen: a point closer to its own nucleus than
 *     (1-a)R_in/2, where R_in is the distance to the nearest other nucleus,
 *     has weight exactly 1. This is not an approximation given the clipped
 *     cell function: inside that radius every mu_ab with b != a is below
 *     -a, so every s saturates at 1 and the normalized weight is exactly 1.
 *
 *  No atomic size adjustment, matching ERKALE.
 *
 *  Weights are multiplied in place.
 */
inline void partition_erkale(const Kokkos::View<Point *> &atom_centers,
                             const Kokkos::View<Point *> &quadrature_points,
                             const Kokkos::View<int *> &point_owner,
                             Kokkos::View<double *> &weights,
                             double a = 0.7) {
  const int natoms = static_cast<int>(atom_centers.extent(0));
  const size_t npts = quadrature_points.extent(0);
  if (natoms < 2 || npts == 0)
    return; // a single centre partitions trivially

  // Internuclear distances, and the nearest-neighbour distance per atom.
  Kokkos::View<double **> R("R", natoms, natoms);
  Kokkos::View<double *> Rin("Rin", natoms);
  Kokkos::parallel_for(
      "interatomic distances",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                        {natoms, natoms}),
      KOKKOS_LAMBDA(const int i, const int j) {
        R(i, j) = dist(atom_centers(i), atom_centers(j));
      });
  Kokkos::parallel_for(
      "nearest neighbour", Kokkos::RangePolicy<ExecSpace>(0, natoms),
      KOKKOS_LAMBDA(const int i) {
        double m = Kokkos::Experimental::finite_max_v<double>;
        for (int j = 0; j < natoms; ++j)
          if (j != i)
            m = Kokkos::min(m, R(i, j));
        Rin(i) = m;
      });
  Kokkos::fence();

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename TeamPolicy::member_type;
  using scratch_view =
      Kokkos::View<double *, ExecSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const size_t bytes = 2 * scratch_view::shmem_size(natoms);
  auto policy = TeamPolicy(static_cast<int>(npts), Kokkos::AUTO)
                    .set_scratch_size(0, Kokkos::PerTeam(bytes));

  Kokkos::parallel_for(
      "ERKALE Becke weights", policy,
      KOKKOS_LAMBDA(const member_type &team) {
        const int g = team.league_rank();
        const int owner = point_owner(g);
        const Point pt = quadrature_points(g);

        // Stratmann prescreen. Inside this radius every other atom's cell
        // function saturates, so the weight is exactly 1 and the whole
        // O(natoms^2) product can be skipped.
        const double screen = 0.5 * (1.0 - a) * Rin(owner);
        if (dist(pt, atom_centers(owner)) < screen)
          return;

        scratch_view r(team.team_scratch(0), natoms);
        scratch_view cell(team.team_scratch(0), natoms);

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, natoms),
                             [=](const int i) {
                               r(i) = dist(pt, atom_centers(i));
                             });
        team.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, natoms), [=](const int i) {
              double w = 1.0;
              for (int j = 0; j < natoms; ++j) {
                if (i == j)
                  continue;
                // uncapped internuclear distance -- this is the line that
                // differs from the LKO variant
                const double mu = (r(i) - r(j)) / R(i, j);
                w *= detail::erkale_s(mu, a);
              }
              cell(i) = w;
            });
        team.team_barrier();

        double norm = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(team, natoms),
            [=](const int i, double &acc) { acc += cell(i); }, norm);

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          weights(g) *= cell(owner) / norm;
        });
      });
  Kokkos::fence();
}

/**
 *  @brief Original Becke partitioning, with no clipping and no distance cap.
 *
 *  Provided so the LKO cap in partition_becke_team can be compared against
 *  the scheme it approximates, which is the only way to say what that cap
 *  costs.
 */
inline void partition_becke_exact(const Kokkos::View<Point *> &atom_centers,
                                  const Kokkos::View<Point *> &quadrature_points,
                                  const Kokkos::View<int *> &point_owner,
                                  Kokkos::View<double *> &weights) {
  const int natoms = static_cast<int>(atom_centers.extent(0));
  const size_t npts = quadrature_points.extent(0);
  if (natoms < 2 || npts == 0)
    return;

  Kokkos::View<double **> R("R", natoms, natoms);
  Kokkos::parallel_for(
      "interatomic distances",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                        {natoms, natoms}),
      KOKKOS_LAMBDA(const int i, const int j) {
        R(i, j) = dist(atom_centers(i), atom_centers(j));
      });
  Kokkos::fence();

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename TeamPolicy::member_type;
  using scratch_view =
      Kokkos::View<double *, ExecSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const size_t bytes = 2 * scratch_view::shmem_size(natoms);
  auto policy = TeamPolicy(static_cast<int>(npts), Kokkos::AUTO)
                    .set_scratch_size(0, Kokkos::PerTeam(bytes));

  Kokkos::parallel_for(
      "Becke weights (exact)", policy,
      KOKKOS_LAMBDA(const member_type &team) {
        const int g = team.league_rank();
        const int owner = point_owner(g);
        const Point pt = quadrature_points(g);

        scratch_view r(team.team_scratch(0), natoms);
        scratch_view cell(team.team_scratch(0), natoms);

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, natoms),
            [=](const int i) { r(i) = dist(pt, atom_centers(i)); });
        team.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, natoms), [=](const int i) {
              double w = 1.0;
              for (int j = 0; j < natoms; ++j) {
                if (i == j)
                  continue;
                w *= detail::becke_s((r(i) - r(j)) / R(i, j));
              }
              cell(i) = w;
            });
        team.team_barrier();

        double norm = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(team, natoms),
            [=](const int i, double &acc) { acc += cell(i); }, norm);

        Kokkos::single(Kokkos::PerTeam(team),
                       [=]() { weights(g) *= cell(owner) / norm; });
      });
  Kokkos::fence();
}

} // namespace Nukexc
