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
// a partition of unity, and the cell-function family differs only in one
// function of one variable. The choice matters for parity: two codes using
// "Becke weights" can disagree at the third digit if one clips the argument
// and the other does not.
//
//   Becke        the original, s(mu) = 1/2 (1 - p(p(p(mu)))) with
//                p(x) = 3x/2 - x^3/2. Smooth everywhere, never exactly 0 or
//                1, so every atom contributes at every point.
//                J. Chem. Phys. 88 (1988) 2547.
//
//   ERKALE       s(mu) = 1/2 (1 - p(p(q(mu, a)))), q clipping mu/a to
//                [-1, 1] with a = 0.7. TWO iterations of p rather than
//                three, because q has already done the work of the first.
//                The clip makes the cell function exactly 0 or 1 outside
//                |mu| > a, which is what allows the Stratmann shortcut.
//
//   Stratmann    s(mu) = 1/2 (1 - z(mu/a)) with
//                z(x) = (35x - 35x^3 + 21x^5 - 5x^7)/16 clipped at |x| = 1
//                and a = 0.64. z is the lowest-degree odd polynomial with
//                z(+-1) = +-1 and three vanishing derivatives there, which
//                is where its one virtue over the iterated p lies: it
//                saturates with the same smoothness in a single evaluation.
//                Stratmann, Scuseria and Frisch, Chem. Phys. Lett. 257
//                (1996) 213. NOT the same function as ERKALE's above --
//                both saturate, so both admit the exact prescreen, but they
//                are different polynomials with different a.
//
//   LKO          Becke's cell function with the internuclear distance capped
//                at R_cutoff in the mu denominator (Laqua, Kussmann,
//                Ochsenfeld, J. Chem. Phys. 149 (2018) 204111). The
//                accompanying neighbour truncation is NOT here yet, so this
//                pays the full O(natoms^2) per point while returning LKO's
//                approximate answer rather than Becke's exact one -- see
//                issue #5.
//
//   Hirshfeld    stockholder weights from superposed free-atom densities,
//                g_A = n_A(r).
//
//   Delley       the same with g_A = n_A(r)/r^2, which peaks the weight more
//                sharply at the nucleus. DMol3's partition.
//
// The last two are not cell functions at all: they need the free-atom
// densities, so they live in hirshfeld.hpp and take an argument the others
// do not.
//
// ONE DRIVER, SEVERAL CELL FUNCTIONS
// ----------------------------------
// Everything above the cell function is identical in all of them: the
// distance cache, the product over the other centres, the normalization and
// the in-place multiply. It is written once here, templated on a policy that
// supplies three things -- the cell function, what distance goes in the mu
// denominator, and the radius inside which the owner's weight is exactly
// one. A new member of the family is a dozen lines, which is the point: the
// schemes are meant to be compared, not chosen once.

#include <Kokkos_Core.hpp>

#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"

namespace Nukexc {

/// Which cell function to use.
enum class PartitionScheme {
  Becke,     ///< original Becke, three iterations of p
  ERKALE,    ///< clipped variant with a = 0.7, ERKALE's becke_weights
  Stratmann, ///< Stratmann-Scuseria-Frisch, degree-7 polynomial, a = 0.64
  LKO,       ///< Becke with the internuclear distance capped (see above)
  /// The two density-based schemes. Not cell functions at all: they need the
  /// free-atom densities, so they live in hirshfeld.hpp and take an extra
  /// argument the others do not.
  Hirshfeld, ///< stockholder, g_A = n_A(r)
  Delley     ///< g_A = n_A(r)/r^2, DMol3's partition
};

namespace detail {

/// Becke's iteration polynomial p(x) = 3x/2 - x^3/2.
///
/// x*x*x, not pow(x, 3). The cell function calls this three times per atom
/// pair per grid point, so the libm call the routine this replaced made here
/// was the single most expensive thing in grid construction: putting pow
/// back, measured, costs a factor of twelve on the alkane and ArborX tests
/// (10.0 s -> 121.8 s and 3.1 s -> 40.6 s). Issue #5 lists it as a
/// micro-optimization; it is the largest single win in this file.
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

/// Stratmann-Scuseria-Frisch z(x) = (35x - 35x^3 + 21x^5 - 5x^7)/16,
/// saturating exactly outside |x| > 1.
///
/// Its derivative is z'(x) = 35 (1 - x^2)^3 / 16, so z reaches +-1 with
/// three vanishing derivatives -- the property the iterated p obtains by
/// iterating, and the reason this is the lowest degree that will do.
KOKKOS_INLINE_FUNCTION
double stratmann_z(double x) {
  if (x <= -1.0)
    return -1.0;
  if (x >= 1.0)
    return 1.0;
  const double x2 = x * x;
  return 0.0625 * x * (35.0 + x2 * (-35.0 + x2 * (21.0 - 5.0 * x2)));
}

/// Stratmann's cell function s(mu) = 1/2 (1 - z(mu/a)), a = 0.64 in SSF.
KOKKOS_INLINE_FUNCTION
double stratmann_s(double mu, double a) {
  return 0.5 * (1.0 - stratmann_z(mu / a));
}

// --------------------------------------------------------------------------
//  Cell-function policies
// --------------------------------------------------------------------------
//
// Each supplies s(mu), the distance that goes in the mu denominator, and the
// radius around the owning nucleus inside which its weight is exactly one
// (zero meaning no such radius, so no prescreen).

struct BeckeCell {
  KOKKOS_INLINE_FUNCTION double s(double mu) const { return becke_s(mu); }
  KOKKOS_INLINE_FUNCTION double distance(double R) const { return R; }
  /// Becke's cell function only saturates asymptotically, so there is no
  /// radius at which a shortcut would be exact.
  KOKKOS_INLINE_FUNCTION double screen_radius(double) const { return 0.0; }
};

struct ErkaleCell {
  double a = 0.7;
  KOKKOS_INLINE_FUNCTION double s(double mu) const { return erkale_s(mu, a); }
  KOKKOS_INLINE_FUNCTION double distance(double R) const { return R; }
  KOKKOS_INLINE_FUNCTION double screen_radius(double Rin) const {
    return 0.5 * (1.0 - a) * Rin;
  }
};

struct StratmannCell {
  double a = 0.64;
  KOKKOS_INLINE_FUNCTION double s(double mu) const {
    return stratmann_s(mu, a);
  }
  KOKKOS_INLINE_FUNCTION double distance(double R) const { return R; }
  KOKKOS_INLINE_FUNCTION double screen_radius(double Rin) const {
    return 0.5 * (1.0 - a) * Rin;
  }
};

struct LKOCell {
  double R_cutoff = 5.0;
  KOKKOS_INLINE_FUNCTION double s(double mu) const { return becke_s(mu); }
  KOKKOS_INLINE_FUNCTION double distance(double R) const {
    return Kokkos::min(R, R_cutoff);
  }
  KOKKOS_INLINE_FUNCTION double screen_radius(double) const { return 0.0; }
};

} // namespace detail

/**
 *  @brief Partition a flat grid's weights by a cell function.
 *
 *  One team per quadrature point; the team caches the point's distance to
 *  every centre, forms each centre's cell product over the others, reduces
 *  the normalization and multiplies the owner's share into the weight in
 *  place.
 *
 *  Two screenings, both from issue #5. A point closer to its owner than the
 *  policy's screening radius takes weight exactly 1 and the whole
 *  O(natoms^2) product is skipped; and a cell product that has fallen below
 *  1e-13 stops early, since the remaining factors cannot lift it.
 *
 *  Not done here: the pairwise trick of evaluating s(mu_ij) once and using
 *  s(-mu) = 1 - s(mu) for the transposed pair. It halves the polynomial
 *  evaluations in a serial loop over pairs, but this loop gives one THREAD
 *  per centre i, and the transposed update lands on another thread's
 *  accumulator. The factor two would cost the parallel decomposition, so it
 *  is not free here even though it is free on a CPU written serially. For a
 *  saturating cell function that is not an approximation: inside that radius
 *  every mu_ab with b != a is below -a, so every s is exactly 1 and the
 *  normalized weight is exactly 1. A policy whose cell function does not
 *  saturate returns a zero radius and pays the full product everywhere.
 *
 *  No atomic size adjustment, matching ERKALE.
 */
template <typename Cell>
inline void partition_cell_function(
    const Kokkos::View<Point *> &atom_centers,
    const Kokkos::View<Point *> &quadrature_points,
    const Kokkos::View<int *> &point_owner, Kokkos::View<double *> &weights,
    const Cell cell) {
  const int natoms = static_cast<int>(atom_centers.extent(0));
  const size_t npts = quadrature_points.extent(0);
  if (natoms < 2 || npts == 0)
    return; // a single centre partitions trivially

  // The RECIPROCAL internuclear distance, with the policy's cap already
  // applied, so the inner loop multiplies instead of dividing -- one
  // division per (point, atom pair) is what this kernel spent most of its
  // time on. The diagonal is never read.
  //
  // Nearest-neighbour distances come along for the prescreen radius. Both
  // are O(natoms^2) once, against O(natoms^2) per point below.
  Kokkos::View<double **> invR("invR", natoms, natoms);
  Kokkos::View<double *> Rin("Rin", natoms);
  Kokkos::parallel_for(
      "reciprocal interatomic distances",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                        {natoms, natoms}),
      KOKKOS_LAMBDA(const int i, const int j) {
        invR(i, j) = (i == j) ? 0.0
                              : 1.0 / cell.distance(
                                          dist(atom_centers(i),
                                               atom_centers(j)));
      });
  Kokkos::parallel_for(
      "nearest neighbour", Kokkos::RangePolicy<ExecSpace>(0, natoms),
      KOKKOS_LAMBDA(const int i) {
        double m = Kokkos::Experimental::finite_max_v<double>;
        for (int j = 0; j < natoms; ++j)
          if (j != i)
            m = Kokkos::min(m, dist(atom_centers(i), atom_centers(j)));
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
      "cell function weights", policy,
      KOKKOS_LAMBDA(const member_type &team) {
        const int g = team.league_rank();
        const int owner = point_owner(g);
        const Point pt = quadrature_points(g);

        const double screen = cell.screen_radius(Rin(owner));
        if (screen > 0.0 && dist(pt, atom_centers(owner)) < screen)
          return;

        scratch_view r(team.team_scratch(0), natoms);
        scratch_view cellw(team.team_scratch(0), natoms);

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
                // mu lies in [-1, 1] by the triangle inequality whenever
                // the denominator is the true internuclear distance, so for
                // those schemes this clamp costs a comparison and changes
                // nothing. For a CAPPED denominator it is essential: LKO's
                // min(R, R_cutoff) makes mu as large as the ratio of the
                // true separation to the cap -- six, on a long alkane --
                // and Becke's cell function is an iterated cubic that
                // diverges outside [-1, 1]. Without the clamp the products
                // overflow, every weight comes out non-finite, and the
                // whole grid is then dropped by the weight threshold. The
                // routine this driver replaced clamped inside its
                // compute_mu for precisely this reason.
                double mu = (r(i) - r(j)) * invR(i, j);
                mu = mu > 1.0 ? 1.0 : (mu < -1.0 ? -1.0 : mu);
                w *= cell.s(mu);
                // Only an exact zero ends the product early. Issue #5 also
                // proposes stopping below 1e-13, and that is wrong here: a
                // cell product is NOT normalized, every factor of the
                // nearest atom's is at most 1 and at least 1/2, so the
                // product falls like 2^-natoms and crosses any absolute
                // threshold on a molecule of a few dozen atoms. The ratio
                // it is divided by falls with it, which is why the weights
                // stay O(1) -- but a partial product truncated at a fixed
                // size is not proportional to anything, and an alkane chain
                // then loses its whole grid to the weight threshold. That
                // threshold is only safe once the product runs over a
                // BOUNDED neighbour set, which is the rest of LKO and is
                // not implemented yet.
                //
                // An exact zero needs no such argument, and it is what the
                // clipped cell functions deliver: outside |mu| > a they
                // saturate at zero, so ERKALE and Stratmann get the early
                // exit and Becke, which never reaches zero, pays for all of
                // its factors as it must.
                if (w == 0.0)
                  break;
              }
              cellw(i) = w;
            });
        team.team_barrier();

        double norm = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(team, natoms),
            [=](const int i, double &acc) { acc += cellw(i); }, norm);

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          // The nearest atom has r_i <= r_j for every j, so every mu_ij is
          // non-positive and its cell function is strictly positive: the
          // sum cannot vanish. The guard is for the arithmetic, not the
          // mathematics -- a zero weight is wrong but a NaN poisons every
          // integral downstream.
          weights(g) *= (norm > 0.0) ? cellw(owner) / norm : 0.0;
        });
      });
  Kokkos::fence();
}

/**
 *  @brief ERKALE-compatible Becke partitioning.
 *
 *  Reproduces AngularGrid::becke_weights (src/dftgrid.cpp), including the
 *  clipped cell function with a = 0.7 and two iterations of p, uncapped
 *  internuclear distances in mu -- unlike the LKO variant -- and Stratmann's
 *  prescreen. Weights are multiplied in place.
 */
inline void partition_erkale(const Kokkos::View<Point *> &atom_centers,
                             const Kokkos::View<Point *> &quadrature_points,
                             const Kokkos::View<int *> &point_owner,
                             Kokkos::View<double *> &weights,
                             double a = 0.7) {
  partition_cell_function(atom_centers, quadrature_points, point_owner,
                          weights, detail::ErkaleCell{a});
}

/**
 *  @brief Original Becke partitioning, with no clipping and no distance cap.
 *
 *  The reference the other members of the family are approximations of, and
 *  the only way to say what the LKO cap or a clip actually costs.
 */
inline void partition_becke_exact(
    const Kokkos::View<Point *> &atom_centers,
    const Kokkos::View<Point *> &quadrature_points,
    const Kokkos::View<int *> &point_owner, Kokkos::View<double *> &weights) {
  partition_cell_function(atom_centers, quadrature_points, point_owner,
                          weights, detail::BeckeCell{});
}

/**
 *  @brief Stratmann-Scuseria-Frisch partitioning.
 *
 *  Chem. Phys. Lett. 257 (1996) 213. The cell function saturates outside
 *  |mu| > a, so the prescreen applies exactly as it does to the ERKALE
 *  variant -- with a = 0.64 the screened radius is 0.18 R_in rather than
 *  0.15 R_in, so slightly more points take the shortcut.
 */
inline void partition_stratmann(const Kokkos::View<Point *> &atom_centers,
                                const Kokkos::View<Point *> &quadrature_points,
                                const Kokkos::View<int *> &point_owner,
                                Kokkos::View<double *> &weights,
                                double a = 0.64) {
  partition_cell_function(atom_centers, quadrature_points, point_owner,
                          weights, detail::StratmannCell{a});
}

/**
 *  @brief Becke partitioning with the internuclear distance capped.
 *
 *  Laqua, Kussmann and Ochsenfeld, J. Chem. Phys. 149 (2018) 204111. The
 *  cap alone is what this is: the neighbour truncation that makes it fast is
 *  issue #5, and without it this costs what Becke costs while returning a
 *  different answer. test_partition_schemes.cxx puts a number on the
 *  difference.
 */
inline void partition_lko(const Kokkos::View<Point *> &atom_centers,
                          const Kokkos::View<Point *> &quadrature_points,
                          const Kokkos::View<int *> &point_owner,
                          Kokkos::View<double *> &weights,
                          double R_cutoff = 5.0) {
  partition_cell_function(atom_centers, quadrature_points, point_owner,
                          weights, detail::LKOCell{R_cutoff});
}

} // namespace Nukexc
