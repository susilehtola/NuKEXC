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
//  Density-based partitioning: Hirshfeld and Delley
// ============================================================================
//
// The stockholder partition (Hirshfeld, Theor. Chim. Acta 44 (1977) 129):
//
//     w_A(r) = n_A(|r - R_A|) / sum_B n_B(|r - R_B|)
//
// with n_A the spherical free-atom density. Unlike Becke's, this cell
// function is not a geometric construction with a tuned parameter -- it is
// fixed entirely by the free atoms, so there is no analogue of Becke's
// atomic size adjustment or of the ERKALE "a" to agree on between codes.
// What has to be agreed on instead is the free-atom densities themselves,
// which is why they are compiled in rather than left to the caller.
//
// As a quadrature partition this is a mixed bargain, and worth being clear
// about. In its favour: it follows the actual density rather than a
// nuclear-position heuristic, so it handles very unequal atoms better than
// Becke without any per-element tuning. Against it: n_A decays
// exponentially, so far from the molecule the quotient is a ratio of two
// underflowing numbers and the weights become noise, where Becke's rational
// cell function stays well conditioned everywhere. ERKALE takes the same
// way out as everyone else -- a zero denominator yields zero weight -- and
// so does this. Points beyond every atom's density carry no weight, which
// is correct for anything one would integrate but does mean the partition
// sums to zero rather than one out there.
//
// The main use is not actually quadrature but population analysis and
// density decomposition, where the stockholder definition is the point.
//
// DELLEY
// ------
// Delley's partition (used by DMol3; An all-electron numerical method for
// solving the local density functional for polyatomic molecules,
// J. Chem. Phys. 92 (1990) 508) is the same construction with
//
//     g_A(r) = n_A(|r - R_A|) / |r - R_A|^2
//
// in place of n_A alone. The extra 1/r^2 makes the function more sharply
// peaked at the nucleus, so a point between two atoms gives a larger share
// to the nearer one than Hirshfeld does -- by the factor (r_far/r_near)^2 in
// the ratio -- while leaving an equidistant point split exactly as Hirshfeld
// splits it. It is therefore a third way of building the same kind of
// partition of unity, alongside the geometric cell functions in
// partition_schemes.hpp and the stockholder weights above, which is why it
// belongs beside them rather than replacing either.
//
// The two share everything but that one factor, so there is one driver here
// and two weight policies.

#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "atomic_density.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"

namespace Nukexc {

namespace detail {

/// g_A = n_A(r): the stockholder weight.
struct HirshfeldWeight {
  KOKKOS_INLINE_FUNCTION double operator()(double n, double) const {
    return n;
  }
};

/// g_A = n_A(r)/r^2: Delley's weight.
struct DelleyWeight {
  /// A point sitting exactly on a nucleus would divide by zero. The limit
  /// there is well defined -- the owning atom takes everything -- so the
  /// denominator is floored rather than special-cased, which keeps that
  /// limit and keeps the kernel branchless. The floor is far below any
  /// radius a quadrature places a point at.
  static constexpr double r2_floor = 1e-24;
  KOKKOS_INLINE_FUNCTION double operator()(double n, double r) const {
    return n / Kokkos::max(r * r, r2_floor);
  }
};

} // namespace detail

/**
 *  @brief Multiply quadrature weights by a free-atom-density cell function.
 *
 *  @param atom_centers      nuclear positions
 *  @param Z                 atomic number per nucleus
 *  @param quadrature_points the points
 *  @param point_owner       which atom each point belongs to
 *  @param weights           modified in place
 *  @param table             free-atom densities, e.g. from
 *                           make_thakkar_density_table()
 *
 *  Throws if any atom is missing from the table. A missing element would
 *  otherwise contribute zero density and quietly hand its share of every
 *  point to its neighbours -- a wrong answer with no symptom, which is the
 *  one failure mode worth paying a host-side check to avoid.
 */
template <typename Weight>
inline void partition_density_based(
    const Kokkos::View<Point *> &atom_centers,
    const Kokkos::View<unsigned *> &Z,
    const Kokkos::View<Point *> &quadrature_points,
    const Kokkos::View<int *> &point_owner, Kokkos::View<double *> &weights,
    const AtomicDensityTable &table, const Weight gw) {
  const int natoms = static_cast<int>(atom_centers.extent(0));
  const size_t npts = quadrature_points.extent(0);
  if (static_cast<int>(Z.extent(0)) != natoms)
    throw std::invalid_argument(
        "partition_density_based: atom_centers and Z have different lengths");
  if (natoms == 0 || npts == 0)
    return;

  // Checked on the host, once, before anything is launched: inside the
  // kernel there is nowhere to report it to.
  if (table.max_Z <= 0)
    throw std::invalid_argument(
        "partition_density_based: empty density table; build one with "
        "make_thakkar_density_table()");
  {
    auto Z_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, Z);
    auto off_h =
        Kokkos::create_mirror_view_and_copy(HostSpace{}, table.orb_offset);
    for (int i = 0; i < natoms; ++i) {
      const int z = static_cast<int>(Z_h(i));
      const bool ok =
          z >= 1 && z <= table.max_Z && off_h(z + 1) > off_h(z);
      if (!ok)
        throw std::runtime_error(
            "partition_density_based: no free-atom density for Z = " +
            std::to_string(z) + " (atom " + std::to_string(i) +
            "); the table covers Z = 1.." + std::to_string(table.max_Z));
    }
  }

  if (natoms < 2)
    return; // a single centre owns every point: w = n_A/n_A = 1

  auto pts = quadrature_points;
  auto cen = atom_centers;
  auto own = point_owner;
  auto w = weights;
  auto t = table;
  auto Zv = Z;

  Kokkos::parallel_for(
      "density based weights", Kokkos::RangePolicy<ExecSpace>(0, npts),
      KOKKOS_LAMBDA(const size_t ip) {
        const Point p = pts(ip);
        const int me = own(ip);
        double sum = 0.0, mine = 0.0;
        for (int ia = 0; ia < natoms; ++ia) {
          const double dx = p[0] - cen(ia)[0];
          const double dy = p[1] - cen(ia)[1];
          const double dz = p[2] - cen(ia)[2];
          const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
          const double g = gw(t.density(static_cast<int>(Zv(ia)), r), r);
          sum += g;
          if (ia == me)
            mine = g;
        }
        // Far from every nucleus both numerator and denominator have
        // underflowed and the quotient is meaningless; drop the point.
        w(ip) *= (sum > 0.0) ? mine / sum : 0.0;
      });
  Kokkos::fence();
}

/// Hirshfeld stockholder partitioning, g_A = n_A(r).
inline void partition_hirshfeld(const Kokkos::View<Point *> &atom_centers,
                                const Kokkos::View<unsigned *> &Z,
                                const Kokkos::View<Point *> &quadrature_points,
                                const Kokkos::View<int *> &point_owner,
                                Kokkos::View<double *> &weights,
                                const AtomicDensityTable &table) {
  partition_density_based(atom_centers, Z, quadrature_points, point_owner,
                          weights, table, detail::HirshfeldWeight{});
}

/// Delley partitioning, g_A = n_A(r)/r^2.
inline void partition_delley(const Kokkos::View<Point *> &atom_centers,
                             const Kokkos::View<unsigned *> &Z,
                             const Kokkos::View<Point *> &quadrature_points,
                             const Kokkos::View<int *> &point_owner,
                             Kokkos::View<double *> &weights,
                             const AtomicDensityTable &table) {
  partition_density_based(atom_centers, Z, quadrature_points, point_owner,
                          weights, table, detail::DelleyWeight{});
}

} // namespace Nukexc
