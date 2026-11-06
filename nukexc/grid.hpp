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

#include <Kokkos_Core.hpp>

#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp> // create_pruned_spec (decl)
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/radial/becke.hpp>
#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "molecule.hpp"
#include "nukexc/atomic_properties.hpp"
#include "partitioning.hpp"
#include "stobasis.hpp"

#include <stdexcept>
#include <type_traits>
#include <vector>

//
// ============================================================
//  Flat integration grid
// ============================================================
//
// Builds a spherical grid for `mol`, applies Becke
// partitioning, and returns flattened 1D arrays ready for the
// overlap/kinetic/potential integrals.  All the repetitive Kokkos
// boilerplate lives here so test cases stay readable.

namespace Nukexc {

const std::vector<double> BECKE_SLATER_RADII = {
    0.00,       // 0: Dummy
    0.35, 0.35, // 1: H (Becke modified), 2: He
    1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.35, // 3-10: Li to Ne
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 0.70, // 11-18: Na to Ar
    2.20, 1.80, 1.60, 1.45, 1.40, 1.35, 1.40, 1.35,
    1.35, 1.35,                                     // 19-28: K to Ni
    1.35, 1.35, 1.30, 1.25, 1.15, 1.10, 1.00, 0.90, // 29-36: Cu to Kr
    2.35, 2.00, 1.80, 1.55, 1.45, 1.45, 1.35, 1.35,
    1.40, 1.40,                                     // 37-46: Rb to Pd
    1.40, 1.45, 1.45, 1.45, 1.45, 1.40, 1.35, 1.20, // 47-54: Ag to Xe
    2.60, 2.15, 1.95, 1.85, 1.85, 1.85, 1.85, 1.85,
    1.80, 1.75, // 55-64: Cs to Gd
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.70,
    1.60, 1.45, // 65-74: Tb to W
    1.35, 1.35, 1.35, 1.35, 1.35, 1.40, 1.45, 1.50,
    1.50, 1.45, // 75-84: Re to Po
    1.40, 1.30, // 85-86: At, Rn
    2.80, 2.35, 2.15, 1.95, 1.80, 1.80, 1.75, 1.75,
    1.75, 1.75, // 87-96: Fr to Cm
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, // 97-106: Bk to Sg
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, // 107-116: Bh to Lv
    1.75, 1.75  // 117-118: Ts, Og
};

const std::vector<double> TA_XI = {
    0.00, // 0: Dummy
    0.8,  0.9, 1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9, 1.4, 1.3,
    1.3,  1.2, 1.1, 1.0, 1.0, 1.0, 1.5, 1.4, 1.3, 1.2, 1.2, 1.2,
    1.2,  1.2, 1.2, 1.1, 1.1, 1.1, 1.1, 1.0, 0.9, 0.9, 0.9, 0.9};

// Treutler-Ahlrichs radial mapping exponent alpha (JCP 102, 346 (1995)).
//   r = R * (1+x)^alpha * ln(2/(1-x)) / ln2 ,  with R == the element xi.
// M4 (alpha = 0.6, Eq. 19) is the recommended default; M3 (alpha = 0.0, Eq. 18)
// drops the (1+x)^alpha factor. Pass one of these as make_flat_grid's ta_alpha.
constexpr double TA_M4 = 0.6;
constexpr double TA_M3 = 0.0;

// Periodic-table row (period, 1..7) of element Z. Used by the per-element
// radial sizing to give heavier atoms more radial points.
inline int periodic_row(unsigned Z) {
  if (Z <= 2)
    return 1;
  if (Z <= 10)
    return 2;
  if (Z <= 18)
    return 3;
  if (Z <= 36)
    return 4;
  if (Z <= 54)
    return 5;
  if (Z <= 86)
    return 6;
  return 7;
}

// Per-element radial-sizing policy for make_flat_grid.
enum class RadialSizing {
  Uniform, // every atom gets `nrad` radial points (the previous default)
  PySCF    // GauXC/PySCF per-period pattern (see pyscf_radial_size)
};

// GauXC/PySCF per-period radial multipliers, normalized so period 1 (H/He) is
// 1.0. Calibrated to PySCF's default (level-3) grid (50,75,80,90,95,100,105),
// i.e. the ratios relative to 50. The characteristic shape -- a +50% jump at
// the first row of heavy atoms, then a slow saturation -- is what distinguishes
// the standard scheme from a naive linear-in-period increment. Matches GauXC's
// PySCF* presets and PySCF's RAD_GRIDS table (Treutler-Ahlrichs lineage,
// JCP 102, 346 (1995)).
constexpr double PYSCF_RADIAL_RATIO[7] = {1.0, 1.5, 1.6, 1.8, 1.9, 2.0, 2.1};

// Radial point count for element Z given the period-1 (H/He) base count.
inline size_t pyscf_radial_size(unsigned Z, size_t base_nrad) {
  const double ratio = PYSCF_RADIAL_RATIO[periodic_row(Z) - 1];
  return static_cast<size_t>(base_nrad * ratio + 0.5);
}

struct FlatGrid {

  Kokkos::View<Point *> quad_points;
  Kokkos::View<double *> weights;
  Kokkos::View<Point *> atom_centers;
  Kokkos::View<unsigned *> Z;
  // Owning atom of each surviving quadrature point (length == quad_points).
  // Because ownership is stored per point rather than implied by a fixed
  // (atom, point) 2D shape, centers may carry different numbers of points.
  Kokkos::View<int *> point_owner;
};

// Adaptive-grid knobs (both default OFF, so the default call is the uniform
// unpruned grid used previously):
//   * pruning     -- angular adaptivity: reduce the Lebedev order in the
//                    inner/outer radial shells (IntegratorXX::PruningScheme
//                    Treutler or Robust). Unpruned keeps full order everywhere.
//   * radial_sizing -- radial adaptivity: RadialSizing::Uniform gives every
//                      centre `nrad` points; RadialSizing::PySCF applies the
//                      GauXC/PySCF per-period pattern (nrad is the H/He count,
//                      heavier atoms scale up -- see pyscf_radial_size).
// The two axes are orthogonal, so all four combinations are selectable.
template <typename radial_type, typename angular_type>
FlatGrid
make_flat_grid(const Molecule &mol, const size_t nrad = 50,
               const size_t nang_order = 30,
               const double weight_threshold = 1e-30,
               [[maybe_unused]] const double ta_alpha = TA_M4,
               const IntegratorXX::PruningScheme pruning =
                   IntegratorXX::PruningScheme::Unpruned,
               const RadialSizing radial_sizing = RadialSizing::Uniform,
               bool verbose = false) {

  using namespace IntegratorXX;
  using angular_traits = quadrature_traits<angular_type>;

  const size_t nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(nang_order));
  const unsigned natoms = mol.natoms;

  auto rad_spec = radial_from_type<radial_type>();

  const int npts = nrad * nang;

  Kokkos::View<Point *> ac_dev("atom centers", natoms);
  Kokkos::View<unsigned *> Z_dev("Z", natoms);

  auto ac_h = Kokkos::create_mirror_view(ac_dev);
  auto Z_h = Kokkos::create_mirror_view(Z_dev);

  Kokkos::deep_copy(ac_h, mol.atom_centers);
  Kokkos::deep_copy(Z_h, mol.Z);

  // Build the grid one atom at a time into flat host buffers. Each atom appends
  // its own spherical grid together with an owner tag, so different centers may
  // contribute different numbers of points (irregular grids) without any
  // rectangular (atom, point) shape constraining them.
  std::vector<Point> qp_host;
  std::vector<double> wt_host;
  std::vector<int> owner_host;
  qp_host.reserve((size_t)natoms * npts);
  wt_host.reserve((size_t)natoms * npts);
  owner_host.reserve((size_t)natoms * npts);

  for (unsigned i = 0; i < natoms; ++i) {

    unsigned atomic_number = Z_h(i);

    // Per-atom radial scaling factor R (defaults to 1.0 if Z is out of range).
    // For TA this R is the element-specific xi (JCP 102, 346 (1995), Table 1);
    // for Becke it is half the Bragg-Slater radius (JCP 88, 2547 (1988)).
    constexpr bool is_ta =
        std::is_same_v<radial_type,
                       IntegratorXX::TreutlerAhlrichs<double, double>>;
    constexpr bool is_becke =
        std::is_same_v<radial_type, IntegratorXX::Becke<double, double>>;

    double r_atomic = 1.0;
    if constexpr (is_ta) {
      if (atomic_number < TA_XI.size())
        r_atomic = TA_XI[atomic_number];
    } else if constexpr (is_becke) {
      if (atomic_number < BECKE_SLATER_RADII.size())
        r_atomic =
            0.5 * BECKE_SLATER_RADII[atomic_number] * detail::ang_to_bohr;
    }

    // Per-element radial sizing (optional): the GauXC/PySCF per-period pattern
    // gives heavier atoms more radial points than H/He (see pyscf_radial_size),
    // since their compact cores need finer radial resolution. Angular
    // adaptivity is orthogonal and handled by `pruning` below.
    const size_t nrad_atom = (radial_sizing == RadialSizing::PySCF)
                                 ? pyscf_radial_size(atomic_number, nrad)
                                 : nrad;

    // Only TA takes the M3/M4 exponent (alpha); other schemes have no such
    // parameter, so forward it exclusively on the TA path.
    std::unique_ptr<RadialTraits> rad_traits;
    if constexpr (is_ta)
      rad_traits = make_radial_traits(rad_spec, nrad_atom, r_atomic, ta_alpha);
    else
      rad_traits = make_radial_traits(rad_spec, nrad_atom, r_atomic);

    UnprunedSphericalGridSpecification unp(
        rad_spec, *rad_traits, angular_from_type<angular_type>(), nang);

    // Angular pruning (optional): Unpruned uses the full order on every shell;
    // Treutler/Robust drop the order on inner+outer shells. Either way the atom
    // yields a flat point list whose (variable) length the flat/owner layout
    // downstream handles regardless of shape.
    SphericalGridFactory::spherical_grid_ptr sph;
    if (pruning == PruningScheme::Unpruned)
      sph = SphericalGridFactory::generate_grid(unp);
    else
      sph =
          SphericalGridFactory::generate_grid(create_pruned_spec(pruning, unp));

    // sph->npts() is the count for THIS atom -- keep it local so per-atom grid
    // sizes are free to differ.
    const size_t npts_atom = sph->npts();
    bool detected_negative_weights = false;
    for (size_t j = 0; j < npts_atom; ++j) {
      Point p;
      p[0] = ac_h(i)[0] + sph->points()[j][0];
      p[1] = ac_h(i)[1] + sph->points()[j][1];
      p[2] = ac_h(i)[2] + sph->points()[j][2];
      qp_host.push_back(p);
      wt_host.push_back(sph->weights()[j]);
      if (sph->weights()[j] < 0) {
        detected_negative_weights = true;
      }
      owner_host.push_back((int)i);
    }

    if (detected_negative_weights) {
      throw std::runtime_error("Detected negative weights aborting");
    }
  }

  const size_t total_points = qp_host.size();

  Kokkos::deep_copy(ac_dev, ac_h);
  Kokkos::deep_copy(Z_dev, Z_h);

  // Upload the flat grid to the device.
  Kokkos::View<Point *> qp_flat("quadrature points", total_points);
  Kokkos::View<double *> wt_flat("weights", total_points);
  Kokkos::View<int *> owner_flat("point owner", total_points);
  Kokkos::deep_copy(
      qp_flat, Kokkos::View<Point *, HostSpace>(qp_host.data(), total_points));
  Kokkos::deep_copy(
      wt_flat, Kokkos::View<double *, HostSpace>(wt_host.data(), total_points));
  Kokkos::deep_copy(owner_flat, Kokkos::View<int *, HostSpace>(
                                    owner_host.data(), total_points));

  // partition_lko, not Becke: this is the distance-capped cell function the
  // routine formerly called partition_becke_team computed, kept as the
  // default so the grids this produces do not change. Pass a FixedGridSpec
  // with PartitionScheme::Becke for the uncapped one.
  partition_lko(ac_dev, qp_flat, owner_flat, wt_flat);

  // Remove all weights below the weight threshold, compacting points, weights
  // and owners together so the surviving grid stays self-consistent.
  //
  // The destination of each surviving point comes from a prefix sum, not from
  // an atomic counter. An atomic hands out slots in thread-arrival order, so
  // the same program on the same input produced the grid in a DIFFERENT ORDER
  // on every run: the quadrature then accumulated its sums in a different
  // order and every result moved in its last bits for no reason a caller
  // could see, batching decisions changed run to run, and nothing derived
  // from the point order could be cached or compared. A scan is also cheaper
  // than a contended atomic on one counter.
  Kokkos::View<int *> dest("compaction destinations", total_points + 1);
  Kokkos::parallel_scan(
      "Compaction destinations", Kokkos::RangePolicy<ExecSpace>(0, total_points),
      KOKKOS_LAMBDA(const int g, int &acc, const bool final) {
        if (final)
          dest(g) = acc;
        if (wt_flat(g) > weight_threshold)
          ++acc;
        if (final && g + 1 == static_cast<int>(total_points))
          dest(total_points) = acc;
      });

  int n_kept = 0;
  if (total_points > 0)
    Kokkos::deep_copy(n_kept, Kokkos::subview(dest, total_points));

  // Initialize the compacted views with the correct sizes
  Kokkos::View<Point *> qp_1d("quad points 1D", n_kept);
  Kokkos::View<double *> wt_1d("weights 1D", n_kept);
  Kokkos::View<int *> owner_1d("point owner 1D", n_kept);

  Kokkos::parallel_for(
      "FlattenViews", Kokkos::RangePolicy<ExecSpace>(0, total_points),
      KOKKOS_LAMBDA(const int g) {
        if (wt_flat(g) > weight_threshold) {
          const int d = dest(g);
          wt_1d(d) = wt_flat(g);
          qp_1d(d) = qp_flat(g);
          owner_1d(d) = owner_flat(g);
        }
      });

  if (verbose == true)
    Kokkos::printf("Reduced weight count from %zu to %d (%f %%) for a weight "
                   "threshold of %e\n",
                   total_points, n_kept,
                   100 * (1.0 - n_kept / double(total_points)),
                   weight_threshold);

  return {qp_1d, wt_1d, ac_dev, Z_dev, owner_1d};
}

} // namespace Nukexc
