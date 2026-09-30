/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Fixed molecular grids assembled from the ERKALE-compatible pieces.
//
// The sharpest check available without a reference calculation is that two
// grids sharing nothing but the partitioning converge to the same integral:
// Koster/Chebyshev x Lobatto here, Treutler-Ahlrichs x Lebedev in
// make_flat_grid. Different radial mapping, different angular rule,
// different point counts. If they agree to many digits on a nontrivial
// integrand, both are almost certainly right.

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/erkale_grid.hpp>
#include <nukexc/molecule.hpp>

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

/// Integrate a spherically symmetric sum of Slater densities centred on the
/// nuclei. Normalized so the exact answer is the number of atoms.
double integrate_atomic_densities(const FlatGrid &grid, double zeta) {
  const int n = static_cast<int>(grid.quad_points.extent(0));
  const int nat = static_cast<int>(grid.atom_centers.extent(0));
  auto pts = grid.quad_points;
  auto w = grid.weights;
  auto cen = grid.atom_centers;

  double acc = 0.0;
  const double norm = zeta * zeta * zeta / M_PI; // int exp(-2 zeta r) d3r = 1
  Kokkos::parallel_reduce(
      "integrate", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int g, double &a) {
        double f = 0.0;
        for (int k = 0; k < nat; ++k) {
          const double dx = pts(g)[0] - cen(k)[0];
          const double dy = pts(g)[1] - cen(k)[1];
          const double dz = pts(g)[2] - cen(k)[2];
          const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
          f += norm * Kokkos::exp(-2.0 * zeta * r);
        }
        a += w(g) * f;
      },
      acc);
  return acc;
}

} // namespace

TEST_CASE("A fixed grid integrates atomic densities", "[grid][fixed]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);

  // Two things set the tolerance here, and neither is what one would guess.
  //
  // First, the error is ANGULAR, not radial, and flat in nrad
  // (zeta = 1.5):
  //
  //     nrad  lmax   error
  //       99    41   5.9e-08
  //      150    41   5.9e-08     <- more radial points buy nothing
  //       99    59   3.9e-10
  //
  // Second, the DIFFUSE cases are the demanding ones -- a compact density
  // contributes little to the other atoms' grids, where the angular
  // resolution is spent -- and they hit a FLOOR rather than converging
  // (zeta = 0.8):
  //
  //     nrad  lmax   error
  //       99    59   1.6e-09
  //       99    71   8.7e-11
  //       99    83   5.2e-09     <- worse
  //      150    95   3.0e-09
  //
  // Non-monotonic, around the 1e-9 level. The likely cause is the
  // outer-edge weight cancellation documented in
  // tests/reference/integratorxx-upstream.md: 1 - x is formed by
  // subtracting from 1, so the outermost radial weights carry a relative
  // error that grows with nrad, and a diffuse density is precisely what
  // weights those points heavily. The tolerance below sits above that floor
  // rather than pretending it is not there.
  FixedGridSpec spec;
  spec.nrad = 99;
  spec.lmax = 59;
  const auto grid = make_fixed_grid(mol, spec);
  REQUIRE(grid.quad_points.extent(0) > 0);

  // Three normalized Slater densities on three nuclei must integrate to 3,
  // whatever the partitioning does, because the partition sums to unity.
  for (double zeta : {0.8, 1.5, 3.0}) {
    const double v = integrate_atomic_densities(grid, zeta);
    INFO("zeta = " << zeta << ", integrated " << v);
    REQUIRE_THAT(v, Catch::Matchers::WithinAbs(3.0, 1e-8));
  }
}

TEST_CASE("Lobatto and Lebedev grids agree on the same integral",
          "[grid][fixed][crosscheck]") {
  // These share only the molecule and the partition-of-unity property:
  // different radial mapping (Koster vs Treutler-Ahlrichs), different
  // angular rule (Lobatto product vs Lebedev), different point counts.
  Molecule mol;
  read_xyz("input/water.xyz", mol);

  FixedGridSpec spec;
  spec.nrad = 99;
  spec.lmax = 59;
  const auto lob = make_fixed_grid(mol, spec);
  const auto leb = make_flat_grid<ta_type, ll_type>(mol, 99, 59);

  INFO("Lobatto points " << lob.quad_points.extent(0) << ", Lebedev points "
                         << leb.quad_points.extent(0));
  // Different rules, so the counts should NOT match; if they did, something
  // is sharing a code path that should not be.
  REQUIRE(lob.quad_points.extent(0) != leb.quad_points.extent(0));

  for (double zeta : {1.0, 2.5}) {
    const double a = integrate_atomic_densities(lob, zeta);
    const double b = integrate_atomic_densities(leb, zeta);
    INFO("zeta = " << zeta << ": Lobatto " << a << ", Lebedev " << b);
    REQUIRE_THAT(a, Catch::Matchers::WithinAbs(b, 1e-8));
  }
}

TEST_CASE("Weight pruning is deterministic and order-preserving",
          "[grid][prune]") {
  // prune_grid uses a prefix sum rather than an atomic counter precisely so
  // the output order is reproducible. Build the same grid twice and require
  // the point arrays to match exactly -- not merely as sets.
  Molecule mol;
  read_xyz("input/water.xyz", mol);

  FixedGridSpec spec;
  spec.nrad = 35;
  spec.lmax = 17;
  spec.weight_threshold = 1e-12; // prune aggressively so it actually acts

  const auto g1 = make_fixed_grid(mol, spec);
  const auto g2 = make_fixed_grid(mol, spec);
  REQUIRE(g1.quad_points.extent(0) == g2.quad_points.extent(0));

  auto p1 = Kokkos::create_mirror_view_and_copy(HostSpace{}, g1.quad_points);
  auto p2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, g2.quad_points);
  auto w1 = Kokkos::create_mirror_view_and_copy(HostSpace{}, g1.weights);
  auto w2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, g2.weights);

  for (size_t i = 0; i < p1.extent(0); ++i) {
    INFO("point " << i);
    for (int k = 0; k < 3; ++k)
      REQUIRE(p1(i)[k] == p2(i)[k]);
    REQUIRE(w1(i) == w2(i));
  }

  // and pruning must actually have removed something at this threshold
  FixedGridSpec unpruned = spec;
  unpruned.weight_threshold = 0.0;
  const auto g0 = make_fixed_grid(mol, unpruned);
  INFO("unpruned " << g0.quad_points.extent(0) << ", pruned "
                   << g1.quad_points.extent(0));
  REQUIRE(g1.quad_points.extent(0) < g0.quad_points.extent(0));
}

TEST_CASE("The partition scheme changes the grid but not the integral",
          "[grid][fixed][partition]") {
  // Every scheme is a partition of unity, so an integral of a smooth
  // molecular function must not depend on which one is used -- only on how
  // efficiently it concentrates weight. This is a different statement from
  // the partition-of-unity test, which checks the weights directly; here it
  // is checked through a converged integral.
  Molecule mol;
  read_xyz("input/water.xyz", mol);

  FixedGridSpec a;
  a.nrad = 99;
  a.lmax = 59;
  a.partition = PartitionScheme::ERKALE;

  FixedGridSpec b = a;
  b.partition = PartitionScheme::Becke;

  const double va = integrate_atomic_densities(make_fixed_grid(mol, a), 1.5);
  const double vb = integrate_atomic_densities(make_fixed_grid(mol, b), 1.5);
  INFO("ERKALE " << va << ", Becke " << vb);
  REQUIRE_THAT(va, Catch::Matchers::WithinAbs(3.0, 1e-8));
  REQUIRE_THAT(vb, Catch::Matchers::WithinAbs(3.0, 1e-8));
}

TEST_CASE("Bad fixed-grid specifications are rejected", "[grid][fixed]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);

  FixedGridSpec s;
  s.nrad = 0;
  REQUIRE_THROWS_AS(make_fixed_grid(mol, s), std::invalid_argument);

  s = FixedGridSpec{};
  s.lmax = -1;
  REQUIRE_THROWS_AS(make_fixed_grid(mol, s), std::invalid_argument);

  s = FixedGridSpec{};
  s.angular = AngularScheme::Lebedev;
  REQUIRE_THROWS_AS(make_fixed_grid(mol, s), std::invalid_argument);
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int result = 0;
  {
    result = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return result;
}
