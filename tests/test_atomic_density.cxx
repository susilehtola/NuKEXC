/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Spherical free-atom densities, and Hirshfeld weights built from them.
//
// There is no ERKALE parity test here, and that is deliberate rather than
// an omission. ERKALE's Hirshfeld class builds its free-atom densities by
// running an atomic SCF in whatever basis the calculation is using and
// tabulating the result on a uniform radial grid with linear interpolation
// (HirshfeldAtom, src/hirshfeld.cpp). Those densities therefore depend on
// the orbital basis, and agreeing with them would mean agreeing with a
// particular basis-set calculation. The Thakkar expansions used here are
// near-Hartree-Fock-limit and basis-set independent, so the two will differ
// by the basis-set error of whatever ERKALE was given -- several digits in
// a small basis. Reproducing that would be reproducing a defect.
//
// What replaces parity is independent physics. Two properties pin the data
// and the evaluator between them, and neither is enforced anywhere in the
// pipeline that produced the table:
//
//   * normalization, int n_A(r) 4 pi r^2 dr = Z;
//   * the Kato cusp condition, n'(0) = -2 Z n(0), which the published
//     expansions satisfy to about 1e-4 (their tabulated CUSP values are
//     0.9999x) and which no step of the translation could have imposed.

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <nukexc/atomic_density.hpp>
#include <nukexc/erkale_grid.hpp>
#include <nukexc/hirshfeld.hpp>
#include <nukexc/koster_radial.hpp>
#include <nukexc/molecule.hpp>

using namespace Nukexc;

namespace {

/// Evaluate the density at a set of radii, through the device path.
std::vector<double> eval_density(const AtomicDensityTable &t, int Z,
                                 const std::vector<double> &rs) {
  const int n = static_cast<int>(rs.size());
  Kokkos::View<double *> r_d("r", n), n_d("n", n);
  Kokkos::deep_copy(r_d, Kokkos::View<const double *, HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                             rs.data(), n));
  Kokkos::parallel_for(
      "density", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i) { n_d(i) = t.density(Z, r_d(i)); });
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, n_d);
  return std::vector<double>(h.data(), h.data() + n);
}

/// Evaluate value and derivative together, through the device path.
void eval_density_deriv(const AtomicDensityTable &t, int Z,
                        const std::vector<double> &rs, std::vector<double> &n,
                        std::vector<double> &dn) {
  const int m = static_cast<int>(rs.size());
  Kokkos::View<double *> r_d("r", m), n_d("n", m), d_d("d", m);
  Kokkos::deep_copy(r_d, Kokkos::View<const double *, HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                             rs.data(), m));
  Kokkos::parallel_for(
      "density deriv", Kokkos::RangePolicy<ExecSpace>(0, m),
      KOKKOS_LAMBDA(const int i) {
        double a, b;
        t.density_and_deriv(Z, r_d(i), a, b);
        n_d(i) = a;
        d_d(i) = b;
      });
  auto nh = Kokkos::create_mirror_view_and_copy(HostSpace{}, n_d);
  auto dh = Kokkos::create_mirror_view_and_copy(HostSpace{}, d_d);
  n.assign(nh.data(), nh.data() + m);
  dn.assign(dh.data(), dh.data() + m);
}

} // namespace

TEST_CASE("Free-atom densities integrate to Z", "[atomden]") {
  const auto t = make_thakkar_density_table();
  REQUIRE(t.max_Z == 103);

  // The residual here is the DATA's, not the quadrature's or the
  // evaluator's, and the distinction is worth stating because it sets the
  // tolerance. The published coefficients are printed to seven decimals,
  // so the tabulated orbitals are only normalized to about 5e-7, and
  //
  //     int n_A 4 pi r^2 dr - Z  =  sum_i occ_i (<R_i|R_i> - 1)
  //
  // exactly. Measured, that is 1.165e-07 for helium and 1.672e-06 for
  // krypton, and the generator reproduces those same numbers from the
  // source files alone. Hydrogen is exactly zero because its 1s is a
  // single primitive with coefficient one. The worst case over the whole
  // table is rhenium at -8.18e-06 -- the defect accumulates over occupied
  // orbitals, so it grows with Z rather than staying at the per-orbital
  // 5e-07 level. The generator checks the same bound, so this tolerance
  // and that one cannot drift apart silently.
  //
  // The data is left as published rather than renormalized: it is a
  // reference table, and silently improving it would make NuKEXC's
  // free-atom densities something other than Thakkar's. Hirshfeld weights
  // are ratios and do not care; anything that does care can renormalize at
  // the point of use.
  for (int Z : {1, 2, 6, 8, 10, 18, 26, 36, 47, 54, 75, 79, 92, 103}) {
    // Two radial grids differing by a factor of four in point count. If
    // the deviation is the data's it is identical between them; if the
    // quadrature were the limit, it would not be.
    double nel[2];
    int k = 0;
    for (size_t npts : {200, 800}) {
      const auto rad = koster_chebyshev_radial(npts, 2.0);
      std::vector<double> rs(rad.r.begin(), rad.r.end());
      const auto n = eval_density(t, Z, rs);
      double acc = 0.0;
      for (size_t i = 0; i < rs.size(); ++i)
        acc += rad.w[i] * rs[i] * rs[i] * 4.0 * M_PI * n[i];
      nel[k++] = acc;
    }

    INFO("Z = " << Z << ", 200 points: " << nel[0] << ", 800 points: "
                << nel[1]);
    // Converged in the radial grid to near machine precision ...
    REQUIRE_THAT(nel[0], Catch::Matchers::WithinAbs(nel[1], 1e-11));
    // ... and what remains is the tabulation's own precision.
    REQUIRE_THAT(nel[1], Catch::Matchers::WithinAbs(double(Z), 1.2e-5));
  }
}

TEST_CASE("Free-atom densities satisfy the Kato cusp condition", "[atomden]") {
  // n'(0) = -2 Z n(0) for the exact atom. This is the sharpest independent
  // check available: nothing in the translation from the published tables
  // to C++ could have imposed it, so a wrong exponent, a dropped
  // normalization, a misread principal quantum number or a misattributed
  // occupation would all show up here. Both source sets are explicitly
  // constrained to satisfy the cusp, so the residual is the fit's, and the
  // worst case over the whole table is 4.5e-04 at arsenic.
  //
  // Swept over every element rather than a sample: it is cheap, and the
  // worst case is at Z = 33, which no hand-picked list would have included.
  const auto t = make_thakkar_density_table();

  double worst = 0.0;
  int worst_Z = 0;
  for (int Z = 1; Z <= t.max_Z; ++Z) {
    std::vector<double> n, dn;
    // Just off the origin: the expansion is smooth here, and it is the
    // ratio rather than either value that is wanted.
    eval_density_deriv(t, Z, {1e-6}, n, dn);
    REQUIRE(n[0] > 0.0);
    const double cusp = -dn[0] / (2.0 * Z * n[0]);
    INFO("Z = " << Z << ", cusp ratio " << cusp);
    REQUIRE_THAT(cusp, Catch::Matchers::WithinAbs(1.0, 1e-3));
    if (std::abs(cusp - 1.0) > worst) {
      worst = std::abs(cusp - 1.0);
      worst_Z = Z;
    }
  }
  INFO("worst cusp deviation " << worst << " at Z = " << worst_Z);
  // Pin the worst case too, so a regression that degraded every element
  // uniformly but stayed inside the per-element bound would still show.
  REQUIRE(worst < 5e-4);
  REQUIRE(worst_Z == 33);
}

TEST_CASE("The analytic density derivative matches finite differences",
          "[atomden]") {
  const auto t = make_thakkar_density_table();
  const std::vector<double> rs = {0.05, 0.2, 0.5, 1.0, 2.0, 4.0};

  for (int Z : {1, 8, 29, 79}) {
    std::vector<double> n, dn;
    eval_density_deriv(t, Z, rs, n, dn);

    // Central differences; h chosen for the usual eps^(1/3) balance.
    const double h = 1e-5;
    std::vector<double> rp, rm;
    for (double r : rs) {
      rp.push_back(r + h);
      rm.push_back(r - h);
    }
    const auto np = eval_density(t, Z, rp);
    const auto nm = eval_density(t, Z, rm);

    for (size_t i = 0; i < rs.size(); ++i) {
      const double fd = (np[i] - nm[i]) / (2.0 * h);
      INFO("Z = " << Z << ", r = " << rs[i] << ": analytic " << dn[i]
                  << ", finite difference " << fd);
      REQUIRE_THAT(dn[i], Catch::Matchers::WithinRel(fd, 1e-6));
    }

    // The value returned alongside the derivative must be the same one
    // density() returns -- they are separate code paths.
    const auto n2 = eval_density(t, Z, rs);
    for (size_t i = 0; i < rs.size(); ++i)
      REQUIRE_THAT(n[i], Catch::Matchers::WithinRel(n2[i], 1e-14));
  }
}

TEST_CASE("Free-atom densities are positive and decay", "[atomden]") {
  const auto t = make_thakkar_density_table();
  for (int Z : {1, 7, 18, 54, 92}) {
    std::vector<double> rs;
    for (double r = 1e-4; r < 20.0; r *= 1.3)
      rs.push_back(r);
    const auto n = eval_density(t, Z, rs);
    for (size_t i = 0; i < rs.size(); ++i) {
      INFO("Z = " << Z << ", r = " << rs[i]);
      REQUIRE(n[i] > 0.0);
      // Spherically averaged atomic densities are monotonically decreasing;
      // shell structure shows as shoulders, not maxima.
      if (i > 0)
        REQUIRE(n[i] < n[i - 1]);
    }
  }
}

TEST_CASE("Every tabulated element is usable", "[atomden]") {
  // The table is CSR with a dense Z index, so a gap would show up as an
  // element that silently has no orbitals rather than as an error.
  const auto t = make_thakkar_density_table();
  auto off = Kokkos::create_mirror_view_and_copy(HostSpace{}, t.orb_offset);
  for (int Z = 1; Z <= t.max_Z; ++Z) {
    INFO("Z = " << Z);
    REQUIRE(off(Z + 1) > off(Z));
  }
  REQUIRE(thakkar::n_electrons(8) == 8.0);
  REQUIRE(thakkar::available(103));
  REQUIRE_FALSE(thakkar::available(104));
}

TEST_CASE("Hirshfeld weights are a partition of unity", "[hirshfeld]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto t = make_thakkar_density_table();

  // Build an unpartitioned grid, then apply Hirshfeld once per owning atom
  // and check the shares add to one at each spatial point. Doing it through
  // the public entry point rather than a cell function keeps the test on
  // the path the library actually takes.
  FixedGridSpec spec;
  spec.nrad = 20;
  spec.lmax = 11;
  spec.weight_threshold = 0.0;

  const int natoms = static_cast<int>(mol.natoms);
  std::vector<std::vector<double>> shares;
  size_t npts = 0;

  for (int owner = 0; owner < natoms; ++owner) {
    FixedGridSpec s = spec;
    s.partition = PartitionScheme::ERKALE; // overwritten below
    auto grid = make_fixed_grid(mol, s);
    // Re-make the raw weights: take a Becke-partitioned grid's geometry but
    // reset the weights to one, then force every point to be owned by
    // `owner` and apply Hirshfeld. The quotient does not depend on the
    // weight it multiplies, so unit weights give the cell function itself.
    Kokkos::deep_copy(grid.weights, 1.0);
    Kokkos::deep_copy(grid.point_owner, owner);
    partition_hirshfeld(grid.atom_centers, grid.Z, grid.quad_points,
                        grid.point_owner, grid.weights, t);
    auto w = Kokkos::create_mirror_view_and_copy(HostSpace{}, grid.weights);
    shares.emplace_back(w.data(), w.data() + w.extent(0));
    npts = w.extent(0);
  }

  REQUIRE(npts > 0);
  int checked = 0;
  for (size_t ip = 0; ip < npts; ++ip) {
    double s = 0.0;
    for (int a = 0; a < natoms; ++a)
      s += shares[a][ip];
    // Points so far out that every free-atom density has underflowed get
    // zero rather than one, by construction; skip those and require that
    // most points are not in that category.
    if (s == 0.0)
      continue;
    INFO("point " << ip << ", shares sum to " << s);
    REQUIRE_THAT(s, Catch::Matchers::WithinAbs(1.0, 1e-13));
    ++checked;
  }
  INFO("checked " << checked << " of " << npts << " points");
  REQUIRE(checked > static_cast<int>(npts) / 2);
}

TEST_CASE("A Hirshfeld grid integrates the promolecular density",
          "[hirshfeld][grid]") {
  // End-to-end: the superposition of free-atom densities over a
  // Hirshfeld-partitioned water grid must give 10 electrons. This exercises
  // the partition and the density together, and would catch a mismatch
  // between the two -- e.g. the partition using a different element's
  // density than the integrand does.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto t = make_thakkar_density_table();

  FixedGridSpec spec;
  spec.nrad = 99;
  spec.lmax = 41;
  spec.partition = PartitionScheme::Hirshfeld;
  spec.atomic_densities = t;
  const auto grid = make_fixed_grid(mol, spec);
  REQUIRE(grid.quad_points.extent(0) > 0);

  const int nat = static_cast<int>(grid.atom_centers.extent(0));
  auto pts = grid.quad_points;
  auto w = grid.weights;
  auto cen = grid.atom_centers;
  auto Z = grid.Z;

  double nel = 0.0;
  Kokkos::parallel_reduce(
      "promolecule", Kokkos::RangePolicy<ExecSpace>(0, pts.extent(0)),
      KOKKOS_LAMBDA(const int g, double &acc) {
        double f = 0.0;
        for (int k = 0; k < nat; ++k) {
          const double dx = pts(g)[0] - cen(k)[0];
          const double dy = pts(g)[1] - cen(k)[1];
          const double dz = pts(g)[2] - cen(k)[2];
          f += t.density(static_cast<int>(Z(k)),
                         Kokkos::sqrt(dx * dx + dy * dy + dz * dz));
        }
        acc += w(g) * f;
      },
      nel);

  INFO("promolecular electron count " << nel);
  REQUIRE_THAT(nel, Catch::Matchers::WithinAbs(10.0, 1e-6));
}

TEST_CASE("Hirshfeld partitioning fails loudly on a missing element",
          "[hirshfeld]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto t = make_thakkar_density_table();

  FixedGridSpec spec;
  spec.nrad = 10;
  spec.lmax = 5;
  auto grid = make_fixed_grid(mol, spec);

  // An element past the end of the table would otherwise contribute zero
  // density and hand its share of every point to its neighbours. The table
  // runs to lawrencium, so this has to reach past the actinides.
  Kokkos::deep_copy(grid.Z, 110u);
  REQUIRE_THROWS_AS(partition_hirshfeld(grid.atom_centers, grid.Z,
                                        grid.quad_points, grid.point_owner,
                                        grid.weights, t),
                    std::runtime_error);

  // And an empty table is an error rather than a silent no-op.
  AtomicDensityTable empty;
  REQUIRE_THROWS_AS(partition_hirshfeld(grid.atom_centers, grid.Z,
                                        grid.quad_points, grid.point_owner,
                                        grid.weights, empty),
                    std::invalid_argument);

  // Selecting the scheme without supplying a table must fail too, rather
  // than producing an unpartitioned grid.
  FixedGridSpec bad = spec;
  bad.partition = PartitionScheme::Hirshfeld;
  REQUIRE_THROWS_AS(make_fixed_grid(mol, bad), std::invalid_argument);
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
