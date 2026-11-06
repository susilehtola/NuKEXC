/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Linear scaling of the screening, on the system that can show it.
//
// Water cannot: at three atoms every basis function reaches every grid box,
// so a screened path and a dense one do identical work. The quantity that
// has to be bounded is the number of SHELLS PER BOX, because the pair
// kernels cost sum_b m_b^2 and the number of boxes is proportional to the
// number of grid points. If m_b approaches a constant as the chain grows,
// total work is proportional to the grid and the method scales linearly; if
// m_b keeps growing, it does not.
//
// Everything here is a COUNT, never a wall time. Counts are deterministic
// and reproducible on a loaded machine; timings on this hardware are not,
// and a scaling claim resting on them would be worthless.
//
// ORIENTATION
// -----------
// The chain is tested along x and along (1,1,1)/sqrt(3). The expectation
// going in was that the diagonal case would be the hard one -- a thin tube
// through a large cube, so that axis-aligned boxes around runs of
// consecutive points would be far larger than the points in them.
//
// Measured, it is the other way round, and the mechanism is worth recording
// because it is the opposite of the intuition. ArborX's Morton64 normalizes
// coordinates to the grid's bounding box, so it subdivides every axis into
// the same number of cells whatever that axis's extent. For C16 the
// bounding box is 38.1 x 5.6 x 3.4 bohr aligned, against
// 24.1 x 22.8 x 24.7 diagonal. The aligned chain therefore gets Morton
// cells eleven times longer along the chain than across it, so its tiles
// are elongated, their bounding boxes fat, and they overlap more shells.
// The diagonal chain gets near-cubic cells and compact tiles, and comes out
// about 16% CHEAPER at large box sizes.
//
// So the pathological orientation here is the axis-aligned one, and the
// defect is the anisotropy of Morton cells rather than the molecule's
// direction. Both orientations are asserted, so whichever way a future
// batching scheme shifts the balance, the test still pins the property that
// matters: that neither grows without bound.

#include <algorithm>
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/grid.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/octree.hpp>

#include "alkane.hpp"
#include "test_basis_sets.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

struct Counts {
  int nboxes = 0;
  int npts = 0;
  double mean_shells = 0.0; ///< mean m_b: the neighbour-list size
  double mean_sq = 0.0;     ///< mean m_b^2: the pair-kernel work per box
};

/// Build the grid, boxes and neighbour list, and count. No timing.
Counts measure(int ncarbon, test::ChainAxis axis, int max_points_per_box,
               size_t nrad, size_t lmax) {
  auto mol = test::alkane(ncarbon, axis);
  const auto host = test::sto3g_basis(mol);
  const GTOEvaluator ev(make_gto_device_basis(host));
  auto grid = make_flat_grid<ta_type, ll_type>(mol, nrad, lmax);

  Counts c;
  c.npts = static_cast<int>(grid.quad_points.extent(0));
  auto bb = create_bounding_boxes(grid, max_points_per_box);
  NeighborList nl;
  build_neighbor_list(ev, bb, max_points_per_box, c.npts, nl);

  auto off = Kokkos::create_mirror_view_and_copy(HostSpace{}, nl.offsets);
  c.nboxes = static_cast<int>(off.extent(0)) - 1;
  REQUIRE(c.nboxes > 0);
  long total = 0, sq = 0;
  for (int b = 0; b < c.nboxes; ++b) {
    const long m = off(b + 1) - off(b);
    total += m;
    sq += m * m;
  }
  c.mean_shells = double(total) / c.nboxes;
  c.mean_sq = double(sq) / c.nboxes;
  return c;
}

} // namespace

TEST_CASE("The generated alkane has the geometry it claims",
          "[scaling][alkane][geometry]") {
  // The scaling results are only meaningful if the two orientations really
  // are aligned and diagonal, and if the molecule is a plausible alkane.
  // Asserted rather than assumed: a broken rotation would make the
  // orientation comparison compare a molecule with itself.
  constexpr double ANG = 1.8897261254578281;

  for (auto axis : {test::ChainAxis::Aligned, test::ChainAxis::Diagonal}) {
    const bool diag = (axis == test::ChainAxis::Diagonal);
    const auto g = test::alkane_geometry(16, axis);

    // C_n H_(2n+2): 16 carbons, 34 hydrogens.
    REQUIRE(g.size() == 50);
    int nc = 0, nh = 0;
    for (unsigned z : g.Z) {
      if (z == 6) ++nc;
      if (z == 1) ++nh;
    }
    REQUIRE(nc == 16);
    REQUIRE(nh == 34);

    std::vector<std::array<double, 3>> C;
    for (size_t i = 0; i < g.size(); ++i)
      if (g.Z[i] == 6)
        C.push_back(g.xyz[i]);

    // Every backbone bond is 1.526 A, and every C-H is 1.090 A.
    for (size_t i = 0; i + 1 < C.size(); ++i) {
      double d = 0.0;
      for (int k = 0; k < 3; ++k) {
        const double t = C[i + 1][k] - C[i][k];
        d += t * t;
      }
      INFO("C-C bond " << i << " = " << std::sqrt(d) / ANG << " A");
      REQUIRE_THAT(std::sqrt(d) / ANG,
                   Catch::Matchers::WithinAbs(1.526, 1e-9));
    }
    for (size_t i = 0; i < g.size(); ++i) {
      if (g.Z[i] != 1)
        continue;
      double best = 1e30;
      for (const auto &c : C) {
        double d = 0.0;
        for (int k = 0; k < 3; ++k) {
          const double t = g.xyz[i][k] - c[k];
          d += t * t;
        }
        best = std::min(best, std::sqrt(d));
      }
      INFO("nearest C-H = " << best / ANG << " A");
      REQUIRE_THAT(best / ANG, Catch::Matchers::WithinAbs(1.090, 1e-9));
    }

    // Orientation, via the bounding box aspect ratio. Aligned is strongly
    // anisotropic; diagonal is nearly cubic. This is the quantity that
    // drives the Morton-cell anisotropy discussed above, so it is the one
    // worth pinning.
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (const auto &p : g.xyz)
      for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], p[k]);
        hi[k] = std::max(hi[k], p[k]);
      }
    double ext[3];
    for (int k = 0; k < 3; ++k)
      ext[k] = hi[k] - lo[k];
    const double aspect = *std::max_element(ext, ext + 3) /
                          *std::min_element(ext, ext + 3);
    INFO((diag ? "diagonal" : "aligned")
         << " extents " << ext[0] << " " << ext[1] << " " << ext[2]
         << ", aspect " << aspect);
    if (diag)
      REQUIRE(aspect < 1.5); // measured 1.08
    else
      REQUIRE(aspect > 5.0); // measured 11.3
  }
}

TEST_CASE("Shells per box saturate as the alkane chain grows",
          "[scaling][alkane]") {
  // The linear-scaling statement. Doubling the chain must not keep doubling
  // the neighbour lists; once the chain is long compared with twice the
  // shell cutoff (about 9 bohr here, so beyond roughly C8) the lists have
  // to level off.
  const int mppb = 128;
  const size_t nrad = 20, lmax = 11;

  for (auto axis : {test::ChainAxis::Aligned, test::ChainAxis::Diagonal}) {
    const char *nm =
        (axis == test::ChainAxis::Aligned) ? "aligned" : "diagonal";

    const auto c8 = measure(8, axis, mppb, nrad, lmax);
    const auto c16 = measure(16, axis, mppb, nrad, lmax);
    const auto c24 = measure(24, axis, mppb, nrad, lmax);

    INFO(nm << ": mean shells/box " << c8.mean_shells << " (C8), "
            << c16.mean_shells << " (C16), " << c24.mean_shells << " (C24)");

    // The grid itself must grow roughly linearly in the chain, or the rest
    // of the test is measuring the wrong thing.
    REQUIRE(c24.npts > 2.5 * c8.npts);
    REQUIRE(c24.npts < 3.5 * c8.npts);

    // Saturation: tripling the chain from C8 to C24 must grow the lists by
    // much less than the threefold the atom count does. Measured at these
    // settings the ratio is about 1.3; 2.0 leaves room for a different
    // grid or cutoff without being so loose it would accept linear growth.
    const double ratio = c24.mean_shells / c8.mean_shells;
    INFO(nm << ": C24/C8 neighbour-list ratio " << ratio);
    REQUIRE(ratio > 1.0);  // it does grow in this pre-asymptotic range
    REQUIRE(ratio < 2.0);  // but nowhere near proportionally

    // And the last doubling must be flatter than the first: the sequence
    // has to be decelerating, which is what distinguishes saturation from
    // slow linear growth.
    const double early = c16.mean_shells / c8.mean_shells;
    const double late = c24.mean_shells / c16.mean_shells;
    INFO(nm << ": growth C8->C16 " << early << ", C16->C24 " << late);
    REQUIRE(late < early);
  }
}

TEST_CASE("The diagonal chain is not the pathological orientation",
          "[scaling][alkane][diagonal]") {
  // Recorded because it is the opposite of what one expects, and because a
  // batching scheme that oriented its boxes should change it. ArborX's
  // Morton codes are normalized to the grid's bounding box, which is
  // strongly anisotropic for an axis-aligned chain and nearly cubic for a
  // diagonal one, so it is the ALIGNED chain whose tiles come out
  // elongated and overlap more shells.
  //
  // Asserted loosely and in the direction measured. If recursive
  // cut-plane batching replaces the Morton tiling -- Blum et al.,
  // Comput. Phys. Commun. 180 (2009) 2175, step (1), after Kahan -- the
  // two orientations should converge towards each other, and this test
  // should be revisited rather than merely retuned.
  const size_t nrad = 20, lmax = 11;

  for (int mppb : {128, 1024}) {
    const auto a = measure(16, test::ChainAxis::Aligned, mppb, nrad, lmax);
    const auto d = measure(16, test::ChainAxis::Diagonal, mppb, nrad, lmax);
    const double ratio = d.mean_sq / a.mean_sq;
    INFO("max_points_per_box " << mppb << ": aligned " << a.mean_sq
                               << ", diagonal " << d.mean_sq << ", ratio "
                               << ratio);
    // Both must be the same order; neither orientation may blow up.
    REQUIRE(ratio > 0.5);
    REQUIRE(ratio < 1.5);
  }
}

TEST_CASE("Bigger boxes cost more pair work per box", "[scaling][alkane]") {
  // A sanity check on the metric rather than on the physics: a box holding
  // more points spans more space and so overlaps more shells. If this did
  // not hold, the neighbour list would not be doing what it claims and the
  // saturation test above would be meaningless.
  const size_t nrad = 20, lmax = 11;
  const auto small = measure(16, test::ChainAxis::Aligned, 128, nrad, lmax);
  const auto big = measure(16, test::ChainAxis::Aligned, 1024, nrad, lmax);

  INFO("128 points/box: " << small.mean_shells << " shells, 1024: "
                          << big.mean_shells);
  REQUIRE(big.mean_shells > small.mean_shells);
  REQUIRE(big.nboxes < small.nboxes);
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
