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
// consecutive points would be far larger than the points in them. That is
// right, and it took two separate fixes to remove it; the measurements that
// led there are worth recording, because the first scheme hid the effect
// rather than curing it.
//
// With BatchScheme::MortonTiles the diagonal chain measured CHEAPER than the
// aligned one, which is the opposite of the expectation. The reason is a
// defect, not a virtue: ArborX's Morton64 normalizes coordinates to the
// grid's bounding box, so it subdivides every axis into the same number of
// cells whatever that axis's extent. For C16 the bounding box is
// 38.1 x 5.6 x 3.4 bohr aligned against 24.1 x 22.8 x 24.7 diagonal, so the
// ALIGNED chain gets Morton cells eleven times longer along the chain than
// across it, elongated tiles, and fat bounding boxes. Morton tiling is
// orientation dependent in both directions at once, and on this system the
// two dependences happened to cross.
//
// BatchScheme::CutPlane removes the first half. Its recursion is rotation
// equivariant by construction -- rotate the points and the centroid, scatter
// matrix and principal axis all rotate with them, so the batches contain the
// same points -- and it cut aligned pair work on C24 at 2048 points per box
// by 56%, from 4932 to 2179. The diagonal case barely moved, 3985 to 3835,
// and so became the expensive orientation: the partitions were now identical
// but the AXIS-ALIGNED BOX around a rotated batch is bigger, 2938 against
// 1642 bohr^3 of mean box volume for the same point sets.
//
// The second half is removed by screening against the points rather than
// against the box, which is Blum's step (2) and is exact. Oriented bounding
// boxes would have narrowed the gap; testing the points closes it, because
// the points in a batch are the same set under rotation whatever box is
// drawn around them. With both in place the two orientations agree to within
// 1% at every box size measured, and the pair work on C24 at 2048 points per
// box is 2036 aligned against 2026 diagonal, down 2.4x from where the
// original Morton-plus-box scheme left it.
//
// Everything asserted below is therefore the conjunction of the two: a
// regression in either one shows up as an orientation dependence.

#include <algorithm>
#include <cmath>
#include <cstdio>
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
  double mean_box_vol = 0.0; ///< mean AABB volume of a batch
  double mean_box_diag = 0.0; ///< mean AABB diagonal of a batch
};

/// Build the grid, boxes and neighbour list, and count. No timing.
Counts measure(int ncarbon, test::ChainAxis axis, int max_points_per_box,
               size_t nrad, size_t lmax,
               BatchScheme scheme = BatchScheme::CutPlane) {
  auto mol = test::alkane(ncarbon, axis);
  const auto host = test::sto3g_basis(mol);
  const GTOEvaluator ev(make_gto_device_basis(host));
  auto grid = make_flat_grid<ta_type, ll_type>(mol, nrad, lmax);

  Counts c;
  c.npts = static_cast<int>(grid.quad_points.extent(0));
  auto bb = create_bounding_boxes(grid, max_points_per_box, scheme);
  NeighborList nl;
  build_neighbor_list(ev, bb, max_points_per_box, grid.quad_points, nl);

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

  auto bbh = Kokkos::create_mirror_view_and_copy(HostSpace{}, bb);
  double vol = 0.0, diag = 0.0;
  for (int b = 0; b < c.nboxes; ++b) {
    double e[3], d2 = 0.0, v = 1.0;
    for (int k = 0; k < 3; ++k) {
      e[k] = bbh(b).maxCorner()[k] - bbh(b).minCorner()[k];
      d2 += e[k] * e[k];
      v *= e[k];
    }
    vol += v;
    diag += std::sqrt(d2);
  }
  c.mean_box_vol = vol / c.nboxes;
  c.mean_box_diag = diag / c.nboxes;
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

TEST_CASE("Batching and screening do not care how the molecule is oriented",
          "[scaling][alkane][diagonal]") {
  // The property the implementation is supposed to have: the same molecule
  // rotated costs the same to integrate. Pair work is sum_b m_b^2, so the
  // ratio between the two orientations is the thing to pin, and it has to be
  // near one at every box size -- a scheme that is orientation independent
  // only for small batches has merely hidden the dependence below the
  // resolution of the test.
  const size_t nrad = 20, lmax = 11;

  for (int nc : {16, 24}) {
    for (int mppb : {128, 512, 2048}) {
      const auto a = measure(nc, test::ChainAxis::Aligned, mppb, nrad, lmax);
      const auto d = measure(nc, test::ChainAxis::Diagonal, mppb, nrad, lmax);

      // The two orientations get essentially the same grid, so essentially
      // the same number of batches. Not exactly: make_flat_grid discards
      // points whose partition weight falls below 1e-30, and rotating the
      // molecule perturbs those near-zero Becke weights by rounding, so a
      // few points land on the other side of that cut. Measured at C16 the
      // difference is one box in 378. Points worth 1e-30 change nothing
      // except this count.
      INFO("boxes: aligned " << a.nboxes << ", diagonal " << d.nboxes
                             << "; points " << a.npts << " and " << d.npts);
      REQUIRE_THAT(double(d.nboxes) / a.nboxes,
                   Catch::Matchers::WithinAbs(1.0, 0.01));
      REQUIRE_THAT(double(d.npts) / a.npts,
                   Catch::Matchers::WithinAbs(1.0, 0.01));

      const double ratio = d.mean_sq / a.mean_sq;
      INFO("C" << nc << ", " << mppb << " points/box: aligned " << a.mean_sq
                << ", diagonal " << d.mean_sq << ", ratio " << ratio);
      // Measured within 1% at every setting here. 10% is loose enough to
      // survive a different grid or cutoff, and still an order of magnitude
      // tighter than the 1.8x that box-based screening gives.
      REQUIRE_THAT(ratio, Catch::Matchers::WithinAbs(1.0, 0.10));
    }
  }
}

TEST_CASE("Cut-plane batching beats Morton tiling, and is what makes the "
          "orientations agree",
          "[scaling][alkane][diagonal]") {
  // Guards the reason BatchScheme::CutPlane is the default. Exact screening
  // alone is not enough: the Morton partition is itself anisotropic, so even
  // with the boxes out of the picture it leaves a large orientation
  // dependence. Measured on C24 at 2048 points per box, Morton gives 3976
  // aligned against 2182 diagonal; cut-plane gives 2036 against 2026.
  const size_t nrad = 20, lmax = 11;
  const int nc = 24, mppb = 2048;

  const auto ma =
      measure(nc, test::ChainAxis::Aligned, mppb, nrad, lmax,
              BatchScheme::MortonTiles);
  const auto md =
      measure(nc, test::ChainAxis::Diagonal, mppb, nrad, lmax,
              BatchScheme::MortonTiles);
  const auto ca = measure(nc, test::ChainAxis::Aligned, mppb, nrad, lmax,
                          BatchScheme::CutPlane);
  const auto cd = measure(nc, test::ChainAxis::Diagonal, mppb, nrad, lmax,
                          BatchScheme::CutPlane);

  const double morton_dev = std::abs(md.mean_sq / ma.mean_sq - 1.0);
  const double cut_dev = std::abs(cd.mean_sq / ca.mean_sq - 1.0);
  INFO("orientation sensitivity: Morton " << morton_dev << ", cut-plane "
                                          << cut_dev);
  REQUIRE(morton_dev > 0.2); // Morton really is orientation dependent
  REQUIRE(cut_dev < 0.1);    // cut-plane really is not
  REQUIRE(cut_dev < morton_dev);

  // And it is not paying for that with more work: cut-plane must be at
  // least as cheap as Morton in the orientation Morton handles best.
  INFO("pair work: Morton " << std::min(ma.mean_sq, md.mean_sq)
                            << ", cut-plane "
                            << std::max(ca.mean_sq, cd.mean_sq));
  REQUIRE(std::max(ca.mean_sq, cd.mean_sq) <
          std::min(ma.mean_sq, md.mean_sq));

  // The mechanism, so a failure above says which half broke: cut-plane
  // batches are more compact, which shows up directly as box volume.
  INFO("mean box volume: Morton aligned " << ma.mean_box_vol
                                          << ", cut-plane aligned "
                                          << ca.mean_box_vol);
  REQUIRE(ca.mean_box_vol < ma.mean_box_vol);
  REQUIRE(cd.mean_box_vol < md.mean_box_vol);
}

TEST_CASE("The neighbour list is exactly the blocks that reach the box",
          "[scaling][alkane][neighborlist]") {
  // The correctness statement behind the numbers above, checked by brute
  // force against every (box, block) pair: a block belongs in box b's list
  // if and only if some point of box b lies within that block's cutoff.
  // Both directions matter. A missing entry is a wrong answer; a spurious
  // one is wasted work that no integration test would ever report, which is
  // exactly how a loose neighbour list survives unnoticed.
  const size_t nrad = 14, lmax = 11;
  const int mppb = 256;

  for (auto axis : {test::ChainAxis::Aligned, test::ChainAxis::Diagonal}) {
    auto mol = test::alkane(6, axis);
    const auto host = test::sto3g_basis(mol);
    const GTOEvaluator ev(make_gto_device_basis(host));
    auto grid = make_flat_grid<ta_type, ll_type>(mol, nrad, lmax);
    const int npts = static_cast<int>(grid.quad_points.extent(0));

    auto bb = create_bounding_boxes(grid, mppb);
    NeighborList nl;
    build_neighbor_list(ev, bb, mppb, grid.quad_points, nl);

    auto off = Kokkos::create_mirror_view_and_copy(HostSpace{}, nl.offsets);
    auto nbr = Kokkos::create_mirror_view_and_copy(HostSpace{}, nl.neighbors);
    auto pts =
        Kokkos::create_mirror_view_and_copy(HostSpace{}, grid.quad_points);
    auto cen = Kokkos::create_mirror_view_and_copy(HostSpace{}, ev.centers());
    auto rad = Kokkos::create_mirror_view_and_copy(HostSpace{}, ev.radii());

    const int nblocks = static_cast<int>(cen.extent(0));
    const int nboxes = static_cast<int>(off.extent(0)) - 1;
    REQUIRE(nboxes == (npts + mppb - 1) / mppb);

    long listed = 0;
    for (int b = 0; b < nboxes; ++b) {
      const int p0 = b * mppb;
      const int p1 = std::min(p0 + mppb, npts);

      // Truth: brute force over this box's points.
      std::vector<bool> truth(nblocks, false);
      for (int j = 0; j < nblocks; ++j) {
        const double r2 = rad(j) * rad(j);
        for (int p = p0; p < p1 && !truth[j]; ++p) {
          double d2 = 0.0;
          for (int k = 0; k < 3; ++k) {
            const double t = pts(p)[k] - cen(j)[k];
            d2 += t * t;
          }
          if (d2 <= r2)
            truth[j] = true;
        }
      }

      std::vector<bool> got(nblocks, false);
      for (int e = off(b); e < off(b + 1); ++e) {
        INFO("box " << b << " lists block " << nbr(e) << " twice or out of "
                    << "range");
        REQUIRE(nbr(e) >= 0);
        REQUIRE(nbr(e) < nblocks);
        REQUIRE(!got[nbr(e)]);
        got[nbr(e)] = true;
        ++listed;
      }
      // Sorted, which the screened kernels rely on.
      for (int e = off(b) + 1; e < off(b + 1); ++e)
        REQUIRE(nbr(e - 1) < nbr(e));

      for (int j = 0; j < nblocks; ++j) {
        INFO("box " << b << ", block " << j << ": listed " << got[j]
                    << ", reaches a point " << truth[j]);
        REQUIRE(got[j] == truth[j]);
      }
    }
    // A list that came out empty would pass every assertion above.
    REQUIRE(listed > nboxes);
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
