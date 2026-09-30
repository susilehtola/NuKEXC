/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The Gauss-Lobatto x uniform-phi angular grid.
//
// Two independent things are checked. That it reproduces ERKALE's
// lobatto_shell point for point -- which pins the ring rule, the phi offset
// and the weights. And that it actually integrates spherical harmonics to
// the order it claims, which the parity check alone would not catch if both
// codes shared a mistake.

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include <nukexc/lobatto_angular.hpp>
#include <nukexc/solid_harmonics_transform.hpp>
#include <nukexc/gto_shell.hpp>

using namespace Nukexc;

namespace {

struct ShellRef {
  int l = 0, nth = 0;
  std::vector<std::array<double, 4>> pw; // x, y, z, w
};

ShellRef load(int l) {
  const std::string path = "reference/lobatto_l" + std::to_string(l) + ".ref";
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open " + path);
  ShellRef r;
  std::string line;
  size_t n = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ss(line);
    std::string tag;
    if (line.rfind("l ", 0) == 0) { ss >> tag >> r.l; continue; }
    if (line.rfind("nth", 0) == 0) { ss >> tag >> r.nth; continue; }
    if (line.rfind("npts", 0) == 0) { ss >> tag >> n; continue; }
    double x, y, z, w;
    if (ss >> x >> y >> z >> w)
      r.pw.push_back({x, y, z, w});
  }
  REQUIRE(r.pw.size() == n);
  return r;
}

/// Real solid harmonic r^l Y_lm evaluated on a unit vector, i.e. Y_lm.
double ylm_on_unit(int l, int m, const std::array<double, 3> &d) {
  const auto T = ylm_transmat(l);
  const auto cart = cartesian_components(l);
  const int nc = cart_size(l);
  double v = 0.0;
  for (int c = 0; c < nc; ++c)
    v += T[static_cast<size_t>(m + l) * nc + c] * std::pow(d[0], cart[c].nx) *
         std::pow(d[1], cart[c].ny) * std::pow(d[2], cart[c].nz);
  return v;
}

} // namespace

TEST_CASE("The Lobatto shell reproduces ERKALE", "[angular][erkale][parity]") {
  for (int l : {5, 11, 17, 29}) {
    INFO("l = " << l);
    const auto ref = load(l);
    const auto g = lobatto_angular_grid(l);

    // Point count is itself a check on the ring rule: it depends on the
    // sin^n(theta) threshold and the l+1 cap, so getting it right by
    // accident is unlikely.
    REQUIRE(g.size() == ref.pw.size());

    double worst_p = 0.0, worst_w = 0.0;
    for (size_t i = 0; i < g.size(); ++i) {
      for (int k = 0; k < 3; ++k)
        worst_p = std::max(worst_p, std::abs(g.dir[i][k] - ref.pw[i][k]));
      worst_w = std::max(worst_w, std::abs(g.w[i] - ref.pw[i][3]) /
                                      std::max(1e-300, ref.pw[i][3]));
    }
    INFO("worst point deviation " << worst_p << ", worst relative weight "
                                  << worst_w);
    REQUIRE(worst_p < 1e-13);
    REQUIRE(worst_w < 1e-13);
  }
}

TEST_CASE("Weights sum to the surface of the unit sphere", "[angular]") {
  for (int l : {3, 5, 11, 17, 29, 41}) {
    const auto g = lobatto_angular_grid(l);
    double s = 0.0;
    for (double w : g.w)
      s += w;
    INFO("l = " << l << ", sum " << s);
    REQUIRE_THAT(s, Catch::Matchers::WithinRel(4.0 * M_PI, 1e-13));
  }
}

TEST_CASE("Points lie on the unit sphere", "[angular]") {
  for (int l : {5, 17, 29}) {
    const auto g = lobatto_angular_grid(l);
    for (size_t i = 0; i < g.size(); ++i) {
      const auto &d = g.dir[i];
      const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      INFO("l = " << l << ", point " << i);
      REQUIRE_THAT(r, Catch::Matchers::WithinAbs(1.0, 1e-13));
    }
  }
}

TEST_CASE("The grid integrates spherical harmonics to its claimed order",
          "[angular]") {
  // Independent of ERKALE: the Gram matrix of the real spherical harmonics
  // over the grid must be the identity for every degree the rule claims to
  // handle. This is what would catch a ring rule that agreed with ERKALE
  // but was wrong -- e.g. too few phi points, which shows up as the
  // off-diagonal m != m' elements failing while the norms still look fine.
  for (int l : {5, 11, 17}) {
    const auto g = lobatto_angular_grid(l);
    INFO("grid order l = " << l << ", " << g.size() << " points");

    // The product rule is exact for degree <= l; check every degree up to
    // there, and both diagonal and off-diagonal elements.
    for (int deg = 0; deg <= l / 2; ++deg) {
      const int nlm = 2 * deg + 1;
      for (int a = 0; a < nlm; ++a)
        for (int b = 0; b < nlm; ++b) {
          double acc = 0.0;
          for (size_t i = 0; i < g.size(); ++i)
            acc += g.w[i] * ylm_on_unit(deg, a - deg, g.dir[i]) *
                   ylm_on_unit(deg, b - deg, g.dir[i]);
          INFO("degree " << deg << ", m " << (a - deg) << " vs "
                         << (b - deg) << ": " << acc);
          REQUIRE_THAT(acc, Catch::Matchers::WithinAbs(a == b ? 1.0 : 0.0,
                                                       1e-10));
        }
    }
  }
}

TEST_CASE("The phi offset staggers alternate rings", "[angular]") {
  // Without the offset every ring starts at phi = 0 and the points line up
  // along meridians, giving the grid a spurious n_phi-fold symmetry. Check
  // the offset is actually applied: on a grid with at least two multi-point
  // rings, no two rings at different theta should share a phi value.
  const auto g = lobatto_angular_grid(17);

  // group by z (i.e. by ring)
  std::vector<double> zs;
  for (const auto &d : g.dir) {
    bool seen = false;
    for (double z : zs)
      if (std::abs(z - d[2]) < 1e-12)
        seen = true;
    if (!seen)
      zs.push_back(d[2]);
  }
  REQUIRE(zs.size() >= 4);

  // find two adjacent rings that both have several points
  int staggered = 0;
  for (size_t k = 0; k + 1 < zs.size(); ++k) {
    std::vector<double> pa, pb;
    for (const auto &d : g.dir) {
      if (std::abs(d[2] - zs[k]) < 1e-12)
        pa.push_back(std::atan2(d[1], d[0]));
      if (std::abs(d[2] - zs[k + 1]) < 1e-12)
        pb.push_back(std::atan2(d[1], d[0]));
    }
    if (pa.size() < 4 || pb.size() != pa.size())
      continue;
    // same ring size, so an unstaggered pair would coincide exactly
    double closest = 1e9;
    for (double x : pa)
      for (double y : pb)
        closest = std::min(closest, std::abs(x - y));
    if (closest > 1e-6)
      ++staggered;
  }
  INFO("staggered adjacent same-size ring pairs: " << staggered);
  REQUIRE(staggered > 0);
}

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
