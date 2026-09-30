/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Koster's radial mapping, checked against ERKALE's radial_chebyshev.
//
// The quadrature primitive is IntegratorXX's GaussChebyshev2Modified, whose
// node and weight expressions are the same as ERKALE's. So what is actually
// under test is the MAPPING and the way it composes with that primitive --
// including node ORDER, which the two generate differently and which the
// reference comparison would miss if it sorted.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include <nukexc/koster_radial.hpp>

using namespace Nukexc;

namespace {

struct RadialRef {
  std::vector<double> r, w;
};

RadialRef load(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open " + path +
                             " (see tests/reference/README.md)");
  RadialRef ref;
  std::string line;
  size_t n = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ss(line);
    std::string tag;
    if (line.rfind("npts", 0) == 0) {
      ss >> tag >> n;
      ref.r.reserve(n);
      ref.w.reserve(n);
      continue;
    }
    double r, w;
    std::istringstream vs(line);
    if (vs >> r >> w) {
      ref.r.push_back(r);
      ref.w.push_back(w);
    }
  }
  REQUIRE(ref.r.size() == n);
  return ref;
}

} // namespace

TEST_CASE("Koster radial grid reproduces ERKALE", "[radial][erkale][parity]") {
  for (int n : {25, 50, 75}) {
    INFO("nrad = " << n);
    const auto ref = load("reference/radial_chebyshev_" + std::to_string(n) +
                          ".ref");

    const RadialGrid q = koster_chebyshev_radial(n);
    const auto &pts = q.r;
    const auto &wts = q.w;

    // ERKALE drops any node whose radius or weight is not finite; at these
    // sizes none are, so the counts should agree exactly. If they ever
    // diverge that is itself worth knowing.
    REQUIRE(pts.size() == ref.r.size());

    // Node ORDER matters: IntegratorXX and ERKALE generate the Chebyshev
    // nodes in different directions, and a grid that agreed only as a set
    // would still pair radii with the wrong weights downstream. Compare
    // element by element, in order.
    double worst_r = 0.0, worst_w = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
      worst_r = std::max(worst_r, std::abs(pts[i] - ref.r[i]) /
                                      std::max(1.0, std::abs(ref.r[i])));
      worst_w = std::max(worst_w, std::abs(wts[i] - ref.w[i]) /
                                      std::max(1e-300, std::abs(ref.w[i])));
    }
    // Exact agreement is not expected, and the tolerance has to GROW with n
    // for a reason worth recording. Both codes form the outermost node as
    // 1 - (something very close to 1), so 1 - x -- which the logarithmic map
    // and its 1/(1-x) Jacobian both need -- is computed with heavy
    // cancellation. Measured relative error in 1 - x against a
    // cancellation-free rearrangement: 4.8e-12 at n=25, 2.3e-11 at n=50,
    // 4.2e-10 at n=75, 2.6e-8 at n=150.
    //
    // The two codes lose the SAME digits, so they still agree far better
    // than either is accurate; what is left here is only the difference in
    // their evaluation order (IntegratorXX computes half the nodes and
    // reflects, ERKALE computes all directly). Observed: ~5e-14 at n=25 and
    // ~2e-11 at n=50. See tests/reference/integratorxx-upstream.md.
    INFO("worst relative deviation: radius " << worst_r << ", weight "
                                             << worst_w);
    REQUIRE(worst_r < 1e-9);
    REQUIRE(worst_w < 1e-8);
  }
}

TEST_CASE("The mapping and its Jacobian are mutually consistent",
          "[radial]") {
  // Independent of ERKALE: dr/dx from the closed form must match a central
  // difference of r(x). This is what would catch a Jacobian that had been
  // transcribed with the wrong power or a missing ln 2.
  const KosterRadialTraits t(10);
  const double h = 1e-6;
  for (double x : {-0.9, -0.5, 0.0, 0.5, 0.9, 0.99}) {
    const double fd = (t.radial_transform(x + h) - t.radial_transform(x - h)) /
                      (2 * h);
    const double an = t.radial_jacobian(x);
    INFO("x = " << x << ", analytic " << an << ", finite difference " << fd);
    REQUIRE_THAT(an, Catch::Matchers::WithinRel(fd, 1e-8));
  }
}

TEST_CASE("The mapping has the parameter-free behaviour it is chosen for",
          "[radial]") {
  const KosterRadialTraits t(10);
  // r(0) = ln 2 / ln 2 = 1, the fixed point that makes the scheme
  // element-independent: no atomic size parameter enters anywhere.
  REQUIRE_THAT(t.radial_transform(0.0), Catch::Matchers::WithinAbs(1.0, 1e-14));
  // r(-1) = ln(2/2)/ln2 = 0
  REQUIRE_THAT(t.radial_transform(-1.0), Catch::Matchers::WithinAbs(0.0, 1e-14));
  // monotone increasing towards the outer edge
  double prev = -1.0;
  for (double x = -0.99; x < 0.999; x += 0.05) {
    const double r = t.radial_transform(x);
    REQUIRE(r > prev);
    prev = r;
  }
}

TEST_CASE("The radial rule integrates hydrogenic densities", "[radial]") {
  // A physical check the reference comparison cannot give: the weights here
  // exclude r^2, so integral 4 pi r^2 rho(r) dr must recover the electron
  // count. exp(-2 Z r) densities for a few Z, normalized as 1s orbitals.
  for (double Z : {1.0, 2.0, 6.0}) {
    const RadialGrid q = koster_chebyshev_radial(200);
    const auto &r = q.r;
    const auto &w = q.w;

    // rho(r) = Z^3/pi exp(-2 Z r), so integral 4 pi r^2 rho dr = 1
    double acc = 0.0;
    for (size_t i = 0; i < r.size(); ++i)
      acc += w[i] * 4.0 * M_PI * r[i] * r[i] *
             (Z * Z * Z / M_PI) * std::exp(-2.0 * Z * r[i]);

    INFO("Z = " << Z << ", integrated " << acc);
    REQUIRE_THAT(acc, Catch::Matchers::WithinAbs(1.0, 1e-9));
  }
}

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
