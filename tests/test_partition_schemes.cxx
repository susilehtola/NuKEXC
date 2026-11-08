/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Atomic partitioning schemes.
//
// Two things need pinning down. That the ERKALE cell function is ERKALE's,
// sampled densely enough that no feature hides between samples; and that the
// partition really is one of unity, which is the property everything
// downstream silently assumes and which no single-scheme test can check.
//
// The third thing measured here is what the LKO distance cap in
// partition_lko actually costs, since "it is an approximation" is not
// a useful statement without a number.

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <nukexc/molecule.hpp>
#include <nukexc/partition_schemes.hpp>
#include <nukexc/partitioning.hpp>

using namespace Nukexc;

namespace {

/// A handful of centres with deliberately uneven spacing, so that near
/// neighbours and far ones both occur.
Kokkos::View<Point *> make_centers() {
  const double c[4][3] = {{0.0, 0.0, 0.0},
                          {0.0, 0.0, 1.8},
                          {0.0, 1.6, -0.6},
                          {7.5, 0.0, 0.0}}; // deliberately beyond 5 bohr
  Kokkos::View<Point *> d("centers", 4);
  auto h = Kokkos::create_mirror_view(d);
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 3; ++k)
      h(i)[k] = c[i][k];
  Kokkos::deep_copy(d, h);
  return d;
}

/// A deterministic cloud of sample points spanning the whole system.
Kokkos::View<Point *> make_points(int n) {
  Kokkos::View<Point *> d("pts", n);
  auto h = Kokkos::create_mirror_view(d);
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i);
    h(i)[0] = 9.0 * std::sin(0.37 * t) * std::cos(0.11 * t);
    h(i)[1] = 4.0 * std::sin(0.53 * t + 1.1);
    h(i)[2] = 4.0 * std::cos(0.29 * t + 0.4);
  }
  Kokkos::deep_copy(d, h);
  return d;
}

Kokkos::View<int *> make_owner(int n, int natoms) {
  Kokkos::View<int *> d("owner", n);
  auto h = Kokkos::create_mirror_view(d);
  for (int i = 0; i < n; ++i)
    h(i) = i % natoms;
  Kokkos::deep_copy(d, h);
  return d;
}

Kokkos::View<double *> ones(int n) {
  Kokkos::View<double *> d("w", n);
  Kokkos::deep_copy(d, 1.0);
  return d;
}

std::vector<double> host(const Kokkos::View<double *> &d) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, d);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}

} // namespace

TEST_CASE("The ERKALE cell function matches ERKALE", "[partition][erkale]") {
  std::ifstream in("reference/becke_erkale_cellfn.ref");
  REQUIRE(in);

  double a = 0.0;
  size_t n = 0;
  std::vector<double> mu, s;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ss(line);
    if (line.rfind("a ", 0) == 0) {
      std::string t;
      ss >> t >> a;
      continue;
    }
    if (line.rfind("npts", 0) == 0) {
      std::string t;
      ss >> t >> n;
      continue;
    }
    double x, y;
    if (ss >> x >> y) {
      mu.push_back(x);
      s.push_back(y);
    }
  }
  REQUIRE(a == 0.7);
  REQUIRE(mu.size() == n);

  // 2001 samples across [-1, 1] -- dense enough that the two kinks where the
  // clip engages, at mu = +-a, cannot fall between samples.
  double worst = 0.0;
  for (size_t i = 0; i < mu.size(); ++i)
    worst = std::max(worst, std::abs(detail::erkale_s(mu[i], a) - s[i]));
  INFO("worst deviation " << worst);
  REQUIRE(worst < 1e-15);

  // The clip is the defining feature: exactly saturated outside |mu| > a,
  // which is what licenses the Stratmann prescreen.
  REQUIRE(detail::erkale_s(-0.9, a) == 1.0);
  REQUIRE(detail::erkale_s(0.9, a) == 0.0);
  REQUIRE_THAT(detail::erkale_s(0.0, a),
               Catch::Matchers::WithinAbs(0.5, 1e-15));
}

TEST_CASE("Becke's cell function saturates only asymptotically",
          "[partition]") {
  // The contrast that matters: the unclipped function never reaches 0 or 1
  // on the open interval, so every atom contributes at every point and no
  // prescreen is exact.
  REQUIRE(detail::becke_s(-0.9) < 1.0);
  REQUIRE(detail::becke_s(0.9) > 0.0);
  REQUIRE(detail::becke_s(-1.0) == 1.0);
  REQUIRE(detail::becke_s(1.0) == 0.0);
  REQUIRE_THAT(detail::becke_s(0.0), Catch::Matchers::WithinAbs(0.5, 1e-15));
  // antisymmetry, s(-mu) = 1 - s(mu)
  for (double m : {-0.8, -0.3, 0.1, 0.55, 0.95})
    REQUIRE_THAT(detail::becke_s(-m),
                 Catch::Matchers::WithinAbs(1.0 - detail::becke_s(m), 1e-15));
}

TEST_CASE("The Stratmann cell function saturates with flat derivatives",
          "[partition][stratmann]") {
  const double a = 0.64;

  // z saturates exactly at the ends and stays there.
  REQUIRE(detail::stratmann_z(1.0) == 1.0);
  REQUIRE(detail::stratmann_z(-1.0) == -1.0);
  REQUIRE(detail::stratmann_z(1.5) == 1.0);
  REQUIRE(detail::stratmann_z(-2.0) == -1.0);

  // The identity that defines the polynomial: z'(x) = 35 (1 - x^2)^3 / 16.
  // The cubed factor is why the first three derivatives vanish at +-1, which
  // is the whole reason for degree seven -- and checking it numerically
  // catches a mistyped coefficient, which inspection does not.
  for (double x : {-0.9, -0.5, -0.1, 0.0, 0.3, 0.75, 0.95}) {
    const double h = 1e-5;
    const double num =
        (detail::stratmann_z(x + h) - detail::stratmann_z(x - h)) / (2.0 * h);
    const double ana = 35.0 * std::pow(1.0 - x * x, 3) / 16.0;
    INFO("x = " << x << ": numerical " << num << ", analytic " << ana);
    REQUIRE_THAT(num, Catch::Matchers::WithinRel(ana, 1e-6));
  }

  // The cell function: saturated outside |mu| > a, one half at the midpoint,
  // antisymmetric -- the same three properties the ERKALE one has.
  REQUIRE(detail::stratmann_s(-0.7, a) == 1.0);
  REQUIRE(detail::stratmann_s(0.7, a) == 0.0);
  REQUIRE_THAT(detail::stratmann_s(0.0, a),
               Catch::Matchers::WithinAbs(0.5, 1e-15));
  for (double m : {-0.5, -0.2, 0.1, 0.4, 0.6})
    REQUIRE_THAT(detail::stratmann_s(-m, a),
                 Catch::Matchers::WithinAbs(
                     1.0 - detail::stratmann_s(m, a), 1e-15));

  // And it is a different function from ERKALE's, which is the point of
  // having both: same saturation property, different polynomial and
  // different a, so they are not interchangeable at the digits that matter.
  const double d = std::abs(detail::stratmann_s(0.3, a) -
                            detail::erkale_s(0.3, 0.7));
  INFO("|s_SSF(0.3) - s_ERKALE(0.3)| = " << d);
  REQUIRE(d > 1e-3);
}

TEST_CASE("Every scheme is a partition of unity", "[partition]") {
  // The property everything downstream assumes: summing the weight a point
  // receives over all possible owners must give back the original weight.
  // A single-scheme run cannot detect a normalization slip; this can.
  const auto centers = make_centers();
  const int natoms = static_cast<int>(centers.extent(0));
  const int npts = 400;
  const auto pts = make_points(npts);

  const char *names[] = {"Becke", "ERKALE", "Stratmann", "LKO"};
  for (int scheme = 0; scheme < 4; ++scheme) {
    INFO(names[scheme]);
    std::vector<double> total(npts, 0.0);

    for (int owner = 0; owner < natoms; ++owner) {
      Kokkos::View<int *> own("own", npts);
      Kokkos::deep_copy(own, owner);
      auto w = ones(npts);
      switch (scheme) {
      case 0:
        partition_becke_exact(centers, pts, own, w);
        break;
      case 1:
        partition_erkale(centers, pts, own, w);
        break;
      case 2:
        partition_stratmann(centers, pts, own, w);
        break;
      default:
        partition_lko(centers, pts, own, w);
        break;
      }
      const auto wh = host(w);
      for (int g = 0; g < npts; ++g)
        total[g] += wh[g];
    }

    for (int g = 0; g < npts; ++g) {
      INFO("point " << g);
      REQUIRE_THAT(total[g], Catch::Matchers::WithinAbs(1.0, 1e-12));
    }
  }
}

TEST_CASE("The Stratmann prescreen is exact, not approximate",
          "[partition][erkale]") {
  // Inside the screening radius the prescreen returns weight 1 without
  // evaluating anything. That is only legitimate because the clipped cell
  // function saturates exactly there. Verify by disabling the shortcut --
  // via a = 0.7 run against points just inside and just outside the radius
  // -- and checking the function is continuous across it.
  const auto centers = make_centers();
  auto ch = Kokkos::create_mirror_view_and_copy(HostSpace{}, centers);

  // nearest neighbour of atom 0 is atom 1 at 1.8 bohr
  const double Rin = 1.8;
  const double a = 0.7;
  const double screen = 0.5 * (1.0 - a) * Rin;

  // points straddling the screening radius along a direction away from all
  // other atoms
  const int n = 2;
  Kokkos::View<Point *> pts("pts", n);
  auto ph = Kokkos::create_mirror_view(pts);
  ph(0)[0] = -(screen * 0.99); ph(0)[1] = 0.0; ph(0)[2] = 0.0;
  ph(1)[0] = -(screen * 1.01); ph(1)[1] = 0.0; ph(1)[2] = 0.0;
  Kokkos::deep_copy(pts, ph);

  Kokkos::View<int *> own("own", n);
  Kokkos::deep_copy(own, 0);
  auto w = ones(n);
  partition_erkale(centers, pts, own, w);
  const auto wh = host(w);

  INFO("inside " << wh[0] << ", outside " << wh[1]);
  REQUIRE_THAT(wh[0], Catch::Matchers::WithinAbs(1.0, 1e-15)); // prescreened
  // Just outside, the full evaluation must also give (essentially) 1 --
  // otherwise the prescreen would introduce a discontinuity.
  REQUIRE_THAT(wh[1], Catch::Matchers::WithinAbs(1.0, 1e-12));

  // The same for Stratmann, whose a is 0.64 and whose radius is therefore
  // larger. The shortcut is a property of saturation, not of one scheme.
  const double a_ssf = 0.64;
  const double screen_ssf = 0.5 * (1.0 - a_ssf) * Rin;
  REQUIRE(screen_ssf > screen); // more points take the shortcut
  auto ph2 = Kokkos::create_mirror_view(pts);
  ph2(0)[0] = -(screen_ssf * 0.99);
  ph2(0)[1] = 0.0;
  ph2(0)[2] = 0.0;
  ph2(1)[0] = -(screen_ssf * 1.01);
  ph2(1)[1] = 0.0;
  ph2(1)[2] = 0.0;
  Kokkos::deep_copy(pts, ph2);
  auto w2 = ones(n);
  partition_stratmann(centers, pts, own, w2);
  const auto wh2 = host(w2);
  INFO("SSF inside " << wh2[0] << ", outside " << wh2[1]);
  REQUIRE_THAT(wh2[0], Catch::Matchers::WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(wh2[1], Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("The capped scheme survives separations far beyond the cap",
          "[partition][lko]") {
  // The cap puts the TRUE separation in the numerator of mu and a capped one
  // in the denominator, so mu is no longer bounded by one: on a chain whose
  // ends are 26 bohr apart and a cap of 5, it reaches five. Becke's cell
  // function is an iterated cubic, and p(x) for x beyond about two is
  // chaotic and then divergent -- p(6) = -99, p(-99) = 4.8e5, and the third
  // iteration overflows. The clamp on mu is what keeps that from happening,
  // and nothing in a short-geometry test exercises it: four centres spanning
  // 7.5 bohr only reach mu = 1.5, where the iteration still converges.
  const int natoms = 12;
  Kokkos::View<Point *> centers("centers", natoms);
  auto ch = Kokkos::create_mirror_view(centers);
  for (int i = 0; i < natoms; ++i) {
    ch(i)[0] = 2.4 * i; // a 26 bohr line, five times the 5 bohr cap
    ch(i)[1] = 0.0;
    ch(i)[2] = 0.0;
  }
  Kokkos::deep_copy(centers, ch);

  const int npts = 200;
  Kokkos::View<Point *> pts("pts", npts);
  auto ph = Kokkos::create_mirror_view(pts);
  for (int i = 0; i < npts; ++i) {
    const double t = static_cast<double>(i);
    ph(i)[0] = 0.13 * t;                      // along the chain
    ph(i)[1] = 2.0 * std::sin(0.41 * t);      // and off it
    ph(i)[2] = 1.5 * std::cos(0.23 * t + 0.7);
  }
  Kokkos::deep_copy(pts, ph);

  std::vector<double> total(npts, 0.0);
  for (int owner = 0; owner < natoms; ++owner) {
    Kokkos::View<int *> own("own", npts);
    Kokkos::deep_copy(own, owner);
    auto w = ones(npts);
    partition_lko(centers, pts, own, w);
    const auto wh = host(w);
    for (int g = 0; g < npts; ++g) {
      INFO("point " << g << ", owner " << owner << ", weight " << wh[g]);
      REQUIRE(std::isfinite(wh[g]));
      REQUIRE(wh[g] >= 0.0);
      REQUIRE(wh[g] <= 1.0);
      total[g] += wh[g];
    }
  }
  for (int g = 0; g < npts; ++g) {
    INFO("point " << g);
    REQUIRE_THAT(total[g], Catch::Matchers::WithinAbs(1.0, 1e-12));
  }
}

TEST_CASE("The LKO distance cap changes the answer measurably",
          "[partition][lko]") {
  // partition_lko caps the internuclear distance at 5 bohr. This puts
  // a number on what that costs, using a geometry with one centre at 7.5
  // bohr so the cap actually engages.
  const auto centers = make_centers();
  const int npts = 400;
  const auto pts = make_points(npts);
  const auto own = make_owner(npts, static_cast<int>(centers.extent(0)));

  auto w_exact = ones(npts);
  partition_becke_exact(centers, pts, own, w_exact);

  auto w_lko = ones(npts);
  partition_lko(centers, pts, own, w_lko);

  const auto a = host(w_exact);
  const auto b = host(w_lko);

  double worst = 0.0;
  int worst_at = 0;
  for (int g = 0; g < npts; ++g)
    if (std::abs(a[g] - b[g]) > worst) {
      worst = std::abs(a[g] - b[g]);
      worst_at = g;
    }

  INFO("worst weight difference " << worst << " at point " << worst_at
                                  << " (exact " << a[worst_at] << ", capped "
                                  << b[worst_at] << ")");
  // The point of the test is the magnitude, not a pass/fail: record that the
  // cap is a real approximation rather than a rounding detail, and fail if
  // it ever becomes either negligible (cap removed silently) or wild.
  //
  // Measured on this geometry: 0.146 absolute, an individual point weight
  // moving from 0.618 to 0.764 -- a 24 % relative change. Integrated over a
  // grid much of that cancels, but it is not a rounding detail, and it is
  // the reason the capped scheme is named LKO here rather than Becke.
  REQUIRE(worst > 1e-6);
  REQUIRE(worst < 1.0);
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
