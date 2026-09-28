/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Libxc conventions that the XC integrators rely on.
//
// This used to be the upstream libxc example program: it printed a few numbers
// and asserted nothing, so it could not fail. What matters here is not that
// libxc runs but that its output means what nukexc/xc_integrals.hpp assumes:
//
//   * exc is the energy density PER PARTICLE, so the energy is
//     int w rho exc, not int w exc;
//   * vrho is d(rho*exc)/drho and vsigma is d(rho*exc)/dsigma, which is what
//     makes the Z-matrix construction correct;
//   * the polarized layout is interleaved (up, down) per point.
//
// Each is checked against an independent reference: a closed form where one
// exists, finite differences otherwise.

#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>
#include <xc.h>
#include <xc_funcs.h>

namespace {

/// Slater exchange energy density per particle, exact:
///   eps_x = -(3/4) (3 rho / pi)^(1/3)
double slater_exc(double rho) {
  return -0.75 * std::cbrt(3.0 * rho / M_PI);
}

struct Functional {
  xc_func_type f;
  explicit Functional(int id, int nspin) {
    if (xc_func_init(&f, id, nspin) != 0)
      throw std::runtime_error("libxc functional " + std::to_string(id) +
                               " not found");
  }
  ~Functional() { xc_func_end(&f); }
  Functional(const Functional &) = delete;
  Functional &operator=(const Functional &) = delete;
};

} // namespace

TEST_CASE("Libxc is the version we expect to link", "[libxc]") {
  int major = 0, minor = 0, micro = 0;
  xc_version(&major, &minor, &micro);
  INFO("Libxc " << major << "." << minor << "." << micro);
  // NuKEXC needs kxc/lxc for the higher response orders later on, which
  // arrived well before 5.x; guard against linking something ancient.
  REQUIRE(major >= 5);
}

TEST_CASE("Slater exchange matches its closed form", "[libxc][lda]") {
  Functional fn(XC_LDA_X, XC_UNPOLARIZED);

  const std::vector<double> rho = {1e-4, 0.1, 0.2, 0.35, 1.0, 12.5};
  std::vector<double> exc(rho.size()), vrho(rho.size());
  xc_lda_exc_vxc(&fn.f, static_cast<int>(rho.size()), rho.data(), exc.data(),
                 vrho.data());

  for (size_t i = 0; i < rho.size(); ++i) {
    INFO("rho = " << rho[i]);
    // exc is per particle
    REQUIRE_THAT(exc[i], Catch::Matchers::WithinRel(slater_exc(rho[i]), 1e-12));
    // v_x = d(rho eps_x)/drho = (4/3) eps_x
    REQUIRE_THAT(vrho[i], Catch::Matchers::WithinRel(
                              4.0 / 3.0 * slater_exc(rho[i]), 1e-12));
  }
}

TEST_CASE("vrho is the derivative of the energy density", "[libxc][lda]") {
  // Independent of any closed form: finite-difference d(rho*exc)/drho.
  Functional fn(XC_LDA_C_VWN, XC_UNPOLARIZED);

  auto energy_density = [&](double r) {
    double e = 0.0;
    xc_lda_exc(&fn.f, 1, &r, &e);
    return r * e;
  };

  for (double rho : {0.05, 0.3, 1.7}) {
    double exc = 0.0, vrho = 0.0;
    xc_lda_exc_vxc(&fn.f, 1, &rho, &exc, &vrho);

    const double h = 1e-6 * rho;
    const double fd =
        (energy_density(rho + h) - energy_density(rho - h)) / (2 * h);

    INFO("rho = " << rho);
    REQUIRE_THAT(vrho, Catch::Matchers::WithinRel(fd, 1e-6));
  }
}

TEST_CASE("GGA vrho and vsigma are the derivatives we assume",
          "[libxc][gga]") {
  Functional fn(XC_GGA_X_PBE, XC_UNPOLARIZED);

  auto energy_density = [&](double r, double s) {
    double e = 0.0;
    xc_gga_exc(&fn.f, 1, &r, &s, &e);
    return r * e;
  };

  const double cases[3][2] = {{0.08, 0.02}, {0.4, 0.3}, {2.0, 5.0}};
  for (const auto &c : cases) {
    const double rho = c[0], sigma = c[1];
    double exc = 0.0, vrho = 0.0, vsigma = 0.0;
    xc_gga_exc_vxc(&fn.f, 1, &rho, &sigma, &exc, &vrho, &vsigma);

    const double hr = 1e-6 * rho;
    const double fd_rho =
        (energy_density(rho + hr, sigma) - energy_density(rho - hr, sigma)) /
        (2 * hr);
    const double hs = 1e-6 * sigma;
    const double fd_sigma =
        (energy_density(rho, sigma + hs) - energy_density(rho, sigma - hs)) /
        (2 * hs);

    INFO("rho = " << rho << ", sigma = " << sigma);
    REQUIRE_THAT(vrho, Catch::Matchers::WithinRel(fd_rho, 1e-6));
    REQUIRE_THAT(vsigma, Catch::Matchers::WithinRel(fd_sigma, 1e-6));
  }
}

TEST_CASE("Polarized arrays are interleaved and obey spin scaling",
          "[libxc][polarized]") {
  // compute_lsda/compute_gga_lsda interleave (up, down) per point by hand, so
  // the layout is load-bearing. Exchange also satisfies the spin-scaling
  // relation Ex[ra, rb] = (Ex[2ra] + Ex[2rb]) / 2, which pins the ordering
  // down independently: swapping the channels must swap vrho, not change exc.
  Functional pol(XC_LDA_X, XC_POLARIZED);
  Functional unpol(XC_LDA_X, XC_UNPOLARIZED);

  const double ra = 0.3, rb = 0.1;
  double rho_pol[2] = {ra, rb};
  double exc = 0.0, vrho[2] = {0.0, 0.0};
  xc_lda_exc_vxc(&pol.f, 1, rho_pol, &exc, vrho);

  // spin scaling, via the unpolarized functional at doubled densities
  double r2a = 2 * ra, r2b = 2 * rb, ea = 0.0, eb = 0.0;
  xc_lda_exc(&unpol.f, 1, &r2a, &ea);
  xc_lda_exc(&unpol.f, 1, &r2b, &eb);
  // exc is per particle, so weight the two channels by their densities
  const double expect = (2 * ra * ea + 2 * rb * eb) / (2 * (ra + rb));
  REQUIRE_THAT(exc, Catch::Matchers::WithinRel(expect, 1e-12));

  // each channel's potential is the closed-form one at twice its density
  REQUIRE_THAT(vrho[0], Catch::Matchers::WithinRel(
                            4.0 / 3.0 * slater_exc(2 * ra), 1e-12));
  REQUIRE_THAT(vrho[1], Catch::Matchers::WithinRel(
                            4.0 / 3.0 * slater_exc(2 * rb), 1e-12));

  // swapping the channels swaps the potentials and leaves the energy alone
  double swapped[2] = {rb, ra};
  double exc_s = 0.0, vrho_s[2] = {0.0, 0.0};
  xc_lda_exc_vxc(&pol.f, 1, swapped, &exc_s, vrho_s);
  REQUIRE_THAT(exc_s, Catch::Matchers::WithinRel(exc, 1e-14));
  REQUIRE_THAT(vrho_s[0], Catch::Matchers::WithinRel(vrho[1], 1e-14));
  REQUIRE_THAT(vrho_s[1], Catch::Matchers::WithinRel(vrho[0], 1e-14));
}

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
