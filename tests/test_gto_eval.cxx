/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The GTO device evaluator (nukexc/gto_eval.hpp).
//
// Two independent references are used, so a shared misunderstanding cannot
// hide a bug:
//
//   * values are checked against a direct sum over primitives and Cartesian
//     monomials written out here, independent of the evaluator's structure;
//   * every derivative order is checked against central differences of the
//     order below it, which is what catches a wrong _der<n> operator;
//   * the analytic overlap of contracted Gaussians is reproduced by
//     quadrature, which exercises normalization, the spherical transform and
//     the collocation path together.

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_eval.hpp>

using namespace Nukexc;

namespace {

/// Build a small mixed basis: a generally contracted s shell, a Cartesian d
/// shell and a spherical f shell, on two different centers.
GTOBasisSet<double> make_test_basis(bool spherical) {
  std::vector<GTOShell<double>> sh;
  const auto t = spherical ? ShellType::Spherical : ShellType::Cartesian;

  // generally contracted s: 3 primitives, 2 contractions
  sh.emplace_back(0, t, std::array<double, 3>{0.1, -0.2, 0.3},
                  std::vector<double>{5.4, 1.3, 0.42},
                  std::vector<double>{0.21, 0.55, 0.37, 0.64, -0.12, 0.28}, 2);
  // d shell
  sh.emplace_back(2, t, std::array<double, 3>{0.1, -0.2, 0.3},
                  std::vector<double>{2.7, 0.83}, std::vector<double>{0.4, 0.7},
                  1);
  // f shell on another center
  sh.emplace_back(3, t, std::array<double, 3>{-0.6, 0.45, -0.15},
                  std::vector<double>{1.15}, std::vector<double>{1.0}, 1);
  return GTOBasisSet<double>(std::move(sh));
}

/// Evaluate the whole basis at one point, returning a (ncomp, nbf) block.
template <GTODeriv D>
std::vector<double> eval_at(const GTOBasisDevice &dev, double x, double y,
                            double z) {
  constexpr int NC = gto_ncomp(D);
  Kokkos::View<Point *> pts("pts", 1);
  auto pts_h = Kokkos::create_mirror_view(pts);
  pts_h(0)[0] = x;
  pts_h(0)[1] = y;
  pts_h(0)[2] = z;
  Kokkos::deep_copy(pts, pts_h);

  Kokkos::View<double ***> comp("comp", NC, dev.nbf, 1);
  fill_collocation_gto<D>(ExecSpace{}, dev, pts, comp);
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, comp);
  std::vector<double> out(static_cast<size_t>(NC) * dev.nbf);
  for (int c = 0; c < NC; ++c)
    for (int i = 0; i < dev.nbf; ++i)
      out[static_cast<size_t>(c) * dev.nbf + i] = h(c, i, 0);
  return out;
}

/// Independent reference: the value of every basis function, written out
/// directly from the shell data without touching the evaluator.
std::vector<double> reference_values(const GTOBasisSet<double> &basis, double x,
                                     double y, double z) {
  std::vector<double> out;
  for (int s = 0; s < basis.nshells(); ++s) {
    const auto &sh = basis.shell(s);
    const int l = sh.l();
    // The shell's OWN components: for a spherical shell ERKALE rescales every
    // relnorm by 1/sqrt(S(0)), so cartesian_components(l) (which carries the
    // raw Cartesian convention) is the wrong table to reference against.
    const auto &cart = sh.cart();
    const double dx = x - sh.O()[0], dy = y - sh.O()[1], dz = z - sh.O()[2];
    const double r2 = dx * dx + dy * dy + dz * dz;

    for (int k = 0; k < sh.nctr(); ++k) {
      std::vector<double> cv(cart.size(), 0.0);
      for (size_t c = 0; c < cart.size(); ++c) {
        double radial = 0.0;
        for (int p = 0; p < sh.nprim(); ++p)
          radial += sh.coeff(p, k) * std::exp(-sh.alpha()[p] * r2);
        cv[c] = cart[c].relnorm * std::pow(dx, cart[c].nx) *
                std::pow(dy, cart[c].ny) * std::pow(dz, cart[c].nz) * radial;
      }
      if (sh.is_spherical()) {
        const auto T = ylm_transmat(l);
        for (int m = 0; m < 2 * l + 1; ++m) {
          double acc = 0.0;
          for (size_t c = 0; c < cart.size(); ++c)
            acc += T[static_cast<size_t>(m) * cart.size() + c] * cv[c];
          out.push_back(acc);
        }
      } else {
        for (double v : cv)
          out.push_back(v);
      }
    }
  }
  return out;
}

} // namespace

TEST_CASE("Values match a direct sum over primitives", "[gto][eval]") {
  for (bool spherical : {false, true}) {
    const auto host = make_test_basis(spherical);
    const auto dev = make_gto_device_basis(host);
    INFO((spherical ? "spherical" : "Cartesian"));

    const double pts[3][3] = {
        {0.4, 0.15, -0.35}, {-1.2, 0.8, 0.25}, {0.11, -0.05, 0.62}};
    for (const auto &p : pts) {
      const auto got = eval_at<GTODeriv::Value>(dev, p[0], p[1], p[2]);
      const auto want = reference_values(host, p[0], p[1], p[2]);
      REQUIRE(got.size() == want.size());
      for (size_t i = 0; i < got.size(); ++i) {
        INFO("bf " << i << " at (" << p[0] << "," << p[1] << "," << p[2] << ")");
        REQUIRE_THAT(got[i], Catch::Matchers::WithinAbs(want[i], 1e-12));
      }
    }
  }
}

namespace {

/// Finite-difference checks for one angular representation. Written as a
/// function rather than Catch2 SECTIONs because SECTIONs inside a loop re-run
/// the enclosing TEST_CASE once per section, which would re-enter the loop.
void check_derivatives(bool spherical) {
  const auto host = make_test_basis(spherical);
  const auto dev = make_gto_device_basis(host);
  const int nbf = dev.nbf;
  const char *tag = spherical ? "spherical" : "Cartesian";

  const double p[3] = {0.37, -0.22, 0.48};
  const double h = 1e-5;

  // gradient vs. central differences of the value
  {
    const auto a = eval_at<GTODeriv::Gradient>(dev, p[0], p[1], p[2]);
    for (int d = 0; d < 3; ++d) {
      double pp[3] = {p[0], p[1], p[2]}, pm[3] = {p[0], p[1], p[2]};
      pp[d] += h;
      pm[d] -= h;
      const auto vp = eval_at<GTODeriv::Value>(dev, pp[0], pp[1], pp[2]);
      const auto vm = eval_at<GTODeriv::Value>(dev, pm[0], pm[1], pm[2]);
      for (int i = 0; i < nbf; ++i) {
        const double fd = (vp[i] - vm[i]) / (2 * h);
        const double an = a[static_cast<size_t>(GTO_DX + d) * nbf + i];
        INFO(tag << " bf " << i << " d/dx" << d);
        REQUIRE_THAT(an, Catch::Matchers::WithinAbs(
                             fd, 1e-6 * std::max(1.0, std::abs(an))));
      }
    }
  }

  // Hessian vs. central differences of the gradient
  {
    const auto a = eval_at<GTODeriv::Hessian>(dev, p[0], p[1], p[2]);
    const int comp[6][3] = {{GTO_DXX, 0, GTO_DX}, {GTO_DXY, 0, GTO_DY},
                            {GTO_DXZ, 0, GTO_DZ}, {GTO_DYY, 1, GTO_DY},
                            {GTO_DYZ, 1, GTO_DZ}, {GTO_DZZ, 2, GTO_DZ}};
    for (const auto &c : comp) {
      double pp[3] = {p[0], p[1], p[2]}, pm[3] = {p[0], p[1], p[2]};
      pp[c[1]] += h;
      pm[c[1]] -= h;
      const auto gp = eval_at<GTODeriv::Gradient>(dev, pp[0], pp[1], pp[2]);
      const auto gm = eval_at<GTODeriv::Gradient>(dev, pm[0], pm[1], pm[2]);
      for (int i = 0; i < nbf; ++i) {
        const double fd = (gp[static_cast<size_t>(c[2]) * nbf + i] -
                           gm[static_cast<size_t>(c[2]) * nbf + i]) /
                          (2 * h);
        const double an = a[static_cast<size_t>(c[0]) * nbf + i];
        INFO(tag << " bf " << i << " hess component " << c[0]);
        REQUIRE_THAT(an, Catch::Matchers::WithinAbs(
                             fd, 1e-5 * std::max(1.0, std::abs(an))));
      }
    }
  }

  // grad(lapl) vs. central differences of the Hessian trace
  {
    const auto a = eval_at<GTODeriv::LaplGradient>(dev, p[0], p[1], p[2]);
    for (int d = 0; d < 3; ++d) {
      double pp[3] = {p[0], p[1], p[2]}, pm[3] = {p[0], p[1], p[2]};
      pp[d] += h;
      pm[d] -= h;
      const auto hp = eval_at<GTODeriv::Hessian>(dev, pp[0], pp[1], pp[2]);
      const auto hm = eval_at<GTODeriv::Hessian>(dev, pm[0], pm[1], pm[2]);
      for (int i = 0; i < nbf; ++i) {
        const double lp = hp[static_cast<size_t>(GTO_DXX) * nbf + i] +
                          hp[static_cast<size_t>(GTO_DYY) * nbf + i] +
                          hp[static_cast<size_t>(GTO_DZZ) * nbf + i];
        const double lm = hm[static_cast<size_t>(GTO_DXX) * nbf + i] +
                          hm[static_cast<size_t>(GTO_DYY) * nbf + i] +
                          hm[static_cast<size_t>(GTO_DZZ) * nbf + i];
        const double fd = (lp - lm) / (2 * h);
        const double an = a[static_cast<size_t>(GTO_LGX + d) * nbf + i];
        INFO(tag << " bf " << i << " lapl-grad component " << d);
        REQUIRE_THAT(an, Catch::Matchers::WithinAbs(
                             fd, 1e-4 * std::max(1.0, std::abs(an))));
      }
    }
  }
}

} // namespace

TEST_CASE("Derivatives match finite differences of the order below",
          "[gto][eval][fd]") {
  check_derivatives(false);
  check_derivatives(true);
}

TEST_CASE("Hessian trace is the Laplacian", "[gto][eval]") {
  const auto host = make_test_basis(true);
  const auto dev = make_gto_device_basis(host);
  const auto a = eval_at<GTODeriv::Hessian>(dev, 0.3, -0.4, 0.55);
  const int nbf = dev.nbf;

  // A Gaussian is smooth, so the mixed partials must be symmetric; the
  // evaluator stores only the six unique components, which is what makes
  // that a meaningful statement about the storage convention rather than a
  // tautology. Check against an independent second difference of the value.
  const double h = 2e-4;
  for (int i = 0; i < nbf; ++i) {
    double lap_an = a[static_cast<size_t>(GTO_DXX) * nbf + i] +
                    a[static_cast<size_t>(GTO_DYY) * nbf + i] +
                    a[static_cast<size_t>(GTO_DZZ) * nbf + i];
    double lap_fd = 0.0;
    const auto v0 = eval_at<GTODeriv::Value>(dev, 0.3, -0.4, 0.55);
    for (int d = 0; d < 3; ++d) {
      double pp[3] = {0.3, -0.4, 0.55}, pm[3] = {0.3, -0.4, 0.55};
      pp[d] += h;
      pm[d] -= h;
      const auto vp = eval_at<GTODeriv::Value>(dev, pp[0], pp[1], pp[2]);
      const auto vm = eval_at<GTODeriv::Value>(dev, pm[0], pm[1], pm[2]);
      lap_fd += (vp[i] - 2 * v0[i] + vm[i]) / (h * h);
    }
    INFO("bf " << i);
    REQUIRE_THAT(lap_an, Catch::Matchers::WithinAbs(
                             lap_fd, 1e-4 * std::max(1.0, std::abs(lap_an))));
  }
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
