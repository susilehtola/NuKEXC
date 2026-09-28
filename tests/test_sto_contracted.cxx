/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The contracted STO path (nukexc/sto_contracted_basis.hpp).
//
// The key property to pin down is that it agrees with the primitive path
// (nukexc/stobasis.hpp) wherever the two describe the same function, since a
// one-primitive contraction *is* a primitive. Everything beyond that -- shared
// angular factors over primitives with different n -- is what the primitive
// path cannot express, and is checked against direct evaluation of the sum.

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <nukexc/sto_contracted_basis.hpp>
#include <nukexc/stobasis.hpp>

#include "sto_basis_reader.hpp"

using namespace Nukexc;

namespace {

/// Evaluate one contracted basis function at a point, on the device.
double eval_contracted(const STOContractedBasisSet &b, int i, double x,
                       double y, double z) {
  Kokkos::View<double *> out("out", 1);
  Kokkos::parallel_for(
      "eval", Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int) {
        const auto sh = load_contracted_shell(b, i);
        out(0) = contracted_basis_eval(b, sh, x, y, z);
      });
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);
  return h(0);
}

/// Evaluate value and gradient of one contracted basis function.
void eval_contracted_grad(const STOContractedBasisSet &b, int i, double x,
                          double y, double z, double &v, double &gx,
                          double &gy, double &gz) {
  Kokkos::View<double *> out("out", 4);
  Kokkos::parallel_for(
      "eval_grad", Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int) {
        const auto sh = load_contracted_shell(b, i);
        double vv, a, c, d;
        contracted_basis_eval_with_grad(b, sh, x, y, z, vv, a, c, d);
        out(0) = vv;
        out(1) = a;
        out(2) = c;
        out(3) = d;
      });
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);
  v = h(0);
  gx = h(1);
  gy = h(2);
  gz = h(3);
}

/// Evaluate one primitive basis function, for the cross-check.
double eval_primitive(const STOBasisSet &b, int i, double x, double y,
                      double z) {
  Kokkos::View<double *> out("out", 1);
  Kokkos::parallel_for(
      "eval_prim", Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int) {
        const auto sh = load_shell(b, i);
        out(0) = basis_eval_fast(sh, x, y, z);
      });
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);
  return h(0);
}

} // namespace

TEST_CASE("A one-primitive contraction equals the primitive path",
          "[sto][contracted]") {
  const int n = 2, l = 1, m = -1;
  const double zeta = 1.7;
  const double cen[3] = {0.3, -0.2, 0.5};

  // primitive path
  const STOBasisSet prim =
      make_manual_basis({{n, l, m, zeta, cen[0], cen[1], cen[2]}});

  // contracted path, a single unit coefficient
  const int off[2] = {0, 1};
  const double one = 1.0;
  STOContractedBasisInput in;
  in.nbf = 1;
  in.l = &l;
  in.m = &m;
  in.centers = cen;
  in.prim_offset = off;
  in.n = &n;
  in.zeta = &zeta;
  in.coeff = &one;
  const STOContractedBasisSet con = make_sto_contracted_basis(in);

  REQUIRE(con.nbf() == 1);
  REQUIRE(con.nprim() == 1);

  const double pts[4][3] = {
      {0.0, 0.0, 0.0}, {1.0, 0.5, -0.3}, {-2.0, 0.1, 1.4}, {0.31, -0.19, 0.52}};
  for (const auto &p : pts) {
    const double a = eval_primitive(prim, 0, p[0], p[1], p[2]);
    const double b = eval_contracted(con, 0, p[0], p[1], p[2]);
    REQUIRE_THAT(b, Catch::Matchers::WithinRel(a, 1e-13) ||
                        Catch::Matchers::WithinAbs(a, 1e-300));
  }

  // The two paths should also agree on the screening radius.
  auto rc_prim =
      Kokkos::create_mirror_view_and_copy(HostSpace{}, prim.cutoff_radii);
  auto rc_con =
      Kokkos::create_mirror_view_and_copy(HostSpace{}, con.cutoff_radii);
  REQUIRE_THAT(rc_con(0), Catch::Matchers::WithinRel(rc_prim(0), 1e-6));
}

TEST_CASE("Contracted functions are normalized", "[sto][contracted]") {
  // Primitives with *different* n sharing one angular factor -- the case the
  // primitive path cannot represent.
  const int l = 0, m = 0;
  const int n[4] = {2, 1, 2, 1};
  const double zeta[4] = {24.975698, 12.554759, 10.357096, 7.071190};
  const double c[4] = {-0.0004756, -0.1852813, -0.1195096, -0.6980503};
  const int off[2] = {0, 4};
  const double cen[3] = {0.0, 0.0, 0.0};

  STOContractedBasisInput in;
  in.nbf = 1;
  in.l = &l;
  in.m = &m;
  in.centers = cen;
  in.prim_offset = off;
  in.n = n;
  in.zeta = zeta;
  in.coeff = c;
  const STOContractedBasisSet b = make_sto_contracted_basis(in);

  // <chi|chi> = int R(r)^2 r^2 dr, since the angular factor is unit-normalized.
  // Simpson on [0, 40] is far past the tail for these exponents.
  auto zeta_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.zeta);
  auto n_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.n);
  auto a_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.coeff);

  auto R = [&](double r) {
    double acc = 0.0;
    for (size_t p = 0; p < b.nprim(); ++p)
      acc += a_h(p) * std::pow(r, n_h(p) - 1) * std::exp(-zeta_h(p) * r);
    return acc;
  };

  const int N = 400000;
  const double rmax = 40.0, h = rmax / N;
  double integral = 0.0;
  for (int i = 0; i <= N; ++i) {
    const double r = i * h;
    const double f = R(r) * R(r) * r * r;
    const double w = (i == 0 || i == N) ? 1.0 : ((i % 2) ? 4.0 : 2.0);
    integral += w * f;
  }
  integral *= h / 3.0;

  REQUIRE_THAT(integral, Catch::Matchers::WithinRel(1.0, 1e-9));
}

TEST_CASE("Contracted gradient matches finite differences",
          "[sto][contracted]") {
  const int l = 1, m = 1;
  const int n[3] = {2, 3, 2};
  const double zeta[3] = {3.4, 1.9, 0.87};
  const double c[3] = {0.31, -0.44, 0.62};
  const int off[2] = {0, 3};
  const double cen[3] = {-0.15, 0.25, 0.4};

  STOContractedBasisInput in;
  in.nbf = 1;
  in.l = &l;
  in.m = &m;
  in.centers = cen;
  in.prim_offset = off;
  in.n = n;
  in.zeta = zeta;
  in.coeff = c;
  const STOContractedBasisSet b = make_sto_contracted_basis(in);

  const double pts[3][3] = {
      {0.7, -0.4, 1.1}, {-1.3, 0.9, -0.2}, {0.05, 0.05, 0.05}};

  for (const auto &p : pts) {
    double v, gx, gy, gz;
    eval_contracted_grad(b, 0, p[0], p[1], p[2], v, gx, gy, gz);

    // central differences; h chosen for ~1e-10 truncation on a smooth function
    const double h = 1e-5;
    const double fdx = (eval_contracted(b, 0, p[0] + h, p[1], p[2]) -
                        eval_contracted(b, 0, p[0] - h, p[1], p[2])) /
                       (2 * h);
    const double fdy = (eval_contracted(b, 0, p[0], p[1] + h, p[2]) -
                        eval_contracted(b, 0, p[0], p[1] - h, p[2])) /
                       (2 * h);
    const double fdz = (eval_contracted(b, 0, p[0], p[1], p[2] + h) -
                        eval_contracted(b, 0, p[0], p[1], p[2] - h)) /
                       (2 * h);

    const double scale =
        std::max({1.0, std::abs(gx), std::abs(gy), std::abs(gz)});
    REQUIRE_THAT(gx, Catch::Matchers::WithinAbs(fdx, 1e-7 * scale));
    REQUIRE_THAT(gy, Catch::Matchers::WithinAbs(fdy, 1e-7 * scale));
    REQUIRE_THAT(gz, Catch::Matchers::WithinAbs(fdz, 1e-7 * scale));

    // the value returned alongside the gradient must match the value-only path
    REQUIRE_THAT(v, Catch::Matchers::WithinRel(
                        eval_contracted(b, 0, p[0], p[1], p[2]), 1e-14));
  }
}

TEST_CASE("Malformed contracted input throws", "[sto][contracted]") {
  const int l = 0, m = 0, n = 1;
  const double zeta = 1.0, c = 1.0, cen[3] = {0, 0, 0};
  const int off[2] = {0, 1};

  STOContractedBasisInput ok;
  ok.nbf = 1;
  ok.l = &l;
  ok.m = &m;
  ok.centers = cen;
  ok.prim_offset = off;
  ok.n = &n;
  ok.zeta = &zeta;
  ok.coeff = &c;
  REQUIRE_NOTHROW(make_sto_contracted_basis(ok));

  SECTION("null array") {
    auto bad = ok;
    bad.zeta = nullptr;
    REQUIRE_THROWS_AS(make_sto_contracted_basis(bad), std::invalid_argument);
  }
  SECTION("empty contraction") {
    const int bad_off[2] = {0, 0};
    auto bad = ok;
    bad.prim_offset = bad_off;
    REQUIRE_THROWS_AS(make_sto_contracted_basis(bad), std::invalid_argument);
  }
  SECTION("l >= n") {
    const int bad_l = 1;
    auto bad = ok;
    bad.l = &bad_l;
    REQUIRE_THROWS_AS(make_sto_contracted_basis(bad), std::invalid_argument);
  }
  SECTION("|m| > l") {
    const int bad_m = 3;
    auto bad = ok;
    bad.m = &bad_m;
    REQUIRE_THROWS_AS(make_sto_contracted_basis(bad), std::invalid_argument);
  }
}

TEST_CASE("Thakkar oxygen reads as a contracted set", "[sto][contracted][io]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);

  const auto b = test::load_thakkar_contracted(mol, "input/k99light/neutral");

  // Water: O contributes 1s, 2s (S block, 2 orbitals) and 2p (P block, 1
  // orbital x 3 m) = 5 functions; each H contributes its 1s = 1 function.
  REQUIRE(b.nbf() == 5 + 1 + 1);
  REQUIRE(b.nprim() > b.nbf()); // genuinely contracted

  // Oxygen's S block has 8 primitives shared by both S orbitals.
  auto off = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.prim_offset);
  REQUIRE(off(1) - off(0) == 8);
  REQUIRE(off(2) - off(1) == 8);

  // Normalization survived the read.
  auto zeta_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.zeta);
  auto n_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.n);
  auto a_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.coeff);
  for (int i = 0; i < 2; ++i) {
    double s = 0.0;
    for (int p = off(i); p < off(i + 1); ++p)
      for (int q = off(i); q < off(i + 1); ++q) {
        const int npq = n_h(p) + n_h(q);
        s += a_h(p) * a_h(q) *
             static_cast<double>(factorial(npq)) /
             std::pow(zeta_h(p) + zeta_h(q), npq + 1);
      }
    REQUIRE_THAT(s, Catch::Matchers::WithinAbs(1.0, 1e-10));
  }

  // A missing basis directory must be loud.
  REQUIRE_THROWS_AS(test::load_thakkar_contracted(mol, "input/does-not-exist"),
                    std::runtime_error);
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
