/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Parity with ERKALE, not merely self-consistency.
//
// Everything else in the GTO suite validates NuKEXC against its own closed
// forms. That catches implementation slips but not a shared misunderstanding
// of the conventions -- if the evaluator, the reference and the normalization
// all agreed on the wrong phase or ordering, every test would still pass.
//
// Here the reference comes from ERKALE itself: tests/reference/*.ref carries
// both the basis (shells, exponents, contraction-major coefficients, centers)
// and the analytic overlap that ERKALE computes for it. Reading the basis from
// the same file as the matrix removes any doubt that the two codes were given
// the same basis, which is the easiest way to make such a comparison
// meaningless.
//
// See tests/reference/README.md for how the file is regenerated.

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/grid.hpp>
#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/density.hpp>
#include <nukexc/molecule.hpp>

#include "erkale_ref_reader.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

using Reference = test::ErkaleReference;

/// The parser is shared with test_erkale_xc.cxx; see erkale_ref_reader.hpp.
/// The overlap is a named block in the file, read as nbf x nbf row-major.
Reference load_reference(const std::string &path) {
  auto r = test::load_erkale_reference(path);
  r.matrix("overlap"); // fail here rather than at the first element access
  return r;
}

const char *kRef = "reference/water_ccpvdz_erkale.ref";
const char *kRef631 = "reference/water_631gdp_erkale.ref";

} // namespace

TEST_CASE("The reference basis round-trips", "[erkale][parity]") {
  const auto ref = load_reference(kRef);
  // water / cc-pVDZ: O gives 3s2p1d = 14, each H 2s1p = 5.
  REQUIRE(ref.nbf == 24);
  REQUIRE(ref.basis.nshells() == 11);
  REQUIRE(ref.basis.max_l() == 2);
  REQUIRE(static_cast<int>(ref.block("overlap").size()) == 24 * 24);

  // cc-pVDZ oxygen s is generally contracted: one exponent set, two columns.
  bool found_general = false;
  for (int i = 0; i < ref.basis.nshells(); ++i)
    if (ref.basis.shell(i).nctr() > 1)
      found_general = true;
  REQUIRE(found_general);
}

TEST_CASE("Normalization reproduces ERKALE's overlap diagonal",
          "[erkale][parity]") {
  // The diagonal alone pins down the contraction normalization and, for the
  // spherical shells, the 1/sqrt(S(0)) rescaling -- independently of any
  // two-center machinery.
  const auto ref = load_reference(kRef);
  const int n = ref.nbf;

  int k = 0;
  for (int s = 0; s < ref.basis.nshells(); ++s) {
    const auto norms = ref.basis.shell(s).function_norms();
    for (double v : norms) {
      INFO("function " << k);
      REQUIRE_THAT(v, Catch::Matchers::WithinAbs(
                          ref.block("overlap")[static_cast<size_t>(k) * n + k], 1e-12));
      ++k;
    }
  }
  REQUIRE(k == n);
}

TEST_CASE("Quadrature overlap reproduces ERKALE's analytic overlap",
          "[erkale][parity][grid]") {
  const auto ref = load_reference(kRef);
  const int nbf = ref.nbf;
  const auto dev = make_gto_device_basis(ref.basis);

  // The molecule the reference was generated for, in bohr -- taken from the
  // shell centers so the grid sits on exactly the nuclei ERKALE used.
  Molecule mol;
  read_xyz("reference/water.xyz", mol);

  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 120, 41);
  const int npts = static_cast<int>(grid.quad_points.extent(0));

  Kokkos::View<double ***> comp("comp", 1, nbf, npts);
  fill_collocation_gto<GTODeriv::Value>(ExecSpace{}, dev, grid.quad_points,
                                        comp);
  Kokkos::fence();

  Kokkos::View<double **> S("S", nbf, nbf);
  auto w = grid.weights;
  Kokkos::parallel_for(
      "overlap",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nbf, nbf}),
      KOKKOS_LAMBDA(const int i, const int j) {
        double acc = 0.0;
        for (int g = 0; g < npts; ++g)
          acc += w(g) * comp(0, i, g) * comp(0, j, g);
        S(i, j) = acc;
      });
  Kokkos::fence();
  auto S_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, S);

  double worst = 0.0;
  int wi = 0, wj = 0;
  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j < nbf; ++j) {
      const double d =
          std::abs(S_h(i, j) - ref.block("overlap")[static_cast<size_t>(i) * nbf + j]);
      if (d > worst) {
        worst = d;
        wi = i;
        wj = j;
      }
    }

  INFO("worst element (" << wi << "," << wj << "): NuKEXC " << S_h(wi, wj)
                         << " vs ERKALE "
                         << ref.block("overlap")[static_cast<size_t>(wi) * nbf + wj]
                         << ", deviation " << worst);
  // Quadrature error, not a convention mismatch: a wrong phase, ordering or
  // normalization would show up at O(1), not at this level.
  REQUIRE(worst < 1e-6);
}

TEST_CASE("Water density integrates to the electron count",
          "[erkale][parity][grid][density]") {
  // 6-31G(d,p) -- the polarized 6-31G family; ERKALE's basis library has no
  // bare 6-31G*, and 6-31G(d,p) is a superset of it.
  //
  // No SCF is needed to make this a real test. For ANY set of S-orthonormal
  // vectors c_i, the density rho = sum_i occ_i |sum_mu c_mu,i phi_mu|^2
  // integrates exactly to sum_i occ_i, because
  //
  //     integral rho = sum_i occ_i c_i^T S c_i = sum_i occ_i.
  //
  // So orthonormalizing arbitrary vectors against ERKALE's overlap gives an
  // exactly known target, and any deviation is quadrature error in NuKEXC's
  // grid, collocation and density contraction -- which is precisely what this
  // is meant to measure.
  const auto ref = load_reference(kRef631);
  const int nbf = ref.nbf;
  REQUIRE(nbf == 24);

  const int nocc = 5; // water has 10 electrons
  const double occ = 2.0;

  // Modified Gram-Schmidt in the S metric, on a deterministic starting set.
  std::vector<std::vector<double>> C(nocc, std::vector<double>(nbf, 0.0));
  auto Sdot = [&](const std::vector<double> &a, const std::vector<double> &b) {
    double acc = 0.0;
    for (int i = 0; i < nbf; ++i)
      for (int j = 0; j < nbf; ++j)
        acc += a[i] * ref.block("overlap")[static_cast<size_t>(i) * nbf + j] * b[j];
    return acc;
  };
  for (int k = 0; k < nocc; ++k) {
    for (int i = 0; i < nbf; ++i)
      C[k][i] = std::sin(0.7 * (k + 1) * (i + 1)) + 0.3 * std::cos(0.11 * i);
    for (int m = 0; m < k; ++m) {
      const double proj = Sdot(C[m], C[k]);
      for (int i = 0; i < nbf; ++i)
        C[k][i] -= proj * C[m][i];
    }
    const double nrm = std::sqrt(Sdot(C[k], C[k]));
    REQUIRE(nrm > 1e-8);
    for (int i = 0; i < nbf; ++i)
      C[k][i] /= nrm;
  }
  // confirm the starting point really is orthonormal in the S metric
  for (int a = 0; a < nocc; ++a)
    for (int b = 0; b < nocc; ++b) {
      INFO("orbital pair " << a << "," << b);
      REQUIRE_THAT(Sdot(C[a], C[b]),
                   Catch::Matchers::WithinAbs(a == b ? 1.0 : 0.0, 1e-10));
    }

  DeviceView2DLeft mo("mo", nbf, nocc);
  auto mo_h = Kokkos::create_mirror_view(mo);
  for (int i = 0; i < nbf; ++i)
    for (int k = 0; k < nocc; ++k)
      mo_h(i, k) = C[k][i];
  Kokkos::deep_copy(mo, mo_h);

  DeviceView1D occv("occ", nocc);
  Kokkos::deep_copy(occv, occ);

  Molecule mol;
  read_xyz("reference/water.xyz", mol);
  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 120, 41);

  // Through the generic evaluator path, so this exercises the concept too.
  const GTOEvaluator ev(make_gto_device_basis(ref.basis));
  DeviceView1D rho = compute_density(ev, grid, mo, occv);
  Kokkos::fence();

  double nel = 0.0;
  auto w = grid.weights;
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  Kokkos::parallel_reduce(
      "integrate rho", Kokkos::RangePolicy<ExecSpace>(0, npts),
      KOKKOS_LAMBDA(const int g, double &acc) { acc += w(g) * rho(g); }, nel);

  INFO("integrated density " << nel << " vs exact " << occ * nocc
                             << ", deviation " << std::abs(nel - occ * nocc));
  REQUIRE_THAT(nel, Catch::Matchers::WithinAbs(occ * nocc, 1e-6));
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
