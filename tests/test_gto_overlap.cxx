/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// End-to-end check of the Gaussian path: the overlap matrix obtained by
// numerical quadrature on a molecular grid must reproduce the closed form.
//
// This is the test that ties Phase 1 together. It exercises, in one number,
//   * the contraction normalization,
//   * the Cartesian component ordering and relative normalization,
//   * the Cartesian-to-spherical transform,
//   * the device evaluator,
//   * the Becke-partitioned atomic grid and its weights.
//
// The closed form is the standard Gaussian product theorem, written out here
// so it shares no code with the evaluator:
//
//   <a|b> = exp(-mu R_AB^2) * Sx * Sy * Sz
//   mu    = alpha*beta/(alpha+beta),  gamma = alpha+beta
//   Sx    = sum_{i=0}^{(la+lb)/2} f_{2i}(la,lb,PAx,PBx) (2i-1)!! / (2 gamma)^i
//                                 * sqrt(pi/gamma)
//   f_j   = sum_k C(la,k) C(lb,j-k) PAx^(la-k) PBx^(lb-j+k)

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/grid.hpp>
#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/molecule.hpp>

#include "gto_analytic_overlap.hpp"
#include "test_basis_sets.hpp"

using namespace Nukexc;
using Nukexc::test::analytic_overlap;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;



TEST_CASE("Same-center overlap diagonal is unity", "[gto][overlap]") {
  // Cheap guard on the closed form itself before it is used as a reference.
  for (auto type : {ShellType::Cartesian, ShellType::Spherical}) {
    std::vector<GTOShell<double>> sh;
    sh.emplace_back(2, type, std::array<double, 3>{0, 0, 0},
                    std::vector<double>{2.7, 0.83},
                    std::vector<double>{0.4, 0.7}, 1);
    const GTOBasisSet<double> b(std::move(sh));
    const auto S = analytic_overlap(b);
    const auto n = b.shell(0).function_norms();
    for (int i = 0; i < b.nbf(); ++i) {
      INFO("function " << i);
      REQUIRE_THAT(S[static_cast<size_t>(i) * b.nbf() + i],
                   Catch::Matchers::WithinRel(n[static_cast<size_t>(i)], 1e-12));
    }
  }
}

TEST_CASE("Quadrature overlap reproduces the closed form",
          "[gto][overlap][grid]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);

  for (auto type : {ShellType::Cartesian, ShellType::Spherical}) {
    const auto host = test::sto3g_basis(mol, type);
    const auto dev = make_gto_device_basis(host);
    const int nbf = dev.nbf;
    INFO((type == ShellType::Spherical ? "spherical" : "Cartesian"));

    const auto Sref = analytic_overlap(host);

    // A tight grid: the integrand is a product of two Gaussians, so this is
    // about resolving the cusp region, not high angular momentum.
    // The limiting error here is ANGULAR, not radial: the worst element is an
    // O-H overlap between functions on different centers, and it converges
    // with the Lebedev order while being flat in nrad. Measured worst
    // deviation at nrad = 100:
    //
    //     L = 11   9.3e-04        L = 29   2.7e-07
    //     L = 17   2.0e-05        L = 35   3.8e-07
    //     L = 23   2.9e-06        L = 41   1.6e-08
    //
    // (the wobble at 29-35 is the usual non-monotonicity across Lebedev
    // orders). L = 41 with 100 radial points runs in about 1.4 s.
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 100, 41);
    const int npts = static_cast<int>(grid.quad_points.extent(0));

    Kokkos::View<double ***> comp("comp", 1, nbf, npts);
    fill_collocation_gto<GTODeriv::Value>(ExecSpace{}, dev, grid.quad_points,
                                          comp);
    Kokkos::fence();

    // S_ij = sum_g w_g phi_i(g) phi_j(g)
    Kokkos::View<double **> S("S", nbf, nbf);
    auto w = grid.weights;
    Kokkos::parallel_for(
        "overlap", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                       {0, 0}, {nbf, nbf}),
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
            std::abs(S_h(i, j) - Sref[static_cast<size_t>(i) * nbf + j]);
        if (d > worst) {
          worst = d;
          wi = i;
          wj = j;
        }
      }
    INFO("worst element (" << wi << "," << wj << "): quadrature "
                           << S_h(wi, wj) << " vs analytic "
                           << Sref[static_cast<size_t>(wi) * nbf + wj]);
    REQUIRE(worst < 1e-7);
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
