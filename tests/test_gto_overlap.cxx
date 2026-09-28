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

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

double binom(int n, int k) {
  if (k < 0 || k > n)
    return 0.0;
  double r = 1.0;
  for (int i = 0; i < k; ++i)
    r = r * (n - i) / (i + 1);
  return std::round(r);
}

double dfact_odd(int n) { // (n)!! for odd n, with (-1)!! = 1
  double r = 1.0;
  for (int k = n; k > 1; k -= 2)
    r *= k;
  return r;
}

/// f_j(la, lb, a, b) = sum_k C(la,k) C(lb,j-k) a^(la-k) b^(lb-j+k)
double overlap_f(int j, int la, int lb, double a, double b) {
  double acc = 0.0;
  for (int k = std::max(0, j - lb); k <= std::min(j, la); ++k)
    acc += binom(la, k) * binom(lb, j - k) * std::pow(a, la - k) *
           std::pow(b, lb - j + k);
  return acc;
}

/// One-dimensional primitive overlap factor.
double overlap_1d(int la, int lb, double pa, double pb, double gamma) {
  double acc = 0.0;
  for (int i = 0; i <= (la + lb) / 2; ++i)
    acc += overlap_f(2 * i, la, lb, pa, pb) * dfact_odd(2 * i - 1) /
           std::pow(2.0 * gamma, i);
  return acc * std::sqrt(M_PI / gamma);
}

/// Closed-form overlap matrix of a GTO basis, in the library's function order.
std::vector<double> analytic_overlap(const GTOBasisSet<double> &basis) {
  const int nbf = basis.nbf();
  std::vector<double> S(static_cast<size_t>(nbf) * nbf, 0.0);

  for (int is = 0; is < basis.nshells(); ++is) {
    const auto &A = basis.shell(is);
    const auto &cA = A.cart();
    const int la = A.l(), nca = static_cast<int>(cA.size());
    const auto TA = ylm_transmat(la);

    for (int js = 0; js < basis.nshells(); ++js) {
      const auto &B = basis.shell(js);
      const auto &cB = B.cart();
      const int lb = B.l(), ncb = static_cast<int>(cB.size());
      const auto TB = ylm_transmat(lb);

      const double dx = A.O()[0] - B.O()[0], dy = A.O()[1] - B.O()[1],
                   dz = A.O()[2] - B.O()[2];
      const double R2 = dx * dx + dy * dy + dz * dz;

      for (int ka = 0; ka < A.nctr(); ++ka)
        for (int kb = 0; kb < B.nctr(); ++kb) {

          // Cartesian block for this contraction pair.
          std::vector<double> blk(static_cast<size_t>(nca) * ncb, 0.0);
          for (int pa = 0; pa < A.nprim(); ++pa)
            for (int pb = 0; pb < B.nprim(); ++pb) {
              const double al = A.alpha()[pa], be = B.alpha()[pb];
              const double gamma = al + be;
              const double pref = std::exp(-al * be / gamma * R2) *
                                  A.coeff(pa, ka) * B.coeff(pb, kb);
              // Gaussian product center
              const double Px = (al * A.O()[0] + be * B.O()[0]) / gamma;
              const double Py = (al * A.O()[1] + be * B.O()[1]) / gamma;
              const double Pz = (al * A.O()[2] + be * B.O()[2]) / gamma;

              for (int ia = 0; ia < nca; ++ia)
                for (int ib = 0; ib < ncb; ++ib) {
                  const double sx =
                      overlap_1d(cA[ia].nx, cB[ib].nx, Px - A.O()[0],
                                 Px - B.O()[0], gamma);
                  const double sy =
                      overlap_1d(cA[ia].ny, cB[ib].ny, Py - A.O()[1],
                                 Py - B.O()[1], gamma);
                  const double sz =
                      overlap_1d(cA[ia].nz, cB[ib].nz, Pz - A.O()[2],
                                 Pz - B.O()[2], gamma);
                  blk[static_cast<size_t>(ia) * ncb + ib] +=
                      pref * sx * sy * sz;
                }
            }

          // relative normalizations
          for (int ia = 0; ia < nca; ++ia)
            for (int ib = 0; ib < ncb; ++ib)
              blk[static_cast<size_t>(ia) * ncb + ib] *=
                  cA[ia].relnorm * cB[ib].relnorm;

          // project to the angular representation of each shell
          const int na = A.is_spherical() ? 2 * la + 1 : nca;
          const int nb = B.is_spherical() ? 2 * lb + 1 : ncb;
          const int r0 = basis.shell_offset(is) + ka * na;
          const int c0 = basis.shell_offset(js) + kb * nb;

          for (int a = 0; a < na; ++a)
            for (int b = 0; b < nb; ++b) {
              double acc = 0.0;
              for (int ia = 0; ia < nca; ++ia) {
                const double ta =
                    A.is_spherical()
                        ? TA[static_cast<size_t>(a) * nca + ia]
                        : (ia == a ? 1.0 : 0.0);
                if (ta == 0.0)
                  continue;
                for (int ib = 0; ib < ncb; ++ib) {
                  const double tb =
                      B.is_spherical()
                          ? TB[static_cast<size_t>(b) * ncb + ib]
                          : (ib == b ? 1.0 : 0.0);
                  acc += ta * tb * blk[static_cast<size_t>(ia) * ncb + ib];
                }
              }
              S[static_cast<size_t>(r0 + a) * nbf + (c0 + b)] = acc;
            }
        }
    }
  }
  return S;
}

/// A small STO-3G-like water basis, built through the caller-array API.
GTOBasisSet<double> water_basis(const Molecule &mol, ShellType type) {
  auto cen = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.atom_centers);
  auto Z = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.Z);

  std::vector<GTOShell<double>> sh;
  for (size_t i = 0; i < mol.natoms; ++i) {
    const std::array<double, 3> O = {cen(i)[0], cen(i)[1], cen(i)[2]};
    if (Z(i) == 1) {
      sh.emplace_back(0, type, O, std::vector<double>{3.425250914, 0.6239137298,
                                                      0.1688554040},
                      std::vector<double>{0.1543289673, 0.5353281423,
                                          0.4446345422},
                      1);
    } else {
      // O: 1s, and a generally contracted 2s/2p sharing one exponent set
      sh.emplace_back(0, type, O,
                      std::vector<double>{130.7093214, 23.80886605,
                                          6.443608313},
                      std::vector<double>{0.1543289673, 0.5353281423,
                                          0.4446345422},
                      1);
      sh.emplace_back(0, type, O,
                      std::vector<double>{5.033151319, 1.169596125,
                                          0.3803889600},
                      std::vector<double>{-0.09996722919, 0.3995128261,
                                          0.7001154689},
                      1);
      sh.emplace_back(1, type, O,
                      std::vector<double>{5.033151319, 1.169596125,
                                          0.3803889600},
                      std::vector<double>{0.1559162750, 0.6076837186,
                                          0.3919573931},
                      1);
    }
  }
  return GTOBasisSet<double>(std::move(sh));
}

} // namespace

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
    const auto host = water_basis(mol, type);
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
