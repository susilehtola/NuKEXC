/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The generated XC chain rule, against a numerical derivative of the energy.
//
// WHY THIS TEST EXISTS
// --------------------
// Every other check on the kernel-assembled XC matrix compares it against
// something: the hand-written LDA and GGA assembly, the nesting of the
// families into one another, the closed-shell identity between the
// restricted and unrestricted channels, and ERKALE. Each of those is worth
// having, and none of them answers the one question that matters most.
//
// The XC matrix is a derivative,
//
//     V^s_uv = d Exc / d P^s_uv,
//
// and libxckernel derives that derivative symbolically and emits the
// contraction for it. Comparing two codes that both call libxckernel cannot
// tell whether the derivation is right -- and since ERKALE's AngularGrid now
// routes its own assembly through xck_contract, the ERKALE comparison in
// test_erkale_xc.cxx is exactly that: an excellent check on the ingredients
// and the wiring, and no check at all on the chain rule itself. See the
// caveat section of tests/reference/README.md.
//
// A numerical derivative needs no second implementation of the derivative.
// It needs only the energy, and the energy is an integral,
//
//     Exc = sum_g w_g rho_g zk_g,
//
// over Libxc's energy density on the ingredients NuKEXC builds from a
// density matrix. The ingredients are what the ERKALE comparison validates;
// zk is Libxc's; the sum is three lines, written out below rather than
// borrowed from xc_energy_restricted so that no generated code appears on
// the reference side of the comparison even in the trivial role of
// contracting a dot product. The two tests therefore close over each other:
// ERKALE says the ingredients are right, finite differences say the
// derivative with respect to them is right.
//
// WHAT IS PERTURBED
// -----------------
// Exc depends on P only through rho, sigma, tau and the Laplacian, each of
// which is a double sum over P_uv. Treating the entries as independent
// would mean perturbing a symmetric matrix asymmetrically, so the
// perturbation here is symmetric -- P_ij and P_ji both move by h -- and the
// directional derivative that comes back is
//
//     dE/dh = V_ij + V_ji = 2 V_ij    (i != j),
//     dE/dh = V_ii                    (i == j),
//
// using only that V is symmetric, which test_xc_kernel_session.cxx asserts
// separately. Every element of the upper triangle is checked, for both spin
// channels, which is what makes this a test of the contraction and not only
// of a few entries of it.
//
// THE TOLERANCE IS MEASURED, NOT PICKED
// -------------------------------------
// Each derivative is taken at two step sizes and Richardson-extrapolated.
// The change between the two estimates -- the drift -- is the size of the
// term the extrapolation removed, so it says how well the step resolves
// this functional and it carries the tolerance. A functional too rough for
// a finite difference on this grid fails the drift assertion rather than
// quietly widening the comparison.
//
// The drift is an estimate of a leading error term whose coefficient can
// pass through zero from one matrix element to the next, and where it does
// the estimate collapses while the residual does not. So the per-element
// drift is floored by a relative level, which the measurements below fix:
//
//     worst |V_analytic - V_numerical|, relative to the largest element
//
//     LDA_X, PBE, TPSS, restricted and both channels    < 5.3e-10
//     MGGA_X_BR89_EXPLICIT                                2.8e-8
//
// BR89 is fifty times worse than everything else and still eight orders of
// magnitude below the quantity, which is the shape to expect: it is the only
// functional here that reads the Laplacian, and a Laplacian built from a
// collocation Hessian is the noisiest ingredient on the grid. The floor is
// set at 1e-7, above both groups and three orders below anything a wrong
// chain-rule term could plausibly contribute.
//
// Agreement this good invites the suspicion that the comparison is not
// live, so it was checked by breaking it: scaling vtau by 1 + 1e-4 before
// the kernel contracts it -- a 1e-4 relative error in one operand of one
// term -- fails the alpha channel of MGGA_X_TPSS at 3.6e-7 against a
// tolerance of 1.4e-7. The test therefore resolves a wrong tau term down
// to about 4e-5 relative -- far below a missing or mis-signed term, which
// is what a misderivation looks like.

#include <algorithm>
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include <xc.h>
#include <xc_funcs.h>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/host_interface.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/xc_kernel_session.hpp>

#include "test_basis_sets.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using Session = XCKernelSession<GTOEvaluator>;

namespace {

/// A functional and the kernel family it is evaluated in.
struct Case {
  int id;
  XCFamily fam;
  const char *name;
};

/// The four families, each with a functional that genuinely depends on
/// everything the family carries.
///
/// `MGGA_X_BR89_EXPLICIT` rather than TPSS for the Laplacian family: TPSS
/// does not use the Laplacian, so Libxc returns vlapl = 0 for it and the
/// Laplacian term of the chain rule would be multiplied by zero. BR89 is
/// the only functional in Libxc's meta-GGA set here that makes that term
/// carry weight, so it is the only one for which this test says anything
/// about `MetaGGALapl` at all.
const Case CASES[] = {
    {XC_LDA_X, XCFamily::LDA, "LDA_X"},
    {XC_GGA_X_PBE, XCFamily::GGA, "GGA_X_PBE"},
    {XC_GGA_C_PBE, XCFamily::GGA, "GGA_C_PBE"},
    {XC_MGGA_X_TPSS, XCFamily::MetaGGATau, "MGGA_X_TPSS"},
    {XC_MGGA_C_TPSS, XCFamily::MetaGGATau, "MGGA_C_TPSS"},
    {XC_MGGA_X_BR89_EXPLICIT, XCFamily::MetaGGALapl, "MGGA_X_BR89_EXPLICIT"}};

/// The same density matrix recipe the rest of the XC tests use: rank one
/// plus a positive diagonal shift, so positive semidefinite with room to
/// spare -- the perturbations below are four orders of magnitude smaller
/// than the diagonal shift, so no step takes the density negative.
std::vector<double> make_P(int nbf, double freq = 0.9) {
  std::vector<double> P(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j <= i; ++j) {
      const double v =
          0.04 * std::cos(freq * (i + 1)) * std::cos(freq * (j + 1));
      P[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf] = v;
      P[static_cast<size_t>(j) + static_cast<size_t>(i) * nbf] = v;
    }
  for (int i = 0; i < nbf; ++i)
    P[static_cast<size_t>(i) + static_cast<size_t>(i) * nbf] += 0.05;
  return P;
}

std::vector<double> scaled(const std::vector<double> &P, double f) {
  std::vector<double> out(P);
  for (double &v : out)
    v *= f;
  return out;
}

/// Add @p h to the (i, j) and (j, i) entries of a column-major matrix.
void bump(std::vector<double> &P, int nbf, int i, int j, double h) {
  P[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf] += h;
  if (i != j)
    P[static_cast<size_t>(j) + static_cast<size_t>(i) * nbf] += h;
}

/// dE/dh for a symmetric bump of element (i, j) equals this times V_ij.
double multiplicity(int i, int j) { return (i == j) ? 1.0 : 2.0; }

/// Exc for a restricted density matrix: Libxc's energy density on NuKEXC's
/// ingredients, integrated. The functional's derivatives are never asked
/// for, and neither is a generated kernel.
double energy(Session &s, xc_func_type &func, XCFamily fam,
              const std::vector<double> &P, int nbf) {
  const HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};
  s.set_density(Pv);
  const int ng = s.npoints();
  std::vector<double> zk(ng, 0.0);
  const std::vector<double> zero(ng, 0.0);
  switch (fam) {
  case XCFamily::LDA:
    xc_lda_exc(&func, ng, s.rho().data(), zk.data());
    break;
  case XCFamily::GGA:
    xc_gga_exc(&func, ng, s.rho().data(), s.sigma().data(), zk.data());
    break;
  case XCFamily::MetaGGATau:
  case XCFamily::MetaGGALapl:
    xc_mgga_exc(&func, ng, s.rho().data(), s.sigma().data(),
                fam == XCFamily::MetaGGALapl ? s.lapl().data() : zero.data(),
                s.tau().data(), zk.data());
    break;
  }
  const std::vector<double> &w = s.weights();
  const std::vector<double> &rho = s.rho();
  double e = 0.0;
  for (int g = 0; g < ng; ++g)
    e += w[g] * rho[g] * zk[g];
  return e;
}

/// Exc for a pair of spin density matrices. The energy expression does not
/// know about spin: it is the total density against zk either way.
double energy(Session &s, xc_func_type &func, XCFamily fam,
              const std::vector<double> &Pa, const std::vector<double> &Pb,
              int nbf) {
  const HostMatrixView Pav{Pa.data(), nbf, MatrixLayout::ColumnMajor};
  const HostMatrixView Pbv{Pb.data(), nbf, MatrixLayout::ColumnMajor};
  s.set_density(Pav, Pbv);
  const int ng = s.npoints();
  std::vector<double> zk(ng, 0.0);
  const std::vector<double> zero(2 * ng, 0.0);
  switch (fam) {
  case XCFamily::LDA:
    xc_lda_exc(&func, ng, s.rho_spin().data(), zk.data());
    break;
  case XCFamily::GGA:
    xc_gga_exc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
               zk.data());
    break;
  case XCFamily::MetaGGATau:
  case XCFamily::MetaGGALapl:
    xc_mgga_exc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
                fam == XCFamily::MetaGGALapl ? s.lapl_spin().data()
                                             : zero.data(),
                s.tau_spin().data(), zk.data());
    break;
  }
  const std::vector<double> &w = s.weights();
  const std::vector<double> &rs = s.rho_spin();
  double e = 0.0;
  for (int g = 0; g < ng; ++g)
    e += w[g] * (rs[2 * g] + rs[2 * g + 1]) * zk[g];
  return e;
}

/// A Richardson-extrapolated central difference, with the evidence for its
/// own accuracy.
struct Derivative {
  /// The extrapolated estimate, (4 d(h/2) - d(h)) / 3.
  double value = 0.0;
  /// d(h) - d(h/2). For a central difference the error is C h^2, so this is
  /// 3/4 C h^2 -- the term the extrapolation removed, and an upper bound in
  /// order of magnitude on the one it did not.
  double drift = 0.0;
};

/// Central differences of @p f at steps h and h/2, extrapolated.
template <typename F> Derivative differentiate(F &&f, double h) {
  const double d1 = (f(h) - f(-h)) / (2.0 * h);
  const double hh = 0.5 * h;
  const double d2 = (f(hh) - f(-hh)) / (2.0 * hh);
  Derivative out;
  out.value = (4.0 * d2 - d1) / 3.0;
  out.drift = d1 - d2;
  return out;
}

/// Largest magnitude in a matrix, so tolerances can be stated relative to
/// the quantity being checked rather than to each element.
double scale_of(const std::vector<double> &V) {
  double s = 0.0;
  for (double v : V)
    s = std::max(s, std::abs(v));
  return s;
}

/// The step. Small enough that the h^2 truncation error is ~1e-8 on these
/// densities, large enough that cancellation in E(h) - E(-h) -- about
/// 1e-16 of an energy of order 10, divided by 2h -- stays below that.
constexpr double STEP = 4e-4;

struct Fixture {
  Molecule mol;
  GTOBasisSet<double> host;
  FlatGrid grid;
  Fixture(size_t nrad = 35, size_t lmax = 17) {
    read_xyz("input/water.xyz", mol);
    host = test::sto3g_basis(mol);
    grid = make_flat_grid<ta_type, ll_type>(mol, nrad, lmax);
  }
};

} // namespace

TEST_CASE("The restricted XC matrix is the derivative of the energy",
          "[xckernel][findiff]") {
  Fixture f;
  const GTOEvaluator ev(make_gto_device_basis(f.host));
  const int nbf = ev.nbf();
  const auto P0 = make_P(nbf);

  for (const Case &c : CASES) {
    INFO("functional " << c.name);
    xc_func_type func;
    REQUIRE(xc_func_init(&func, c.id, XC_UNPOLARIZED) == 0);

    Session s(ev, f.grid, c.fam);

    // The analytic matrix, from the generated kernel.
    const int ng = s.npoints();
    std::vector<double> zk(ng), vrho(ng), vsigma(ng), vlapl(ng), vtau(ng);
    const std::vector<double> zero(ng, 0.0);
    {
      const HostMatrixView Pv{P0.data(), nbf, MatrixLayout::ColumnMajor};
      s.set_density(Pv);
      if (c.fam == XCFamily::LDA)
        xc_lda_exc_vxc(&func, ng, s.rho().data(), zk.data(), vrho.data());
      else if (c.fam == XCFamily::GGA)
        xc_gga_exc_vxc(&func, ng, s.rho().data(), s.sigma().data(), zk.data(),
                       vrho.data(), vsigma.data());
      else
        xc_mgga_exc_vxc(&func, ng, s.rho().data(), s.sigma().data(),
                        c.fam == XCFamily::MetaGGALapl ? s.lapl().data()
                                                       : zero.data(),
                        s.tau().data(), zk.data(), vrho.data(), vsigma.data(),
                        vlapl.data(), vtau.data());
    }
    std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef R{V.data(), nbf, MatrixLayout::ColumnMajor};
    XCKernelDerivs d{.vrho = vrho.data()};
    if (c.fam != XCFamily::LDA)
      d.vsigma = vsigma.data();
    if (c.fam == XCFamily::MetaGGATau || c.fam == XCFamily::MetaGGALapl)
      d.vtau = vtau.data();
    if (c.fam == XCFamily::MetaGGALapl)
      d.vlapl = vlapl.data();
    s.accumulate_fock(d, zk.data(), R);

    const double sc = scale_of(V);
    INFO("largest |V| = " << sc);
    REQUIRE(sc > 1e-3); // not a comparison of empty matrices

    // Every element of the upper triangle.
    double worst = 0.0, worst_drift = 0.0;
    int wi = -1, wj = -1;
    for (int i = 0; i < nbf; ++i)
      for (int j = i; j < nbf; ++j) {
        const Derivative num = differentiate(
            [&](double h) {
              auto P = P0;
              bump(P, nbf, i, j, h);
              return energy(s, func, c.fam, P, nbf);
            },
            STEP);
        const double analytic =
            multiplicity(i, j) *
            V[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf];
        const double err = std::abs(analytic - num.value);
        if (err > worst) {
          worst = err;
          worst_drift = std::abs(num.drift);
          wi = i;
          wj = j;
        }
        // A functional the finite difference cannot resolve on this grid
        // must fail here, not inflate the comparison below.
        INFO("element (" << i << ", " << j << "): analytic " << analytic
                         << ", numerical " << num.value << ", drift "
                         << num.drift);
        REQUIRE(std::abs(num.drift) < 1e-4 * sc);
        REQUIRE(err < std::max(3.0 * std::abs(num.drift), 1e-7 * sc));
      }
    INFO("worst element (" << wi << ", " << wj << "): error " << worst
                           << ", its drift " << worst_drift << ", relative "
                           << worst / sc);
    xc_func_end(&func);
  }
}

TEST_CASE("Both unrestricted XC matrices are derivatives of the energy",
          "[xckernel][findiff][unrestricted]") {
  // The one that matters most: the unrestricted meta-GGA chain rule has no
  // independent check anywhere else in the suite. The closed-shell identity
  // relates it to the restricted kernel, and ERKALE shares the generator,
  // so until this test the derivation itself rested on libxckernel alone.
  //
  // Pa and Pb are different matrices, so sigma_ab differs from sigma_aa and
  // sigma_bb and the cross term carries weight: a channel differentiating
  // with respect to the wrong spin's gradient cannot hide.
  Fixture f;
  const GTOEvaluator ev(make_gto_device_basis(f.host));
  const int nbf = ev.nbf();
  const auto Pa0 = scaled(make_P(nbf, 0.9), 0.6);
  const auto Pb0 = scaled(make_P(nbf, 1.7), 0.4);

  for (const Case &c : CASES) {
    INFO("functional " << c.name);
    xc_func_type func;
    REQUIRE(xc_func_init(&func, c.id, XC_POLARIZED) == 0);

    Session s(ev, f.grid, c.fam);
    const int ng = s.npoints();

    std::vector<double> zk(ng), vrho(2 * ng), vsigma(3 * ng), vlapl(2 * ng),
        vtau(2 * ng);
    const std::vector<double> zero(2 * ng, 0.0);
    {
      const HostMatrixView Pav{Pa0.data(), nbf, MatrixLayout::ColumnMajor};
      const HostMatrixView Pbv{Pb0.data(), nbf, MatrixLayout::ColumnMajor};
      s.set_density(Pav, Pbv);
      if (c.fam == XCFamily::LDA)
        xc_lda_exc_vxc(&func, ng, s.rho_spin().data(), zk.data(), vrho.data());
      else if (c.fam == XCFamily::GGA)
        xc_gga_exc_vxc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
                       zk.data(), vrho.data(), vsigma.data());
      else
        xc_mgga_exc_vxc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
                        c.fam == XCFamily::MetaGGALapl ? s.lapl_spin().data()
                                                       : zero.data(),
                        s.tau_spin().data(), zk.data(), vrho.data(),
                        vsigma.data(), vlapl.data(), vtau.data());
    }
    std::vector<double> Va(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<double> Vb(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef Ra{Va.data(), nbf, MatrixLayout::ColumnMajor};
    HostMatrixRef Rb{Vb.data(), nbf, MatrixLayout::ColumnMajor};
    XCPackedDerivsU pd{.vrho = vrho.data()};
    if (c.fam != XCFamily::LDA)
      pd.vsigma = vsigma.data();
    if (c.fam == XCFamily::MetaGGATau || c.fam == XCFamily::MetaGGALapl)
      pd.vtau = vtau.data();
    if (c.fam == XCFamily::MetaGGALapl)
      pd.vlapl = vlapl.data();
    s.accumulate_fock(pd, zk.data(), Ra, Rb);

    const double sca = scale_of(Va), scb = scale_of(Vb);
    INFO("largest |Va| = " << sca << ", |Vb| = " << scb);
    REQUIRE(sca > 1e-3);
    REQUIRE(scb > 1e-3);

    struct Channel {
      const char *name;
      const std::vector<double> *V;
      double sc;
      bool alpha;
    };
    const Channel channels[] = {{"alpha", &Va, sca, true},
                                {"beta", &Vb, scb, false}};

    for (const Channel &ch : channels) {
      INFO("channel " << ch.name);
      double worst = 0.0, worst_drift = 0.0;
      int wi = -1, wj = -1;
      for (int i = 0; i < nbf; ++i)
        for (int j = i; j < nbf; ++j) {
          const Derivative num = differentiate(
              [&](double h) {
                auto Pa = Pa0;
                auto Pb = Pb0;
                bump(ch.alpha ? Pa : Pb, nbf, i, j, h);
                return energy(s, func, c.fam, Pa, Pb, nbf);
              },
              STEP);
          const double analytic =
              multiplicity(i, j) *
              (*ch.V)[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf];
          const double err = std::abs(analytic - num.value);
          if (err > worst) {
            worst = err;
            worst_drift = std::abs(num.drift);
            wi = i;
            wj = j;
          }
          INFO("element (" << i << ", " << j << "): analytic " << analytic
                           << ", numerical " << num.value << ", drift "
                           << num.drift);
          REQUIRE(std::abs(num.drift) < 1e-4 * ch.sc);
          REQUIRE(err < std::max(3.0 * std::abs(num.drift), 1e-7 * ch.sc));
        }
      INFO("worst element (" << wi << ", " << wj << "): error " << worst
                             << ", its drift " << worst_drift << ", relative "
                             << worst / ch.sc);
    }

    xc_func_end(&func);
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
