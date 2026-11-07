/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// ============================================================================
//  Unrestricted exchange-correlation matrices against ERKALE
// ============================================================================
//
// The acceptance test for the meta-GGA path. Everything else that checks it
// checks it against NuKEXC: the kernels against NuKEXC's own hand-written LDA
// and GGA assembly, the unrestricted kernels against the closed-shell limit of
// the restricted ones. Those catch implementation slips but not a shared
// misunderstanding -- if the collocation, the ingredients and the test all
// agreed on the wrong convention for tau, or on the wrong factor in the XC
// matrix, every one of them would still pass. A second program computing the
// same quantity from its own basis evaluation, its own grid and its own
// ingredients is what rules that out.
//
// How much it rules out depends on how much the two programs share, and they
// share more than they used to -- see the next section, which is deliberately
// placed before anything else here.
//
// The reference carries the basis AND the density matrices AND the results, so
// nothing here has to assume the two codes were handed the same input. See
// tests/reference/README.md.
//
// WHAT THIS DOES AND DOES NOT ESTABLISH
// -------------------------------------
// ERKALE at the commit that produced the reference assembles its own XC
// matrices through libxckernel -- it calls xck_contract("ua_o1") and
// ("ub_o1"), the same generated kernels NuKEXC calls. So this test does NOT
// independently verify the chain rule inside those kernels: a mistake in the
// generator's meta-GGA derivation would be present on both sides and would
// cancel. Saying so here rather than letting "validated against ERKALE" be
// read as more than it is.
//
// That gap is closed in test_xc_finite_difference.cxx, which needs no second
// implementation of the derivative: it differentiates the energy numerically
// with respect to every element of both density matrices and compares against
// the kernels' matrices. The two tests lean on each other -- this one says the
// ingredients and the conventions are right, that one says the derivative with
// respect to them is right -- and neither is sufficient alone, since the
// numerical derivative is built on the same ingredients it cannot check.
//
// What it does verify independently, because the two codes share no code for
// any of it: the basis evaluation and its ordering, the quadrature grid, the
// construction of rho, sigma, tau and the Laplacian from a density matrix,
// the Libxc calling conventions for a polarized functional, the energy
// expression, and -- not least -- the OPERAND WIRING. ERKALE passes operands
// by name through a dispatch table keyed on strings; NuKEXC passes them
// through libxckernel's named structs. Those are independent routes to the
// same kernel, so a swapped vsigma_aa and vsigma_ab, or a vlapl in the vrho
// slot, cannot pass here. That was the failure mode worth being afraid of.
//
// WHAT SETS THE TOLERANCE
// -----------------------
// The two codes integrate on different grids -- ERKALE uses Chebyshev radial
// shells and NuKEXC Treutler-Ahlrichs -- so the comparison is at the grid
// error level, not at rounding. The reference therefore carries each quantity
// on a COARSE and a FINE grid, and the change between them is what the fine
// grid is worth. The assertions scale with that rather than with a number
// picked to make the test pass, because the comparison cannot be better than
// the reference's own grid error, and for a functional like BR89 that error
// is four orders of magnitude larger than for LDA.
//
// Measured, as relative deviations from ERKALE's fine grid, against ERKALE's
// own coarse-to-fine change in the last column:
//
//     functional      dExc     dHa      dHb      ERKALE Exc / Ha
//     LDA_X           1.7e-8   6.1e-8   6.9e-8   1.1e-7 / 6.0e-7
//     GGA_X_PBE       5.0e-8   2.0e-6   1.4e-6   6.4e-7 / 6.5e-6
//     GGA_XC_PBE      1.1e-8   2.0e-6   1.4e-6   7.2e-7 / 5.5e-6
//     MGGA_X_TPSS     8.9e-8   1.9e-6   3.3e-6   1.1e-6 / 5.0e-6
//     MGGA_XC_TPSS    5.5e-8   2.7e-6   3.5e-6   1.2e-6 / 6.5e-6
//     MGGA_X_SCAN     4.1e-6   2.7e-4   7.1e-5   9.0e-6 / 7.0e-4
//     MGGA_X_BR89     2.4e-5   4.1e-4   6.3e-4   2.6e-4 / 9.6e-4
//
// Every entry is BELOW the reference's own grid uncertainty -- NuKEXC lands
// inside ERKALE's error bar for all seven, the two meta-GGA families
// included. The tightest ratio is BR89's beta matrix at 1.5x, which is why
// the factor below is 3 and not 1.
//
// This is not a weak test despite the loose-looking numbers. A wrong operand,
// a missing cross term or a transposed derivative is wrong by O(1), not by
// O(1e-6): there are four to eight orders of margin between what is asserted
// and what such a mistake would cost.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <xc.h>
#include <xc_funcs.h>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/grid.hpp>
#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/xc_kernel_session.hpp>

#include "erkale_ref_reader.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using GTOSession = XCKernelSession<GTOEvaluator>;

namespace {

const char *kRef = "reference/water_631gdp_xc_erkale.ref";

/// A functional as the reference names it: one id, or an exchange and a
/// correlation id whose contributions add.
struct Case {
  const char *name;
  int x_id;
  int c_id; ///< 0 for none
};

/// Libxc functional handle that closes itself.
struct Func {
  xc_func_type f;
  explicit Func(int id, int nspin) {
    if (xc_func_init(&f, id, nspin) != 0)
      throw std::runtime_error("xc_func_init failed for id " +
                               std::to_string(id));
  }
  ~Func() { xc_func_end(&f); }
  Func(const Func &) = delete;
  Func &operator=(const Func &) = delete;
};

/// The more demanding of two families, so a sum of functionals is evaluated
/// with the ingredients the most demanding term needs.
XCFamily wider(XCFamily a, XCFamily b) {
  const auto rank = [](XCFamily f) {
    switch (f) {
    case XCFamily::LDA:
      return 0;
    case XCFamily::GGA:
      return 1;
    case XCFamily::MetaGGATau:
      return 2;
    case XCFamily::MetaGGALapl:
      return 3;
    }
    return 0;
  };
  return rank(a) >= rank(b) ? a : b;
}

double worst_abs_diff(const std::vector<double> &a, const std::vector<double> &b,
                      double &scale) {
  double w = 0.0;
  scale = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    w = std::max(w, std::abs(a[i] - b[i]));
    scale = std::max(scale, std::abs(b[i]));
  }
  return w;
}

} // namespace

TEST_CASE("NuKEXC reproduces ERKALE's unrestricted XC matrices",
          "[erkale][parity][xc][unrestricted][mgga]") {
  const auto ref = test::load_erkale_reference(kRef);
  const int nbf = ref.nbf;
  REQUIRE(nbf == 24);

  const auto &Pa = ref.matrix("Pa");
  const auto &Pb = ref.matrix("Pb");
  // ERKALE writes row-major; the densities are symmetric, but saying so
  // explicitly costs nothing and documents the file.
  const HostMatrixView Pav{Pa.data(), nbf, MatrixLayout::RowMajor};
  const HostMatrixView Pbv{Pb.data(), nbf, MatrixLayout::RowMajor};

  // The nuclei come from the geometry, the basis from the reference: the
  // reference's shell centers are the nuclei ERKALE used, so the grid sits
  // where ERKALE's did even though the quadrature rule differs.
  Molecule mol;
  read_xyz("reference/water.xyz", mol);
  const auto dev = make_gto_device_basis(ref.basis);
  const GTOEvaluator ev(dev);
  REQUIRE(ev.nbf() == nbf);

  const FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 99, 35);
  const int ng = static_cast<int>(grid.quad_points.extent(0));
  INFO("NuKEXC grid: " << ng << " points");

  const Case cases[] = {
      {"LDA_X", XC_LDA_X, 0},
      {"GGA_X_PBE", XC_GGA_X_PBE, 0},
      {"GGA_XC_PBE", XC_GGA_X_PBE, XC_GGA_C_PBE},
      {"MGGA_X_TPSS", XC_MGGA_X_TPSS, 0},
      {"MGGA_XC_TPSS", XC_MGGA_X_TPSS, XC_MGGA_C_TPSS},
      {"MGGA_X_SCAN", XC_MGGA_X_SCAN, 0},
      {"MGGA_X_BR89", XC_MGGA_X_BR89_EXPLICIT, 0},
  };

  for (const auto &c : cases) {
    INFO("functional " << c.name);

    Func fx(c.x_id, XC_POLARIZED);
    XCFamily fam = xc_family_of(fx.f);
    std::unique_ptr<Func> fc;
    if (c.c_id) {
      fc = std::make_unique<Func>(c.c_id, XC_POLARIZED);
      fam = wider(fam, xc_family_of(fc->f));
    }

    GTOSession s(ev, grid, fam);
    s.set_density(Pav, Pbv);

    // Libxc's polarized arrays. A sum of functionals adds term by term in
    // every one of them, zk included, so the two are simply accumulated.
    std::vector<double> zk(ng, 0.0), vrho(2 * ng, 0.0), vsigma(3 * ng, 0.0),
        vtau(2 * ng, 0.0), vlapl(2 * ng, 0.0);
    std::vector<double> zk1(ng), vrho1(2 * ng), vsigma1(3 * ng), vtau1(2 * ng),
        vlapl1(2 * ng);
    const std::vector<double> lapl_zero(2 * ng, 0.0);

    const auto evaluate = [&](const xc_func_type &func) {
      std::fill(zk1.begin(), zk1.end(), 0.0);
      std::fill(vrho1.begin(), vrho1.end(), 0.0);
      std::fill(vsigma1.begin(), vsigma1.end(), 0.0);
      std::fill(vtau1.begin(), vtau1.end(), 0.0);
      std::fill(vlapl1.begin(), vlapl1.end(), 0.0);

      switch (xc_family_of(func)) {
      case XCFamily::LDA:
        xc_lda_exc_vxc(&func, ng, s.rho_spin().data(), zk1.data(),
                       vrho1.data());
        break;
      case XCFamily::GGA:
        xc_gga_exc_vxc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
                       zk1.data(), vrho1.data(), vsigma1.data());
        break;
      case XCFamily::MetaGGATau:
      case XCFamily::MetaGGALapl:
        // A tau-only functional in a Laplacian session still gets the
        // Laplacian passed -- it ignores it -- but a Laplacian functional
        // in a tau-only session would be handed zeros, which is why the
        // family is the wider of the two and not each term's own.
        xc_mgga_exc_vxc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
                        fam == XCFamily::MetaGGALapl ? s.lapl_spin().data()
                                                     : lapl_zero.data(),
                        s.tau_spin().data(), zk1.data(), vrho1.data(),
                        vsigma1.data(), vlapl1.data(), vtau1.data());
        break;
      }
      for (int i = 0; i < ng; ++i)
        zk[i] += zk1[i];
      for (int i = 0; i < 2 * ng; ++i) {
        vrho[i] += vrho1[i];
        vtau[i] += vtau1[i];
        vlapl[i] += vlapl1[i];
      }
      for (int i = 0; i < 3 * ng; ++i)
        vsigma[i] += vsigma1[i];
    };

    evaluate(fx.f);
    if (fc)
      evaluate(fc->f);

    std::vector<double> Va(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<double> Vb(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef Ra{Va.data(), nbf, MatrixLayout::RowMajor};
    HostMatrixRef Rb{Vb.data(), nbf, MatrixLayout::RowMajor};

    XCPackedDerivsU d{.vrho = vrho.data()};
    if (fam != XCFamily::LDA)
      d.vsigma = vsigma.data();
    if (fam == XCFamily::MetaGGATau || fam == XCFamily::MetaGGALapl)
      d.vtau = vtau.data();
    if (fam == XCFamily::MetaGGALapl)
      d.vlapl = vlapl.data();

    const XCOutputPolarized got = s.accumulate_fock(d, zk.data(), Ra, Rb);

    // --- what ERKALE's fine grid is worth, read off the reference ---
    const std::string pf = std::string(c.name) + "_fine_";
    const std::string pc = std::string(c.name) + "_coarse_";
    const double erk_Exc = ref.scalar(pf + "Exc");
    const double erk_Nel = ref.scalar(pf + "Nel");
    const auto &erk_Ha = ref.matrix(pf + "Ha");
    const auto &erk_Hb = ref.matrix(pf + "Hb");

    double hscale_c = 0.0;
    const double erk_conv_H =
        worst_abs_diff(ref.matrix(pc + "Ha"), erk_Ha, hscale_c);
    const double erk_conv_Exc =
        std::abs(ref.scalar(pc + "Exc") - erk_Exc) / std::abs(erk_Exc);
    INFO("ERKALE coarse->fine: Exc " << erk_conv_Exc << " relative, Ha "
                                     << erk_conv_H / hscale_c
                                     << " relative; points "
                                     << ref.scalar(pf + "npoints"));

    // The reference has to be converged for any of this to mean anything.
    // BR89 is the loosest at 2.6e-4 on Exc and 9.6e-4 on Ha; a reference an
    // order of magnitude worse than that is not one worth comparing against.
    REQUIRE(erk_conv_Exc < 1e-3);
    REQUIRE(erk_conv_H < 1e-2 * hscale_c);

    // --- the comparison ---
    const double dExc = std::abs(got.Exc - erk_Exc) / std::abs(erk_Exc);
    const double dNel = std::abs(got.Nel - erk_Nel);
    double sa = 0.0, sb = 0.0;
    const double wa = worst_abs_diff(Va, erk_Ha, sa);
    const double wb = worst_abs_diff(Vb, erk_Hb, sb);

    INFO("Exc NuKEXC " << got.Exc << " ERKALE " << erk_Exc << " relative "
                       << dExc);
    INFO("Nel NuKEXC " << got.Nel << " ERKALE " << erk_Nel << " absolute "
                       << dNel);
    INFO("Ha worst " << wa << " scale " << sa << " relative " << wa / sa
                     << "; Hb worst " << wb << " relative " << wb / sb);

    // The reference's own grid error, with a floor: a hypothetical perfectly
    // converged reference would otherwise demand exact agreement, and
    // NuKEXC's grid error does not vanish just because ERKALE's did. The
    // floor is set an order above the best agreement measured (6e-8).
    const double tol_Exc = std::max(3.0 * erk_conv_Exc, 1e-7);
    const double tol_H = std::max(3.0 * erk_conv_H / hscale_c, 1e-7);
    INFO("tolerances: Exc " << tol_Exc << ", H " << tol_H);

    REQUIRE(sa > 1e-3); // not a comparison of two empty matrices
    REQUIRE(sb > 1e-3);
    // The electron count is a property of the grid and the density alone --
    // no functional enters it -- so it is the one quantity here that should
    // agree far better than the XC matrices do. Measured: 1.4e-9.
    REQUIRE(dNel < 1e-7);
    REQUIRE(dExc < tol_Exc);
    REQUIRE(wa < tol_H * sa);
    REQUIRE(wb < tol_H * sb);
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
