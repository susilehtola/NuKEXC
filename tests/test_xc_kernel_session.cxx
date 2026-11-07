/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The kernel-backed XC path, against the hand-written one and against
// itself.
//
// Two kinds of check, because neither suffices alone.
//
// PARITY. For LDA and GGA, XCSession's hand-written assembly is validated
// against ERKALE, so it is a real oracle: the kernels must reproduce it.
// NuKEXC's screening has to be turned off for that comparison, because the
// kernels see every point; with ERKALE's default 1e-10 thresholds the two
// differ by about 3e-10 on the Fock matrix, which is the cost of the
// screening rather than an error in either.
//
// REDUCTION. For the meta-GGAs there is no hand-written path to compare
// against -- that is the whole reason for the kernels. But the families
// nest: a meta-GGA kernel handed vtau = 0 must reproduce the GGA kernel
// exactly, and the full meta-GGA handed vlapl = 0 must reproduce the
// tau-only one. Those identities exercise precisely the wiring that is
// easy to get wrong, since xck_mgga_r_o1 takes its derivative arrays as
// vlapl, vrho, vsigma, vtau -- deliberately not Libxc's order -- and all
// four are const double* of length npts, so a permutation is otherwise
// undetectable.

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
#include <nukexc/stobasis.hpp>
#include <nukexc/xc_kernel_session.hpp>

#include "gto_analytic_overlap.hpp"
#include "test_basis_sets.hpp"
#include "sto_basis_reader.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using Session = XCKernelSession<STOPrimitiveEvaluator>;
using GTOSession = XCKernelSession<GTOEvaluator>;

namespace {

struct Fixture {
  Molecule mol;
  STOBasisSet basis;
  FlatGrid grid;
  Fixture(size_t nrad = 50, size_t lmax = 21) {
    read_xyz("input/water.xyz", mol);
    basis = test::load_adf_basis(mol, "input/zorabasis/TZP", 1e-10);
    grid = make_flat_grid<ta_type, ll_type>(mol, nrad, lmax);
  }
};

/// A plausible symmetric density matrix, scaled to a sane electron count.
///
/// Rank-one plus a positive diagonal shift, so it is positive semidefinite
/// and the density it gives is nonnegative everywhere -- which matters once
/// it is split between spin channels, since Libxc is entitled to misbehave
/// on a negative density. @p freq picks a different such matrix, for a
/// genuinely spin-polarized pair.
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

double worst_diff(const std::vector<double> &a, const std::vector<double> &b,
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

TEST_CASE("The kernel path reproduces the hand-written LDA and GGA",
          "[xckernel][parity]") {
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  auto P = make_P(nbf);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};

  struct Case {
    int id;
    XCFamily fam;
    const char *name;
  };
  const Case cases[] = {{XC_LDA_X, XCFamily::LDA, "LDA_X"},
                        {XC_GGA_X_PBE, XCFamily::GGA, "GGA_X_PBE"},
                        {XC_GGA_C_PBE, XCFamily::GGA, "GGA_C_PBE"}};

  for (const auto &c : cases) {
    INFO("functional " << c.name);
    xc_func_type func;
    REQUIRE(xc_func_init(&func, c.id, XC_UNPOLARIZED) == 0);

    // Reference: the hand-written path, screening disabled so that both
    // see the same points.
    std::vector<double> Vref(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef Rref{Vref.data(), nbf, MatrixLayout::ColumnMajor};
    XCThresholds thr;
    thr.density = 0.0;
    thr.basis = 0.0;
    thr.weight = 0.0;
    const XCOutput ref = eval_vxc(ev, f.grid, Pv, Rref, func, thr);

    // Kernel path.
    Session s(ev, f.grid, c.fam);
    s.set_density(Pv);
    const int ng = s.npoints();
    std::vector<double> zk(ng), vrho(ng), vsigma(ng);
    if (c.fam == XCFamily::LDA)
      xc_lda_exc_vxc(&func, ng, s.rho().data(), zk.data(), vrho.data());
    else
      xc_gga_exc_vxc(&func, ng, s.rho().data(), s.sigma().data(), zk.data(),
                     vrho.data(), vsigma.data());

    std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef R{V.data(), nbf, MatrixLayout::ColumnMajor};
    XCKernelDerivs d{.vrho = vrho.data()};
    if (c.fam != XCFamily::LDA)
      d.vsigma = vsigma.data();
    const XCOutput got = s.accumulate_fock(d, zk.data(), R);

    INFO("Exc hand-written " << ref.Exc << ", kernel " << got.Exc);
    REQUIRE_THAT(got.Nel, Catch::Matchers::WithinRel(ref.Nel, 1e-12));
    REQUIRE_THAT(got.Exc, Catch::Matchers::WithinRel(ref.Exc, 1e-11));

    double scale = 0.0;
    const double w = worst_diff(V, Vref, scale);
    INFO("worst |V_kernel - V_handwritten| = " << w << ", scale " << scale
                                               << ", relative " << w / scale);
    REQUIRE(scale > 1e-3);
    REQUIRE(w < 1e-11 * scale);

    xc_func_end(&func);
  }
}

TEST_CASE("The kernel Fock matrix is symmetric", "[xckernel]") {
  // accumulate_fock copies the kernel's (nbf, nbf) block out without caring
  // which of the two orderings it wrote, which is only safe because the XC
  // Fock matrix is symmetric. Assert it rather than rely on it.
  Fixture f(30, 17);
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  auto P = make_P(nbf);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};

  xc_func_type func;
  REQUIRE(xc_func_init(&func, XC_GGA_X_PBE, XC_UNPOLARIZED) == 0);
  Session s(ev, f.grid, XCFamily::GGA);
  s.set_density(Pv);
  const int ng = s.npoints();
  std::vector<double> zk(ng), vrho(ng), vsigma(ng);
  xc_gga_exc_vxc(&func, ng, s.rho().data(), s.sigma().data(), zk.data(),
                 vrho.data(), vsigma.data());

  std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef R{V.data(), nbf, MatrixLayout::ColumnMajor};
  s.accumulate_fock({.vrho = vrho.data(), .vsigma = vsigma.data()}, zk.data(),
                    R);
  xc_func_end(&func);

  double worst = 0.0, scale = 0.0;
  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j < nbf; ++j) {
      const double a = V[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf];
      const double b = V[static_cast<size_t>(j) + static_cast<size_t>(i) * nbf];
      worst = std::max(worst, std::abs(a - b));
      scale = std::max(scale, std::abs(a));
    }
  INFO("worst asymmetry " << worst << ", scale " << scale);
  REQUIRE(scale > 1e-3);
  REQUIRE(worst < 1e-13 * scale);
}

TEST_CASE("A meta-GGA kernel with vtau = 0 reduces to the GGA kernel",
          "[xckernel][reduction]") {
  // The families nest, so this is exact: the tau terms are linear in vtau
  // and vanish with it. It is the check that the meta kernels' operands are
  // wired correctly, which matters most for xck_mgga_r_o1, whose
  // derivative order is vlapl, vrho, vsigma, vtau.
  Fixture f(40, 17);
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  auto P = make_P(nbf);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};

  xc_func_type func;
  REQUIRE(xc_func_init(&func, XC_GGA_X_PBE, XC_UNPOLARIZED) == 0);

  // GGA reference, through the GGA kernel.
  Session g(ev, f.grid, XCFamily::GGA);
  g.set_density(Pv);
  const int ng = g.npoints();
  std::vector<double> zk(ng), vrho(ng), vsigma(ng);
  xc_gga_exc_vxc(&func, ng, g.rho().data(), g.sigma().data(), zk.data(),
                 vrho.data(), vsigma.data());
  std::vector<double> Vg(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef Rg{Vg.data(), nbf, MatrixLayout::ColumnMajor};
  g.accumulate_fock({.vrho = vrho.data(), .vsigma = vsigma.data()}, zk.data(),
                    Rg);
  xc_func_end(&func);

  const std::vector<double> zero(ng, 0.0);

  SECTION("tau-only meta-GGA") {
    Session m(ev, f.grid, XCFamily::MetaGGATau);
    m.set_density(Pv);
    // The ingredients must agree where they overlap: same density, same
    // sigma, computed through a different collocation split.
    for (int i = 0; i < ng; i += 97)
      REQUIRE_THAT(m.rho()[i], Catch::Matchers::WithinRel(g.rho()[i], 1e-14));

    std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef R{V.data(), nbf, MatrixLayout::ColumnMajor};
    m.accumulate_fock({.vrho = vrho.data(),
                       .vsigma = vsigma.data(),
                       .vtau = zero.data()},
                      zk.data(), R);
    double scale = 0.0;
    const double w = worst_diff(V, Vg, scale);
    INFO("worst difference " << w << ", scale " << scale);
    REQUIRE(scale > 1e-3);
    REQUIRE(w < 1e-12 * scale);
  }

}

TEST_CASE("The full meta-GGA reduces to the GGA with vtau = vlapl = 0",
          "[xckernel][reduction][gto]") {
  // Gaussians, because the full meta-GGA reads the collocation Hessian.
  Fixture f(40, 17);
  const auto host = test::sto3g_basis(f.mol);
  const GTOEvaluator ev(make_gto_device_basis(host));
  const int nbf = ev.nbf();
  INFO("GTO nbf = " << nbf);
  auto P = make_P(nbf);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};

  xc_func_type func;
  REQUIRE(xc_func_init(&func, XC_GGA_X_PBE, XC_UNPOLARIZED) == 0);
  GTOSession g(ev, f.grid, XCFamily::GGA);
  g.set_density(Pv);
  const int ng = g.npoints();
  std::vector<double> zk(ng), vrho(ng), vsigma(ng);
  xc_gga_exc_vxc(&func, ng, g.rho().data(), g.sigma().data(), zk.data(),
                 vrho.data(), vsigma.data());
  xc_func_end(&func);

  std::vector<double> Vg(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef Rg{Vg.data(), nbf, MatrixLayout::ColumnMajor};
  g.accumulate_fock({.vrho = vrho.data(), .vsigma = vsigma.data()}, zk.data(),
                    Rg);

  const std::vector<double> zero(ng, 0.0);
  GTOSession m(ev, f.grid, XCFamily::MetaGGALapl);
  m.set_density(Pv);
  std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef R{V.data(), nbf, MatrixLayout::ColumnMajor};
  m.accumulate_fock({.vrho = vrho.data(),
                     .vsigma = vsigma.data(),
                     .vtau = zero.data(),
                     .vlapl = zero.data()},
                    zk.data(), R);

  double scale = 0.0;
  const double w = worst_diff(V, Vg, scale);
  INFO("worst difference " << w << ", scale " << scale);
  REQUIRE(scale > 1e-4);
  REQUIRE(w < 1e-12 * scale);
}

TEST_CASE("The full meta-GGA with vlapl = 0 reduces to the tau-only one",
          "[xckernel][reduction][gto]") {
  // Same nesting one level up, and with a NONZERO vtau this time, so the
  // tau machinery is actually exercised rather than switched off. If vlapl
  // and vtau were transposed in the operand array this would fail while
  // the previous test still passed.
  Fixture f(40, 17);
  const auto host = test::sto3g_basis(f.mol);
  const GTOEvaluator ev(make_gto_device_basis(host));
  const int nbf = ev.nbf();
  auto P = make_P(nbf);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};

  GTOSession t(ev, f.grid, XCFamily::MetaGGATau);
  GTOSession l(ev, f.grid, XCFamily::MetaGGALapl);
  t.set_density(Pv);
  l.set_density(Pv);
  const int ng = t.npoints();

  // A real meta-GGA, so vtau is genuinely nonzero.
  xc_func_type func;
  REQUIRE(xc_func_init(&func, XC_MGGA_X_TPSS, XC_UNPOLARIZED) == 0);
  std::vector<double> zk(ng), vrho(ng), vsigma(ng), vlapl(ng), vtau(ng);
  const std::vector<double> lapl_in(ng, 0.0);
  xc_mgga_exc_vxc(&func, ng, t.rho().data(), t.sigma().data(), lapl_in.data(),
                  t.tau().data(), zk.data(), vrho.data(), vsigma.data(),
                  vlapl.data(), vtau.data());
  xc_func_end(&func);

  double vtau_scale = 0.0;
  for (double v : vtau)
    vtau_scale = std::max(vtau_scale, std::abs(v));
  INFO("max |vtau| = " << vtau_scale);
  REQUIRE(vtau_scale > 1e-6); // otherwise the test proves nothing

  std::vector<double> Vt(static_cast<size_t>(nbf) * nbf, 0.0);
  std::vector<double> Vl(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef Rt{Vt.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixRef Rl{Vl.data(), nbf, MatrixLayout::ColumnMajor};
  const std::vector<double> zero(ng, 0.0);

  t.accumulate_fock({.vrho = vrho.data(),
                     .vsigma = vsigma.data(),
                     .vtau = vtau.data()},
                    zk.data(), Rt);
  l.accumulate_fock({.vrho = vrho.data(),
                     .vsigma = vsigma.data(),
                     .vtau = vtau.data(),
                     .vlapl = zero.data()},
                    zk.data(), Rl);

  double scale = 0.0;
  const double w = worst_diff(Vl, Vt, scale);
  INFO("worst difference " << w << ", scale " << scale << ", relative "
                           << w / scale);
  REQUIRE(scale > 1e-3);
  REQUIRE(w < 1e-12 * scale);
}

TEST_CASE("A real meta-GGA runs and conserves the electron count",
          "[xckernel][mgga]") {
  Fixture f(60, 23);
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  auto P = make_P(nbf);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};

  Session s(ev, f.grid, XCFamily::MetaGGATau);
  s.set_density(Pv);
  const int ng = s.npoints();

  // tau must be positive and bounded below by the von Weizsacker form
  // wherever there is density; a sign error would otherwise only show as a
  // slightly wrong energy.
  const auto &rho = s.rho();
  const auto &sig = s.sigma();
  const auto &tau = s.tau();
  for (int i = 0; i < ng; ++i) {
    REQUIRE(tau[i] >= 0.0);
    if (rho[i] > 1e-8)
      REQUIRE(tau[i] >= sig[i] / (8.0 * rho[i]) - 1e-12);
  }

  xc_func_type func;
  REQUIRE(xc_func_init(&func, XC_MGGA_X_TPSS, XC_UNPOLARIZED) == 0);
  std::vector<double> zk(ng), vrho(ng), vsigma(ng), vlapl(ng), vtau(ng);
  const std::vector<double> lapl_in(ng, 0.0);
  xc_mgga_exc_vxc(&func, ng, rho.data(), sig.data(), lapl_in.data(),
                  tau.data(), zk.data(), vrho.data(), vsigma.data(),
                  vlapl.data(), vtau.data());
  xc_func_end(&func);

  std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef R{V.data(), nbf, MatrixLayout::ColumnMajor};
  const XCOutput out = s.accumulate_fock(
      {.vrho = vrho.data(), .vsigma = vsigma.data(), .vtau = vtau.data()},
      zk.data(), R);

  INFO("Exc = " << out.Exc << ", Nel = " << out.Nel);
  REQUIRE(std::isfinite(out.Exc));
  REQUIRE(out.Exc < 0.0); // an exchange functional
  for (double v : V)
    REQUIRE(std::isfinite(v));
}

TEST_CASE("Missing operands are rejected loudly", "[xckernel]") {
  Fixture f(20, 11);
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  auto P = make_P(nbf);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};

  Session s(ev, f.grid, XCFamily::MetaGGATau);
  const int ng = s.npoints();
  std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef R{V.data(), nbf, MatrixLayout::ColumnMajor};
  const std::vector<double> ones(ng, 1.0);

  // Using the session before a density is set is a programming error, not
  // a silent zero.
  REQUIRE_THROWS_AS(s.accumulate_fock({.vrho = ones.data()}, nullptr, R),
                    std::logic_error);

  s.set_density(Pv);
  // vtau omitted for a meta-GGA would be dereferenced inside the kernel.
  REQUIRE_THROWS_AS(
      s.accumulate_fock({.vrho = ones.data(), .vsigma = ones.data()}, nullptr,
                        R),
      std::invalid_argument);
  // vsigma omitted likewise.
  REQUIRE_THROWS_AS(
      s.accumulate_fock({.vrho = ones.data(), .vtau = ones.data()}, nullptr, R),
      std::invalid_argument);
  // A Laplacian meta-GGA on a Slater basis is rejected at construction:
  // STOPrimitiveEvaluator::max_deriv is Gradient, so the collocation
  // Hessian the kernel indexes does not exist. Before max_deriv was
  // advertised, BasisEvaluatorWith<E, Hessian> was satisfied by this
  // evaluator -- eval_block<Hessian> is callable and fails on a
  // static_assert only once instantiated -- so this surfaced as a
  // compile error inside stobasis.hpp instead.
  REQUIRE_THROWS_AS(Session(ev, f.grid, XCFamily::MetaGGALapl),
                    std::invalid_argument);

  // vlapl is required only by the full meta-GGA, which needs Gaussians.
  const auto host = test::sto3g_basis(f.mol);
  const GTOEvaluator gev(make_gto_device_basis(host));
  const int gnbf = gev.nbf();
  auto gP = make_P(gnbf);
  HostMatrixView gPv{gP.data(), gnbf, MatrixLayout::ColumnMajor};
  std::vector<double> gV(static_cast<size_t>(gnbf) * gnbf, 0.0);
  HostMatrixRef gR{gV.data(), gnbf, MatrixLayout::ColumnMajor};
  GTOSession l(gev, f.grid, XCFamily::MetaGGALapl);
  l.set_density(gPv);
  REQUIRE_THROWS_AS(l.accumulate_fock({.vrho = ones.data(),
                                       .vsigma = ones.data(),
                                       .vtau = ones.data()},
                                      nullptr, gR),
                    std::invalid_argument);
}

TEST_CASE("The collocation order each family needs is explicit",
          "[xckernel]") {
  // The kernel indexes chi + k*nbf*npts for every component it uses, so a
  // tower built to too low an order is read out of bounds. These are the
  // ncomp values the generated manifests declare.
  REQUIRE(basis_ncomp(collocation_deriv(XCFamily::LDA)) == 1);
  REQUIRE(basis_ncomp(collocation_deriv(XCFamily::GGA)) == 4);
  REQUIRE(basis_ncomp(collocation_deriv(XCFamily::MetaGGATau)) == 4);
  REQUIRE(basis_ncomp(collocation_deriv(XCFamily::MetaGGALapl)) == 10);
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

// ===========================================================================
//  Unrestricted (spin-polarized) path
// ===========================================================================

TEST_CASE("The kernel path reproduces the hand-written unrestricted LDA "
          "and GGA",
          "[xckernel][parity][unrestricted]") {
  // Same argument as the restricted parity test: host_interface's
  // eval_vxc(Pa, Pb) is validated against ERKALE, so it is the oracle, and
  // it is the only one available for the unrestricted kernels. Pa and Pb are
  // genuinely different matrices -- with Pa == Pb the cross term
  // sigma_ab = grad rho_a . grad rho_b would equal sigma_aa and a channel
  // reading the wrong sigma derivative would still agree.
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const auto Pa = scaled(make_P(nbf, 0.9), 0.6);
  const auto Pb = scaled(make_P(nbf, 1.7), 0.4);
  HostMatrixView Pav{Pa.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixView Pbv{Pb.data(), nbf, MatrixLayout::ColumnMajor};

  struct Case {
    int id;
    XCFamily fam;
    const char *name;
  };
  const Case cases[] = {{XC_LDA_X, XCFamily::LDA, "LDA_X"},
                        {XC_GGA_X_PBE, XCFamily::GGA, "GGA_X_PBE"},
                        {XC_GGA_C_PBE, XCFamily::GGA, "GGA_C_PBE"}};

  for (const auto &c : cases) {
    INFO("functional " << c.name);
    xc_func_type func;
    REQUIRE(xc_func_init(&func, c.id, XC_POLARIZED) == 0);

    std::vector<double> Varef(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<double> Vbref(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef Rar{Varef.data(), nbf, MatrixLayout::ColumnMajor};
    HostMatrixRef Rbr{Vbref.data(), nbf, MatrixLayout::ColumnMajor};
    // The oracle screens points whose total density is below its threshold
    // and the kernel path screens nothing, so the two are compared with the
    // screen effectively off. At ERKALE's default 1e-10 the dropped points
    // are worth 2e-11 in Exc on this density -- small, but a hundred times
    // the agreement the kernels actually reach, which would have had to be
    // hidden behind a loose tolerance. Lowering it to 1e-14 brings Exc to
    // 1.8e-15 relative. Not lower: eval_vxc would then hand Libxc densities
    // where it is entitled to return a NaN, and the nan_points assertion
    // below is what notices if that starts happening.
    const XCOutputPolarized ref = eval_vxc(ev, f.grid, Pav, Pbv, Rar, Rbr,
                                           func, XCThresholds{.density = 1e-14});
    REQUIRE(ref.nan_points == 0);

    Session s(ev, f.grid, c.fam);
    s.set_density(Pav, Pbv);
    const int ng = s.npoints();

    std::vector<double> zk(ng), vrho(2 * ng), vsigma(3 * ng);
    if (c.fam == XCFamily::LDA)
      xc_lda_exc_vxc(&func, ng, s.rho_spin().data(), zk.data(), vrho.data());
    else
      xc_gga_exc_vxc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
                     zk.data(), vrho.data(), vsigma.data());

    std::vector<double> Va(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<double> Vb(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef Ra{Va.data(), nbf, MatrixLayout::ColumnMajor};
    HostMatrixRef Rb{Vb.data(), nbf, MatrixLayout::ColumnMajor};
    XCPackedDerivsU d{.vrho = vrho.data()};
    if (c.fam != XCFamily::LDA)
      d.vsigma = vsigma.data();
    const XCOutputPolarized got = s.accumulate_fock(d, zk.data(), Ra, Rb);

    INFO("Exc hand-written " << ref.Exc << ", kernel " << got.Exc);
    REQUIRE_THAT(got.Nel_alpha,
                 Catch::Matchers::WithinRel(ref.Nel_alpha, 1e-12));
    REQUIRE_THAT(got.Nel_beta,
                 Catch::Matchers::WithinRel(ref.Nel_beta, 1e-12));
    REQUIRE_THAT(got.Exc, Catch::Matchers::WithinRel(ref.Exc, 1e-11));

    // The two channels must differ, or the cross term is untested.
    double spin_split = 0.0;
    for (size_t i = 0; i < Va.size(); ++i)
      spin_split = std::max(spin_split, std::abs(Va[i] - Vb[i]));

    double sa = 0.0, sb = 0.0;
    const double wa = worst_diff(Va, Varef, sa);
    const double wb = worst_diff(Vb, Vbref, sb);
    INFO("alpha worst " << wa << " scale " << sa << "; beta worst " << wb
                        << " scale " << sb << "; alpha-beta split "
                        << spin_split);
    REQUIRE(sa > 1e-3);
    REQUIRE(sb > 1e-3);
    REQUIRE(spin_split > 1e-4);
    // Measured: 1.1e-15, 6.9e-16 and 1.2e-15 for the three functionals.
    REQUIRE(wa < 1e-11 * sa);
    REQUIRE(wb < 1e-11 * sb);

    xc_func_end(&func);
  }
}

TEST_CASE("A closed-shell density gives the same XC potential restricted "
          "or unrestricted",
          "[xckernel][unrestricted][mgga][gto]") {
  // The acceptance test for the unrestricted meta-GGA kernels, which have no
  // hand-written oracle in NuKEXC and no ERKALE comparison yet.
  //
  // Restricted P parametrizes P^a = P^b = P/2, so for a closed-shell density
  //
  //     dExc/dP_uv = 1/2 (dExc/dP^a_uv + dExc/dP^b_uv) = dExc/dP^a_uv
  //
  // by symmetry. The unrestricted channels must therefore each equal the
  // restricted matrix exactly, not up to a factor. The relation is checked
  // here for LDA and GGA too, where both sides are independently validated
  // against the hand-written paths -- so if the identity itself were wrong
  // those cases would say so, and the meta-GGA cases would not be left
  // carrying an unverified premise.
  //
  // It holds bit for bit, for all four families: halving a density matrix is
  // exact in binary, and from there the two kernels' arithmetic coincides.
  // That is a stronger result than the tolerances below ask for, and the
  // tolerances are deliberately not tightened to zero -- a Libxc release
  // that reorders its polarized expressions would break an exact comparison
  // without anything being wrong.
  //
  // Exact agreement invites the suspicion that the comparison is vacuous, so
  // it was checked against a perturbation: with Pa, Pb = P/2 (1 +- 1e-6) the
  // alpha matrix moves 2.5e-7 relative and the two channels separate by
  // 5e-7, first order in the perturbation as they should be. The scale
  // assertion below is what keeps that from silently regressing to a
  // comparison of two zero matrices.
  Fixture f(40, 17);
  const auto host = test::sto3g_basis(f.mol);
  const GTOEvaluator ev(make_gto_device_basis(host));
  const int nbf = ev.nbf();
  const auto P = make_P(nbf);
  const auto Ph = scaled(P, 0.5);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixView Phv{Ph.data(), nbf, MatrixLayout::ColumnMajor};

  struct Case {
    int id;
    XCFamily fam;
    const char *name;
  };
  const Case cases[] = {
      {XC_LDA_X, XCFamily::LDA, "LDA_X"},
      {XC_GGA_X_PBE, XCFamily::GGA, "GGA_X_PBE"},
      {XC_MGGA_X_TPSS, XCFamily::MetaGGATau, "MGGA_X_TPSS (tau)"},
      {XC_MGGA_X_TPSS, XCFamily::MetaGGALapl, "MGGA_X_TPSS (tau and lapl)"}};

  for (const auto &c : cases) {
    INFO("functional " << c.name);
    GTOSession r(ev, f.grid, c.fam);
    GTOSession u(ev, f.grid, c.fam);
    r.set_density(Pv);
    u.set_density(Phv, Phv);
    const int ng = r.npoints();

    // The alpha and beta ingredients must come out equal, which is worth
    // asserting separately: it isolates a mistake in the density half from
    // a mistake in the Fock half.
    const auto &rs = u.rho_spin();
    for (int g = 0; g < ng; ++g) {
      REQUIRE_THAT(rs[2 * g + 1], Catch::Matchers::WithinAbs(rs[2 * g], 1e-14));
      REQUIRE_THAT(rs[2 * g] + rs[2 * g + 1],
                   Catch::Matchers::WithinRel(r.rho()[g], 1e-12));
    }

    // --- restricted ---
    std::vector<double> zk(ng), vrho(ng), vsigma(ng), vlapl(ng), vtau(ng);
    const std::vector<double> lapl_zero(ng, 0.0);
    {
      xc_func_type func;
      REQUIRE(xc_func_init(&func, c.id, XC_UNPOLARIZED) == 0);
      if (c.fam == XCFamily::LDA)
        xc_lda_exc_vxc(&func, ng, r.rho().data(), zk.data(), vrho.data());
      else if (c.fam == XCFamily::GGA)
        xc_gga_exc_vxc(&func, ng, r.rho().data(), r.sigma().data(), zk.data(),
                       vrho.data(), vsigma.data());
      else
        xc_mgga_exc_vxc(&func, ng, r.rho().data(), r.sigma().data(),
                        c.fam == XCFamily::MetaGGALapl ? r.lapl().data()
                                                       : lapl_zero.data(),
                        r.tau().data(), zk.data(), vrho.data(), vsigma.data(),
                        vlapl.data(), vtau.data());
      xc_func_end(&func);
    }
    std::vector<double> Vr(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef Rr{Vr.data(), nbf, MatrixLayout::ColumnMajor};
    XCKernelDerivs dr{.vrho = vrho.data()};
    if (c.fam != XCFamily::LDA)
      dr.vsigma = vsigma.data();
    if (c.fam == XCFamily::MetaGGATau || c.fam == XCFamily::MetaGGALapl)
      dr.vtau = vtau.data();
    if (c.fam == XCFamily::MetaGGALapl)
      dr.vlapl = vlapl.data();
    const XCOutput ref = r.accumulate_fock(dr, zk.data(), Rr);

    // --- unrestricted ---
    std::vector<double> zku(ng), vrhou(2 * ng), vsigmau(3 * ng),
        vlaplu(2 * ng), vtauu(2 * ng);
    const std::vector<double> lapl_zero_u(2 * ng, 0.0);
    {
      xc_func_type func;
      REQUIRE(xc_func_init(&func, c.id, XC_POLARIZED) == 0);
      if (c.fam == XCFamily::LDA)
        xc_lda_exc_vxc(&func, ng, u.rho_spin().data(), zku.data(),
                       vrhou.data());
      else if (c.fam == XCFamily::GGA)
        xc_gga_exc_vxc(&func, ng, u.rho_spin().data(), u.sigma_spin().data(),
                       zku.data(), vrhou.data(), vsigmau.data());
      else
        xc_mgga_exc_vxc(&func, ng, u.rho_spin().data(), u.sigma_spin().data(),
                        c.fam == XCFamily::MetaGGALapl ? u.lapl_spin().data()
                                                       : lapl_zero_u.data(),
                        u.tau_spin().data(), zku.data(), vrhou.data(),
                        vsigmau.data(), vlaplu.data(), vtauu.data());
      xc_func_end(&func);
    }
    std::vector<double> Va(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<double> Vb(static_cast<size_t>(nbf) * nbf, 0.0);
    HostMatrixRef Ra{Va.data(), nbf, MatrixLayout::ColumnMajor};
    HostMatrixRef Rb{Vb.data(), nbf, MatrixLayout::ColumnMajor};
    XCPackedDerivsU du{.vrho = vrhou.data()};
    if (c.fam != XCFamily::LDA)
      du.vsigma = vsigmau.data();
    if (c.fam == XCFamily::MetaGGATau || c.fam == XCFamily::MetaGGALapl)
      du.vtau = vtauu.data();
    if (c.fam == XCFamily::MetaGGALapl)
      du.vlapl = vlaplu.data();
    const XCOutputPolarized got = u.accumulate_fock(du, zku.data(), Ra, Rb);

    // vtau must be nonzero for the meta-GGA cases, or they prove nothing.
    if (c.fam != XCFamily::LDA && c.fam != XCFamily::GGA) {
      double vt = 0.0;
      for (double v : vtauu)
        vt = std::max(vt, std::abs(v));
      INFO("max |vtau_spin| = " << vt);
      REQUIRE(vt > 1e-6);
    }

    REQUIRE_THAT(got.Nel, Catch::Matchers::WithinRel(ref.Nel, 1e-12));
    REQUIRE_THAT(got.Nel_alpha,
                 Catch::Matchers::WithinRel(0.5 * ref.Nel, 1e-12));
    INFO("Exc restricted " << ref.Exc << ", unrestricted " << got.Exc);
    REQUIRE_THAT(got.Exc, Catch::Matchers::WithinRel(ref.Exc, 1e-11));

    double sa = 0.0, sb = 0.0;
    const double wa = worst_diff(Va, Vr, sa);
    const double wab = worst_diff(Va, Vb, sb);
    INFO("worst |V_alpha - V_restricted| = " << wa << ", scale " << sa
                                             << ", relative " << wa / sa);
    INFO("worst |V_alpha - V_beta| = " << wab);
    REQUIRE(sa > 1e-3);
    REQUIRE(wab < 1e-13 * sa);
    REQUIRE(wa < 1e-11 * sa);
  }
}

TEST_CASE("The unrestricted kernel Fock matrices are symmetric",
          "[xckernel][unrestricted]") {
  // accumulate_fock copies each kernel's (nbf, nbf) block out without caring
  // which of the two orderings it wrote, exactly as the restricted path
  // does. Assert it per channel rather than rely on it.
  Fixture f(30, 17);
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const auto Pa = scaled(make_P(nbf, 0.9), 0.6);
  const auto Pb = scaled(make_P(nbf, 1.7), 0.4);
  HostMatrixView Pav{Pa.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixView Pbv{Pb.data(), nbf, MatrixLayout::ColumnMajor};

  Session s(ev, f.grid, XCFamily::GGA);
  s.set_density(Pav, Pbv);
  const int ng = s.npoints();

  xc_func_type func;
  REQUIRE(xc_func_init(&func, XC_GGA_X_PBE, XC_POLARIZED) == 0);
  std::vector<double> zk(ng), vrho(2 * ng), vsigma(3 * ng);
  xc_gga_exc_vxc(&func, ng, s.rho_spin().data(), s.sigma_spin().data(),
                 zk.data(), vrho.data(), vsigma.data());
  xc_func_end(&func);

  std::vector<double> Va(static_cast<size_t>(nbf) * nbf, 0.0);
  std::vector<double> Vb(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef Ra{Va.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixRef Rb{Vb.data(), nbf, MatrixLayout::ColumnMajor};
  s.accumulate_fock({.vrho = vrho.data(), .vsigma = vsigma.data()}, zk.data(),
                    Ra, Rb);

  for (const auto *V : {&Va, &Vb}) {
    double worst = 0.0, scale = 0.0;
    for (int i = 0; i < nbf; ++i)
      for (int j = 0; j < nbf; ++j) {
        const double a = (*V)[static_cast<size_t>(i) +
                              static_cast<size_t>(j) * nbf];
        const double b = (*V)[static_cast<size_t>(j) +
                              static_cast<size_t>(i) * nbf];
        worst = std::max(worst, std::abs(a - b));
        scale = std::max(scale, std::abs(a));
      }
    INFO("worst asymmetry " << worst << ", scale " << scale);
    REQUIRE(scale > 1e-3);
    REQUIRE(worst < 1e-13 * scale);
  }
}

TEST_CASE("Mixing the spin cases up is caught", "[xckernel][unrestricted]") {
  // The session holds one density at a time. Reaching for the ingredients of
  // the spin case that is not currently set used to be possible and would
  // have returned whatever the previous call left in the buffers, which is
  // the kind of mistake that shows up as a slightly wrong energy rather than
  // as a crash.
  Fixture f(20, 11);
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const auto P = make_P(nbf);
  const auto Ph = scaled(P, 0.5);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixView Phv{Ph.data(), nbf, MatrixLayout::ColumnMajor};

  Session s(ev, f.grid, XCFamily::GGA);
  const int ng = static_cast<int>(f.grid.quad_points.extent(0));
  std::vector<double> ones2(2 * ng, 1.0), ones3(3 * ng, 1.0);
  std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
  std::vector<double> W(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef Ra{V.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixRef Rb{W.data(), nbf, MatrixLayout::ColumnMajor};

  // Nothing set yet: both spin cases refuse.
  REQUIRE_THROWS_AS(s.rho(), std::logic_error);
  REQUIRE_THROWS_AS(s.rho_spin(), std::logic_error);

  s.set_density(Pv);
  REQUIRE_NOTHROW(s.rho());
  REQUIRE_THROWS_AS(s.rho_spin(), std::logic_error);
  REQUIRE_THROWS_AS(s.sigma_spin(), std::logic_error);
  REQUIRE_THROWS_AS(s.nel_alpha(), std::logic_error);
  REQUIRE_THROWS_AS(
      s.accumulate_fock({.vrho = ones2.data(), .vsigma = ones3.data()},
                        nullptr, Ra, Rb),
      std::logic_error);

  s.set_density(Phv, Phv);
  REQUIRE_NOTHROW(s.rho_spin());
  REQUIRE_NOTHROW(s.sigma_spin());
  REQUIRE_THROWS_AS(s.rho(), std::logic_error);
  REQUIRE_THROWS_AS(s.sigma(), std::logic_error);
  REQUIRE_THROWS_AS(s.nel(), std::logic_error);

  // A tau-only session has no polarized Laplacian to hand out either.
  REQUIRE_THROWS_AS(s.tau_spin(), std::logic_error);
  REQUIRE_THROWS_AS(s.lapl_spin(), std::logic_error);

  // Missing operands, as for the restricted path.
  REQUIRE_THROWS_AS(s.accumulate_fock({.vrho = ones2.data()}, nullptr, Ra, Rb),
                    std::invalid_argument);
  // One channel alone is not a request.
  REQUIRE_THROWS_AS(
      s.accumulate_fock({.vrho = ones2.data(), .vsigma = ones3.data()},
                        nullptr, Ra, HostMatrixRef{nullptr, nbf}),
      std::invalid_argument);
  // Neither channel is: the energy alone.
  REQUIRE_NOTHROW(s.accumulate_fock(
      {.vrho = ones2.data(), .vsigma = ones3.data()}, nullptr,
      HostMatrixRef{nullptr, nbf}, HostMatrixRef{nullptr, nbf}));
}
