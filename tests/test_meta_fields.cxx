/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The meta-GGA density ingredients: tau and the density Laplacian.
//
// tau gets an EXACT oracle rather than a physical one. compute_core_
// hamiltonian forms the kinetic matrix by quadrature on the same grid,
// T_uv = 1/2 sum_g w_g grad chi_u . grad chi_v, so
//
//     sum_g w_g tau_g  ==  Tr(P T)
//
// identically -- the same products and weights, only a different summation
// order. Any error in the tau contraction (a missing half, a wrong
// transpose, the gradient of the wrong function) breaks it well outside
// rounding, whereas a test against an analytic kinetic energy would be
// limited by the quadrature and could not separate a contraction bug from
// grid error.
//
// WHY TWO BASES
// -------------
// tau needs only gradients, so it is tested on the Slater basis, which is
// what keeps compute_core_hamiltonian's kinetic matrix available as the
// oracle: all five core-Hamiltonian variants go through load_shell, which
// static_asserts one basis function per block, so none of them can be
// instantiated with the Gaussian evaluator's multi-function blocks.
//
// The density Laplacian needs the collocation Hessian, which the Slater
// evaluators do not provide (max_deriv == BasisDeriv::Gradient), so it is
// tested on Gaussians -- and there the reference is a finite-difference
// Laplacian of the density rather than a matrix identity.

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/basis_evaluator.hpp>
#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/density.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/stobasis.hpp>

#include "gto_test_basis.hpp"
#include "sto_basis_reader.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

/// Gradient-order collocation, split for the density contractions.
struct Grad {
  DeviceView2DLeft chi, gx, gy, gz;
};

/// Gradient-order collocation plus the Laplacian (the Hessian's trace).
struct Hess {
  DeviceView2DLeft chi, gx, gy, gz, lapl;
};

template <typename Evaluator>
Grad build_grad(const Evaluator &ev, const Kokkos::View<Point *> &pts) {
  const int n = ev.nbf();
  const int ng = static_cast<int>(pts.extent(0));
  Kokkos::View<double ***> comp("comp", basis_ncomp(BasisDeriv::Gradient), n,
                                ng);
  fill_collocation<BasisDeriv::Gradient>(ExecSpace{}, ev, pts, comp);
  Grad c;
  c.chi = DeviceView2DLeft("chi", n, ng);
  c.gx = DeviceView2DLeft("gx", n, ng);
  c.gy = DeviceView2DLeft("gy", n, ng);
  c.gz = DeviceView2DLeft("gz", n, ng);
  auto chi = c.chi, gx = c.gx, gy = c.gy, gz = c.gz;
  Kokkos::parallel_for(
      "split grad",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, ng}),
      KOKKOS_LAMBDA(const int i, const int g) {
        chi(i, g) = comp(BF_VAL, i, g);
        gx(i, g) = comp(BF_DX, i, g);
        gy(i, g) = comp(BF_DY, i, g);
        gz(i, g) = comp(BF_DZ, i, g);
      });
  Kokkos::fence();
  return c;
}

Hess build_hess(const GTOEvaluator &ev, const Kokkos::View<Point *> &pts) {
  const int n = ev.nbf();
  const int ng = static_cast<int>(pts.extent(0));
  Kokkos::View<double ***> comp("comp", basis_ncomp(BasisDeriv::Hessian), n,
                                ng);
  fill_collocation<BasisDeriv::Hessian>(ExecSpace{}, ev, pts, comp);
  Hess c;
  c.chi = DeviceView2DLeft("chi", n, ng);
  c.gx = DeviceView2DLeft("gx", n, ng);
  c.gy = DeviceView2DLeft("gy", n, ng);
  c.gz = DeviceView2DLeft("gz", n, ng);
  c.lapl = DeviceView2DLeft("lapl", n, ng);
  auto chi = c.chi, gx = c.gx, gy = c.gy, gz = c.gz, lp = c.lapl;
  Kokkos::parallel_for(
      "split hess",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, ng}),
      KOKKOS_LAMBDA(const int i, const int g) {
        chi(i, g) = comp(BF_VAL, i, g);
        gx(i, g) = comp(BF_DX, i, g);
        gy(i, g) = comp(BF_DY, i, g);
        gz(i, g) = comp(BF_DZ, i, g);
        lp(i, g) =
            comp(BF_DXX, i, g) + comp(BF_DYY, i, g) + comp(BF_DZZ, i, g);
      });
  Kokkos::fence();
  return c;
}

/// A symmetric, non-diagonal density matrix of a plausible scale.
std::vector<double> make_P_host(int nbf) {
  std::vector<double> P(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j <= i; ++j) {
      const double v =
          0.04 * std::cos(0.9 * (i + 1)) * std::cos(0.9 * (j + 1));
      P[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf] = v;
      P[static_cast<size_t>(j) + static_cast<size_t>(i) * nbf] = v;
    }
  for (int i = 0; i < nbf; ++i)
    P[static_cast<size_t>(i) + static_cast<size_t>(i) * nbf] += 0.05;
  return P;
}

DeviceView2DLeft upload_P(const std::vector<double> &P, int nbf) {
  DeviceView2DLeft d("P", nbf, nbf);
  auto h = Kokkos::create_mirror_view(d);
  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j < nbf; ++j)
      h(i, j) = P[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf];
  Kokkos::deep_copy(d, h);
  return d;
}

double integrate(const DeviceView1D &f, const Kokkos::View<double *> &w) {
  double acc = 0.0;
  Kokkos::parallel_reduce(
      "integrate", Kokkos::RangePolicy<ExecSpace>(0, f.extent(0)),
      KOKKOS_LAMBDA(const int g, double &a) { a += w(g) * f(g); }, acc);
  return acc;
}

} // namespace

TEST_CASE("tau integrates to Tr(P T) exactly", "[meta][tau]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto basis = test::load_adf_basis(mol, "input/zorabasis/TZP", 1e-10);
  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 60, 23);
  const STOPrimitiveEvaluator ev(basis);
  const int nbf = ev.nbf();
  const int ng = static_cast<int>(grid.quad_points.extent(0));
  INFO("nbf = " << nbf << ", ng = " << ng);

  const auto core = compute_core_hamiltonian(ev, grid);
  const auto Ph = make_P_host(nbf);
  const auto P = upload_P(Ph, nbf);
  const auto c = build_grad(ev, grid.quad_points);

  // Empty Laplacian collocation: tau only, which is all a Slater basis can
  // supply and all this identity needs.
  DeviceView1D tau("tau", ng), lapl;
  compute_meta_fields_dm(c.chi, c.gx, c.gy, c.gz, DeviceView2DLeft{}, P, tau,
                         lapl);
  Kokkos::fence();

  auto Th = Kokkos::create_mirror_view_and_copy(HostSpace{}, core.kinetic);
  double trace = 0.0;
  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j < nbf; ++j)
      trace += Ph[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf] *
               Th(j, i);

  const double integral = integrate(tau, grid.weights);
  INFO("integral w tau = " << integral << ", Tr(P T) = " << trace);
  REQUIRE(std::abs(trace) > 1.0); // vacuous at zero
  REQUIRE_THAT(integral, Catch::Matchers::WithinRel(trace, 1e-11));
}

TEST_CASE("tau is positive and respects the von Weizsacker bound",
          "[meta][tau]") {
  // tau >= |grad rho|^2 / (8 rho) pointwise, with equality only for a
  // one-orbital density. Independent of the kinetic matrix, so it catches a
  // tau of the right size but the wrong shape -- which the integral
  // identity alone would not.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto basis = test::load_adf_basis(mol, "input/zorabasis/TZP", 1e-10);
  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 50, 21);
  const STOPrimitiveEvaluator ev(basis);
  const int nbf = ev.nbf();
  const int ng = static_cast<int>(grid.quad_points.extent(0));

  const auto P = upload_P(make_P_host(nbf), nbf);
  const auto c = build_grad(ev, grid.quad_points);

  DeviceView1D tau("tau", ng), lapl;
  compute_meta_fields_dm(c.chi, c.gx, c.gy, c.gz, DeviceView2DLeft{}, P, tau,
                         lapl);
  DeviceView1D rho("rho", ng), gx("gx", ng), gy("gy", ng), gz("gz", ng),
      sigma("sigma", ng);
  compute_density_and_sigma_dm(c.chi, c.gx, c.gy, c.gz, P, rho, gx, gy, gz,
                              sigma);
  Kokkos::fence();

  auto th = Kokkos::create_mirror_view_and_copy(HostSpace{}, tau);
  auto rh = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);
  auto sh = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma);

  int checked = 0, negative = 0;
  double worst = 0.0;
  for (int g = 0; g < ng; ++g) {
    if (th(g) < 0.0)
      ++negative;
    if (rh(g) < 1e-8)
      continue; // bound is 0/0 where there is no density
    const double vw = sh(g) / (8.0 * rh(g));
    if (vw - th(g) > 1e-10 * std::max(1.0, std::abs(th(g))))
      worst = std::max(worst, vw - th(g));
    ++checked;
  }
  INFO("checked " << checked << " of " << ng << ", " << negative
                  << " negative, worst violation " << worst);
  REQUIRE(checked > ng / 20);
  REQUIRE(negative == 0);
  REQUIRE(worst == 0.0);
}

TEST_CASE("The density Laplacian integrates to zero", "[meta][lapl]") {
  // Vanishes over all space for a decaying density, by the divergence
  // theorem. Limited by the quadrature, unlike the tau identity, so the
  // tolerance is relative to the scale of the integrand.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::gto_basis(mol);
  const GTOEvaluator ev(make_gto_device_basis(host));
  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 99, 41);
  const int nbf = ev.nbf();
  const int ng = static_cast<int>(grid.quad_points.extent(0));
  INFO("GTO nbf = " << nbf << ", ng = " << ng);

  const auto P = upload_P(make_P_host(nbf), nbf);
  const auto c = build_hess(ev, grid.quad_points);

  DeviceView1D tau("tau", ng), lapl("lapl", ng);
  compute_meta_fields_dm(c.chi, c.gx, c.gy, c.gz, c.lapl, P, tau, lapl);
  Kokkos::fence();

  auto lh = Kokkos::create_mirror_view_and_copy(HostSpace{}, lapl);
  auto wh = Kokkos::create_mirror_view_and_copy(HostSpace{}, grid.weights);
  double scale = 0.0;
  for (int g = 0; g < ng; ++g)
    scale += wh(g) * std::abs(lh(g));

  const double integral = integrate(lapl, grid.weights);
  INFO("integral w lapl rho = " << integral
                               << ", integral w |lapl rho| = " << scale);
  REQUIRE(scale > 1.0);
  REQUIRE(std::abs(integral) < 1e-3 * scale);
}

TEST_CASE("The density Laplacian matches finite differences", "[meta][lapl]") {
  // Independent of the integral identity and of the collocation Hessian's
  // packing: differentiate the density numerically. This is what catches a
  // wrong Hessian component order, which the integral test would miss
  // because a permuted trace is still a trace.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::gto_basis(mol);
  const GTOEvaluator ev(make_gto_device_basis(host));
  const int nbf = ev.nbf();
  const auto P = upload_P(make_P_host(nbf), nbf);

  auto cen = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.atom_centers);
  std::vector<Point> probe;
  for (double d : {0.35, 0.8, 1.6}) {
    Point p;
    p[0] = cen(0)[0] + 0.53 * d;
    p[1] = cen(0)[1] + 0.71 * d;
    p[2] = cen(0)[2] + 0.46 * d;
    probe.push_back(p);
  }

  // 1 + 6 point stencil per probe point.
  const double h = 5e-4;
  std::vector<Point> all;
  for (const auto &p : probe) {
    all.push_back(p);
    for (int d = 0; d < 3; ++d)
      for (int s : {+1, -1}) {
        Point q = p;
        q[d] += s * h;
        all.push_back(q);
      }
  }
  const int np = static_cast<int>(all.size());
  Kokkos::View<Point *> pts("probe", np);
  auto ph = Kokkos::create_mirror_view(pts);
  for (int i = 0; i < np; ++i)
    ph(i) = all[i];
  Kokkos::deep_copy(pts, ph);

  const auto c = build_hess(ev, pts);
  DeviceView1D tau("tau", np), lapl("lapl", np);
  compute_meta_fields_dm(c.chi, c.gx, c.gy, c.gz, c.lapl, P, tau, lapl);
  const auto rho_d = compute_density_dm(c.chi, P);
  Kokkos::fence();
  auto lh = Kokkos::create_mirror_view_and_copy(HostSpace{}, lapl);
  auto rho = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_d);

  for (size_t k = 0; k < probe.size(); ++k) {
    const int base = static_cast<int>(k) * 7;
    double fd = 0.0;
    for (int d = 0; d < 3; ++d)
      fd += (rho(base + 1 + 2 * d) + rho(base + 2 + 2 * d) -
             2.0 * rho(base)) /
            (h * h);
    INFO("probe " << k << ": analytic " << lh(base) << ", finite difference "
                  << fd);
    REQUIRE_THAT(lh(base), Catch::Matchers::WithinRel(fd, 1e-4));
  }
}

TEST_CASE("A mismatched Laplacian collocation is rejected", "[meta]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto basis = test::load_adf_basis(mol, "input/zorabasis/TZP", 1e-10);
  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 20, 11);
  const STOPrimitiveEvaluator ev(basis);
  const int nbf = ev.nbf();
  const int ng = static_cast<int>(grid.quad_points.extent(0));

  const auto P = upload_P(make_P_host(nbf), nbf);
  const auto c = build_grad(ev, grid.quad_points);
  DeviceView1D tau("tau", ng), lapl("lapl", ng);

  // Right rank, wrong shape: empty means "tau only", but a nonempty view of
  // the wrong size is a mistake rather than a request.
  DeviceView2DLeft wrong("wrong", nbf, ng / 2);
  REQUIRE_THROWS_AS(
      compute_meta_fields_dm(c.chi, c.gx, c.gy, c.gz, wrong, P, tau, lapl),
      std::invalid_argument);
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
