/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The screened XC quadrature against the dense one.
//
// WHY THIS TEST EXISTS
// --------------------
// The screened polarized quadrature had no test at all: its
// only caller was the standalone SCF driver, which checks a converged total
// energy and so cannot say whether the screened quadrature is right, only
// whether it is close enough not to break convergence. That left the whole
// screened path -- the neighbour list over shells, the per-box packing, the
// local-to-global function map and the density contraction -- resting on an
// integrated quantity.
//
// The dense path is the reference because it shares nothing with the
// screened one but the basis and the grid: it builds a full (nbf, npts)
// collocation, contracts through the occupied orbitals, and reaches V with
// a single global GEMM. The screened path packs per box, contracts through
// the density matrix, and scatters. Agreement therefore pins the local
// numbering and the packing at once; an indexing error would permute rows
// rather than perturb them, and no tolerance would hide it.
//
// THE BASIS MUST HAVE MULTI-FUNCTION SHELLS
// -----------------------------------------
// A Slater basis has one function per shell, so shell indices and function
// indices coincide and a kernel that confuses them still passes. STO-3G on
// water has a p shell, so it does not. Both bases are run: the Gaussian one
// to exercise the case, the Slater one because that is what the SCF driver
// uses.
//
// WHAT THE TOLERANCE MEANS
// ------------------------
// Screening drops contributions from shells whose cutoff radius does not
// reach a box, so the two paths need not agree bit for bit -- but on these
// fixtures the dropped tails turn out to be below round-off: the worst
// relative difference measured is 2.1e-13, so the tolerance is 1e-11 and not
// the 1e-8 that was here when the number had never been looked at. It is
// relative to the largest element of the reference matrix, since the small
// elements of V are differences of larger numbers.
//
// The comparison also got stronger than it was. The screened path now takes
// its potential from libxckernel's generated contraction while the dense one
// still has the chain rule written out by hand, so this is a parity test
// between two independent derivations of the same functional derivative --
// not, as it was, between two spellings of one.

#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/basis_evaluator.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>
#include <nukexc/xc_integrals.hpp>
#include <nukexc/xc_screened.hpp>

#include <xc.h>
#include <xc_funcs.h>

#include "sto_basis_reader.hpp"
#include "test_basis_sets.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

/// Deterministic occupied orbitals. Any real coefficients give a
/// non-negative density, so these need not be physical -- what they must be
/// is dense enough that every basis function contributes, which eigenvectors
/// of a model Hamiltonian would not guarantee.
DeviceView2DLeft model_orbitals(const int nbf, const int n_occ,
                                const int seed) {
  DeviceView2DLeft C("C", nbf, n_occ);
  auto h = Kokkos::create_mirror_view(C);
  for (int i = 0; i < n_occ; ++i) {
    double norm = 0.0;
    for (int mu = 0; mu < nbf; ++mu) {
      const double x = std::cos(0.7 * (mu + 1) * (i + 1) + 0.31 * seed) *
                       std::exp(-0.08 * mu);
      h(mu, i) = x;
      norm += x * x;
    }
    const double s = 1.0 / std::sqrt(norm);
    for (int mu = 0; mu < nbf; ++mu)
      h(mu, i) *= s;
  }
  Kokkos::deep_copy(C, h);
  return C;
}

DeviceView1D unit_occupations(const int n_occ) {
  DeviceView1D occ("occ", n_occ);
  Kokkos::deep_copy(occ, 1.0);
  return occ;
}

std::vector<double> to_host(const DeviceView2DLeft &m) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, m);
  const int n0 = static_cast<int>(m.extent(0));
  const int n1 = static_cast<int>(m.extent(1));
  std::vector<double> out(static_cast<size_t>(n0) * n1);
  for (int i = 0; i < n0; ++i)
    for (int j = 0; j < n1; ++j)
      out[static_cast<size_t>(i) * n1 + j] = h(i, j);
  return out;
}

/// Worst absolute difference and the reference's scale.
double worst(const std::vector<double> &got, const std::vector<double> &ref,
             double &scale) {
  REQUIRE(got.size() == ref.size());
  double w = 0.0;
  scale = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    w = std::max(w, std::abs(got[i] - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  return w;
}

/// Everything both paths need: a grid, its boxes, and the dense collocation
/// on that same -- already permuted -- grid.
template <typename Evaluator> struct Fixture {
  FlatGrid grid;
  NeighborList nl;
  DeviceView2DLeft chi, gx, gy, gz;
  DeviceView2DLeft Ca, Cb;
  DeviceView1D occa, occb;
};

template <typename Evaluator>
Fixture<Evaluator> make_fixture(const Molecule &mol, const Evaluator &ev,
                                const int nrad, const int nang,
                                const int n_occ_a, const int n_occ_b) {
  Fixture<Evaluator> f;
  f.grid = make_flat_grid<ta_type, ll_type>(mol, nrad, nang);

  // create_bounding_boxes permutes the grid in place, so it has to run
  // before the collocation is filled: both paths must see the same order.
  const int mppb = 128;
  auto bb = create_bounding_boxes(f.grid, mppb);
  build_neighbor_list(ev, bb, mppb, f.grid.quad_points, f.nl);

  const int nbf = ev.nbf();
  const int npts = static_cast<int>(f.grid.quad_points.extent(0));
  f.chi = DeviceView2DLeft("chi", nbf, npts);
  f.gx = DeviceView2DLeft("gx", nbf, npts);
  f.gy = DeviceView2DLeft("gy", nbf, npts);
  f.gz = DeviceView2DLeft("gz", nbf, npts);
  fill_collocation_value_grad(ExecSpace{}, ev, f.grid.quad_points, f.chi, f.gx,
                              f.gy, f.gz);
  Kokkos::fence();

  f.Ca = model_orbitals(nbf, n_occ_a, 0);
  f.Cb = model_orbitals(nbf, n_occ_b, 1);
  f.occa = unit_occupations(n_occ_a);
  f.occb = unit_occupations(n_occ_b);
  return f;
}

struct Functional {
  xc_func_type func;
  explicit Functional(int id) {
    REQUIRE(xc_func_init(&func, id, XC_POLARIZED) == 0);
  }
  ~Functional() { xc_func_end(&func); }
  Functional(const Functional &) = delete;
  Functional &operator=(const Functional &) = delete;
};

void compare(const XC_result_polarized &scr, const XC_result_polarized &dense,
             const double tol) {
  INFO("energy: screened " << scr.energy << ", dense " << dense.energy);
  REQUIRE(std::abs(dense.energy) > 1e-6);
  REQUIRE_THAT(scr.energy, Catch::Matchers::WithinRel(dense.energy, tol));

  struct Pair {
    const char *name;
    DeviceView2DLeft a, b;
  };
  for (const auto &pr :
       {Pair{"V alpha", scr.potential_alpha, dense.potential_alpha},
        Pair{"V beta", scr.potential_beta, dense.potential_beta}}) {
    const auto got = to_host(pr.a);
    const auto ref = to_host(pr.b);
    double scale = 0.0;
    const double w = worst(got, ref, scale);
    INFO(pr.name << ": worst |screened - dense| = " << w << ", scale "
                 << scale << ", relative " << w / scale);
    REQUIRE(scale > 1e-6);
    REQUIRE(w < tol * scale);
  }
}

} // namespace

TEST_CASE("The screened LSDA matches the dense one for a Gaussian basis",
          "[xc][screened][lsda][gto]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::sto3g_basis(mol);
  const GTOEvaluator ev(make_gto_device_basis(host));
  REQUIRE(ev.nblock() < ev.nbf()); // multi-function shells, or vacuous

  const auto f = make_fixture(mol, ev, 75, 29, 5, 4);
  const Functional fn(XC_LDA_X);

  const auto dense = compute_lsda(f.chi, f.grid.weights, f.Ca, f.occa, f.Cb,
                                  f.occb, fn.func);
  const auto scr = compute_xc_polarized_sparse(ev, f.grid, f.nl, f.Ca, f.occa, f.Cb,
                                       f.occb, fn.func);
  compare(scr, dense, 1e-11);
}

TEST_CASE("The screened GGA matches the dense one for a Gaussian basis",
          "[xc][screened][gga][gto]") {
  // The gradient term is the one that can go wrong silently: it sits on one
  // index only, so a packing error in the gradient block shows up as an
  // asymmetric V rather than a wrong number, and the symmetrization then
  // hides half of it.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::sto3g_basis(mol);
  const GTOEvaluator ev(make_gto_device_basis(host));
  REQUIRE(ev.nblock() < ev.nbf());

  const auto f = make_fixture(mol, ev, 75, 29, 5, 4);
  const Functional fn(XC_GGA_X_PBE);

  const auto dense = compute_gga_lsda(f.chi, f.gx, f.gy, f.gz, f.grid.weights,
                                      f.Ca, f.occa, f.Cb, f.occb, fn.func);
  const auto scr = compute_xc_polarized_sparse(ev, f.grid, f.nl, f.Ca, f.occa,
                                           f.Cb, f.occb, fn.func);
  compare(scr, dense, 1e-11);
}

TEST_CASE("The screened XC quadrature also serves a Slater basis",
          "[xc][screened][sto]") {
  // What the SCF driver actually runs -- and, since the Slater evaluator
  // became shell-based, a case that CAN catch a shell/function confusion:
  // TZP on water is 17 shells carrying 31 functions. It used to be 31
  // single-function shells, where the two indexings coincide and any
  // confusion between them hides.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto basis = test::load_adf_basis(mol, "input/zorabasis/TZP", 1e-10);
  const STOPrimitiveEvaluator ev(basis);
  REQUIRE(ev.nblock() < ev.nbf());

  const auto f = make_fixture(mol, ev, 50, 21, 5, 4);

  const Functional lda(XC_LDA_X);
  compare(compute_xc_polarized_sparse(ev, f.grid, f.nl, f.Ca, f.occa, f.Cb, f.occb,
                              lda.func),
          compute_lsda(f.chi, f.grid.weights, f.Ca, f.occa, f.Cb, f.occb,
                       lda.func),
          1e-11);

  const Functional gga(XC_GGA_X_PBE);
  compare(compute_xc_polarized_sparse(ev, f.grid, f.nl, f.Ca, f.occa, f.Cb,
                                  f.occb, gga.func),
          compute_gga_lsda(f.chi, f.gx, f.gy, f.gz, f.grid.weights, f.Ca,
                           f.occa, f.Cb, f.occb, gga.func),
          1e-11);
}

TEST_CASE("The screened density is the same density the dense path uses",
          "[xc][screened][density]") {
  // Isolates the half of the comparison that the potential test cannot: the
  // screened path contracts through the density matrix and the dense one
  // through the occupied orbitals, so if the two densities differ the
  // potentials differ for a reason that has nothing to do with the
  // packing. rho is also where the old kernel paid a factor N_occ in
  // evaluations, and the number of electrons it integrates to is the one
  // quantity a reader can check against the input by hand.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::sto3g_basis(mol);
  const GTOEvaluator ev(make_gto_device_basis(host));

  const auto f = make_fixture(mol, ev, 99, 41, 5, 4);
  const PackedBlockLayout L = build_packed_block_layout(ev, f.nl);

  const int npts = static_cast<int>(f.grid.quad_points.extent(0));
  DeviceView1D rho("rho", npts), gx("gx", npts), gy("gy", npts),
      gz("gz", npts);
  packed_density<BasisDeriv::Gradient>(
      ev, f.grid, f.nl, L, density_matrix_from_orbitals(f.Ca, f.occa), rho, gx,
      gy, gz);

  DeviceView1D rho_ref("rho ref", npts), gxr("gxr", npts), gyr("gyr", npts),
      gzr("gzr", npts), sig("sig", npts);
  compute_density_and_sigma(f.chi, f.gx, f.gy, f.gz, f.Ca, f.occa, rho_ref,
                            gxr, gyr, gzr, sig);
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_ref);
  auto hw = Kokkos::create_mirror_view_and_copy(HostSpace{}, f.grid.weights);
  auto hgx = Kokkos::create_mirror_view_and_copy(HostSpace{}, gx);
  auto hgxr = Kokkos::create_mirror_view_and_copy(HostSpace{}, gxr);

  double worst_rho = 0.0, worst_grad = 0.0, peak = 0.0, nelec = 0.0,
         nelec_ref = 0.0, gpeak = 0.0;
  for (int g = 0; g < npts; ++g) {
    worst_rho = std::max(worst_rho, std::abs(h(g) - hr(g)));
    worst_grad = std::max(worst_grad, std::abs(hgx(g) - hgxr(g)));
    peak = std::max(peak, std::abs(hr(g)));
    gpeak = std::max(gpeak, std::abs(hgxr(g)));
    nelec += hw(g) * h(g);
    nelec_ref += hw(g) * hr(g);
  }
  INFO("peak rho " << peak << ", worst |drho| " << worst_rho
                   << ", worst |dgrad_x| " << worst_grad << " of " << gpeak);
  REQUIRE(peak > 1e-3);
  REQUIRE(worst_rho < 1e-10 * peak);
  REQUIRE(worst_grad < 1e-10 * gpeak);

  // The integral is sum_i occ_i C_i^T S C_i. The model coefficients are
  // normalized as vectors rather than in the overlap metric, so this is NOT
  // the orbital count -- it is near it only because S has a unit diagonal.
  // What the integral is good for is that it is a global check: a packing
  // error that cancelled in the worst-element comparison above would still
  // move it.
  INFO("electrons: packed " << nelec << ", dense " << nelec_ref);
  REQUIRE_THAT(nelec, Catch::Matchers::WithinRel(nelec_ref, 1e-10));
  REQUIRE(nelec > 1.0);
  REQUIRE(nelec < 2.0 * 5.0);
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
