/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The core Hamiltonian over SHELLS rather than single basis functions.
//
// The quadrature does not care what the radial part of a basis function is.
// It needs shells on centres, a number of functions per shell, a range per
// shell for screening, and a way to evaluate a shell's functions and their
// gradients at a point. That is the whole of the BasisEvaluator concept, and
// a routine written against it should work for Gaussians, Slater functions
// and numerical atomic orbitals alike.
//
// compute_core_hamiltonian did not. It was templated over the concept, but
// walked basis functions one at a time through load_shell, whose
// static_assert requires one function per block. Every basis with a p, d or
// f shell violates that, so the routine could not be instantiated with the
// Gaussian evaluator at all -- which meant the Gaussian path could not form
// S, T or V_n, and therefore could not drive an SCF. The Slater evaluators
// happen to have one function per shell, which is why the defect was
// invisible.
//
// So the first thing these tests assert is that the thing compiles and runs
// for a multi-function basis, and the second is that it gets the right
// answer, against a closed form derived by a completely different route.

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/basis_evaluator.hpp>
#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>

#include "gto_analytic_overlap.hpp"
#include "sto_basis_reader.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

double worst_rel(const std::vector<double> &got, const std::vector<double> &ref,
                 double &scale) {
  double w = 0.0;
  scale = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    w = std::max(w, std::abs(got[i] - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  return w;
}

std::vector<double> to_host(const DeviceView2DLeft &m) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, m);
  const int n = static_cast<int>(m.extent(0));
  std::vector<double> out(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      out[static_cast<size_t>(i) * n + j] = h(i, j);
  return out;
}

} // namespace

TEST_CASE("The core Hamiltonian accepts a multi-function-shell basis",
          "[coreh][shells][gto]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::water_basis(mol, ShellType::Spherical);
  const GTOEvaluator ev(make_gto_device_basis(host));

  // The basis must actually exercise the case, or the test is vacuous: at
  // least one shell holding more than one function.
  int biggest = 0;
  for (int b = 0; b < ev.nblock(); ++b)
    biggest = std::max(biggest, ev.block_size(b));
  INFO("nbf = " << ev.nbf() << ", nblock = " << ev.nblock()
                << ", largest shell = " << biggest);
  REQUIRE(ev.nblock() < ev.nbf()); // some shell holds several functions
  REQUIRE(biggest >= 3);           // the p shell

  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 99, 41);
  const auto core = compute_core_hamiltonian(ev, grid);

  REQUIRE(core.overlap.extent(0) == static_cast<size_t>(ev.nbf()));
  const auto S = to_host(core.overlap);
  for (double v : S)
    REQUIRE(std::isfinite(v));
}

TEST_CASE("The quadrature overlap matches the closed form for Gaussians",
          "[coreh][shells][gto]") {
  // The real check. compute_core_hamiltonian's S is a pure quadrature
  // construction; analytic_overlap is a closed-form Gaussian integral.
  // Nothing is shared between them but the basis, so agreement pins the
  // shell indexing, the function ordering within a shell, the
  // normalizations and the weights all at once. A wrong block_first_bf
  // would permute rows; a wrong block_size would drop or duplicate
  // functions; both show up here and neither would show in a trace.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::water_basis(mol, ShellType::Spherical);
  const GTOEvaluator ev(make_gto_device_basis(host));
  const int nbf = ev.nbf();

  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 150, 59);
  const auto core = compute_core_hamiltonian(ev, grid);

  const auto S = to_host(core.overlap);
  const auto Sref = test::analytic_overlap(host);
  REQUIRE(S.size() == Sref.size());

  double scale = 0.0;
  const double w = worst_rel(S, Sref, scale);
  INFO("nbf = " << nbf << ", worst |S_quad - S_exact| = " << w << ", scale "
                << scale);
  REQUIRE(scale > 0.5); // a sane overlap has unit diagonal
  // Quadrature-limited, not algebra-limited: a grid this size resolves a
  // STO-3G density to around 1e-9 (see test_gto_overlap for the same
  // comparison at several grid sizes).
  REQUIRE(w < 1e-7);
}

TEST_CASE("The core Hamiltonian is symmetric for a shell basis",
          "[coreh][shells][gto]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::water_basis(mol, ShellType::Spherical);
  const GTOEvaluator ev(make_gto_device_basis(host));
  const int nbf = ev.nbf();

  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 75, 29);
  const auto core = compute_core_hamiltonian(ev, grid);

  // S, T and V_n are all A A^T products, so symmetry is structural rather
  // than physical -- but an indexing error in the shell scatter would
  // break it, which is why it is worth asserting.
  for (const auto &m : {core.overlap, core.kinetic, core.nuclear,
                        core.hamiltonian}) {
    const auto h = to_host(m);
    double worst = 0.0, scale = 0.0;
    for (int i = 0; i < nbf; ++i)
      for (int j = 0; j < nbf; ++j) {
        worst = std::max(worst, std::abs(h[static_cast<size_t>(i) * nbf + j] -
                                         h[static_cast<size_t>(j) * nbf + i]));
        scale = std::max(scale, std::abs(h[static_cast<size_t>(i) * nbf + j]));
      }
    INFO("worst asymmetry " << worst << ", scale " << scale);
    REQUIRE(worst < 1e-12 * std::max(1.0, scale));
  }
}

TEST_CASE("Kinetic and nuclear energies have the right sign and scale",
          "[coreh][shells][gto]") {
  // Weak but independent: T is positive definite, so its diagonal is
  // positive; V_n is attractive, so its diagonal is negative. A sign error
  // in the sqrt-weighted construction would flip one of them while leaving
  // the overlap -- the only thing checked against a closed form above --
  // untouched.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::water_basis(mol, ShellType::Spherical);
  const GTOEvaluator ev(make_gto_device_basis(host));
  const int nbf = ev.nbf();

  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 99, 41);
  const auto core = compute_core_hamiltonian(ev, grid);
  const auto T = to_host(core.kinetic);
  const auto V = to_host(core.nuclear);
  const auto H = to_host(core.hamiltonian);

  for (int i = 0; i < nbf; ++i) {
    const size_t d = static_cast<size_t>(i) * nbf + i;
    INFO("function " << i << ": T = " << T[d] << ", V = " << V[d]);
    REQUIRE(T[d] > 0.0);
    REQUIRE(V[d] < 0.0);
    REQUIRE_THAT(H[d], Catch::Matchers::WithinRel(T[d] + V[d], 1e-12));
  }
}

TEST_CASE("Slater and Gaussian paths both go through the shell abstraction",
          "[coreh][shells]") {
  // Both evaluators must drive the same routine. The Slater one has one
  // function per shell, so nblock == nbf; the Gaussian one does not. That
  // the same code serves both is the property the abstraction is for.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 50, 21);

  const auto sto = test::load_adf_basis(mol, "input/zorabasis/TZP", 1e-10);
  const STOPrimitiveEvaluator sev(sto);
  REQUIRE(sev.nblock() == sev.nbf()); // single-function shells

  const auto gto = test::water_basis(mol, ShellType::Spherical);
  const GTOEvaluator gev(make_gto_device_basis(gto));
  REQUIRE(gev.nblock() < gev.nbf()); // multi-function shells

  const auto scoreh = compute_core_hamiltonian(sev, grid);
  const auto gcoreh = compute_core_hamiltonian(gev, grid);

  // Different bases, so the matrices differ; what must hold for both is
  // that the overlap diagonal is near unity, since both are normalized.
  for (const auto &pr :
       {std::pair{scoreh.overlap, sev.nbf()}, {gcoreh.overlap, gev.nbf()}}) {
    const auto S = to_host(pr.first);
    const int n = pr.second;
    for (int i = 0; i < n; ++i) {
      const double d = S[static_cast<size_t>(i) * n + i];
      INFO("overlap diagonal " << i << " = " << d);
      REQUIRE(d > 0.5);
      REQUIRE(d < 1.5);
    }
  }
}

TEST_CASE("The screened path agrees with the dense one for a shell basis",
          "[coreh][shells][screened]") {
  // The dense path is checked against a closed form above, so it is the
  // reference here. The screened path reaches the same matrices by a
  // completely different route: an ArborX neighbour list over shells, a
  // team per spatial box, and atomic scatter of function pairs. Agreement
  // therefore exercises the local-to-global basis-function map -- the
  // prefix sums and the binary search that recover (shell, component) from
  // a local function index -- which has no analogue in the dense path.
  //
  // Before this change the screened kernels indexed the neighbour list as
  // though it held basis functions. It holds blocks. For a Slater basis
  // those coincide; for this one they do not, so the old code would have
  // been wrong rather than merely uncompilable.
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const auto host = test::water_basis(mol, ShellType::Spherical);
  const GTOEvaluator ev(make_gto_device_basis(host));
  const int nbf = ev.nbf();
  REQUIRE(ev.nblock() < nbf);

  auto grid = make_flat_grid<ta_type, ll_type>(mol, 75, 29);

  // create_bounding_boxes permutes the grid in place (Morton order), so it
  // must run before anything reads the points -- both paths then see the
  // same permuted grid.
  const int max_points_per_box = 128;
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  auto bb = create_bounding_boxes(grid, max_points_per_box);
  NeighborList nl;
  build_neighbor_list(ev, bb, max_points_per_box, npts, nl);

  const auto dense = compute_core_hamiltonian(ev, grid);
  const auto scr = compute_core_hamiltonian_screened(ev, grid, nl);

  struct Pair {
    const char *name;
    DeviceView2DLeft a, b;
  };
  for (const auto &pr : {Pair{"overlap", scr.overlap, dense.overlap},
                         Pair{"kinetic", scr.kinetic, dense.kinetic},
                         Pair{"nuclear", scr.nuclear, dense.nuclear}}) {
    const auto got = to_host(pr.a);
    const auto ref = to_host(pr.b);
    double scale = 0.0;
    const double w = worst_rel(got, ref, scale);
    INFO(pr.name << ": worst |screened - dense| = " << w << ", scale "
                 << scale << ", relative " << w / scale);
    REQUIRE(scale > 1e-3);
    // Screening drops contributions below the shells' cutoff radii, so this
    // is not expected to be exact -- the cutoffs are what the tolerance is
    // measuring.
    REQUIRE(w < 1e-8 * scale);
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
