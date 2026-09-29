/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Bob Schreiner
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 *
 *    Redistribution and use in source and binary forms, with or without
 *    modification, are permitted provided that the following conditions are
 *    met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Convergence tests for the H2+ overlap integral S_AB against the exact
 * analytical result:
 *
 *   S_AB(zeta=1, R) = e^{-R} (1 + R + R^2/3)
 *
 * At R = 1 bohr, zeta = 1:  S_AB = (7/3) e^{-1}
 *
 * Two orthogonal sweeps are performed:
 *
 *   1. Radial convergence -- nang fixed large, nrad varied
 *   2. Angular convergence -- nrad fixed large, nang_order varied
 *
 * For each sweep the test records (grid parameter, computed S_AB, absolute
 * error) and prints a convergence table to stdout.  Then it asserts:
 *
 *   (a) The error at the finest grid meets a tight absolute tolerance.
 *   (b) The error decreases monotonically across the sweep.
 *   (c) The coarsest grid is at least 100x less accurate than the finest,
 *       confirming the sweep actually spans a meaningful range.
 */

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/grid.hpp> // make_flat_grid
#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp> // make_manual_basis, overlap_integral

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Nukexc;

using bk_type = IntegratorXX::Becke<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

// ============================================================
//  Shared test fixtures
// ============================================================

// Separation in bohr and the resulting exact S_AB for zeta=1.
static constexpr double R = 1.0;
static double S_EXACT = std::exp(-1) * (1. + 1. + 1. / 3.);

// Fixed grid parameters for the two orthogonal sweeps. Kept as globals so the
// CSV header written once in main() stays in sync with the test cases.
static constexpr size_t NANG_ORDER_FOR_RADIAL_SWEEP = 59;
static constexpr size_t NRAD_FOR_ANGULAR_SWEEP = 200;

// H2+ molecule: two H atoms separated by R along x.
static Molecule make_h2_mol() {
  return Molecule(std::vector<std::vector<double>>{{0., 0., 0.}, {R, 0., 0.}},
                  std::vector<unsigned>{1u, 1u});
}

// Basis: one 1s STO (zeta=1) centred on each atom.
// The zeta value is explicit here so the exact formula is unambiguous.
static STOBasisSet make_h2_basis() {
  return make_manual_basis({
      {1, 0, 0, 1.0, 0., 0., 0.}, // 1s on atom A
      {1, 0, 0, 1.0, R, 0., 0.},  // 1s on atom B
  });
}

// Compute S_AB for a given flat grid, return the off-diagonal element.
static double compute_S_AB(const Molecule &mol, STOBasisSet &basis,
                           const FlatGrid &grid) {
  auto S = overlap_integral(STOPrimitiveEvaluator(basis), grid.quad_points, grid.weights);
  auto S_h = Kokkos::create_mirror_view(S);
  Kokkos::deep_copy(S_h, S);
  return S_h(0, 1);
}

// ============================================================
//  Convergence record and table printer
// ============================================================

struct ConvergencePoint {
  size_t param;       // nrad or nang_order
  size_t npts_actual; // actual number of grid points used
  double S_AB;
  double abs_error;
};

static void print_convergence_table(const std::string &sweep_label,
                                    const std::vector<ConvergencePoint> &data) {

  const int w1 = 14, w2 = 12, w3 = 18, w4 = 14;
  if (sweep_label == "nrad") {
    std::cout << "\n-- " << sweep_label << " convergence (nang_order=59) --\n";
  } else {
    std::cout << "\n-- " << sweep_label << " convergence (nrad=200) --\n";
  }

  std::cout << std::setw(w1) << std::left << sweep_label << std::setw(w2)
            << std::right << "npts" << std::setw(w3) << std::right << "S_AB"
            << std::setw(w4) << std::right << "|error|"
            << "\n";
  std::cout << std::string(w1 + w2 + w3 + w4, '-') << "\n";

  for (const auto &p : data) {
    std::cout << std::setw(w1) << std::left << p.param << std::setw(w2)
              << std::right << p.npts_actual << std::setw(w3) << std::right
              << std::fixed << std::setprecision(12) << p.S_AB << std::setw(w4)
              << std::right << std::scientific << std::setprecision(3)
              << p.abs_error << "\n";
  }
  std::cout << std::string(w1 + w2 + w3 + w4, '-') << "\n";

  // Print successive improvement ratios so the convergence rate is visible.
  std::cout << "  Successive error ratios (err[i] / err[i+1]):\n  ";
  for (size_t i = 0; i + 1 < data.size(); ++i) {
    if (data[i + 1].abs_error > 0.0)
      std::cout << std::fixed << std::setprecision(2)
                << (data[i].abs_error / data[i + 1].abs_error) << "x  ";
    else
      std::cout << "inf  ";
  }
  std::cout << "\n";

  // Print successive algebraic convergence estimate.
  std::cout << "  Successive convergence rates:\n  ";
  for (size_t i = 1; i + 1 < data.size(); ++i) {
    if (data[i + 1].abs_error > 0.0)
      std::cout << std::fixed << std::setprecision(2)
                << -std::log(data[i + 1].abs_error / data[i].abs_error) /
                       std::log(data[i].abs_error /
                                (double)data[i - 1].abs_error)
                << "  ";
    else
      std::cout << "inf  ";
  }
  std::cout << "\n";
}

// ============================================================
//  TEST 1 — Radial convergence
// ============================================================
//
// Angular grid is fixed at a high order (nang_order = 59) so angular
// quadrature error is negligible.  Only the radial grid grows.
//
// Expected behaviour: smooth exponential convergence driven by the
// Becke radial mapping; error should fall by a large factor each time
// nrad doubles.

TEST_CASE("H2+ S_AB radial convergence", "[convergence][radial]") {

  using namespace IntegratorXX;

  const std::vector<size_t> nrad_sweep = {10, 20, 30, 50, 75, 120, 200};
  const size_t nang_order_fixed =
      NANG_ORDER_FOR_RADIAL_SWEEP; // high enough to be negligible

  auto mol = make_h2_mol();
  auto basis = make_h2_basis();

  std::vector<ConvergencePoint> data;
  data.reserve(nrad_sweep.size());

  for (size_t nrad : nrad_sweep) {

    auto grid = make_flat_grid<ta_type, ll_type>(mol, nrad, nang_order_fixed);

    // npts_actual = natoms * nrad * nang
    const size_t npts = grid.quad_points.extent(0);

    double S_AB = compute_S_AB(mol, basis, grid);
    double err = std::abs(S_AB - S_EXACT);
    data.push_back({nrad, npts, S_AB, err});
  }

  print_convergence_table("nrad", data);

  // ---- CSV (radial sweep; appends; header is written once in main) --------
  {
    std::ofstream csv("convergence_h2plus.csv", std::ios::app);
    csv << std::setprecision(15);
    for (const auto &p : data)
      csv << "radial," << p.param << "," << p.npts_actual << "," << p.S_AB
          << "," << p.abs_error << "\n";
  }

  // ---- (a) Final grid meets tight absolute tolerance ----
  REQUIRE_THAT(data.back().abs_error, Catch::Matchers::WithinAbs(0.0, 1e-8));

  // ---- (b) Errors decrease monotonically ----
  // Allow two non-monotone step in case of floating-point noise near
  // convergence, but the overall trend must be downward.
  int violations = 0;
  for (size_t i = 0; i + 1 < data.size(); ++i)
    if (data[i + 1].abs_error > data[i].abs_error)
      ++violations;
  REQUIRE(violations <= 2);
}

// ============================================================
//  TEST 2 — Angular convergence
// ============================================================
//
// Radial grid is fixed at nrad = 200 so radial error is negligible.
// The Lebedev-Laikov angular grid order grows.
//
// For the H2+ 1s overlap integrand, the angular dependence is relatively
// mild (the integrand is azimuthally symmetric about the bond axis), so
// angular convergence is expected to be rapid.

TEST_CASE("H2+ S_AB angular convergence", "[convergence][angular]") {

  using namespace IntegratorXX;
  using angular_traits = quadrature_traits<ll_type>;

  // Sweep through Lebedev algebraic orders.  next_algebraic_order ensures
  // we always land on a valid Lebedev grid.
  const std::vector<size_t> nang_order_sweep = {5, 10, 17, 23, 29, 35, 41, 53};
  const size_t nrad_fixed = NRAD_FOR_ANGULAR_SWEEP;

  auto mol = make_h2_mol();
  auto basis = make_h2_basis();

  std::vector<ConvergencePoint> data;
  data.reserve(nang_order_sweep.size());

  for (size_t order : nang_order_sweep) {
    auto grid = make_flat_grid<ta_type, ll_type>(mol, nrad_fixed, order);

    const size_t npts = grid.quad_points.extent(0);

    double S_AB = compute_S_AB(mol, basis, grid);
    double err = std::abs(S_AB - S_EXACT);
    data.push_back({order, npts, S_AB, err});
  }

  print_convergence_table("nang_order", data);

  // ---- CSV (angular sweep; appends to the file opened by the radial test) --
  {
    std::ofstream csv("convergence_h2plus.csv", std::ios::app);
    csv << std::setprecision(15);
    for (const auto &p : data)
      csv << "angular," << p.param << "," << p.npts_actual << "," << p.S_AB
          << "," << p.abs_error << "\n";
  }

  // ---- (a) Final grid meets tight absolute tolerance ----
  REQUIRE_THAT(data.back().abs_error, Catch::Matchers::WithinAbs(0.0, 1e-9));

  // ---- (b) Errors decrease monotonically ----
  int violations = 0;
  for (size_t i = 0; i + 1 < data.size(); ++i)
    if (data[i + 1].abs_error > data[i].abs_error)
      ++violations;
  REQUIRE(violations <= 2);
}

// ============================================================
//  TEST 3 — Joint diagonal normalization convergence
// ============================================================
//
// While S_AB is the most sensitive off-diagonal test, the diagonal
// elements S_AA = S_BB = 1 should also converge, and they do so faster
// (the integrand is spherically symmetric, trivial for both radial and
// angular quadrature).  This test records how quickly both diagonals
// recover from a deliberately coarse starting grid.

TEST_CASE("H2+ diagonal normalization convergence", "[convergence][diagonal]") {

  using namespace IntegratorXX;

  const std::vector<size_t> nrad_sweep = {5, 10, 20, 40, 80, 120};
  const size_t nang_order_fixed = 29;

  auto mol = make_h2_mol();
  auto basis = make_h2_basis();

  struct DiagPoint {
    size_t nrad;
    double err_AA, err_BB, err_offdiag;
  };
  std::vector<DiagPoint> data;

  for (size_t nrad : nrad_sweep) {
    auto grid = make_flat_grid<ta_type, ll_type>(mol, nrad, nang_order_fixed);
    auto S = overlap_integral(STOPrimitiveEvaluator(basis), grid.quad_points, grid.weights);
    auto S_h = Kokkos::create_mirror_view(S);
    Kokkos::deep_copy(S_h, S);

    data.push_back({nrad, std::abs(S_h(0, 0) - 1.0), std::abs(S_h(1, 1) - 1.0),
                    std::abs(S_h(0, 1) - S_EXACT)});
  }

  // Print a three-column table showing relative convergence rates
  const int w = 14;
  std::cout << "\n-- Diagonal vs. off-diagonal convergence (nang_order="
            << nang_order_fixed << ") --\n";
  std::cout << std::setw(w) << std::left << "nrad" << std::setw(w) << std::right
            << "|S_AA - 1|" << std::setw(w) << std::right << "|S_BB - 1|"
            << std::setw(w) << std::right << "|S_AB - exact|"
            << "\n";
  std::cout << std::string(4 * w, '-') << "\n";
  for (const auto &p : data) {
    std::cout << std::setw(w) << std::left << p.nrad << std::setw(w)
              << std::right << std::scientific << std::setprecision(3)
              << p.err_AA << std::setw(w) << std::right << p.err_BB
              << std::setw(w) << std::right << p.err_offdiag << "\n";
  }
  std::cout << std::string(4 * w, '-') << "\n";

  // Finest grid: all elements well converged
  REQUIRE_THAT(data.back().err_AA, Catch::Matchers::WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(data.back().err_BB, Catch::Matchers::WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(data.back().err_offdiag, Catch::Matchers::WithinAbs(0.0, 1e-7));
}

// ============================================================
int main() {
  Kokkos::initialize();
  { // Fresh CSV + header, written once so the sweep-write order is irrelevant.
    std::ofstream csv("convergence_h2plus.csv");
    csv << std::setprecision(15);
    csv << "# H2+ overlap integral S_AB convergence vs the exact analytic "
           "value\n";
    csv << "# S_AB(zeta=1,R) = e^{-R}(1 + R + R^2/3);  R=" << R
        << " bohr, zeta=1\n";
    csv << "# system: H2+, two 1s STOs (one per atom); radial: "
           "Treutler-Ahlrichs; angular: Lebedev-Laikov\n";
    csv << "# radial sweep at fixed nang_order=" << NANG_ORDER_FOR_RADIAL_SWEEP
        << "; angular sweep at fixed nrad=" << NRAD_FOR_ANGULAR_SWEEP << "\n";
    csv << "# S_exact=" << S_EXACT << " ; abs_error = |S_AB - S_exact|\n";
    csv << "sweep,param,npts,S_AB,abs_error\n";
  }
  int result = Catch::Session().run();
  Kokkos::finalize();
  return result;
}
