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
 * Convergence tests for the H2+ ground state energy against the literature
 * result:
 *
 *
 *  * Two orthogonal sweeps are performed:
 *
 *   1. Radial convergence -- nang fixed large, nrad varied
 *   2. Angular convergence -- nrad fixed large, nang_order varied
 *
 * For each sweep the test records (grid parameter, computed gs_energy, absolute
 * error) and prints a convergence table to stdout.  Then it asserts:
 *
 *   (a) The error at the finest grid meets a tight absolute tolerance.
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

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/grid.hpp> // make_flat_grid
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp> // make_manual_basis, overlap_integral

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "sto_basis_reader.hpp"

using namespace Nukexc;

using bk_type = IntegratorXX::Becke<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

static constexpr double R = 1.0;

// H2+ molecule: two H atoms separated by R along x.
static Molecule make_h2_mol() {
  return Molecule(std::vector<std::vector<double>>{{0., 0., 0.}, {R, 0., 0.}},
                  std::vector<unsigned>{1u, 1u});
}
struct ConvergencePoint {
  size_t param;       // nrad or nang_order
  size_t npts_actual; // actual number of grid points used
  double gs_energy;
  double abs_error;
};

static void print_convergence_table(const std::string &sweep_label,
                                    const std::vector<ConvergencePoint> &data,
                                    int fixed_value) {

  const int w1 = 14, w2 = 12, w3 = 18, w4 = 14;
  if (sweep_label == "nrad") {
    std::cout << "\n-- " << sweep_label
              << " convergence (nang_order=" << fixed_value << ") --\n";
  } else {
    std::cout << "\n-- " << sweep_label << " convergence (nrad=" << fixed_value
              << ") --\n";
  }

  std::cout << std::setw(w1) << std::left << sweep_label << std::setw(w2)
            << std::right << "npts" << std::setw(w3) << std::right
            << "Ground state (H)" << std::setw(w4) << std::right << "|error|"
            << "\n";
  std::cout << std::string(w1 + w2 + w3 + w4, '-') << "\n";

  for (const auto &p : data) {
    std::cout << std::setw(w1) << std::left << p.param << std::setw(w2)
              << std::right << p.npts_actual << std::setw(w3) << std::right
              << std::fixed << std::setprecision(12) << p.gs_energy
              << std::setw(w4) << std::right << std::scientific
              << std::setprecision(3) << p.abs_error << "\n";
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

TEST_CASE("H2+ Core Hamiltonian radial convergence", "[convergence][radial]") {

  using namespace IntegratorXX;

  std::vector<int> nrad_sweep;
  int nrad_max = 2000;
  for (int i = 20; i < nrad_max; i *= 2) {
    nrad_sweep.push_back(i);
  }

  const size_t nang_order_fixed = 50; // high enough to be negligible

  auto mol = make_h2_mol();
  auto basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P");

  std::vector<ConvergencePoint> data;
  data.reserve(nrad_sweep.size());

  // Compute reference
  double ref_energy;
  {
    auto grid =
        make_flat_grid<bk_type, ll_type>(mol, nrad_max, nang_order_fixed);
    // npts_actual = natoms * nrad * nang
    const size_t npts = grid.quad_points.extent(0);

    // Compute the core Hamiltonian
    CoreHamiltonianResult coreH = compute_core_hamiltonian(STOPrimitiveEvaluator(basis), grid);

    // Initialize the mo_coeff and mo_energies
    // Diagonalise the Hamiltonian
    Diagonalizer digaonalizer(basis.nbf());
    auto X = digaonalizer.compute_transformation(coreH.overlap);
    const int K = X.extent(1);

    DeviceView2DLeft mo_coeff("mo coeff", basis.nbf(), K);
    DeviceView1D mo_energies("mo energies", K);

    digaonalizer.solve(coreH.hamiltonian, mo_coeff, mo_energies);

    auto mo_energies_h = Kokkos::create_mirror_view(mo_energies);
    Kokkos::deep_copy(mo_energies_h, mo_energies);
    double gs_energy = mo_energies_h(0);

    ref_energy = gs_energy;
  }

  for (size_t nrad : nrad_sweep) {
    auto grid = make_flat_grid<bk_type, ll_type>(mol, nrad, nang_order_fixed);

    // npts_actual = natoms * nrad * nang
    const size_t npts = grid.quad_points.extent(0);

    // Compute the core Hamiltonian
    CoreHamiltonianResult coreH = compute_core_hamiltonian(STOPrimitiveEvaluator(basis), grid);

    // Initialize the mo_coeff and mo_energies

    // Diagonalise the Hamiltonian
    Diagonalizer digaonalizer(basis.nbf());
    auto X = digaonalizer.compute_transformation(coreH.overlap);
    const int K = X.extent(1);

    DeviceView2DLeft mo_coeff("mo coeff", basis.nbf(), K);
    DeviceView1D mo_energies("mo energies", K);

    digaonalizer.solve(coreH.hamiltonian, mo_coeff, mo_energies);

    auto mo_energies_h = Kokkos::create_mirror_view(mo_energies);
    Kokkos::deep_copy(mo_energies_h, mo_energies);
    double gs_energy = mo_energies_h(0);
    double err = std::abs(gs_energy - ref_energy);

    data.push_back({nrad, npts, mo_energies_h(0), err});
  }

  print_convergence_table("nrad", data, nang_order_fixed);

  // ---- (a) Final grid meets tight absolute tolerance ----
  REQUIRE_THAT(data.back().abs_error, Catch::Matchers::WithinAbs(0.0, 1e-5));
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

TEST_CASE("H2+ Core Hamiltonian angular convergence",
          "[convergence][angular]") {

  using namespace IntegratorXX;
  using angular_traits = quadrature_traits<ll_type>;

  std::vector<int> nang_order_sweep;
  int nang_max = 50;
  for (int i = 5; i < nang_max; i += 5) {
    if (i == 12 or i == 13 or i == 24 or i == 25 or i == 26 or i == 27)
      continue;

    nang_order_sweep.push_back(i);
  }

  const size_t nrad_fixed = 2000;

  auto mol = make_h2_mol();
  auto basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P");

  // Compute reference
  double ref_energy;
  {
    auto grid = make_flat_grid<bk_type, ll_type>(mol, nrad_fixed, nang_max);
    // npts_actual = natoms * nrad * nang
    const size_t npts = grid.quad_points.extent(0);

    // Compute the core Hamiltonian
    CoreHamiltonianResult coreH = compute_core_hamiltonian(STOPrimitiveEvaluator(basis), grid);

    // Initialize the mo_coeff and mo_energies

    // Diagonalise the Hamiltonian
    Diagonalizer digaonalizer(basis.nbf());
    auto X = digaonalizer.compute_transformation(coreH.overlap);
    const int K = X.extent(1);

    DeviceView2DLeft mo_coeff("mo coeff", basis.nbf(), K);
    DeviceView1D mo_energies("mo energies", K);


    digaonalizer.solve(coreH.hamiltonian, mo_coeff, mo_energies);

    auto mo_energies_h = Kokkos::create_mirror_view(mo_energies);
    Kokkos::deep_copy(mo_energies_h, mo_energies);
    double gs_energy = mo_energies_h(0);

    ref_energy = gs_energy;
  }

  std::vector<ConvergencePoint> data;
  data.reserve(nang_order_sweep.size());

  for (size_t nang_order : nang_order_sweep) {
    auto grid = make_flat_grid<bk_type, ll_type>(mol, nrad_fixed, nang_order);

    // npts_actual = natoms * nrad * nang
    const size_t npts = grid.quad_points.extent(0);

    // Compute the core Hamiltonian
    CoreHamiltonianResult coreH = compute_core_hamiltonian(STOPrimitiveEvaluator(basis), grid);


    // Diagonalise the Hamiltonian
    Diagonalizer digaonalizer(basis.nbf());
    auto X = digaonalizer.compute_transformation(coreH.overlap);
    const int K = X.extent(1);

    DeviceView2DLeft mo_coeff("mo coeff", basis.nbf(), K);
    DeviceView1D mo_energies("mo energies", K);

    digaonalizer.solve(coreH.hamiltonian, mo_coeff, mo_energies);

    auto mo_energies_h = Kokkos::create_mirror_view(mo_energies);
    Kokkos::deep_copy(mo_energies_h, mo_energies);
    double gs_energy = mo_energies_h(0);
    double err = std::abs(gs_energy - ref_energy);

    data.push_back({nang_order, npts, mo_energies_h(0), err});
  }

  print_convergence_table("nang_order", data, nrad_fixed);

  // ---- (a) Final grid meets tight absolute tolerance ----
  REQUIRE_THAT(data.back().abs_error, Catch::Matchers::WithinAbs(0.0, 1e-5));
}

// ============================================================
int main() {
  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();
  return result;
}
