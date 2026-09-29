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

#include <Kokkos_Core.hpp>

#include <catch2/catch_all.hpp>

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/density.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/stobasis.hpp>

#include "sto_basis_reader.hpp"

using namespace Nukexc;

using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

void write_density_vtk(const std::string &filename, const FlatGrid &grid,
                       const DeviceView1D &density) {

  // Mirror views to host
  auto points_h = Kokkos::create_mirror_view(grid.quad_points);
  auto weights_h = Kokkos::create_mirror_view(grid.weights);
  auto density_h = Kokkos::create_mirror_view(density);
  Kokkos::deep_copy(points_h, grid.quad_points);
  Kokkos::deep_copy(weights_h, grid.weights);
  Kokkos::deep_copy(density_h, density);

  const int N = points_h.extent(0);

  std::ofstream out(filename);
  if (!out)
    throw std::runtime_error("Could not open VTK file: " + filename);

  // ---- Header ----
  out << "# vtk DataFile Version 3.0\n";
  out << "NuKEXC electron density\n";
  out << "ASCII\n";
  out << "DATASET UNSTRUCTURED_GRID\n";

  // ---- Points ----
  out << "POINTS " << N << " double\n";
  out << std::scientific << std::setprecision(10);
  for (int g = 0; g < N; ++g) {
    out << points_h(g)[0] << " " << points_h(g)[1] << " " << points_h(g)[2]
        << "\n";
  }

  // ---- Cells: one vertex cell per quadrature point ----
  out << "CELLS " << N << " " << 2 * N << "\n";
  for (int g = 0; g < N; ++g)
    out << "1 " << g << "\n";

  out << "CELL_TYPES " << N << "\n";
  for (int g = 0; g < N; ++g)
    out << "1\n"; // VTK_VERTEX

  // ---- Point data ----
  out << "POINT_DATA " << N << "\n";

  out << "SCALARS density double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (int g = 0; g < N; ++g)
    out << density_h(g) << "\n";

  out << "SCALARS weight double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (int g = 0; g < N; ++g)
    out << weights_h(g) << "\n";
}

// compute_density on one occupied 1s orbital: the quadrature of rho must
// recover the electron count.
TEST_CASE("hydrogen 1s density -- integrates to one electron",
          "[density][hydrogen_1s]") {

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}},
               std::vector<unsigned>{1u});
  auto grid = make_flat_grid<bk_type, ll_type>(mol, 100, 40);
  STOBasisSet basis = make_manual_basis({{1, 0, 0, 1.0, 0., 0., 0.}}); // 1s

  // For a single occupied orbital with coefficient 1.0,
  // the density matrix is a 1x1 identity (one basis function, fully occupied)
  DeviceView2DLeft mo_orbitals("Mo orbitals", 1, 1);
  DeviceView1D mo_coeff("MO coeff", 1);
  auto orbitals_h = Kokkos::create_mirror_view(mo_orbitals);
  auto coeff_h = Kokkos::create_mirror_view(mo_coeff);
  orbitals_h(0, 0) = 1.0;
  coeff_h(0) = 1.0;
  Kokkos::deep_copy(mo_orbitals, orbitals_h);
  Kokkos::deep_copy(mo_coeff, coeff_h);

  DeviceView1D density = compute_density(STOPrimitiveEvaluator(basis), grid, mo_orbitals, mo_coeff);

  // Integrate density against quadrature weights: ∫ρ(r)dr should equal N_elec =
  // 1
  Kokkos::View<double *, ExecSpace> weights = grid.weights;
  double integrated_density = 0.0;
  Kokkos::parallel_reduce(
      "Integrate density", density.extent(0),
      KOKKOS_LAMBDA(const int g, double &sum) {
        sum += density(g) * weights(g);
      },
      integrated_density);

  REQUIRE(integrated_density == Catch::Approx(1.0).epsilon(1e-5));
}

// Two-centre case: the normalized bonding MO built by hand from the overlap
// must give a density that integrates to the single H2+ electron, i.e. the
// quadrature reproduces Tr(DS) = 1 across both centres.
TEST_CASE("H2+ bonding MO density -- integrates to one electron",
          "[density][h2_plus]") {
  const double R = 1.0; // bond length in bohr

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}, {R, 0., 0.}},
               std::vector<unsigned>{1u, 1u});

  auto grid = make_flat_grid<bk_type, ll_type>(mol, 100, 40);

  // Build the basis explicitly so the zeta value is unambiguous.
  STOBasisSet basis = make_manual_basis({
      {1, 0, 0, 1.0, 0., 0., 0.}, // 1s on atom A
      {1, 0, 0, 1.0, R, 0., 0.},  // 1s on atom B
  });

  CoreHamiltonianResult result = compute_core_hamiltonian(STOPrimitiveEvaluator(basis), grid);
  DeviceView2DLeft S = result.overlap;

  auto S_h = Kokkos::create_mirror_view(S);
  Kokkos::deep_copy(S_h, S);

  // ---- Density normalization for the bonding MO ----
  // The bonding MO is ψ = N(φ_A + φ_B), N = 1/sqrt(2 + 2*S_AB)
  // The density matrix is D_ij = c_i * c_j where c_i = N for all i.
  //
  const double N_bond = 1.0 / std::sqrt(2.0 + 2.0 * S_h(0, 1));
  DeviceView2DLeft mo_orbitals("Mo orbitals", 2, 1);
  DeviceView1D mo_coeff("MO coeff", 1);
  auto orbitals_h = Kokkos::create_mirror_view(mo_orbitals);
  auto coeff_h = Kokkos::create_mirror_view(mo_coeff);
  orbitals_h(0, 0) = N_bond;
  orbitals_h(1, 0) = N_bond;
  coeff_h(0) = 1.0;
  Kokkos::deep_copy(mo_orbitals, orbitals_h);
  Kokkos::deep_copy(mo_coeff, coeff_h);

  DeviceView1D density = compute_density(STOPrimitiveEvaluator(basis), grid, mo_orbitals, mo_coeff);
  write_density_vtk("h2plus_density_1s.vtk", grid, density);

  double integrated_density = 0.0;
  Kokkos::parallel_reduce(
      "Integrate H2+ bonding density", density.extent(0),
      KOKKOS_LAMBDA(const int g, double &sum) {
        sum += density(g) * grid.weights(g);
      },
      integrated_density);

  // ∫ρ dr = Tr(DS) = N^2 * (S_00 + S_01 + S_10 + S_11)
  //                = N^2 * (1 + S_AB + S_AB + 1) = N^2 * 2(1 + S_AB) = 1
  REQUIRE_THAT(integrated_density, Catch::Matchers::WithinRel(1.0, 1e-5));
}

// Same normalization check, but on an MO that came out of the diagonalizer
// against a real QZ4P basis rather than being constructed by hand.
TEST_CASE("H2+ QZ4P lowest MO density -- integrates to one electron",
          "[density][h2_plus][qz4p]") {

  const double R = 1.0; // bond length in bohr

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}, {R, 0., 0.}},
               std::vector<unsigned>{1u, 1u});

  auto grid = make_flat_grid<bk_type, ll_type>(mol);

  // Build the basis explicitly so the zeta value is unambiguous.
  STOBasisSet basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P");

  int n_basis = basis.nbf();

  CoreHamiltonianResult hamiltonian;
  hamiltonian = compute_core_hamiltonian(STOPrimitiveEvaluator(basis), grid);

  // Solve the core-Hamiltonian eigenproblem to get the occupied MO.
  Nukexc::Diagonalizer diagonalizer(n_basis);
  auto X = diagonalizer.compute_transformation(hamiltonian.overlap);
  const int K = X.extent(1);
  DeviceView2DLeft mo_coeffs("mo_coeffs", n_basis, K);
  DeviceView1D mo_energies("mo_energies", K);

  diagonalizer.solve(hamiltonian.hamiltonian, mo_coeffs, mo_energies);

  // Density matrix for H2+ (1 electron in the lowest MO):
  //   D(i, j) = n_occ * C(i, 0) * C(j, 0),  n_occ = 1.0
  auto mo_occ_orbitals =
      Kokkos::subview(mo_coeffs, Kokkos::ALL(), std::make_pair(0, 1));

  // Occupation number for H2+: 1 electron
  DeviceView1D mo_occ_coeff("MO occ coeff", 1);
  Kokkos::deep_copy(mo_occ_coeff, 1.0);

  DeviceView1D density =
      compute_density(STOPrimitiveEvaluator(basis), grid, mo_occ_orbitals, mo_occ_coeff);
  write_density_vtk("h2plus_density_QZ4P.vtk", grid, density);

  double integrated_density = 0.0;
  Kokkos::parallel_reduce(
      "Integrate H2+ bonding density", density.extent(0),
      KOKKOS_LAMBDA(const int g, double &sum) {
        sum += density(g) * grid.weights(g);
      },
      integrated_density);

  REQUIRE_THAT(integrated_density, Catch::Matchers::WithinRel(1.0, 1e-5));
}

TEST_CASE("compute_density_and_sigma -- hydrogen 1s rho and sigma",
          "[density][sigma]") {

  ExecSpace space;
  const double ref_integrated_sigma = 1.0 / (2.0 * M_PI);

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}},
               std::vector<unsigned>{1u});
  auto grid = make_flat_grid<bk_type, ll_type>(mol, 200, 20);

  const int N_quad = grid.quad_points.extent(0);
  // Primary basis: single 1s STO
  STOBasisSet basis = make_manual_basis({{1, 0, 0, 1.0, 0., 0., 0.}});

  const int N_bf = basis.nbf();
  // Density matrix: fully occupied single orbital, D_11 = 1
  DeviceView2DLeft mo_orbitals("Mo orbitals", 1, 1);
  DeviceView1D mo_coeff("MO coeff", 1);
  auto orbitals_h = Kokkos::create_mirror_view(mo_orbitals);
  auto coeff_h = Kokkos::create_mirror_view(mo_coeff);
  orbitals_h(0, 0) = 1.0;
  coeff_h(0) = 1.0;
  Kokkos::deep_copy(mo_orbitals, orbitals_h);
  Kokkos::deep_copy(mo_coeff, coeff_h);

  DeviceView1D rho("Rho", N_quad);
  DeviceView1D gx_rho("Rho dx", N_quad);
  DeviceView1D gy_rho("Rho dy", N_quad);
  DeviceView1D gz_rho("Rho dz", N_quad);
  DeviceView1D sigma("Sigma", N_quad);

  DeviceView2DLeft collocation_values("collocation values", N_bf, N_quad);
  DeviceView2DLeft collocation_gx("collocation gx", N_bf, N_quad);
  DeviceView2DLeft collocation_gy("collocation gy", N_bf, N_quad);
  DeviceView2DLeft collocation_gz("collocation gz", N_bf, N_quad);

  fill_collocation(space, basis, grid.quad_points, collocation_values);

  fill_grad_collocation(space, basis, grid.quad_points, collocation_gx,
                        collocation_gy, collocation_gz);

  compute_density_and_sigma(collocation_values, collocation_gx, collocation_gy,
                            collocation_gz, mo_orbitals, mo_coeff, rho, gx_rho,
                            gy_rho, gz_rho, sigma);

  double integrated_sigma = 0.0;
  double integrated_rho = 0.0;
  Kokkos::parallel_reduce(
      "Integrate sigma", grid.quad_points.extent(0),
      KOKKOS_LAMBDA(const int g, double &sum_sigma, double &sum_rho) {
        sum_sigma += sigma(g) * grid.weights(g);
        sum_rho += rho(g) * grid.weights(g);
      },
      integrated_sigma, integrated_rho);

  REQUIRE_THAT(integrated_rho, Catch::Matchers::WithinRel(1.0, 1e-10));
  REQUIRE_THAT(integrated_sigma,
               Catch::Matchers::WithinRel(ref_integrated_sigma, 1e-10));
}

TEST_CASE("compute_density_and_sigma -- helium 1s rho and sigma",
          "[density][sigma][He]") {

  ExecSpace space;
  const double zeta = 1.6875;
  // Exact analytical value for integrated Libxc sigma: 2*zeta^5/pi
  const double ref_integrated_sigma = 2 * std::pow(zeta, 5.) / M_PI;

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}},
               std::vector<unsigned>{2u});
  auto grid = make_flat_grid<bk_type, ll_type>(mol, 200, 20);

  const int N_quad = grid.quad_points.extent(0);
  // Primary basis: single 1s STO
  STOBasisSet basis = make_manual_basis({{1, 0, 0, zeta, 0., 0., 0.}});

  const int N_bf = basis.nbf();
  // Density matrix: fully occupied single orbital, D_11 = 1
  DeviceView2DLeft mo_orbitals("Mo orbitals", 1, 1);
  DeviceView1D mo_coeff("MO coeff", 1);
  auto orbitals_h = Kokkos::create_mirror_view(mo_orbitals);
  auto coeff_h = Kokkos::create_mirror_view(mo_coeff);
  orbitals_h(0, 0) = 1.0;
  coeff_h(0) = 2.0;
  Kokkos::deep_copy(mo_orbitals, orbitals_h);
  Kokkos::deep_copy(mo_coeff, coeff_h);

  DeviceView1D rho("Rho", N_quad);
  DeviceView1D gx_rho("Rho dx", N_quad);
  DeviceView1D gy_rho("Rho dy", N_quad);
  DeviceView1D gz_rho("Rho dz", N_quad);
  DeviceView1D sigma("Sigma", N_quad);

  DeviceView2DLeft collocation_values("collocation values", N_bf, N_quad);
  DeviceView2DLeft collocation_gx("collocation gx", N_bf, N_quad);
  DeviceView2DLeft collocation_gy("collocation gy", N_bf, N_quad);
  DeviceView2DLeft collocation_gz("collocation gz", N_bf, N_quad);

  fill_collocation(space, basis, grid.quad_points, collocation_values);

  fill_grad_collocation(space, basis, grid.quad_points, collocation_gx,
                        collocation_gy, collocation_gz);

  compute_density_and_sigma(collocation_values, collocation_gx, collocation_gy,
                            collocation_gz, mo_orbitals, mo_coeff, rho, gx_rho,
                            gy_rho, gz_rho, sigma);

  double integrated_sigma = 0.0;
  double integrated_rho = 0.0;
  Kokkos::parallel_reduce(
      "Integrate sigma", grid.quad_points.extent(0),
      KOKKOS_LAMBDA(const int g, double &sum_sigma, double &sum_rho) {
        sum_sigma += sigma(g) * grid.weights(g);
        sum_rho += rho(g) * grid.weights(g);
      },
      integrated_sigma, integrated_rho);

  REQUIRE_THAT(integrated_rho, Catch::Matchers::WithinRel(2.0, 1e-10));
  REQUIRE_THAT(integrated_sigma,
               Catch::Matchers::WithinRel(ref_integrated_sigma, 1e-10));
}
int main() {
  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();

  return result;
}
