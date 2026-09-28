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

#include "standalone_helpers.hpp"
#include "sto_basis_reader.hpp"

#include <armadillo>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>
#include <openorbitaloptimizer/scfsolver.hpp>

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/coulomb.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/exact_exchange.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nuclear_repulsion.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/stobasis.hpp>

using bk_type = IntegratorXX::Becke<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;

using lebedv_type = IntegratorXX::LebedevLaikov<double>;
using namespace Nukexc;

// Converged UHF energy decomposition, split into the individual grid-sensitive
// terms (all in Ha):
//   kinetic            = Tr[D T]
//   nuclear_attraction = Tr[D V_ne]
//   coulomb            = 1/2 Tr[D J]
//   exchange           = -1/2 (Tr[D_a K_a] + Tr[D_b K_b])
//   nuclear_repulsion  = E_nuc                       (grid-independent)
//   total              = sum of the above
// one_electron() = kinetic + nuclear_attraction; two_electron() = coulomb +
// exchange. Reporting the terms separately shows which one converges slowest
// and which cancels non-monotonely in the total as the grid is refined.
struct ScfEnergies {
  double total = 0.0;
  double kinetic = 0.0;
  double nuclear_attraction = 0.0;
  double coulomb = 0.0;
  double exchange = 0.0;
  double nuclear_repulsion = 0.0;
  double one_electron() const { return kinetic + nuclear_attraction; }
  double two_electron() const { return coulomb + exchange; }
};

// Run a dense UHF SCF on `grid` and return its converged energy decomposition.
// `basis` is the primary basis, `basis_aux` the RI fitting basis. Closed- or
// open-shell is selected via `charge`/`multiplicity` (water: 0 / 1).
inline ScfEnergies
run_uhf_scf_energy(ExecSpace &space, const Molecule &mol,
                   const STOBasisSet &basis, const STOBasisSet &basis_aux,
                   const FlatGrid &grid, int charge = 0, int multiplicity = 1,
                   double conv_thr = 1e-9, double lin_dep_threshold = 1e-5) {
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);

  // ---- Dense collocations: Phi, Phi_aux, and the (weight-scaled) potential --
  DeviceView2DLeft basis_collocation("Basis collocation", N_bf, N_quad);
  fill_collocation(space, basis, grid.quad_points, basis_collocation);

  DeviceView2DLeft basis_aux_collocation("Aux basis collocation", N_bf_aux,
                                         N_quad);
  fill_collocation(space, basis_aux, grid.quad_points, basis_aux_collocation);

  DeviceView2DLeft potential_collocation_scaled("Potential collocation",
                                                N_bf_aux, N_quad);
  sto_potential_collocation_scaled(space, basis_aux, grid,
                                   potential_collocation_scaled);

  // ---- Auxiliary Coulomb metric (A|B) and its symmetric half-inverse -------
  DeviceView2DLeft aux_overlap("Aux overlap", N_bf_aux, N_bf_aux);
  KokkosBlas::gemm(space, "N", "T", 1.0, basis_aux_collocation,
                   potential_collocation_scaled, 0.0, aux_overlap);
  DeviceView2DLeft aux_overlap_sym("Aux overlap sym", N_bf_aux, N_bf_aux);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf_aux, N_bf_aux}),
      KOKKOS_LAMBDA(int i, int j) {
        aux_overlap_sym(i, j) = 0.5 * (aux_overlap(i, j) + aux_overlap(j, i));
      });
  DeviceView2DLeft half_inverse_X =
      compute_half_inverse(aux_overlap_sym, lin_dep_threshold);

  // ---- Core Hamiltonian, overlap and orthogonaliser X = S^{-1/2} -----------
  auto hcore = compute_core_hamiltonian(basis, grid);
  Diagonalizer diag(N_bf);
  DeviceView2DLeft X =
      diag.compute_transformation(hcore.overlap, lin_dep_threshold);

  arma::mat h_core = kokkos_to_arma(hcore.hamiltonian);
  arma::mat T_arma = kokkos_to_arma(hcore.kinetic);   // kinetic
  arma::mat Vne_arma = kokkos_to_arma(hcore.nuclear); // nuclear attraction
  arma::mat X_arma = kokkos_to_arma(X);
  arma::mat S_arma = kokkos_to_arma(hcore.overlap);

  // ---- Electron counts / OOO block setup -----------------------------------
  const int n_elec = mol.Z_total - charge;
  const double n_alpha = (n_elec + (multiplicity - 1)) / 2.0;
  const double n_beta = (n_elec - (multiplicity - 1)) / 2.0;

  arma::uvec blocks_per_type = {1, 1};
  arma::vec max_occupations = {1.0, 1.0};
  arma::vec number_of_particles = {n_alpha, n_beta};
  std::vector<std::string> block_descriptions = {"alpha", "beta"};

  const double E_nuc = compute_nuclear_repulsion(mol);

  // ---- Core Hamiltonian initial guess -----------------------------
  arma::mat F_orth = X_arma.t() * h_core * X_arma;

  // Filled by the last (converged) Fock build so the caller gets the energy
  // decomposition without a separate re-evaluation.
  ScfEnergies conv{};
  conv.nuclear_repulsion = E_nuc;

  // ---- Dense UHF Fock builder (called synchronously by solver.run()) -------
  auto fock_builder =
      [&, space](const OpenOrbitalOptimizer::DensityMatrix<double, double> &dm)
      -> std::pair<double, OpenOrbitalOptimizer::FockMatrix<double>> {
    const auto &orbitals = dm.first;
    const auto &occupations = dm.second;

    arma::mat C_alpha = X_arma * orbitals[0];
    arma::mat C_beta = X_arma * orbitals[1];
    arma::vec occ_alpha = occupations[0];
    arma::vec occ_beta = occupations[1];

    arma::mat D_alpha = C_alpha * arma::diagmat(occ_alpha) * C_alpha.t();
    arma::mat D_beta = C_beta * arma::diagmat(occ_beta) * C_beta.t();
    arma::mat D_tot = D_alpha + D_beta;

    arma::mat C_combined = arma::join_horiz(C_alpha, C_beta);
    arma::vec occ_combined = arma::join_vert(occ_alpha, occ_beta);
    DeviceView2DLeft k_C_tot = arma_to_kokkos(C_combined, "C_combined");
    DeviceView1D k_occ_tot = arma_to_kokkos1d(occ_combined, "occ_combined");
    DeviceView2DLeft k_C_alpha = arma_to_kokkos(C_alpha, "C_alpha");
    DeviceView2DLeft k_C_beta = arma_to_kokkos(C_beta, "C_beta");
    DeviceView1D k_occ_alpha = arma_to_kokkos1d(occ_alpha, "occ_alpha");
    DeviceView1D k_occ_beta = arma_to_kokkos1d(occ_beta, "occ_beta");

    // Coulomb (total density) and both exact-exchange spin channels.
    DeviceView2DLeft J = compute_coulomb(
        space, k_C_tot, k_occ_tot, basis_collocation, basis_aux_collocation,
        potential_collocation_scaled, half_inverse_X);
    arma::mat J_arma = kokkos_to_arma(J);

    DeviceView2DLeft K_alpha = compute_exact_exchange(
        space, k_C_alpha, k_occ_alpha, basis_collocation, basis_aux_collocation,
        potential_collocation_scaled, half_inverse_X);
    DeviceView2DLeft K_beta = compute_exact_exchange(
        space, k_C_beta, k_occ_beta, basis_collocation, basis_aux_collocation,
        potential_collocation_scaled, half_inverse_X);
    arma::mat K_alpha_arma = kokkos_to_arma(K_alpha);
    arma::mat K_beta_arma = kokkos_to_arma(K_beta);

    const double E_exchange = -0.5 * arma::trace(D_alpha * K_alpha_arma) -
                              0.5 * arma::trace(D_beta * K_beta_arma);
    const double E_kinetic = arma::trace(D_tot * T_arma);
    const double E_nuc_attr = arma::trace(D_tot * Vne_arma);
    const double E_core = E_kinetic + E_nuc_attr; // = Tr[D h_core]
    const double E_coulomb = 0.5 * arma::trace(D_tot * J_arma);
    const double Etot = E_nuc + E_core + E_coulomb + E_exchange;

    conv.total = Etot;
    conv.kinetic = E_kinetic;
    conv.nuclear_attraction = E_nuc_attr;
    conv.coulomb = E_coulomb;
    conv.exchange = E_exchange;

    arma::mat F_alpha = h_core + J_arma - K_alpha_arma;
    arma::mat F_beta = h_core + J_arma - K_beta_arma;
    std::vector<arma::mat> fock_arma = {X_arma.t() * F_alpha * X_arma,
                                        X_arma.t() * F_beta * X_arma};
    return std::make_pair(Etot, fock_arma);
  };

  // ---- Construct and run the SCF solver ------------------------------------
  OpenOrbitalOptimizer::SCFSolver<double, double> solver(
      blocks_per_type, max_occupations, number_of_particles, fock_builder,
      block_descriptions);
  solver.convergence_threshold(conv_thr);
  solver.verbosity(0);
  solver.initialize_with_fock({F_orth, F_orth});
  solver.run();

  return conv;
}
int main() {

  Kokkos::initialize();
  {
    // Read in your molecule
    std::string xyz_file = "input/water.xyz";
    Molecule mol = create_molecule_from_xyz(xyz_file);

    // Choose an STObasis
    std::string basis_dir = "input/zorabasis/TZ2P";
    ;
    STOBasisSet basis, basis_aux;
    double screening_tol = 1e-3; // Not used in dense algorithm
    basis = test::load_adf_basis(mol, basis_dir);
    basis_aux =
        test::load_adf_basis(mol, basis_dir, screening_tol, /*fit=*/true);

    // Create the quadrature grid
    int radial_points = 50;
    int angular_order = 20;
    FlatGrid grid =
        make_flat_grid<ta_type, lebedv_type>(mol, radial_points, angular_order);
    int total_points = grid.quad_points.size();

    // Run an Unrestricted Hartree Fock SCF until convergence
    ExecSpace space;
    int charge = 0;
    int multiplicity = 1;

    auto energies = run_uhf_scf_energy(space, mol, basis, basis_aux, grid,
                                       charge, multiplicity);

    std::cout << std::fixed << std::setprecision(10)
              << "Molecule: water (Z_total=" << mol.Z_total << ")"
              << ", nang_order=" << angular_order << ", nrad=" << radial_points
              << ", npts=" << total_points << "\n E_kin = " << energies.kinetic
              << " Ha\n E_ne = " << energies.nuclear_attraction
              << " Ha\n E_J = " << energies.coulomb
              << " Ha\n E_K = " << energies.exchange
              << " Ha\n E_scf = " << energies.total << " Ha\n";
  }
  Kokkos::finalize();
  return 0;
}
