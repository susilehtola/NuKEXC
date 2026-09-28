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
 * Accuracy-per-point study of the adaptive-grid options on a full molecule
 * (water), for the core-Hamiltonian occupied-orbital energy sum.
 *
 * make_flat_grid exposes two orthogonal, independently toggleable knobs:
 *   * pruning     -- angular adaptivity (Unpruned vs Robust): fewer Lebedev
 *                    points on the inner/outer radial shells.
 *   * radial_sizing -- radial adaptivity: RadialSizing::PySCF gives heavier
 *                      atoms (here O) more radial points than light ones (H),
 *                      following the GauXC/PySCF per-period pattern.
 * That gives four combinations, swept here over grid levels that grow the
 * radial count and Lebedev order together (a fixed angular order would cap
 * every curve at the same angular floor and hide the differences): uniform
 * (Unpruned, radial=Uniform) pruned       (Robust,   radial=Uniform)
 *   per-element  (Unpruned, radial=PySCF)
 *   both         (Robust,   radial=PySCF)
 *
 * (Robust pruning is used rather than Treutler: its middle-shell order tracks
 * the base order instead of being fixed at 11, so it converges to the reference
 * rather than plateauing -- see convergence_studies/convergence_pruning.cxx.)
 *
 * For each combination and base nrad we record the TOTAL number of grid points
 * actually used and the error of the observable. Plotting error vs total points
 * (convergence_studies/plot_adaptive.py) shows accuracy-per-point: the adaptive schemes aim
 * to reach a given accuracy with fewer points than the uniform grid.
 *
 * Observable: sum of the lowest nocc = Z_total/2 core-Hamiltonian MO energies
 * (a grid-sensitive scalar spanning the O 1s core -- radial-stressing -- and
 * the valence orbitals -- angular-stressing).
 *
 * Reference: a fine UNIFORM UNPRUNED grid (nrad_ref, nang_ref) larger than any
 * swept grid; all four schemes converge to it, so error = |E - E_ref| is a fair
 * common yardstick (grid self-convergence, not an analytic value).
 */

#include <Kokkos_Core.hpp>

#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/radial/becke.hpp>
#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/grid.hpp> // make_flat_grid, TA_M4
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp> // load_adf_basis

#include "scf_driver.hpp" // run_uhf_scf_energy
#include "standards.hpp"  // make_water

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "sto_basis_reader.hpp"

using namespace Nukexc;

using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using PruningScheme = IntegratorXX::PruningScheme;

static constexpr double WEIGHT_THRESHOLD = 1e-30;

struct Combo {
  const char *label;
  PruningScheme pruning;
  RadialSizing radial_sizing;
};

// A grid "level": both axes grow together so every combination can converge
// (a fixed angular order would cap all curves at the same angular floor and
// hide the accuracy-per-point differences).
struct Level {
  size_t nrad;
  size_t nang_order;
};

// Converged unrestricted-HF energy decomposition on the grid built for the
// given grid knobs; also returns the total number of grid points used.
static ScfEnergies scf_energy(ExecSpace &space, const Molecule &mol,
                              const STOBasisSet &basis,
                              const STOBasisSet &basis_aux, size_t nrad,
                              size_t nang_order, PruningScheme pruning,
                              RadialSizing radial_sizing, size_t &npts_out) {
  auto grid = make_flat_grid<ta_type, ll_type>(
      mol, nrad, nang_order, WEIGHT_THRESHOLD, TA_M4, pruning, radial_sizing);
  npts_out = grid.quad_points.extent(0);
  return run_uhf_scf_energy(space, mol, basis, basis_aux, grid);
}

int main() {
  Kokkos::initialize();
  int status = 0;
  {
    // Resolution. Default is a coarse, CPU-friendly prototype sweep; set the
    // environment variable NUKEXC_HIRES=1 for the fine (GPU) production sweep.
    // NOTE: Lebedev-Laikov orders 13, 25, 27 have negative weights. Robust
    // pruning's base-6 middle shell hits order 25 at base 31 (nang=30), so
    // nang=30 is avoided here (see convergence_pruning.cxx).
    const bool hires = std::getenv("NUKEXC_HIRES") != nullptr;
    const std::vector<Level> levels =
        hires ? std::vector<Level>{{40, 17},  {60, 21},  {90, 28}, {130, 29},
                                   {200, 35}, {300, 41}, {600, 45}}
              : std::vector<Level>{{30, 15}, {40, 17}, {60, 21}};
    const size_t nrad_ref = hires ? 1000 : 90;
    const size_t nang_ref = hires ? 50 : 28;

    ExecSpace space;
    auto mol = make_water();
    auto basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P");
    auto basis_aux = test::load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-10, true);

    // Reference: fine uniform unpruned grid (finer than every swept level).
    size_t npts_ref = 0;
    const ScfEnergies e_ref =
        scf_energy(space, mol, basis, basis_aux, nrad_ref, nang_ref,
                   PruningScheme::Unpruned, RadialSizing::Uniform, npts_ref);
    std::cout << std::fixed << std::setprecision(10)
              << "Molecule: water (Z_total=" << mol.Z_total << ")\n"
              << "Resolution: " << (hires ? "HI-RES (GPU)" : "prototype (CPU)")
              << "\nReference (UHF, uniform unpruned, nrad=" << nrad_ref
              << ", nang_order=" << nang_ref << ", npts=" << npts_ref
              << "):\n  E_kin = " << e_ref.kinetic
              << " Ha, E_ne = " << e_ref.nuclear_attraction
              << " Ha, E_J = " << e_ref.coulomb
              << " Ha, E_K = " << e_ref.exchange
              << " Ha, E_scf = " << e_ref.total << " Ha\n";

    const std::vector<Combo> combos = {
        {"uniform", PruningScheme::Unpruned, RadialSizing::Uniform},
        {"pruned", PruningScheme::Robust, RadialSizing::Uniform},
        {"per-element", PruningScheme::Unpruned, RadialSizing::PySCF},
        {"both", PruningScheme::Robust, RadialSizing::PySCF},
    };

    const std::string csv_path = "convergence_adaptive.csv";
    std::ofstream csv(csv_path);
    csv << std::setprecision(15);
    csv << "# Adaptive-grid accuracy-per-point study on water (unrestricted HF "
           "one- and two-electron energies)\n";
    csv << "# radial scheme: Treutler M4 ; angular: Lebedev-Laikov ; basis: "
           "QZ4P "
           "(+QZ4P fit)\n";
    csv << "# knobs: pruning (Unpruned/Robust) x radial sizing "
           "(Uniform/PySCF per-period)\n";
    csv << "# grid levels sweep (nrad, nang_order) together so curves "
           "converge\n";
    csv << "# observables (converged UHF, Ha): E_kin = Tr[D T] ; E_ne = "
           "Tr[D V_ne] ; E_J = 1/2 Tr[D J] ; E_K = -1/2 Tr[D K] (exact "
           "exchange) ; E_scf = E_nuc_rep + E_kin + E_ne + E_J + E_K\n";
    csv << "# resolution: " << (hires ? "hi-res (GPU)" : "prototype (CPU)")
        << "\n";
    csv << "# reference = fine uniform unpruned grid (shared self-reference, "
           "NOT analytic):\n";
    csv << "#   nrad_ref=" << nrad_ref << ", nang_order_ref=" << nang_ref
        << ", npts_ref=" << npts_ref << ", E_kin_ref=" << e_ref.kinetic
        << ", E_ne_ref=" << e_ref.nuclear_attraction
        << ", E_J_ref=" << e_ref.coulomb << ", E_K_ref=" << e_ref.exchange
        << ", E_scf_ref=" << e_ref.total << " Ha\n";
    csv << "# err_* = |value - reference| ; npts = total grid points used\n";
    csv << "combo,nrad,nang_order,npts,E_kin,E_ne,E_J,E_K,E_scf,err_kin,"
           "err_ne,err_J,err_K,err_total\n";

    const int w = 14;
    for (const auto &c : combos) {
      std::cout << "\n=== " << c.label << " (pruning="
                << (c.pruning == PruningScheme::Unpruned ? "Unpruned"
                                                         : "Robust")
                << ", radial="
                << (c.radial_sizing == RadialSizing::Uniform ? "Uniform"
                                                             : "PySCF")
                << ") ===\n";
      std::cout << std::setw(w) << std::left << "nrad" << std::setw(w)
                << std::left << "nang" << std::setw(11) << std::right << "npts"
                << std::setw(12) << std::right << "err_kin" << std::setw(12)
                << std::right << "err_ne" << std::setw(12) << std::right
                << "err_J" << std::setw(12) << std::right << "err_K"
                << std::setw(12) << std::right << "err_total"
                << "\n";
      std::cout << std::string(2 * w + 11 + 5 * 12, '-') << "\n";

      for (const auto &lv : levels) {
        size_t npts = 0;
        const ScfEnergies e =
            scf_energy(space, mol, basis, basis_aux, lv.nrad, lv.nang_order,
                       c.pruning, c.radial_sizing, npts);
        const double err_kin = std::abs(e.kinetic - e_ref.kinetic);
        const double err_ne =
            std::abs(e.nuclear_attraction - e_ref.nuclear_attraction);
        const double err_J = std::abs(e.coulomb - e_ref.coulomb);
        const double err_K = std::abs(e.exchange - e_ref.exchange);
        const double err_tot = std::abs(e.total - e_ref.total);

        csv << c.label << ',' << lv.nrad << ',' << lv.nang_order << ',' << npts
            << ',' << e.kinetic << ',' << e.nuclear_attraction << ','
            << e.coulomb << ',' << e.exchange << ',' << e.total << ','
            << err_kin << ',' << err_ne << ',' << err_J << ',' << err_K << ','
            << err_tot << '\n';

        std::cout << std::setw(w) << std::left << lv.nrad << std::setw(w)
                  << std::left << lv.nang_order << std::setw(11) << std::right
                  << npts << std::scientific << std::setprecision(3)
                  << std::setw(12) << std::right << err_kin << std::setw(12)
                  << std::right << err_ne << std::setw(12) << std::right
                  << err_J << std::setw(12) << std::right << err_K
                  << std::setw(12) << std::right << err_tot << "\n";
      }
    }

    csv.close();
    std::cout << "\nWrote " << (combos.size() * levels.size()) << " rows to "
              << std::filesystem::absolute(csv_path).string()
              << "\nPlot with: python convergence_studies/plot_adaptive.py "
              << std::filesystem::absolute(csv_path).string() << "\n";
  }
  Kokkos::finalize();
  return status;
}
