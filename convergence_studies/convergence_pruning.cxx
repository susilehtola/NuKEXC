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
 * Angular pruning-scheme comparison on water (core-Hamiltonian occupied-orbital
 * energy sum), holding radial sizing at Uniform so only the pruning differs.
 *
 * The three schemes (IntegratorXX::PruningScheme) partition each atom's radial
 * shells into regions with different Lebedev orders:
 *
 *   Unpruned : full order on every shell.
 *   Treutler : [0,r/3) order 7, [r/3,r/2) order 11, [r/2,rsz) full.
 *              The low/medium orders are FIXED (7, 11) regardless of the base.
 *   Robust   : [0,r/4) order 7, [r/4,r/2) order (base-6), [r/2,rsz) full.
 *              The medium order tracks the base order, so it refines with it.
 *
 * Because Treutler's inner/middle shells are stuck at orders 7/11, its error
 * plateaus once the base order passes them; Robust's middle region grows with
 * the base, so it keeps converging (closer to Unpruned) while still saving
 * points versus the full grid.
 *
 * Each grid "level" grows nrad and the Lebedev order together so every scheme
 * can converge. We record the TOTAL points used and the error; plotting error
 * vs total points (convergence_studies/plot_pruning.py) shows accuracy-per-point.
 *
 * Reference: a fine UNIFORM UNPRUNED grid (nrad_ref, nang_ref) finer than any
 * swept level; error = |E - E_ref| (grid self-convergence, not analytic).
 */

#include <Kokkos_Core.hpp>

#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
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

struct Scheme {
  const char *label;
  PruningScheme pruning;
};

// A grid "level": both axes grow together so every scheme can converge.
struct Level {
  size_t nrad;
  size_t nang_order;
};

// Converged unrestricted-HF energy decomposition on the grid built for the
// given pruning scheme; also returns the total number of grid points used.
static ScfEnergies scf_energy(ExecSpace &space, const Molecule &mol,
                              const STOBasisSet &basis,
                              const STOBasisSet &basis_aux, size_t nrad,
                              size_t nang_order, PruningScheme pruning,
                              size_t &npts_out) {
  auto grid =
      make_flat_grid<ta_type, ll_type>(mol, nrad, nang_order, WEIGHT_THRESHOLD,
                                       TA_M4, pruning, RadialSizing::Uniform);
  npts_out = grid.quad_points.extent(0);
  return run_uhf_scf_energy(space, mol, basis, basis_aux, grid);
}

int main() {
  Kokkos::initialize();
  int status = 0;
  {
    // Resolution. Default is a coarse, CPU-friendly prototype sweep; set the
    // environment variable NUKEXC_HIRES=1 for the fine (GPU) production sweep.
    // Lebedev-Laikov orders 13, 25, 27 have negative weights. For Robust
    // pruning the middle shell uses order next_algebraic_order(base-6), which
    // hits the negative-weight order 25 when the base is 31 (i.e. nang=30) and
    // 13 when the base is 19 (nang~18). Those nang values are therefore avoided
    // here (nang=30 -> 29): otherwise grid construction aborts on Robust.
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
                   PruningScheme::Unpruned, npts_ref);
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

    const std::vector<Scheme> schemes = {
        {"Unpruned", PruningScheme::Unpruned},
        {"Treutler", PruningScheme::Treutler},
        {"Robust", PruningScheme::Robust},
    };

    const std::string csv_path = "convergence_pruning.csv";
    std::ofstream csv(csv_path);
    csv << std::setprecision(15);
    csv << "# Angular pruning-scheme comparison on water (unrestricted HF "
           "one- and two-electron energies)\n";
    csv << "# radial scheme: Treutler M4 ; angular: Lebedev-Laikov ; basis: "
           "QZ4P "
           "(+QZ4P fit) ; radial sizing=Uniform\n";
    csv << "# schemes: Unpruned, Treutler (fixed 7/11), Robust (7 / base-6)\n";
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
    csv << "scheme,nrad,nang_order,npts,E_kin,E_ne,E_J,E_K,E_scf,err_kin,"
           "err_ne,err_J,err_K,err_total\n";

    const int w = 14;
    for (const auto &s : schemes) {
      std::cout << "\n=== " << s.label << " ===\n";
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
        const ScfEnergies e = scf_energy(space, mol, basis, basis_aux, lv.nrad,
                                         lv.nang_order, s.pruning, npts);
        const double err_kin = std::abs(e.kinetic - e_ref.kinetic);
        const double err_ne =
            std::abs(e.nuclear_attraction - e_ref.nuclear_attraction);
        const double err_J = std::abs(e.coulomb - e_ref.coulomb);
        const double err_K = std::abs(e.exchange - e_ref.exchange);
        const double err_tot = std::abs(e.total - e_ref.total);

        csv << s.label << ',' << lv.nrad << ',' << lv.nang_order << ',' << npts
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
    std::cout << "\nWrote " << (schemes.size() * levels.size()) << " rows to "
              << std::filesystem::absolute(csv_path).string()
              << "\nPlot with: python convergence_studies/plot_pruning.py "
              << std::filesystem::absolute(csv_path).string() << "\n";
  }
  Kokkos::finalize();
  return status;
}
