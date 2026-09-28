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
 * Radial-grid convergence study on a single hydrogen atom, comparing three
 * radial quadrature schemes for the core-Hamiltonian ground state:
 *
 *   - Becke              (JCP 88, 2547 (1988))
 *   - Treutler-Ahlrichs M3   (alpha = 0.0, Eq. 18 of JCP 102, 346 (1995))
 *   - Treutler-Ahlrichs M4   (alpha = 0.6, Eq. 19 of JCP 102, 346 (1995))
 *
 * The TA M3/M4 transform is r = xi * (1+x)^alpha * ln(2/(1-x)) / ln2, so M3 and
 * M4 share the same element scaling xi and differ only by the (1+x)^alpha
 * factor. Becke uses r = R (1+x)/(1-x). All three share the same Lebedev
 * angular grid.
 *
 * A single H atom is the cleanest radial test: one center (Becke partitioning
 * trivial) and a 1s-dominated, nearly spherically symmetric density (the
 * angular grid is a non-factor). We hold the Lebedev order fixed and high and
 * sweep only the radial point count nrad for each scheme.
 *
 * Reference energy
 * ----------------
 * All three schemes integrate the same operator and converge to the same
 * finite-basis energy as nrad -> inf. E_ref is the mean of the three schemes on
 * the finest grid (a shared self-reference isolating radial-grid error, NOT the
 * exact energy). For H in the QZ4P basis this coincides with the exact 1s
 * eigenvalue (-0.5 Ha) to ~1e-13 Ha, so here error = |E - E_ref| is effectively
 * the true grid error.
 *
 * Output: a CSV (columns scheme,nrad,npts,gs_energy,abs_error) that
 * convergence_studies/plot_radial_h.py turns into a convergence figure.
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
#include <nukexc/grid.hpp> // make_flat_grid, TA_M3, TA_M4
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp> // load_adf_basis

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "sto_basis_reader.hpp"

using namespace Nukexc;

using bk_type = IntegratorXX::Becke<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

static constexpr double EXACT_1S = -0.5;         // exact H 1s core energy (Ha)
static constexpr double WEIGHT_THRESHOLD = 1e-30;

// Single hydrogen atom at the origin.
static Molecule make_h_atom() {
  return Molecule(std::vector<std::vector<double>>{{0., 0., 0.}},
                  std::vector<unsigned>{1u});
}

// Build a grid for the given radial scheme (alpha selects TA M3/M4; ignored by
// Becke), assemble the core Hamiltonian and return the lowest MO energy plus
// the point count used.
template <typename radial_type>
static double gs_energy(const STOBasisSet &basis, const Molecule &mol,
                        size_t nrad, size_t nang_order, double alpha,
                        size_t &npts_out) {
  auto grid = make_flat_grid<radial_type, ll_type>(mol, nrad, nang_order,
                                                   WEIGHT_THRESHOLD, alpha);
  npts_out = grid.quad_points.extent(0);

  CoreHamiltonianResult coreH = compute_core_hamiltonian(basis, grid);

  Diagonalizer diagonalizer(basis.nbf());
  auto X = diagonalizer.compute_transformation(coreH.overlap);
  const int K = X.extent(1);

  DeviceView2DLeft mo_coeff("mo coeff", basis.nbf(), K);
  DeviceView1D mo_energies("mo energies", K);

  diagonalizer.solve(coreH.hamiltonian, mo_coeff, mo_energies);

  auto mo_energies_h = Kokkos::create_mirror_view(mo_energies);
  Kokkos::deep_copy(mo_energies_h, mo_energies);
  return mo_energies_h(0);
}

// Sweep nrad for one scheme, write CSV rows tagged with the scheme label and
// print a convergence table.
template <typename radial_type>
static void run_sweep(const char *label, double alpha, const STOBasisSet &basis,
                      const Molecule &mol,
                      const std::vector<size_t> &nrad_sweep, size_t nang_order,
                      double ref_energy, std::ofstream &csv) {
  const int w = 14;
  std::cout << "\n=== " << label << " ===\n";
  std::cout << std::setw(w) << std::left << "nrad" << std::setw(w) << std::right
            << "npts" << std::setw(20) << std::right << "gs_energy (Ha)"
            << std::setw(w) << std::right << "|error|"
            << "\n";
  std::cout << std::string(4 * w, '-') << "\n";

  for (size_t nrad : nrad_sweep) {
    size_t npts = 0;
    const double e =
        gs_energy<radial_type>(basis, mol, nrad, nang_order, alpha, npts);
    const double err = std::abs(e - ref_energy);

    csv << label << ',' << nrad << ',' << npts << ',' << e << ',' << err
        << '\n';

    std::cout << std::setw(w) << std::left << nrad << std::setw(w) << std::right
              << npts << std::setw(20) << std::right << std::fixed
              << std::setprecision(12) << e << std::setw(w) << std::right
              << std::scientific << std::setprecision(3) << err << "\n";
  }
  std::cout << std::string(4 * w, '-') << "\n";
}

int main() {
  Kokkos::initialize();
  int status = 0;
  {
    const std::vector<size_t> nrad_sweep = {10, 20, 40, 80, 160, 320, 640};
    const size_t nrad_ref = 1280;
    const size_t nang_order = 29; // fixed high: angular error negligible for 1s

    auto mol = make_h_atom();
    auto basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P");

    // Shared reference: mean of all three schemes on the finest grid.
    size_t np = 0;
    const double e_bk =
        gs_energy<bk_type>(basis, mol, nrad_ref, nang_order, TA_M4, np);
    const double e_m3 =
        gs_energy<ta_type>(basis, mol, nrad_ref, nang_order, TA_M3, np);
    const double e_m4 =
        gs_energy<ta_type>(basis, mol, nrad_ref, nang_order, TA_M4, np);
    const double ref_energy = (e_bk + e_m3 + e_m4) / 3.0;
    const double spread = std::max({e_bk, e_m3, e_m4}) -
                          std::min({e_bk, e_m3, e_m4});

    std::cout << std::setprecision(12) << std::fixed
              << "Reference grid (nrad=" << nrad_ref
              << ", nang_order=" << nang_order << "):\n"
              << "  E_Becke = " << e_bk << " Ha\n"
              << "  E_TA-M3 = " << e_m3 << " Ha\n"
              << "  E_TA-M4 = " << e_m4 << " Ha\n"
              << "  spread  = " << std::scientific << spread << " Ha\n"
              << "  E_ref   = " << std::fixed << ref_energy << " Ha (mean)\n"
              << "  exact 1s = " << EXACT_1S << " Ha; basis-limited residual "
              << "|E_ref - exact| = " << std::scientific
              << std::abs(ref_energy - EXACT_1S) << " Ha\n";

    // ---- CSV ---------------------------------------------------------------
    const std::string csv_path = "convergence_radial_h.csv";
    std::ofstream csv(csv_path);
    csv << std::setprecision(15);
    csv << "# Radial-scheme convergence on a single H atom: Becke vs TA-M3 vs "
           "TA-M4\n";
    csv << "# observable: core-Hamiltonian lowest MO energy (Ha)\n";
    csv << "# basis: QZ4P (ADF Slater-type, input/zorabasis/QZ4P)\n";
    csv << "# angular: Lebedev-Laikov, fixed nang_order=" << nang_order << "\n";
    csv << "# TA xi(H)=0.8 (element scaling); TA M3: alpha=" << TA_M3
        << ", M4: alpha=" << TA_M4 << "\n";
    csv << "# E_ref = mean of the three schemes on the finest grid "
           "(shared self-reference, NOT analytic):\n";
    csv << "#   nrad_ref=" << nrad_ref << ", E_ref_Becke=" << e_bk
        << ", E_ref_TA-M3=" << e_m3 << ", E_ref_TA-M4=" << e_m4
        << ", spread=" << spread << "\n";
    csv << "#   E_ref=" << ref_energy << " Ha ; exact 1s=" << EXACT_1S
        << " Ha\n";
    csv << "# abs_error = |gs_energy - E_ref|\n";
    csv << "scheme,nrad,npts,gs_energy,abs_error\n";

    run_sweep<bk_type>("Becke", TA_M4 /*ignored*/, basis, mol, nrad_sweep,
                       nang_order, ref_energy, csv);
    run_sweep<ta_type>("TA-M3", TA_M3, basis, mol, nrad_sweep, nang_order,
                       ref_energy, csv);
    run_sweep<ta_type>("TA-M4", TA_M4, basis, mol, nrad_sweep, nang_order,
                       ref_energy, csv);

    csv.close();
    std::cout << "\nWrote " << (3 * nrad_sweep.size()) << " rows to "
              << std::filesystem::absolute(csv_path).string() << "\n"
              << "Plot with: python convergence_studies/plot_radial_h.py "
              << std::filesystem::absolute(csv_path).string() << "\n";
  }
  Kokkos::finalize();
  return status;
}
