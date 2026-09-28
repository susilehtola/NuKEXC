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
 * Two-dimensional radial x angular grid convergence study for the H2+ ground
 * state energy (core Hamiltonian), run for TWO radial quadrature schemes so
 * their convergence can be compared:
 *
 *   - Becke              (IntegratorXX::Becke)
 *   - Treutler-Ahlrichs  (IntegratorXX::TreutlerAhlrichs)
 *
 * Both share the same Lebedev-Laikov angular grid, so any difference between
 * them is purely the radial mapping. For each scheme we sweep the FULL 2D grid
 * -- every combination of (nrad, nang_order) -- which exposes the coupling
 * between radial and angular refinement: refining one stops helping once the
 * other is the limiting factor, so each "error vs nrad" curve flattens onto a
 * plateau whose height is set by the angular order (and vice versa).
 *
 * Reference energy
 * ----------------
 * To compare the two schemes fairly, a SINGLE shared reference is used for both.
 * If each scheme were referenced to its own finest grid, the comparison would
 * hide any constant offset between them. Instead E_ref is the mean of the two
 * schemes evaluated on the finest grid in the sweep box (nrad_ref, nang_ref),
 * both chosen larger than any swept value. The two finest values and their
 * difference are recorded in the CSV header so the reader can see how well the
 * schemes agree in the fully-converged limit. E_ref is therefore a
 * self-reference (grid convergence), NOT the exact/analytic energy.
 *
 * Output: a CSV (columns scheme,nrad,nang_order,npts,gs_energy,abs_error) that
 * convergence_studies/plot_convergence_2d.py turns into a scheme-comparison figure.
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
#include <nukexc/grid.hpp> // make_flat_grid
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp> // load_adf_basis

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

static constexpr double R = 1.0;

// H2+ molecule: two H atoms separated by R along x.
static Molecule make_h2_mol() {
  return Molecule(std::vector<std::vector<double>>{{0., 0., 0.}, {R, 0., 0.}},
                  std::vector<unsigned>{1u, 1u});
}

// Build the grid for (nrad, nang_order) with the given radial scheme, assemble
// the core Hamiltonian and return the lowest MO energy (the H2+ ground state)
// together with the number of quadrature points actually used.
template <typename radial_type>
static double gs_energy(const STOBasisSet &basis, const Molecule &mol,
                        size_t nrad, size_t nang_order, size_t &npts_out) {
  auto grid = make_flat_grid<radial_type, ll_type>(mol, nrad, nang_order);
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

// Sweep the full (nrad, nang_order) box for one radial scheme, write CSV rows
// tagged with the scheme name and print a convergence table.
template <typename radial_type>
static void run_sweep(const char *scheme, const STOBasisSet &basis,
                      const Molecule &mol,
                      const std::vector<size_t> &nrad_sweep,
                      const std::vector<size_t> &nang_sweep, double ref_energy,
                      std::ofstream &csv) {
  const int w = 14;
  std::cout << "\n=== " << scheme << " radial scheme ===\n";
  std::cout << std::setw(w) << std::left << "nrad" << std::setw(w) << std::left
            << "nang_order" << std::setw(w) << std::right << "npts"
            << std::setw(20) << std::right << "gs_energy (Ha)" << std::setw(w)
            << std::right << "|error|"
            << "\n";
  std::cout << std::string(6 * w, '-') << "\n";

  for (size_t nrad : nrad_sweep) {
    for (size_t nang : nang_sweep) {
      size_t npts = 0;
      const double e = gs_energy<radial_type>(basis, mol, nrad, nang, npts);
      const double err = std::abs(e - ref_energy);

      csv << scheme << ',' << nrad << ',' << nang << ',' << npts << ',' << e
          << ',' << err << '\n';

      std::cout << std::setw(w) << std::left << nrad << std::setw(w)
                << std::left << nang << std::setw(w) << std::right << npts
                << std::setw(20) << std::right << std::fixed
                << std::setprecision(12) << e << std::setw(w) << std::right
                << std::scientific << std::setprecision(3) << err << "\n";
    }
    std::cout << std::string(6 * w, '-') << "\n";
  }
}

int main() {
  Kokkos::initialize();
  int status = 0;
  {
    // ---- Sweep box ---------------------------------------------------------
    // Radial points and Lebedev angular orders. The reference uses grids finer
    // than anything swept so every swept point has a well-defined error.
    const std::vector<size_t> nrad_sweep = {10, 20, 40, 80, 160, 320};
    const std::vector<size_t> nang_sweep = {5, 9, 11, 17, 23, 29};

    const size_t nrad_ref = 640;
    const size_t nang_ref = 41;

    auto mol = make_h2_mol();
    auto basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P");

    // ---- Shared reference (mean of both schemes on the finest grid) --------
    size_t npts_ref_bk = 0, npts_ref_ta = 0;
    const double e_ref_bk =
        gs_energy<bk_type>(basis, mol, nrad_ref, nang_ref, npts_ref_bk);
    const double e_ref_ta =
        gs_energy<ta_type>(basis, mol, nrad_ref, nang_ref, npts_ref_ta);
    const double ref_energy = 0.5 * (e_ref_bk + e_ref_ta);
    const double ref_diff = std::abs(e_ref_bk - e_ref_ta);

    std::cout << std::setprecision(12) << std::fixed
              << "Reference grid (nrad=" << nrad_ref
              << ", nang_order=" << nang_ref << "):\n"
              << "  E_Becke = " << e_ref_bk << " Ha\n"
              << "  E_M4    = " << e_ref_ta << " Ha\n"
              << "  |diff|  = " << std::scientific << ref_diff << " Ha\n"
              << "  E_ref   = " << std::fixed << ref_energy
              << " Ha  (mean, shared by both schemes)\n";

    // ---- Open CSV ----------------------------------------------------------
    // The leading '#' lines document exactly how E_ref (used for every
    // abs_error) was produced, so the CSV is self-describing and the plotting
    // script can echo the provenance onto the figure.
    const std::string csv_path = "convergence_2d.csv";
    std::ofstream csv(csv_path);
    csv << std::setprecision(15);
    csv << "# H2+ core-Hamiltonian ground-state (lowest MO) energy vs grid\n";
    csv << "# comparing radial schemes: Becke vs Treutler-Ahlrichs (M4)\n";
    csv << "# molecule: H2+, two H atoms at x=0 and x=" << R << " bohr\n";
    csv << "# basis: QZ4P (ADF Slater-type, input/zorabasis/QZ4P)\n";
    csv << "# angular quadrature (both schemes): Lebedev-Laikov\n";
    csv << "# E_ref = mean of the two radial schemes on the finest grid "
           "(shared, unbiased self-reference; NOT analytic):\n";
    csv << "#   nrad_ref=" << nrad_ref << ", nang_order_ref=" << nang_ref
        << "\n";
    csv << "#   E_ref_Becke=" << e_ref_bk << ", E_ref_M4=" << e_ref_ta
        << ", |diff|=" << ref_diff << "\n";
    csv << "#   E_ref=" << ref_energy << " Ha\n";
    csv << "# abs_error = |gs_energy - E_ref|\n";
    csv << "scheme,nrad,nang_order,npts,gs_energy,abs_error\n";

    // ---- Sweeps ------------------------------------------------------------
    run_sweep<bk_type>("Becke", basis, mol, nrad_sweep, nang_sweep, ref_energy,
                       csv);
    run_sweep<ta_type>("M4", basis, mol, nrad_sweep, nang_sweep, ref_energy,
                       csv);

    csv.close();
    const size_t nrows = 2 * nrad_sweep.size() * nang_sweep.size();
    std::cout << "\nWrote " << nrows << " rows to "
              << std::filesystem::absolute(csv_path).string() << "\n"
              << "Plot with: python convergence_studies/plot_convergence_2d.py "
              << std::filesystem::absolute(csv_path).string() << "\n";
  }
  Kokkos::finalize();
  return status;
}
