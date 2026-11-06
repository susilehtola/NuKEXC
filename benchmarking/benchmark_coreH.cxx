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
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/diagonalizer.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>

#include "standards.hpp"
#include "test_io.hpp"

#include "sto_basis_reader.hpp"

using namespace Nukexc;

// ── Config
// ────────────────────────────────────────────────────────────────────

struct Config {
  std::string basis_dir = "input/zorabasis/TZP";
  std::string algorithm = "screened";
  int nrad = 50;
  int nang = 20;
  double screening_tol = 1e-6;
  int max_points_per_box = 32;
};

Config parse_args(int argc, char *argv[]) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    ArgParser p{argv[i]};

    if (p.arg == "--help" || p.arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"

          << "  --basis=<dir>             Basis set directory       (default: "
          << cfg.basis_dir << ")\n"
          << "  --alg=<string>    Algorithm(screened/scratch/dense) (default: "
          << cfg.algorithm << ")\n"
          << "  --nrad=<int>              Radial grid points        (default: "
          << cfg.nrad << ")\n"
          << "  --nang=<int>              Angular grid points       (default: "
          << cfg.nang << ")\n"
          << "  --tol=<float>             Screening tolerance       (default: "
          << cfg.screening_tol << ")\n"
          << "  --box-size=<int>          Max points per box        (default: "
          << cfg.max_points_per_box << ")\n";
      std::exit(0);
    } else if (!p.string_opt("--basis=", cfg.basis_dir) &&
               !p.string_opt("--alg=", cfg.algorithm) &&
               !p.int_opt("--nrad=", cfg.nrad) &&
               !p.int_opt("--nang=", cfg.nang) &&
               !p.double_opt("--tol=", cfg.screening_tol) &&
               !p.int_opt("--box-size=", cfg.max_points_per_box)) {
      throw std::runtime_error("Unknown argument: " + p.arg + " (try --help)");
    }
  }
  if (cfg.nrad <= 0 || cfg.nang <= 0)
    throw std::runtime_error("--nrad and --nang must be positive");
  return cfg;
}

// ── Helpers ────────────────────────────────────────────────────────────

void print_config(const Config &cfg) {
  print_config_box("SCF Benchmark Configuration",
                   {
                       {"Basis directory", cfg.basis_dir},
                       {"Algorithm", cfg.algorithm},
                       {"Radial points", cfg_val(cfg.nrad)},
                       {"Angular order", cfg_val(cfg.nang)},
                       {"Screening tolerance", cfg_val(cfg.screening_tol)},
                       {"Max points per box", cfg_val(cfg.max_points_per_box)},
                   });
}

struct BenchmarkResult {
  std::string molecule;
  int nbf;
  int grid_points;
  double t_grid;
  double t_neighbors; // 0 if not applicable
  double t_core_hamiltonian;
  double t_total;
};

void print_results(const std::vector<BenchmarkResult> &results, bool screened) {
  const int w = 14;
  const int wm = 12;

  auto hline_top = [&](const std::string &left, const std::string &mid,
                       const std::string &right, const std::string &fill) {
    std::cout << left;
    std::cout << repeat(fill, wm + 2) << mid;
    std::cout << repeat(fill, 6 + 2) << mid;
    std::cout << repeat(fill, 8 + 2) << mid;
    if (screened)
      std::cout << repeat(fill, w + 2) << mid;
    std::cout << repeat(fill, w + 2) << mid;
    std::cout << repeat(fill, w + 2) << right << "\n";
  };

  hline_top("┌", "┬", "┐", "─");
  std::cout << "│ " << std::setw(wm) << std::left << "Molecule"
            << " │ " << std::setw(6) << "N"
            << " │ " << std::setw(8) << "Grid pts";
  if (screened)
    std::cout << " │ " << std::setw(w) << "Neighbors (s)";
  std::cout << " │ " << std::setw(w) << "H_el (s)"
            << " │ " << std::setw(w) << "Total (s)"
            << " │\n";
  hline_top("├", "┼", "┤", "─");

  for (auto &r : results) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "│ " << std::setw(wm) << std::left << r.molecule << " │ "
              << std::setw(6) << r.nbf << " │ " << std::setw(8)
              << r.grid_points;
    if (screened)
      std::cout << " │ " << std::setw(w) << r.t_neighbors;
    std::cout << " │ " << std::setw(w) << r.t_core_hamiltonian << " │ "
              << std::setw(w) << r.t_total << " │\n";
  }
  hline_top("└", "┴", "┘", "─");
  std::cout << std::flush;
}

// ── Molecules ────────────────────────────────────────────────────────────

std::vector<std::pair<std::string, Molecule>> make_molecules() {
  std::vector<std::pair<std::string, Molecule>> mol_list;
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
  mol_list.push_back({"taxol", make_taxol()});
#else 
  mol_list.push_back({"water", make_water()});
  mol_list.push_back({"benzene", make_benzene()});
#endif
  return mol_list;
}

// ── Benchmarks ───────────────────────────────────────────────────

void run_benchmark_fused(const Config &cfg) {
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::cout << "\n── Fused Core Hamiltonian ──\n";

  std::vector<BenchmarkResult> results;
  for (auto &[name, mol] : make_molecules()) {
    STOBasisSet basis = test::load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

    Kokkos::Timer total_timer, grid_timer, core_hamiltonian_timer;
    total_timer.reset();

    grid_timer.reset();
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang);
    double t_grid = grid_timer.seconds();

    core_hamiltonian_timer.reset();
    auto hcore = compute_core_hamiltonian(STOPrimitiveEvaluator(basis), grid);
    double t_core_hamiltonian = core_hamiltonian_timer.seconds();

    const int N = hcore.overlap.extent(0);
    DeviceView2DLeft mo_coeff("mo_coeff", N, N);
    DeviceView1D mo_energies("mo_energies", N);

    results.push_back({name, N, (int)grid.quad_points.extent(0), t_grid, 0.0,
                       t_core_hamiltonian, total_timer.seconds()});
  }
  print_results(results, false);
}

void run_benchmark_screened(const Config &cfg) {
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::cout << "\n── Screened Core Hamiltonian ──\n";

  std::vector<BenchmarkResult> results;
  for (auto &[name, mol] : make_molecules()) {
    STOBasisSet basis = test::load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

    Kokkos::Timer total_timer, grid_timer, nl_timer, core_hamiltonian_timer;

    total_timer.reset();

    grid_timer.reset();
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang);
    double t_grid = grid_timer.seconds();

    nl_timer.reset();
    auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
    NeighborList nl;
    build_neighbor_list(STOPrimitiveEvaluator(basis), bb, cfg.max_points_per_box, grid.quad_points, nl);
    double t_neighbors = nl_timer.seconds();

    core_hamiltonian_timer.reset();
    auto hcore = compute_core_hamiltonian_screened(STOPrimitiveEvaluator(basis), grid, nl);

    double t_core_hamiltonian = core_hamiltonian_timer.seconds();

    const int N = hcore.overlap.extent(0);
    DeviceView2DLeft mo_coeff("mo_coeff", N, N);
    DeviceView1D mo_energies("mo_energies", N);

    results.push_back({name, N, (int)grid.quad_points.extent(0), t_grid,
                       t_neighbors, t_core_hamiltonian, total_timer.seconds()});
  }
  print_results(results, true);
}

void run_benchmark_scratch(const Config &cfg) {
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::cout << "\n── Screened Core Hamiltonian ──\n";

  std::vector<BenchmarkResult> results;
  for (auto &[name, mol] : make_molecules()) {
    STOBasisSet basis = test::load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

    Kokkos::Timer total_timer, grid_timer, nl_timer, core_hamiltonian_timer;
    total_timer.reset();

    grid_timer.reset();
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang);
    double t_grid = grid_timer.seconds();

    nl_timer.reset();
    auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
    NeighborList nl;
    build_neighbor_list(STOPrimitiveEvaluator(basis), bb, cfg.max_points_per_box, grid.quad_points, nl);
    double t_neighbors = nl_timer.seconds();

    core_hamiltonian_timer.reset();
    auto hcore = compute_core_hamiltonian_screened_scratch(STOPrimitiveEvaluator(basis), grid, nl);

    double t_core_hamiltonian = core_hamiltonian_timer.seconds();

    const int N = hcore.overlap.extent(0);
    DeviceView2DLeft mo_coeff("mo_coeff", N, N);
    DeviceView1D mo_energies("mo_energies", N);

    results.push_back({name, N, (int)grid.quad_points.extent(0), t_grid,
                       t_neighbors, t_core_hamiltonian, total_timer.seconds()});
  }
  print_results(results, true);
}

void run_benchmark_tiled(const Config &cfg) {
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::cout << "\n── Screened Core Hamiltonian ──\n";

  std::vector<BenchmarkResult> results;
  for (auto &[name, mol] : make_molecules()) {
    STOBasisSet basis = test::load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

    Kokkos::Timer total_timer, grid_timer, nl_timer, core_hamiltonian_timer;
    total_timer.reset();

    grid_timer.reset();
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang);
    double t_grid = grid_timer.seconds();

    nl_timer.reset();
    auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
    NeighborList nl;
    build_neighbor_list(STOPrimitiveEvaluator(basis), bb, cfg.max_points_per_box, grid.quad_points, nl);
    double t_neighbors = nl_timer.seconds();

    core_hamiltonian_timer.reset();
    auto hcore = compute_core_hamiltonian_screened_tiled(STOPrimitiveEvaluator(basis), grid, nl);

    double t_core_hamiltonian = core_hamiltonian_timer.seconds();

    const int N = hcore.overlap.extent(0);
    DeviceView2DLeft mo_coeff("mo_coeff", N, N);
    DeviceView1D mo_energies("mo_energies", N);

    results.push_back({name, N, (int)grid.quad_points.extent(0), t_grid,
                       t_neighbors, t_core_hamiltonian, total_timer.seconds()});
  }
  print_results(results, true);
}

void run_benchmark_sparse(const Config &cfg) {
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::cout << "\n── Screened Core Hamiltonian ──\n";

  std::vector<BenchmarkResult> results;
  for (auto &[name, mol] : make_molecules()) {
    STOBasisSet basis = test::load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

    Kokkos::Timer total_timer, grid_timer, nl_timer, core_hamiltonian_timer;
    total_timer.reset();

    grid_timer.reset();
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang);
    double t_grid = grid_timer.seconds();

    nl_timer.reset();
    auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
    NeighborList nl;
    build_neighbor_list(STOPrimitiveEvaluator(basis), bb, cfg.max_points_per_box, grid.quad_points, nl);
    double t_neighbors = nl_timer.seconds();

    core_hamiltonian_timer.reset();
    auto hcore = compute_core_hamiltonian_screened_sparse(STOPrimitiveEvaluator(basis), grid, nl);

    double t_core_hamiltonian = core_hamiltonian_timer.seconds();

    const int N = hcore.overlap.extent(0);
    DeviceView2DLeft mo_coeff("mo_coeff", N, N);
    DeviceView1D mo_energies("mo_energies", N);

    results.push_back({name, N, (int)grid.quad_points.extent(0), t_grid,
                       t_neighbors, t_core_hamiltonian, total_timer.seconds()});
  }
  print_results(results, true);
}



// ── Main ─────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  Kokkos::initialize();
  {
    print_config(cfg);

    if (cfg.algorithm == "screened")
      run_benchmark_screened(cfg);
    else if (cfg.algorithm == "scratch")
      run_benchmark_scratch(cfg);
    else if (cfg.algorithm == "tiled")
      run_benchmark_tiled(cfg);
    else if (cfg.algorithm == "dense")
      run_benchmark_fused(cfg);
    else if (cfg.algorithm == "sparse")
      run_benchmark_sparse(cfg);
 
    std::cout << "\n";
  }
  Kokkos::finalize();
  return 0;
}
