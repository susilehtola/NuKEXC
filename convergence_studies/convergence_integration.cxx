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
#include <algorithm>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include <catch2/catch_all.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>

#include "standards.hpp"
#include "test_io.hpp"

#include "sto_basis_reader.hpp"

using namespace Nukexc;

using bk_type = IntegratorXX::Becke<double, double>;
using mk_type = IntegratorXX::MuraKnowles<double, double>;
using mhl_type = IntegratorXX::MurrayHandyLaming<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;

using ah_type = IntegratorXX::AhrensBeylkin<double>;
using de_type = IntegratorXX::Delley<double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using wo_type = IntegratorXX::Womersley<double>;

using RecorderFn = std::function<void(size_t, size_t, size_t, size_t, double,
                                      double, double, double, double)>;
// ── Config
// ────────────────────────────────────────────────────────────────────

struct Config {
  std::string xyz_file = "input/water.xyz";
  std::string basis_dir = "input/zorabasis/TZP";
  int n_max = 7;
  int m_max = 7;
  double screening_tol = 1e-6;
  int max_points_per_box = 64;
  int screening = 1;
};

Config parse_args(int argc, char *argv[]) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    ArgParser p{argv[i]};

    if (p.arg == "--help" || p.arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "  --xyz=<file>        XYZ input file          (default: "
          << cfg.xyz_file << ")\n"
          << "  --basis=<dir>          Basis set directory       (default: "
          << cfg.basis_dir << ")\n"
          << "  --n_max=<int>           Radial grid points        (default: "
          << cfg.n_max << ")\n"
          << "  --m_max=<int>           Angular grid points       (default: "
          << cfg.m_max << ")\n"
          << "  --tol=<float>          Screening tolerance       (default: "
          << cfg.screening_tol << ")\n"
          << "  --box-size=<int>       Max points per box        (default: "
          << cfg.max_points_per_box << ")\n"
          << "  --screening=<int>       Turn on/off screening        (default: "
          << cfg.screening << ")\n";

      std::exit(0);
    } else if (!p.string_opt("--xyz=", cfg.xyz_file) &&
               !p.string_opt("--basis=", cfg.basis_dir) &&
               !p.int_opt("--n_max=", cfg.n_max) &&
               !p.int_opt("--m_max=", cfg.m_max) &&
               !p.double_opt("--tol=", cfg.screening_tol) &&
               !p.int_opt("--box-size=", cfg.max_points_per_box) &&
               !p.int_opt("--screening=", cfg.screening)) {
      throw std::runtime_error("Unknown argument: " + p.arg + " (try --help)");
    }
  }
  if (cfg.n_max <= 0 || cfg.m_max <= 0)
    throw std::runtime_error("--n_max and --m_max must be positive");
  return cfg;
}
template <typename radial_type, typename angular_type>
CoreHamiltonianResult compute_reference(const Config cfg, size_t nrad,
                                        size_t ang_order) {
  Molecule mol;
  read_xyz(cfg.xyz_file, mol);
  STOBasisSet stobasis = test::load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

  FlatGrid grid =
      make_flat_grid<radial_type, angular_type>(mol, nrad, ang_order);

  CoreHamiltonianResult core;
  if (cfg.screening) {
    auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
    NeighborList nl;
    build_neighbor_list(stobasis, bb, cfg.max_points_per_box,
                        grid.quad_points.extent(0), nl);
    core = compute_core_hamiltonian_screened(stobasis, grid, nl);
  } else {
    core = compute_core_hamiltonian(stobasis, grid);
  }
  return core;
}

template <typename radial_type, typename angular_type>
void convergence_analysis(const Config &cfg,
                          const CoreHamiltonianResult &ref_core,
                          const size_t &nrad, const size_t &ang_order,
                          RecorderFn const &recorder) {

  using angular_traits = IntegratorXX::quadrature_traits<angular_type>;

  Molecule mol;
  read_xyz(cfg.xyz_file, mol);
  STOBasisSet stobasis = test::load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

  FlatGrid grid =
      make_flat_grid<radial_type, angular_type>(mol, nrad, ang_order);

  int N = stobasis.nbf();

  CoreHamiltonianResult core = compute_core_hamiltonian(stobasis, grid);

  double max_error_overlap_diag = 0.0;
  double max_error_overlap = 0.0;
  double max_error_hamiltontian = 0.0;
  double max_error_kinetic = 0.0;
  double max_error_nuclear = 0.0;

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j, double &lmax) {
        double err = Kokkos::abs(ref_core.kinetic(i, j) - core.kinetic(i, j));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_kinetic));

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j, double &lmax) {
        double err = Kokkos::abs(ref_core.nuclear(i, j) - core.nuclear(i, j));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_nuclear));

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j, double &lmax) {
        double err =
            Kokkos::abs(ref_core.hamiltonian(i, j) - core.hamiltonian(i, j));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_hamiltontian));

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j, double &lmax) {
        double err = Kokkos::abs(ref_core.overlap(i, j) - core.overlap(i, j));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_overlap));

  Kokkos::parallel_reduce(
      N,
      KOKKOS_LAMBDA(int i, double &lmax) {
        double err = Kokkos::abs(1.0 - core.overlap(i, i));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_overlap_diag));

  size_t dynamic_nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(ang_order));

  // Pass dynamic_nang to the recorder instead of raw ang_order
  recorder(nrad, dynamic_nang,
           grid.quad_points.extent(0) / grid.atom_centers.extent(0),
           grid.quad_points.extent(0), max_error_overlap_diag,

           max_error_overlap, max_error_kinetic, max_error_nuclear,
           max_error_hamiltontian);
}

void print_config(const Config &cfg) {
  print_config_box("Integral Convergence Config.",
                   {
                       {"XYZ file", cfg.xyz_file},
                       {"Basis directory", cfg.basis_dir},
                       {"Radial points", cfg_val(std::pow(2, cfg.n_max))},
                       {"Angular order", cfg_val(cfg.m_max * 10)},
                       {"Screening tolerance", cfg_val(cfg.screening_tol)},
                       {"Max points per box", cfg_val(cfg.max_points_per_box)},
                       {"Screening", cfg_val(cfg.screening)},
                   });
}
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
    using namespace IntegratorXX;
    using radial_type = bk_type;
    using angular_type = ll_type;

    std::vector<double> errors_overlap_diag;
    std::vector<double> errors_overlap;
    std::vector<double> errors_hamiltonian;
    std::vector<double> errors_kinetic;
    std::vector<double> errors_nuclear;
    std::vector<size_t> rad_npts;
    std::vector<size_t> ang_npts;
    std::vector<size_t> atom_npts;
    std::vector<size_t> total_npts;

    auto recorder = [&](size_t rad, size_t ang, size_t atom, size_t total,
                        double error_diag, double error_overlap,
                        double error_kinetic, double error_nuclear,
                        double error_hamiltonian) {
      rad_npts.push_back(rad);
      ang_npts.push_back(ang); // Stores actual point count (e.g., 50, 110...)
      atom_npts.push_back(atom);
      total_npts.push_back(total);
      errors_overlap_diag.push_back(error_diag);
      errors_overlap.push_back(error_overlap);
      errors_kinetic.push_back(error_kinetic);
      errors_nuclear.push_back(error_nuclear);
      errors_hamiltonian.push_back(error_hamiltonian);

      std::cout << std::setw(10) << rad << std::setw(10) << ang << std::setw(15)
                << atom << std::setw(15) << total << std::setw(20) << error_diag
                << std::setw(20) << error_overlap << std::setw(20)
                << error_kinetic << std::setw(20) << error_nuclear
                << std::setw(20) << error_hamiltonian << std::endl;
    };

    int n_max = cfg.n_max;
    int m_max = cfg.m_max;
    int nrad_max = std::pow(2, n_max);
    int nang_order_max = m_max * 10;

    CoreHamiltonianResult ref_core =
        compute_reference<radial_type, angular_type>(cfg, nrad_max,
                                                     nang_order_max);

    std::cout << "\n--- Final Convergence for " << cfg.xyz_file << "  ---\n ";
    std::cout << std::setw(10) << "rad_pts" << std::setw(10) << "ang_pts"
              << std::setw(15) << "pts_per_atom" << std::setw(15) << "pts_total"
              << std::setw(20) << "err_overlap_diag" << std::setw(20)
              << "err_overlap" << std::setw(20) << "err_kinetic"
              << std::setw(20) << "err_nuclear" << std::setw(20)
              << "err_hamiltonian" << std::endl;

    for (unsigned n = 1; n < n_max; ++n) {
      for (unsigned m = 1; m < m_max; ++m) {
        size_t nrad = std::pow(2, n);
        size_t ang_deg = m * 10;
        convergence_analysis<radial_type, angular_type>(cfg, ref_core, nrad,
                                                        ang_deg, recorder);
      }
    }
  }

  Kokkos::finalize();
  return 0;
}
