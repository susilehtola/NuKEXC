/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Timings for the screened XC quadrature.
//
// The screened path's cost is dominated by two things that have nothing to
// do with the functional: how many times a shell is evaluated, and how the
// per-box contraction is organized. This driver measures them -- the LDA
// and GGA potentials on the same grid and the same orbitals -- so that a
// change to either can be stated as a number rather than as an argument.
//
// Run it from the project root (it resolves "input/..." relative to the
// working directory), or from the benchmarking build directory, which gets
// its own copy of input/.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>
#include <nukexc/xc_integrals.hpp>

#include <xc.h>
#include <xc_funcs.h>

#include "standards.hpp"
#include "sto_basis_reader.hpp"
#include "test_io.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

struct Config {
  std::string basis_dir = "input/zorabasis/TZP";
  int nrad = 50;
  int nang = 20;
  double screening_tol = 1e-6;
  int max_points_per_box = 32;
  double occupied_fraction = 0.1; ///< occupied orbitals as a share of nbf
};

Config parse_args(int argc, char *argv[]) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    ArgParser p{argv[i]};
    if (p.arg == "--help" || p.arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "  --basis=<dir>     Basis set directory   (default: "
                << cfg.basis_dir << ")\n"
                << "  --nrad=<int>      Radial grid points    (default: "
                << cfg.nrad << ")\n"
                << "  --nang=<int>      Angular order         (default: "
                << cfg.nang << ")\n"
                << "  --tol=<float>     Screening tolerance   (default: "
                << cfg.screening_tol << ")\n"
                << "  --box-size=<int>  Max points per box    (default: "
                << cfg.max_points_per_box << ")\n"
                << "  --occ=<float>     Occupied fraction     (default: "
                << cfg.occupied_fraction << ")\n";
      std::exit(0);
    } else if (!p.string_opt("--basis=", cfg.basis_dir) &&
               !p.int_opt("--nrad=", cfg.nrad) &&
               !p.int_opt("--nang=", cfg.nang) &&
               !p.double_opt("--tol=", cfg.screening_tol) &&
               !p.int_opt("--box-size=", cfg.max_points_per_box) &&
               !p.double_opt("--occ=", cfg.occupied_fraction)) {
      throw std::runtime_error("Unknown argument: " + p.arg + " (try --help)");
    }
  }
  if (cfg.nrad <= 0 || cfg.nang <= 0)
    throw std::runtime_error("--nrad and --nang must be positive");
  if (cfg.occupied_fraction <= 0.0 || cfg.occupied_fraction > 1.0)
    throw std::runtime_error("--occ must be in (0, 1]");
  return cfg;
}

/// Deterministic orbitals. The timing depends on how many there are, not on
/// what they are; any real coefficients give a non-negative density.
DeviceView2DLeft model_orbitals(const int nbf, const int n_occ) {
  DeviceView2DLeft C("C", nbf, n_occ);
  auto h = Kokkos::create_mirror_view(C);
  for (int i = 0; i < n_occ; ++i) {
    double norm = 0.0;
    for (int mu = 0; mu < nbf; ++mu) {
      const double x = std::cos(0.7 * (mu + 1) * (i + 1)) * std::exp(-0.08 * mu);
      h(mu, i) = x;
      norm += x * x;
    }
    for (int mu = 0; mu < nbf; ++mu)
      h(mu, i) /= std::sqrt(norm);
  }
  Kokkos::deep_copy(C, h);
  return C;
}

struct Row {
  std::string molecule;
  int nbf = 0;
  int n_occ = 0;
  int grid_points = 0;
  double t_neighbors = 0.0;
  double t_lda = 0.0;
  double t_gga = 0.0;
  double e_lda = 0.0;
  double e_gga = 0.0;
};

void print_rows(const std::vector<Row> &rows) {
  std::cout << "\n"
            << std::left << std::setw(12) << "Molecule" << std::right
            << std::setw(7) << "nbf" << std::setw(7) << "nocc"
            << std::setw(10) << "grid" << std::setw(12) << "nbrs (s)"
            << std::setw(12) << "LSDA (s)" << std::setw(12) << "GGA (s)"
            << std::setw(16) << "E_x(LDA)" << std::setw(16) << "E_x(PBE)"
            << "\n";
  for (const auto &r : rows) {
    std::cout << std::left << std::setw(12) << r.molecule << std::right
              << std::setw(7) << r.nbf << std::setw(7) << r.n_occ
              << std::setw(10) << r.grid_points << std::fixed
              << std::setprecision(4) << std::setw(12) << r.t_neighbors
              << std::setw(12) << r.t_lda << std::setw(12) << r.t_gga
              << std::setprecision(8) << std::setw(16) << r.e_lda
              << std::setw(16) << r.e_gga << "\n";
  }
  std::cout << std::flush;
}

} // namespace

int main(int argc, char *argv[]) {
  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }

  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::vector<std::pair<std::string, Molecule>> mols;
    mols.push_back({"water", make_water()});
    mols.push_back({"benzene", make_benzene()});

    xc_func_type lda, gga;
    if (xc_func_init(&lda, XC_LDA_X, XC_POLARIZED) != 0 ||
        xc_func_init(&gga, XC_GGA_X_PBE, XC_POLARIZED) != 0) {
      std::cerr << "libxc refused to initialize a functional\n";
      Kokkos::finalize();
      return 1;
    }

    std::vector<Row> rows;
    try {
      for (auto &[name, mol] : mols) {
        STOBasisSet basis =
            test::load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);
        const STOPrimitiveEvaluator ev(basis);

        Row r;
        r.molecule = name;
        r.nbf = ev.nbf();
        r.n_occ = std::max(
            1, static_cast<int>(cfg.occupied_fraction * ev.nbf() + 0.5));

        FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad,
                                                         cfg.nang);
        r.grid_points = static_cast<int>(grid.quad_points.extent(0));

        Kokkos::Timer t;
        auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
        NeighborList nl;
        build_neighbor_list(ev, bb, cfg.max_points_per_box, grid.quad_points,
                            nl);
        Kokkos::fence();
        r.t_neighbors = t.seconds();

        const auto Ca = model_orbitals(r.nbf, r.n_occ);
        const auto Cb = model_orbitals(r.nbf, std::max(1, r.n_occ - 1));
        DeviceView1D occa("occa", Ca.extent(1)), occb("occb", Cb.extent(1));
        Kokkos::deep_copy(occa, 1.0);
        Kokkos::deep_copy(occb, 1.0);

        t.reset();
        const auto res_lda =
            compute_lsda_sparse(ev, grid, nl, Ca, occa, Cb, occb, lda);
        Kokkos::fence();
        r.t_lda = t.seconds();
        r.e_lda = res_lda.energy;

        t.reset();
        const auto res_gga =
            compute_gga_lsda_sparse(ev, grid, nl, Ca, occa, Cb, occb, gga);
        Kokkos::fence();
        r.t_gga = t.seconds();
        r.e_gga = res_gga.energy;

        rows.push_back(r);
      }
      print_rows(rows);
    } catch (const std::exception &e) {
      std::cerr << "benchmark_xc failed: " << e.what() << "\n";
      status = 1;
    }

    xc_func_end(&lda);
    xc_func_end(&gga);
  }
  Kokkos::finalize();
  return status;
}
