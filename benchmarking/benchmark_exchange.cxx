/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
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


// ============================================================================
//  RI exchange: what the orbital batch costs
// ============================================================================
//
// compute_exact_exchange_screened is the worst-scaling routine in the
// library (issue #8), and it had no benchmark -- which is why the issue
// carries a flop count and no seconds. This is that benchmark.
//
// What it isolates is the ONE structural knob the current loop has. The
// auxiliary potential and the packed collocation are rebuilt once per
// orbital BATCH; the three-centre GEMM runs once per orbital. So the cost
// is
//
//     passes * P + N_occ * Q,   passes = ceil(N_occ / batch)
//
// and sweeping the batch separates P from Q. Everything else about the
// routine -- and the question of whether to hoist the three-centre
// construction out of the orbital loop at all, which is what issue #8
// proposes -- needs those two numbers first.
//
// RUN IT MORE THAN ONCE. The first sweep taken on this machine showed a
// clean minimum at a batch of eight and a penalty beyond it; the next two
// showed a flat curve from four upwards. The difference between adjacent
// batch sizes is smaller than the spread between runs of the same one, and
// only the gap between a batch of one and a batch of four or more survives
// repetition. That is also why max|K| is printed: it is identical across the
// sweep, so a row that differs there is a bug and not a timing.
//
// The setup is the SCF driver's: the Coulomb metric and its half-inverse
// are built once, outside the timing, because an SCF builds them once too.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/coulomb.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/exact_exchange.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>
#include <nukexc/stopotential.hpp>

#include "standards.hpp"
#include "sto_basis_reader.hpp"
#include "test_io.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

struct Config {
  std::string basis_dir = "input/zorabasis/DZ";
  int nrad = 25;
  int nang = 11;
  double screening_tol = 1e-6;
  int max_points_per_box = 128;
  double occupied_fraction = 0.1;
  double lin_dep = 1e-6;
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

/// Largest absolute element, so a wrong batch size shows up as a wrong
/// matrix rather than only as a different time.
double max_abs(const DeviceView2DLeft &m) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, m);
  double w = 0.0;
  for (size_t i = 0; i < h.extent(0); ++i)
    for (size_t j = 0; j < h.extent(1); ++j)
      w = std::max(w, std::abs(h(i, j)));
  return w;
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
    ExecSpace space;
    std::vector<std::pair<std::string, Molecule>> mols;
    mols.push_back({"water", make_water()});
    mols.push_back({"benzene", make_benzene()});

    try {
      for (auto &[name, mol] : mols) {
        STOBasisSet basis =
            test::load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);
        STOBasisSet basis_aux = test::load_adf_basis(
            mol, cfg.basis_dir, cfg.screening_tol, /*fit=*/true);
        const STOPrimitiveEvaluator ev(basis);
        const STOPotentialEvaluator ev_aux(basis_aux);

        const int nbf = ev.nbf();
        const int naux = ev_aux.nbf();
        const int n_occ =
            std::max(1, static_cast<int>(cfg.occupied_fraction * nbf + 0.5));

        FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad,
                                                         cfg.nang);
        const int npts = static_cast<int>(grid.quad_points.extent(0));
        auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
        NeighborList nl;
        build_neighbor_list(ev, bb, cfg.max_points_per_box, grid.quad_points,
                            nl);
        Kokkos::fence();

        // The metric, once -- an SCF builds it once too.
        const auto metric = coulomb_overlap_metric(space, ev_aux, grid);
        const auto X = compute_half_inverse(metric, cfg.lin_dep);
        Kokkos::fence();

        const auto C = model_orbitals(nbf, n_occ);
        DeviceView1D occ("occ", n_occ);
        Kokkos::deep_copy(occ, 1.0);

        std::cout << "\n" << name << ": nbf " << nbf << ", naux " << naux
                  << ", nocc " << n_occ << ", grid " << npts << ", fit rank "
                  << X.extent(1) << "\n"
                  << std::left << std::setw(9) << "batch" << std::right
                  << std::setw(10) << "passes" << std::setw(12) << "seconds"
                  << std::setw(14) << "max|K|" << "\n";

        // Stop AFTER the first batch that takes the whole orbital set in
        // one pass -- that case is the interesting end of the sweep, so
        // breaking before it would leave out the answer.
        for (const int want : {1, 2, 4, 8, 16, 32}) {
          const int passes = (n_occ + want - 1) / want;
          Kokkos::Timer t;
          const auto K = compute_exact_exchange_screened(
              space, C, occ, ev, ev_aux, grid, nl, X, want);
          Kokkos::fence();
          const double secs = t.seconds();
          std::cout << std::left << std::setw(9) << want << std::right
                    << std::setw(10) << passes << std::fixed
                    << std::setprecision(4) << std::setw(12) << secs
                    << std::setprecision(10) << std::setw(14) << max_abs(K)
                    << "\n";
          if (want >= n_occ)
            break;
        }
        std::cout << std::flush;
      }
    } catch (const std::exception &e) {
      std::cerr << "benchmark_exchange failed: " << e.what() << "\n";
      status = 1;
    }
  }
  Kokkos::finalize();
  return status;
}
