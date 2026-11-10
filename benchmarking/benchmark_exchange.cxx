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
// The sweep also answers a question it cannot time directly. Separating P
// from Q shows WHERE the time is, and N_occ * Q is the large majority of
// the total -- so issue #8's proposal to hoist the three-centre
// construction out of the orbital loop is a question about that one GEMM.
// Which ordering is cheaper follows from the LAYOUT alone, so --layout-only
// prints it rather than guessing; layout_cost() below derives it.
//
// What it says, and it is not what the issue assumes. Aligned alkanes, DZ,
// 128-point boxes, from --layout-only --alkane=n:
//
//     system     nbf   nbf/box   floor B/C   pair-first needs   neutral
//                        mean                      nocc >        nocc
//     CH4         18      17.7      0.139            11            5
//     C4H10       60      54.2      0.426            48           17
//     C8H18      116      88.5      0.702           152           33
//     C16H34     228     114.9      0.936           940           65
//     C24H50     340     124.8      1.014          never          97
//     C32H66     452     129.2      1.041          never         129
//     benzene     72      63.6      0.500            65           21
//
// The occupancy pair-first would need RUNS AWAY -- 11, 48, 152, 940 -- and
// past about C24 no occupancy works at all, because the floor B/C passes
// one. The floor is essentially nbf_b / np_b, and nbf_b saturates at the
// screening radius while np_b is pinned by max_points_per_box, so the two
// meet and then cross. The occupancy, meanwhile, only grows linearly. The
// gap therefore WIDENS with system size rather than closing.
//
// That makes box size the lever, not the loop order. C16H34 again, varying
// --box-size:
//
//     box pts    128     256     512    1024    2048
//     nbf/box  114.9   117.9   122.0   126.4   131.3
//     floor    0.936   0.481   0.248   0.129   0.068
//     needs      940     119      85      76      74
//
// nbf/box barely moves -- 115 to 131 while the point count went up
// sixteenfold -- because the screening already localizes the functions, so
// the floor falls as 1/box. The required occupancy then asymptotes to about
// 74 for this basis, which is A/C: the grid-work-only crossover.
//
// So the hoist is not independently worth doing; it is worth doing WITH a
// boxing change, and only above roughly 75 occupied orbitals (about C20).
// At the current 128-point boxes it is a loss at every size measured here.
// Note also that pair-first must hold naux * nbf_b(nbf_b+1)/2 per box --
// 128 MiB for C16H34 at these sizes -- so it needs blocking over the
// auxiliary index, which the orbital-first form does not.
//
// These are flop counts from the measured layout, not timings. The two
// orderings' constant factors are not identical: pair-first's grid step is
// a larger GEMM with more reuse, and its contraction is a clean one. A
// ratio near one should be settled by building the thing; a ratio of 1.04
// that grows with system size should not.
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
//
// ============================================================================
//  Orbital screening: --screen and --screen-only
// ============================================================================
//
// Issue #8's other half asks for density-matrix screening. This loop has no
// function-pair index to screen, so the density matrix reaches it only
// through a factorization P = L L^T -- exact, since K sees nothing of the
// occupied set but P -- whose vectors inherit P's decay where the canonical
// eigenvectors do not. localize.hpp supplies the pivoted Cholesky one. The
// build then skips a box where the orbital has no amplitude.
//
// Both modes use SAP-GUESS orbitals, not the model ones above. A synthetic
// coefficient vector has amplitude everywhere, so it cannot measure this in
// either direction; the guess already carries the decay that decides it,
// which is a property of the gap and the geometry rather than of the last
// few SCF digits.
//
//   --screen       runs the full build at a range of tolerances, for both
//                  factorizations, and reports time and max|dK| against the
//                  unscreened canonical run. The first thing to read is
//                  that the two factorizations agree at ROUND-OFF with the
//                  screening off: that is the identity, not a tolerance.
//   --screen-only  reports only what would be skipped, from
//                  orbital_box_amplitudes -- no auxiliary potential, no
//                  three-centre GEMMs, no fitting metric -- which is what
//                  makes a long chain affordable. Weighted by nbf_b * np_b,
//                  because a count of skipped PAIRS is not a speedup: the
//                  far-field boxes carry few functions and little work.
//
// The answer, from the tables now in exact_exchange.hpp and localize.hpp:
// the localization is perfect and size-independent, and the screening is
// still worth only about a fifth of the three-centre work at C16H34 for
// 1e-6 relative error in K. Tails, not second moments, and this loop has
// one decaying quantity where GauXC's pair screening has two.
//
// A thing worth knowing before trusting any TIMING here: this machine is
// shared, and runs of the same configuration have differed by 40 per cent.
// The work-kept fractions do not have that problem -- they are counted,
// not timed -- which is the other reason --screen-only exists.

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
#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/localize.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>
#include <nukexc/stopotential.hpp>

#include "alkane.hpp"
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
  int alkane = 0;        ///< >0: use C_nH_(2n+2) instead of water + benzene
  bool layout_only = false;
  bool screen = false;      ///< run the orbital-screening study, not the sweep
  bool screen_only = false; ///< report the alive fractions and stop
  std::string sap_dir = "";
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
                << cfg.occupied_fraction << ")\n"
                << "  --alkane=<int>    Use C_nH_(2n+2) instead of the\n"
                << "                    default water + benzene pair\n"
                << "  --layout-only     Print the loop-ordering analysis\n"
                << "                    and skip the timing sweep\n"
                << "  --screen          Run the orbital-screening study on\n"
                << "                    SAP-guess orbitals instead of the\n"
                << "                    batch sweep on model ones\n"
                << "  --screen-only     Report what the orbital screening\n"
                << "                    would skip, without any exchange\n"
                << "                    build or fitting metric\n"
                << "  --sap-dir=<dir>   SAP potentials  (default: built-in)\n";
      std::exit(0);
    } else if (!p.string_opt("--basis=", cfg.basis_dir) &&
               !p.int_opt("--nrad=", cfg.nrad) &&
               !p.int_opt("--nang=", cfg.nang) &&
               !p.double_opt("--tol=", cfg.screening_tol) &&
               !p.int_opt("--box-size=", cfg.max_points_per_box) &&
               !p.int_opt("--alkane=", cfg.alkane) &&
               !p.bool_opt("--layout-only", cfg.layout_only) &&
               !p.bool_opt("--screen", cfg.screen) &&
               !p.bool_opt("--screen-only", cfg.screen_only) &&
               !p.string_opt("--sap-dir=", cfg.sap_dir) &&
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

/// What the two loop orderings cost, from the layout alone.
///
/// The three-centre GEMM is where the time goes, and the two orderings
/// contract the same box-local quantities in a different order. Per box,
/// with x = nbf_b, p = np_b and pairs = x(x+1)/2:
///
///   orbital-first (what the code does)
///       nocc * naux * x * p            -- one GEMM per orbital per box
///   pair-first (what issue #8 proposes: build (Q|mu nu) per box once,
///              then contract with the orbital coefficients)
///       naux * pairs * p               -- build it, on the grid
///     + nocc * naux * x * x            -- contract it, NO point factor
///
/// The second line is the one that decides this, and it is easy to leave
/// out: hoisting the build off the orbital loop does not hoist away the
/// contraction that replaces it, and that contraction costs naux * x^2 per
/// orbital where the direct form costs naux * x * p. So pair-first is
/// cheaper only when x is small against p -- when a box holds many more
/// POINTS than functions.
///
/// Summing over boxes with A = sum pairs*p, B = sum x^2, C = sum x*p,
///
///       pair / orbital = A / (nocc * C)  +  B / C
///
/// so pair-first wins iff nocc > A / (C - B), and NEVER, at any occupancy,
/// once B >= C. That is the whole criterion, and both parts are printed:
/// B/C is a floor pair-first cannot get under no matter how many orbitals
/// there are.
struct LayoutCost {
  int num_boxes = 0;
  double mean_nbf = 0.0, mean_npts = 0.0;
  int max_nbf = 0;
  double floor_ratio = 0.0;  ///< B/C: pair-first's irreducible cost
  double crossover = 0.0;    ///< nocc above which pair-first wins, or <0
};

template <typename Evaluator>
LayoutCost layout_cost(const Evaluator &ev, const NeighborList &nl,
                       const int naux, const int npts) {
  const PackedBlockLayout L = build_packed_block_layout(
      ev, nl, (size_t{64} << 20), aux_chunk_points(naux, npts));
  const auto nbf_h =
      Kokkos::create_mirror_view_and_copy(HostSpace{}, L.map.nbf);
  const auto npts_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, L.npts);

  LayoutCost c;
  c.num_boxes = L.num_boxes;
  c.max_nbf = L.max_nbf;
  double sum_nbf = 0.0, sum_npts = 0.0, A = 0.0, B = 0.0, C = 0.0;
  for (int b = 0; b < L.num_boxes; ++b) {
    const double x = nbf_h(b), p = npts_h(b);
    sum_nbf += x;
    sum_npts += p;
    A += x * (x + 1.0) / 2.0 * p;
    B += x * x;
    C += x * p;
  }
  if (L.num_boxes > 0) {
    c.mean_nbf = sum_nbf / L.num_boxes;
    c.mean_npts = sum_npts / L.num_boxes;
  }
  c.floor_ratio = C > 0.0 ? B / C : 0.0;
  // Negative means the floor alone already costs more than orbital-first
  // pays in total, so no occupancy makes pair-first win.
  c.crossover = C > B ? A / (C - B) : -1.0;
  return c;
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

/// Worst elementwise difference between two matrices of the same shape.
double max_abs_diff(const DeviceView2DLeft &A, const DeviceView2DLeft &B) {
  auto a = Kokkos::create_mirror_view_and_copy(HostSpace{}, A);
  auto b = Kokkos::create_mirror_view_and_copy(HostSpace{}, B);
  double w = 0.0;
  for (size_t i = 0; i < a.extent(0); ++i)
    for (size_t j = 0; j < a.extent(1); ++j)
      w = std::max(w, std::abs(a(i, j) - b(i, j)));
  return w;
}

/// The occupied orbitals of the SAP guess Hamiltonian, T + V_SAP.
///
/// Why not the model orbitals above: the screening is a statement about
/// WHERE an occupied orbital has amplitude, and a synthetic coefficient
/// vector has it everywhere, so model orbitals cannot measure it in either
/// direction. Why not a converged SCF: what decides the screening is the
/// decay of the density matrix, which is a property of the gap and the
/// geometry and is already present in the guess. The guess is also the one
/// density an SCF is guaranteed to pass through.
template <typename Evaluator>
OccupiedFactors sap_guess_occupied(const Evaluator &ev, const FlatGrid &grid,
                                   const int n_occ, const double lin_dep,
                                   const std::string &sap_dir) {
  const int nbf = ev.nbf();
  const auto h =
      compute_core_hamiltonian(ev, grid, /*use_SAP=*/true, sap_dir);
  Diagonalizer diag(nbf);
  const auto Xo = diag.compute_transformation(h.overlap, lin_dep);
  const int K = static_cast<int>(Xo.extent(1));
  if (K < n_occ)
    throw std::runtime_error(
        "sap_guess_occupied: the orthogonal basis is smaller than the "
        "occupied space");

  DeviceView2DLeft Call("MOs", nbf, K);
  DeviceView1D eps("orbital energies", K);
  diag.solve(h.hamiltonian, Call, eps);
  Kokkos::fence();

  OccupiedFactors f;
  f.rank = n_occ;
  f.C = DeviceView2DLeft("occupied MOs", nbf, n_occ);
  Kokkos::deep_copy(
      f.C, Kokkos::subview(Call, Kokkos::ALL(), Kokkos::make_pair(0, n_occ)));
  f.occ = DeviceView1D("occupations", n_occ);
  Kokkos::deep_copy(f.occ, 2.0);
  Kokkos::fence();
  return f;
}

/// How far from its own centre each factor reaches, as a spread in the
/// basis-function index.
///
/// Not a distance -- the index is only a proxy for position -- but for a
/// chain whose functions are numbered along it, the proxy is a faithful
/// one and it needs nothing but the coefficients. A canonical orbital
/// spread over the whole chain scores near nbf/sqrt(12); a factor confined
/// to one bond scores a few.
double mean_index_spread(const DeviceView2DLeft &C) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, C);
  const size_t n = h.extent(0), m = h.extent(1);
  if (m == 0)
    return 0.0;
  double total = 0.0;
  for (size_t k = 0; k < m; ++k) {
    double w = 0.0, c = 0.0, c2 = 0.0;
    for (size_t mu = 0; mu < n; ++mu) {
      const double p = h(mu, k) * h(mu, k);
      w += p;
      c += p * static_cast<double>(mu);
      c2 += p * static_cast<double>(mu) * static_cast<double>(mu);
    }
    if (w > 0.0)
      total += std::sqrt(std::max(0.0, c2 / w - (c / w) * (c / w)));
  }
  return total / static_cast<double>(m);
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
    if (cfg.alkane > 0) {
      // The canonical screening system: nbf/box saturates along the chain
      // while the occupancy keeps growing, which is the only way to reach
      // the crossover below. Aligned, so the boxes are as small as an
      // axis-aligned subdivision can make them -- the diagonal chain would
      // inflate nbf/box and push the crossover UP, flattering the current
      // ordering.
      mols.push_back({"C" + std::to_string(cfg.alkane) + "H" +
                          std::to_string(2 * cfg.alkane + 2),
                      test::alkane(cfg.alkane, test::ChainAxis::Aligned)});
    } else {
      mols.push_back({"water", make_water()});
      mols.push_back({"benzene", make_benzene()});
    }

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
        // The timing sweep uses the model occupancy above, because the
        // sweep is about the batch. The crossover is a statement about the
        // real molecule, so it is compared against the neutral one.
        const int n_occ_neutral = static_cast<int>(mol.Z_total / 2);

        FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad,
                                                         cfg.nang);
        const int npts = static_cast<int>(grid.quad_points.extent(0));
        auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
        NeighborList nl;
        build_neighbor_list(ev, bb, cfg.max_points_per_box, grid.quad_points,
                            nl);
        Kokkos::fence();

        const LayoutCost lc = layout_cost(ev, nl, naux, npts);
        const double mib = 8.0 / (1024.0 * 1024.0);
        std::cout << "\n" << name << ": nbf " << nbf << ", naux " << naux
                  << ", nocc " << n_occ << ", grid " << npts << "\n"
                  << "  boxes " << lc.num_boxes << ", nbf/box mean "
                  << std::fixed << std::setprecision(1) << lc.mean_nbf
                  << " max " << lc.max_nbf << ", pts/box mean "
                  << lc.mean_npts << "\n"
                  << "  pair-first floor B/C " << std::setprecision(3)
                  << lc.floor_ratio << " of orbital-first; "
                  << (lc.crossover < 0.0
                          ? std::string("NO occupancy makes it win")
                          : "wins for nocc > " +
                                std::to_string(
                                    static_cast<int>(lc.crossover + 0.5)))
                  << "; neutral nocc is " << n_occ_neutral << "\n"
                  << "  three-centre tensor: orbital-first (batch 8) "
                  << std::setprecision(1)
                  << static_cast<double>(naux) * nbf * 8 * mib
                  << " MiB, pair-first "
                  << static_cast<double>(naux) * nbf * n_occ * mib
                  << " MiB\n"
                  << std::left << std::setw(9) << "batch" << std::right
                  << std::setw(10) << "passes" << std::setw(12) << "seconds"
                  << std::setw(14) << "max|K|" << "\n";

        // Stop AFTER the first batch that takes the whole orbital set in
        // one pass -- that case is the interesting end of the sweep, so
        // breaking before it would leave out the answer.
        // --layout-only stops here, before the metric: its half-inverse is
        // an naux x naux eigendecomposition, which the layout report does
        // not use and which dominates the run at a few thousand
        // auxiliaries.
        if (cfg.layout_only) {
          std::cout << std::flush;
          continue;
        }

        if (cfg.screen_only) {
          // What the orbital screening would skip, for the two
          // factorizations of the same density matrix. No metric and no
          // exchange build: the amplitudes come from the packing and one
          // GEMM-free pass, which is what makes a long chain affordable.
          const OccupiedFactors canon = sap_guess_occupied(
              ev, grid, n_occ_neutral, cfg.lin_dep, cfg.sap_dir);
          const OccupiedFactors chol =
              cholesky_localized_occupied(canon.C, canon.occ);
          const OccupiedFactors *sets[2] = {&canon, &chol};
          const char *labels[2] = {"canonical", "Cholesky"};
          const double tols[] = {1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4};

          std::cout << "  rank " << canon.rank << " / " << chol.rank
                    << ", index spread " << std::setprecision(1)
                    << mean_index_spread(canon.C) << " / "
                    << mean_index_spread(chol.C) << " of a delocalized "
                    << static_cast<double>(nbf) / std::sqrt(12.0) << "\n"
                    << std::left << std::setw(14) << "work kept %"
                    << std::right;
          for (const double t : tols)
            std::cout << std::scientific << std::setprecision(0)
                      << std::setw(9) << t;
          std::cout << "\n";

          for (int fi = 0; fi < 2; ++fi) {
            std::vector<double> work;
            const auto amp = orbital_box_amplitudes(
                space, ev, grid, nl, sets[fi]->C, 0, &work);
            std::cout << std::left << std::setw(14) << labels[fi]
                      << std::right;
            for (const double t : tols) {
              // Weighted by nbf_b * np_b, because that is what a kept
              // (box, orbital) pair costs. A plain count of pairs would
              // credit the far-field boxes, which carry few functions and
              // little work, with the same weight as the dense ones.
              double kept = 0.0, total = 0.0;
              for (size_t b = 0; b < amp.extent(0); ++b)
                for (size_t k = 0; k < amp.extent(1); ++k) {
                  total += work[b];
                  if (amp(b, k) >= t)
                    kept += work[b];
                }
              std::cout << std::fixed << std::setprecision(1) << std::setw(9)
                        << (total > 0.0 ? 100.0 * kept / total : 0.0);
            }
            std::cout << "\n" << std::flush;
          }
          continue;
        }

        // The metric, once -- an SCF builds it once too.
        const auto metric = coulomb_overlap_metric(space, ev_aux, grid);
        const auto X = compute_half_inverse(metric, cfg.lin_dep);
        Kokkos::fence();
        std::cout << "  fit rank " << X.extent(1) << "\n";

        if (cfg.screen) {
          // The occupied set the screening is asked about, and the same
          // density matrix re-factorized. Both must give the same K: that
          // is the first row pair of the table, and a difference there is
          // a bug in the factorization, not a screening error.
          const OccupiedFactors canon = sap_guess_occupied(
              ev, grid, n_occ_neutral, cfg.lin_dep, cfg.sap_dir);
          const OccupiedFactors chol =
              cholesky_localized_occupied(canon.C, canon.occ);
          std::cout << "  SAP guess: " << canon.rank
                    << " canonical orbitals, mean index spread "
                    << std::setprecision(1) << mean_index_spread(canon.C)
                    << "\n  Cholesky:  " << chol.rank
                    << " vectors, mean index spread "
                    << mean_index_spread(chol.C) << "\n"
                    << std::left << std::setw(12) << "orbitals" << std::right
                    << std::setw(10) << "tol" << std::setw(9) << "alive%"
                    << std::setw(11) << "seconds" << std::setw(15) << "max|K|"
                    << std::setw(12) << "max|dK|" << "\n";

          const OccupiedFactors *sets[2] = {&canon, &chol};
          const char *labels[2] = {"canonical", "Cholesky"};
          DeviceView2DLeft Kref;
          for (int fi = 0; fi < 2; ++fi) {
            for (const double tol : {0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4}) {
              ExchangeScreeningStats st;
              Kokkos::Timer t;
              const auto Kmat = compute_exact_exchange_screened(
                  space, sets[fi]->C, sets[fi]->occ, ev, ev_aux, grid, nl, X,
                  8, tol, &st);
              Kokkos::fence();
              const double secs = t.seconds();
              if (Kref.extent(0) == 0)
                Kref = Kmat;
              std::cout << std::left << std::setw(12) << labels[fi]
                        << std::right << std::scientific
                        << std::setprecision(0) << std::setw(10) << tol
                        << std::fixed << std::setprecision(1) << std::setw(9)
                        << 100.0 * st.alive_fraction() << std::setprecision(4)
                        << std::setw(11) << secs << std::scientific
                        << std::setprecision(6) << std::setw(15)
                        << max_abs(Kmat) << std::setprecision(2)
                        << std::setw(12) << max_abs_diff(Kmat, Kref) << "\n"
                        << std::flush;
            }
          }
          std::cout << std::flush;
          continue;
        }

        const auto C = model_orbitals(nbf, n_occ);
        DeviceView1D occ("occ", n_occ);
        Kokkos::deep_copy(occ, 1.0);

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
