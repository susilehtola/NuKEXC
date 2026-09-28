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

#include <integratorxx/generators/spherical_factory.hpp>
#include <iomanip>
#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/coulomb.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/exact_exchange.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nuclear_repulsion.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>
#include <nukexc/xc_integrals.hpp>

#include <openorbitaloptimizer/scfsolver.hpp>
#include <xc.h>
#include <xc_funcs.h>

#include "standalone_helpers.hpp"
#include "test_io.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "sto_basis_reader.hpp"

using namespace Nukexc;
using bk_type = IntegratorXX::Becke<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

// ===========================================================================
// Config
// ===========================================================================
struct Config {
  std::string xyz_file = "input/water.xyz";
  std::string basis_file = "input/zorabasis_cholesky/TZ2P.cholesky";
  std::string method = "hf";       // "hf" or "dft"
  std::string algorithm = "dense"; // "dense", "sparse" or "tiled"
  int nrad = 100;
  int nang = 30;
  double lin_dep_threshold = 1e-6;
  double conv_thr = 1e-8;
  int charge = 0;
  int multiplicity = 1;
  int max_points_per_box = 32; // runtime via --box-size=
  int tile_size = 32; // neighbor tile (--tile=, only used by --alg=tiled)
  std::string xfunc = "lda_x";
  std::string cfunc = "lda_c_pw";
  double dens_thr = 1e-10;      // libxc density threshold (runtime --dens-thr=)
  std::string pruning = "none"; // grid angular pruning: none/robust/treutler
  std::string radial = "uniform"; // per-element radial sizing: uniform/pyscf
  std::string guess = "core";     // initial guess: core/gwh/sap
  std::string sap_dir = "";       // SAP potential dir (empty: built-in default)
  bool nl_histogram = false;      // runtime via --nl-histogram
};

Config parse_args(int argc, char *argv[]) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    ArgParser p{argv[i]};

    if (p.arg == "--help" || p.arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "  --xyz=<file>          Molecule XYZ file          (default: "
          << cfg.xyz_file << ")\n"
          << "  --basis=<file>        Basis set file             (default: "
          << cfg.basis_file << ")\n"
          << "  --method=<string>     'hf' or 'dft'              (default: "
          << cfg.method << ")\n"
          << "  --alg=<string>        'dense','sparse' or 'tiled'(default: "
          << cfg.algorithm << ")\n"
          << "  --box-size=<int>      Max quadrature points/box  (default: "
          << cfg.max_points_per_box << ")  [used by --alg=sparse,tiled]\n"
          << "  --tile=<int>          Neighbor tile size         (default: "
          << cfg.tile_size << ")  [only used by --alg=tiled]\n"
          << "  --nrad=<int>          Radial grid points         (default: "
          << cfg.nrad << ")\n"
          << "  --nang=<int>          Angular grid points        (default: "
          << cfg.nang << ")\n"
          << "  --lin-dep=<float>     Linear dep. threshold      (default: "
          << cfg.lin_dep_threshold << ")\n"
          << "  --conv-thr=<float>    SCF convergence threshold  (default: "
          << cfg.conv_thr << ")\n"
          << "  --charge=<int>        Molecular charge           (default: "
          << cfg.charge << ")\n"
          << "  --multiplicity=<int>  Spin multiplicity          (default: "
          << cfg.multiplicity << ")\n"
          << "  --xfunc=<name>        Exchange functional (dft)  (default: "
          << cfg.xfunc << ")\n"
          << "  --cfunc=<name>        Correlation functional(dft)(default: "
          << cfg.cfunc << ")\n"
          << "  --dens-thr=<float>    Libxc density threshold     (default: "
          << cfg.dens_thr << ")  [below this the functional is zeroed]\n"
          << "  --pruning=<name>      Angular pruning none/robust/treutler "
          << "(default: " << cfg.pruning << ")\n"
          << "  --radial=<name>       Radial sizing uniform/pyscf   (default: "
          << cfg.radial << ")\n"
          << "  --guess=<name>        Initial guess core/gwh/sap    (default: "
          << cfg.guess << ")  [core is more robust for near-linearly\n"
          << "                        dependent Slater sets; sap is the "
          << "superposition of atomic potentials]\n"
          << "  --sap-dir=<dir>       SAP potential directory     (default: "
          << "input/potentials/nr/LDAX/)  [only used by --guess=sap;\n"
          << "                        NUKEXC_SAP_DIR also works]\n"
          << "  --nl-histogram        Print neighbor-list histogram and write "
          << "pgfplots files  [used by --alg=sparse,tiled]\n"
          << "\n"
          << "  Supported functional names (only relevant for --method=dft):\n"
          << "    LDA exchange:     lda_x\n"
          << "    LDA correlation:  lda_c_pw, lda_c_vwn\n"
          << "    GGA exchange:     gga_x_pbe, gga_x_b88, gga_x_pw91\n"
          << "    GGA correlation:  gga_c_pbe, gga_c_lyp, gga_c_pw91\n";
      std::exit(0);
    } else if (p.arg == "--nl-histogram") {
      cfg.nl_histogram = true;
    } else if (!p.string_opt("--xyz=", cfg.xyz_file) &&
               !p.string_opt("--basis=", cfg.basis_file) &&
               !p.string_opt("--method=", cfg.method) &&
               !p.string_opt("--alg=", cfg.algorithm) &&
               !p.int_opt("--box-size=", cfg.max_points_per_box) &&
               !p.int_opt("--tile=", cfg.tile_size) &&
               !p.int_opt("--nrad=", cfg.nrad) &&
               !p.int_opt("--nang=", cfg.nang) &&
               !p.double_opt("--lin-dep=", cfg.lin_dep_threshold) &&
               !p.double_opt("--conv-thr=", cfg.conv_thr) &&
               !p.int_opt("--charge=", cfg.charge) &&
               !p.int_opt("--multiplicity=", cfg.multiplicity) &&
               !p.string_opt("--xfunc=", cfg.xfunc) &&
               !p.string_opt("--cfunc=", cfg.cfunc) &&
               !p.double_opt("--dens-thr=", cfg.dens_thr) &&
               !p.string_opt("--pruning=", cfg.pruning) &&
               !p.string_opt("--radial=", cfg.radial) &&
               !p.string_opt("--guess=", cfg.guess) &&
               !p.string_opt("--sap-dir=", cfg.sap_dir)) {
      throw std::runtime_error("Unknown argument: " + p.arg + " (try --help)");
    }
  }
  if (cfg.nrad <= 0 || cfg.nang <= 0)
    throw std::runtime_error("--nrad and --nang must be positive");
  if (cfg.max_points_per_box <= 0)
    throw std::runtime_error("--box-size must be positive");
  if (cfg.tile_size <= 0)
    throw std::runtime_error("--tile must be positive");
  if (cfg.method != "hf" && cfg.method != "dft")
    throw std::runtime_error("Unknown --method: " + cfg.method +
                             " (expected 'hf' or 'dft')");
  if (cfg.algorithm != "dense" && cfg.algorithm != "sparse" &&
      cfg.algorithm != "tiled")
    throw std::runtime_error("Unknown --alg: " + cfg.algorithm +
                             " (expected 'dense', 'sparse' or 'tiled')");
  if (cfg.pruning != "none" && cfg.pruning != "robust" &&
      cfg.pruning != "treutler")
    throw std::runtime_error("Unknown --pruning: " + cfg.pruning +
                             " (expected 'none', 'robust' or 'treutler')");
  if (cfg.radial != "uniform" && cfg.radial != "pyscf")
    throw std::runtime_error("Unknown --radial: " + cfg.radial +
                             " (expected 'uniform' or 'pyscf')");
  // Validated here rather than at the point of use so a typo fails before the
  // grid, basis and integrals are built.
  if (cfg.guess != "gwh" && cfg.guess != "core" && cfg.guess != "sap")
    throw std::runtime_error("Unknown --guess: " + cfg.guess +
                             " (expected 'gwh', 'core' or 'sap')");
  return cfg;
}

void print_config(const Config &cfg) {
  std::vector<std::pair<std::string, std::string>> rows = {
      {"Molecule file", cfg.xyz_file},
      {"Basis file", cfg.basis_file},
      {"Method", cfg.method},
      {"Algorithm", cfg.algorithm},
      {"Box size (pts/box)", cfg_val(cfg.max_points_per_box)},
  };
  if (cfg.algorithm == "tiled")
    rows.push_back({"Neighbor tile size", cfg_val(cfg.tile_size)});
  rows.push_back({"Radial points", cfg_val(cfg.nrad)});
  rows.push_back({"Angular order", cfg_val(cfg.nang)});
  rows.push_back({"Lin. dep. threshold", cfg_val(cfg.lin_dep_threshold)});
  rows.push_back({"Conv. threshold", cfg_val(cfg.conv_thr)});
  rows.push_back({"Initial guess", cfg.guess});
  if (cfg.guess == "sap")
    rows.push_back({"SAP potentials", cfg.sap_dir.empty()
                                          ? "input/potentials/nr/LDAX/"
                                          : cfg.sap_dir});
  rows.push_back({"Charge", cfg_val(cfg.charge)});
  rows.push_back({"Multiplicity", cfg_val(cfg.multiplicity)});
  if (cfg.method == "dft") {
    rows.push_back({"Exchange functional", cfg.xfunc});
    rows.push_back({"Correlation function", cfg.cfunc});
    rows.push_back({"Libxc density thresh.", cfg_val(cfg.dens_thr)});
  }

  print_config_box(
      cfg.method == "dft" ? "DFT Configuration" : "HF Configuration", rows);
}

int main(int argc, char *argv[]) {
  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  print_config(cfg);

  Kokkos::Timer program_timer;
  TimingRegistry startup_timing;
  TimingRegistry fock_cumulative_timing;

  Kokkos::initialize();
  {
    const bool is_dft = (cfg.method == "dft");
    const bool is_sparse = (cfg.algorithm == "sparse");
    const bool is_tiled = (cfg.algorithm == "tiled");
    // Both "sparse" and "tiled" use the neighbor-list quadrature path; they
    // differ only in which integral kernels are invoked.
    const bool use_neighbor_path = is_sparse || is_tiled;
    const int tile = cfg.tile_size;
    // The neighbor-list path evaluates the basis on the fly inside every
    // kernel it uses -- (A|B), J, K and, since the sparse XC quadrature
    // landed, the LSDA/GGA functionals too. Nothing on that path reads a
    // collocation matrix any more, so none of the four N_bf x N_quad tables
    // get allocated or filled for --alg=sparse/tiled.
    const bool need_dense_aux_colloc = !use_neighbor_path; // dense (A|B) + J/K
    const bool need_basis_colloc = !use_neighbor_path;
    // Only the dense GGA quadrature consumes the gradient collocation;
    // compute_gga_lsda_sparse differentiates the basis pointwise instead.
    const bool need_grad_colloc = is_dft && !use_neighbor_path;

    Molecule mol;
    {
      TIME_SCOPE(startup_timing, "Read XYZ");
      read_xyz(cfg.xyz_file, mol);
    }

    const double screening_tol = 1e-10;

    FlatGrid grid;
    {
      TIME_SCOPE(startup_timing, "Build molecular grid");
      const IntegratorXX::PruningScheme prune_scheme =
          cfg.pruning == "none"       ? IntegratorXX::PruningScheme::Unpruned
          : cfg.pruning == "treutler" ? IntegratorXX::PruningScheme::Treutler
                                      : IntegratorXX::PruningScheme::Robust;
      const RadialSizing radial_scheme =
          cfg.radial == "uniform" ? RadialSizing::Uniform : RadialSizing::PySCF;
      grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang,
                                              screening_tol * 1e-5, TA_M4,
                                              prune_scheme, radial_scheme);
    }

    STOBasisSet basis;
    STOBasisSet basis_aux;
    {
      TIME_SCOPE(startup_timing, "Load basis sets");
      {
        TIME_SCOPE(startup_timing, "Load primary basis");
        basis = test::load_adf_basis(mol, cfg.basis_file, screening_tol);
      }
      {
        TIME_SCOPE(startup_timing, "Load auxiliary (fitting) basis");
        basis_aux =
            test::load_adf_basis(mol, cfg.basis_file, screening_tol, /*fit=*/true);
      }
    }

    // ---- Resolve functionals from runtime strings (DFT only) -------------
    FunctionalInfo x_info{};
    FunctionalInfo c_info{};
    xc_func_type func_x{};
    xc_func_type func_c{};
    bool has_separate_c = false;
    double a_exx = 0.0;

    if (is_dft) {
      TIME_SCOPE(startup_timing, "Resolve/init XC functionals");

      x_info = lookup_functional(cfg.xfunc);
      if (xc_func_init(&func_x, x_info.xc_id, XC_POLARIZED) != 0) {
        throw std::runtime_error(
            "Failed to initialize Libxc exchange functional '" +
            x_info.canonical_name + "'");
      }
      // Libxc's own default (1e-15) is far looser than production codes use,
      // and leaves the GGA potentials enormous where the density has decayed
      // to roundoff: B3LYP at rho=1e-11 with a stale gradient returns
      // e_xc ~ -3e7. Multiplied by the r^2-growing radial weights of the
      // outer shells that injects large spurious contributions into the Fock
      // matrix. 1e-10 zeroes those points and leaves physically populated
      // regions untouched. Note this is a cleanliness measure, not a
      // convergence fix: the open-shell SCF stalls trace to the initial guess
      // and to the exactly degenerate open shells of the free atoms, and
      // raising this threshold as far as 1e-5 does not cure them.
      xc_func_set_dens_threshold(&func_x, cfg.dens_thr);

      // Libxc defines: XC_EXCHANGE=0, XC_CORRELATION=1,
      // XC_EXCHANGE_CORRELATION=2
      const bool is_combined_xc =
          (func_x.info->kind == XC_EXCHANGE_CORRELATION);
      c_info = lookup_functional(cfg.cfunc);

      if (is_combined_xc) {
        // Combined functionals (B3LYP, M06, ...) already include correlation.
        has_separate_c = false;
      } else {
        if (xc_func_init(&func_c, c_info.xc_id, XC_POLARIZED) != 0) {
          xc_func_end(&func_x);
          throw std::runtime_error(
              "Failed to initialize Libxc correlation functional '" +
              c_info.canonical_name + "'");
        }
        xc_func_set_dens_threshold(&func_c, cfg.dens_thr);
        has_separate_c = true;
      }

      a_exx = xc_hyb_exx_coef(&func_x);
    }

    // ---- Sizes -------------------------------------------------------------
    const int N_bf = basis.nbf();
    const int N_bf_aux = basis_aux.nbf();
    const int N_quad = grid.quad_points.extent(0);

    // Emitted in a fixed, greppable form so batch drivers (e.g.
    // benchmarking/benchmark_w411.py) can recover the problem size for scaling
    // plots.
    std::cout << "Basis functions (primary): " << N_bf << "\n";
    std::cout << "Basis functions (aux)    : " << N_bf_aux << "\n";
    std::cout << "Quadrature points        : " << N_quad << "\n";

    ExecSpace space;

    // ---- Neighbor lists (sparse path only) ---------------------------------
    // NOTE: create_bounding_boxes() PERMUTES grid.quad_points and grid.weights
    // in place (Morton ordering). It must therefore run BEFORE any collocation
    // is filled, otherwise the collocation columns refer to the pre-permutation
    // point ordering while grid.weights uses the post-permutation ordering,
    // which silently corrupts every subsequent quadrature

    NeighborList nl;     // primary basis neighbor list (K/J sparse)
    NeighborList nl_aux; // auxiliary basis neighbor list (aux overlap sparse)

    if (use_neighbor_path) {
      TIME_SCOPE(startup_timing, "Build neighbor lists (sparse)");
      {
        TIME_SCOPE(startup_timing, "Bounding boxes + neighbor list (aux)");
        auto bb_aux = create_bounding_boxes(grid, cfg.max_points_per_box);
        build_neighbor_list(STOPrimitiveEvaluator(basis_aux), bb_aux, cfg.max_points_per_box, N_quad,
                            nl_aux, cfg.nl_histogram, "neighbor_histogram_aux");
      }
      {
        TIME_SCOPE(startup_timing, "Bounding boxes + neighbor list (primary)");
        auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
        build_neighbor_list(STOPrimitiveEvaluator(basis), bb, cfg.max_points_per_box, N_quad, nl,
                            cfg.nl_histogram, "neighbor_histogram");
      }
    }

    // ---- Collocations (dense path and/or DFT quadrature) ------------------
    DeviceView2DLeft basis_collocation;
    DeviceView2DLeft basis_aux_collocation;
    DeviceView2DLeft potential_collocation_scaled;
    DeviceView2DLeft basis_collocation_gx, basis_collocation_gy,
        basis_collocation_gz;

    DeviceView2DLeft aux_overlap("Aux overlap", N_bf_aux, N_bf_aux);
    DeviceView2DLeft aux_overlap_sym("Aux overlap sym", N_bf_aux, N_bf_aux);

    if (need_basis_colloc) {
      TIME_SCOPE(startup_timing, "Fill primary basis collocation");
      basis_collocation = DeviceView2DLeft("Basis collocation", N_bf, N_quad);
      fill_collocation(space, basis, grid.quad_points, basis_collocation);
    }

    if (need_grad_colloc) {
      TIME_SCOPE(startup_timing, "Fill primary basis gradient collocation");
      basis_collocation_gx =
          DeviceView2DLeft("Basis collocation dX", N_bf, N_quad);
      basis_collocation_gy =
          DeviceView2DLeft("Basis collocation dY", N_bf, N_quad);
      basis_collocation_gz =
          DeviceView2DLeft("Basis collocation dZ", N_bf, N_quad);
      fill_grad_collocation(space, basis, grid.quad_points,
                            basis_collocation_gx, basis_collocation_gy,
                            basis_collocation_gz);
    }

    if (need_dense_aux_colloc) {
      TIME_SCOPE(startup_timing,
                 "Fill auxiliary basis collocation + potential");
      basis_aux_collocation =
          DeviceView2DLeft("Auxiliary Basis collocation", N_bf_aux, N_quad);
      potential_collocation_scaled =
          DeviceView2DLeft("Potential collocation", N_bf_aux, N_quad);
      fill_collocation(space, basis_aux, grid.quad_points,
                       basis_aux_collocation);
      sto_potential_collocation_scaled(space, basis_aux, grid,
                                       potential_collocation_scaled);
    }

    // ---- Auxiliary overlap (A|B), dense / sparse / tiled -------------------
    if (is_tiled) {
      TIME_SCOPE(startup_timing, "Auxiliary Coulomb overlap (tiled)");
      aux_overlap_sym =
          coulomb_overlap_integral_tiled(space, STOPotentialEvaluator(basis_aux), grid, nl_aux, tile);
    } else if (is_sparse) {
      TIME_SCOPE(startup_timing, "Auxiliary Coulomb overlap (sparse)");
      aux_overlap_sym =
          coulomb_overlap_integral_sparse(space, STOPotentialEvaluator(basis_aux), grid, nl_aux);
    } else {
      TIME_SCOPE(startup_timing, "Auxiliary Coulomb overlap (dense GEMM)");
      KokkosBlas::gemm(space, "N", "T", 1.0, basis_aux_collocation,
                       potential_collocation_scaled, 0.0, aux_overlap);
      Kokkos::parallel_for(
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf_aux, N_bf_aux}),
          KOKKOS_LAMBDA(int i, int j) {
            aux_overlap_sym(i, j) =
                0.5 * (aux_overlap(i, j) + aux_overlap(j, i));
          });
    }

    DeviceView2DLeft half_inverse_X;
    {
      TIME_SCOPE(startup_timing, "Half-inverse of auxiliary overlap");
      half_inverse_X =
          compute_half_inverse(aux_overlap_sym, cfg.lin_dep_threshold);
    }

    // ---- Core Hamiltonian ---------------------------------------------------
    decltype(compute_core_hamiltonian(basis, grid)) hcore;
    {
      TIME_SCOPE(startup_timing, "Core Hamiltonian (T + V_ne + S)");
      hcore = compute_core_hamiltonian(basis, grid);
    }

    // ---- SAP guess Hamiltonian ---------------------------------------------
    // A second pass over the grid with the bare Z replaced by the tabulated
    // screened charges Z_eff^A(r), giving T + V_SAP. Only the guess uses it:
    // the SCF itself keeps the true h_core below, so the converged energy is
    // unchanged and only the starting orbitals differ.
    arma::mat h_sap;
    if (cfg.guess == "sap") {
      TIME_SCOPE(startup_timing, "SAP guess Hamiltonian (T + V_SAP)");
      auto hsap =
          compute_core_hamiltonian(basis, grid, /*use_SAP=*/true, cfg.sap_dir);
      h_sap = kokkos_to_arma(hsap.hamiltonian);
    }

    // ---- Orthogonalisation matrix X
    // ------------------------------------------
    DeviceView2DLeft X;
    {
      TIME_SCOPE(startup_timing, "Orthogonalization matrix X = S^-1/2");
      Diagonalizer diag(N_bf);
      X = diag.compute_transformation(hcore.overlap, cfg.lin_dep_threshold);
    }

    arma::mat h_core, X_arma, S_arma;
    {
      TIME_SCOPE(startup_timing, "Copy core matrices to host (arma)");
      h_core = kokkos_to_arma(hcore.hamiltonian);
      X_arma = kokkos_to_arma(X);
      S_arma = kokkos_to_arma(hcore.overlap);
    }

    // Derive electron counts from geometry + charge + multiplicity
    int n_elec = mol.Z_total - cfg.charge;
    std::cout << "Number of Electrons: " << n_elec << std::endl;
    std::cout << "Z_total: " << mol.Z_total << std::endl;
    if ((n_elec + cfg.multiplicity - 1) % 2 != 0)
      throw std::runtime_error(
          "Charge and multiplicity are inconsistent with the number of "
          "electrons");
    double n_alpha = (n_elec + (cfg.multiplicity - 1)) / 2.0;
    double n_beta = (n_elec - (cfg.multiplicity - 1)) / 2.0;

    // ---- OOO setup
    // ------------------------------------------------------------
    arma::uvec blocks_per_type = {1, 1};    // 2 spin blocks (alpha and beta)
    arma::vec max_occupations = {1.0, 1.0}; // max 1 electron per spin channel
    arma::vec number_of_particles = {n_alpha, n_beta};
    std::vector<std::string> block_descriptions = {"alpha", "beta"};

    double E_nuc;
    {
      TIME_SCOPE(startup_timing, "Nuclear repulsion energy");
      E_nuc = compute_nuclear_repulsion(mol);
    }

    // ---- Initial guess -----------------------------------------------------
    arma::mat F_guess_orth;
    {
      TIME_SCOPE(startup_timing, "Initial guess (" + cfg.guess + ")");
      arma::mat F_guess(N_bf, N_bf, arma::fill::zeros);
      if (cfg.guess == "core") {
        // Bare core-Hamiltonian guess. GWH estimates off-diagonals from the
        // diagonal of h_core and the overlap, which populates near-null
        // directions when the basis is close to linearly dependent; the core
        // guess has no such estimate and is the diagnostic fallback.
        F_guess = h_core;
      } else if (cfg.guess == "gwh") {
        const double K_gwh = 1.75;
        for (int i = 0; i < N_bf; ++i) {
          for (int j = 0; j < N_bf; ++j) {
            if (i == j) {
              F_guess(i, j) = h_core(i, i);
            } else {
              F_guess(i, j) =
                  0.5 * K_gwh * (h_core(i, i) + h_core(j, j)) * S_arma(i, j);
            }
          }
        }
      } else if (cfg.guess == "sap") {
        // Superposition of atomic potentials, S. Lehtola, J. Chem. Theory
        // Comput. 15, 1593 (2019), DOI 10.1021/acs.jctc.8b01089.
        //
        // h_sap is already T + V_SAP. Adding h_core would double-count the
        // nuclear attraction, since V_SAP contains -Z/r in the r -> 0 limit.
        F_guess = h_sap;
      }
      F_guess_orth = X_arma.t() * F_guess * X_arma;
    }

    startup_timing.report("Startup Timing Breakdown");

    // ---- Fock builder (unified HF/DFT, dense/sparse) ------------------------
    // A shared call counter so we can label each SCF iteration's timing
    // report; the underlying int is shared across lambda copies.
    auto call_count = std::make_shared<int>(0);

    auto fock_builder =
        [space, grid, basis, basis_aux, nl, basis_collocation,
         basis_collocation_gx, basis_collocation_gy, basis_collocation_gz,
         basis_aux_collocation, potential_collocation_scaled, h_core, X_arma,
         half_inverse_X, N_bf, E_nuc, S_arma, cfg, func_c, func_x, x_info,
         c_info, has_separate_c, a_exx, is_dft = is_dft, is_sparse = is_sparse,
         is_tiled = is_tiled, tile = tile, call_count, &fock_cumulative_timing](
            const OpenOrbitalOptimizer::DensityMatrix<double, double> &dm)
        -> std::pair<double, OpenOrbitalOptimizer::FockMatrix<double>> {
      TimingRegistry fock_timing;
      ++(*call_count);

      const auto &orbitals = dm.first;     // vector<arma::mat>, one per block
      const auto &occupations = dm.second; // vector<arma::vec>, one per block

      arma::mat C_alpha, C_beta;
      arma::vec occ_alpha, occ_beta;
      arma::mat D_alpha, D_beta, D_tot;
      {
        TIME_SCOPE(fock_timing, "Build density matrices");
        C_alpha = X_arma * orbitals[0];
        C_beta = X_arma * orbitals[1];
        occ_alpha = occupations[0]; // 0 or 1
        occ_beta = occupations[1];  // 0 or 1

        D_alpha = C_alpha * arma::diagmat(occ_alpha) * C_alpha.t();
        D_beta = C_beta * arma::diagmat(occ_beta) * C_beta.t();
        D_tot = D_alpha + D_beta;
      }

#ifndef NDEBUG
      std::cout << "Tr[D_alpha * S] = " << arma::trace(D_alpha * S_arma)
                << "\n";
      std::cout << "Tr[D_beta  * S] = " << arma::trace(D_beta * S_arma) << "\n";
      std::cout << "Tr[D_tot   * S] = " << arma::trace(D_tot * S_arma) << "\n";
#endif

      // Combined alpha+beta orbitals/occupations, used for the total-density
      // Coulomb (J) build (same total density regardless of HF vs DFT).
      DeviceView2DLeft k_C_tot, k_C_alpha, k_C_beta;
      DeviceView1D k_occ_tot, k_occ_alpha, k_occ_beta;
      {
        TIME_SCOPE(fock_timing, "Host->device transfer of orbitals");
        arma::mat C_combined = arma::join_horiz(C_alpha, C_beta);
        arma::vec occ_combined = arma::join_vert(occ_alpha, occ_beta);

        k_C_tot = arma_to_kokkos(C_combined, "C_combined");
        k_occ_tot = arma_to_kokkos1d(occ_combined, "occ_combined");
        k_C_alpha = arma_to_kokkos(C_alpha, "C_alpha");
        k_C_beta = arma_to_kokkos(C_beta, "C_beta");
        k_occ_alpha = arma_to_kokkos1d(occ_alpha, "occ_alpha");
        k_occ_beta = arma_to_kokkos1d(occ_beta, "occ_beta");
      }

      // ---- Coulomb (J) build: same for HF and DFT, dense or sparse --------
      arma::mat J_arma;
      {
        TIME_SCOPE(fock_timing, "Coulomb (J) build");
        DeviceView2DLeft J;
        if (is_tiled) {
          J = compute_coulomb_tiled(space, k_C_tot, k_occ_tot, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux),
                                    grid, nl, half_inverse_X, tile);
        } else if (is_sparse) {
          J = compute_coulomb_sparse(space, k_C_tot, k_occ_tot, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux), grid, nl, half_inverse_X);
        } else {
          J = compute_coulomb(space, k_C_tot, k_occ_tot, basis_collocation,
                              basis_aux_collocation,
                              potential_collocation_scaled, half_inverse_X);
        }
        J_arma = kokkos_to_arma(J);
      }

      double E_exchange = 0.0;       // HF: exact exchange energy
                                     // DFT: exchange-functional energy (or
                                     // combined XC energy for combined funcs)
      double E_correlation = 0.0;    // DFT correlation-functional energy only
      double E_exact_exchange = 0.0; // DFT hybrid exact-exchange energy

      arma::mat K_alpha_arma = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat K_beta_arma = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat Vx_alpha = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat Vx_beta = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat Vc_alpha = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat Vc_beta = arma::zeros<arma::mat>(N_bf, N_bf);

      if (!is_dft) {
        // ---- Pure HF exact exchange -----------------------------------
        TIME_SCOPE(fock_timing, "Exact exchange (K) build");
        DeviceView2DLeft K_alpha, K_beta;
        {
          TIME_SCOPE(fock_timing, "  K alpha");
          if (is_tiled) {
            K_alpha = compute_exact_exchange_tiled(space, k_C_alpha, k_occ_alpha, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux), grid, nl,
                half_inverse_X, tile);
          } else if (is_sparse) {
            K_alpha = compute_exact_exchange_sparse(space, k_C_alpha, k_occ_alpha, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux), grid, nl,
                half_inverse_X);
          } else {
            K_alpha = compute_exact_exchange(
                space, k_C_alpha, k_occ_alpha, basis_collocation,
                basis_aux_collocation, potential_collocation_scaled,
                half_inverse_X);
          }
        }
        {
          TIME_SCOPE(fock_timing, "  K beta");
          if (is_tiled) {
            K_beta = compute_exact_exchange_tiled(space, k_C_beta, k_occ_beta, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux), grid, nl,
                                                  half_inverse_X, tile);
          } else if (is_sparse) {
            K_beta = compute_exact_exchange_sparse(space, k_C_beta, k_occ_beta, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux), grid, nl,
                                                   half_inverse_X);
          } else {
            K_beta = compute_exact_exchange(
                space, k_C_beta, k_occ_beta, basis_collocation,
                basis_aux_collocation, potential_collocation_scaled,
                half_inverse_X);
          }
        }
        K_alpha_arma = kokkos_to_arma(K_alpha);
        K_beta_arma = kokkos_to_arma(K_beta);
        E_exchange = -0.5 * arma::trace(D_alpha * K_alpha_arma) -
                     0.5 * arma::trace(D_beta * K_beta_arma);
      } else {
        // ---- DFT exchange-correlation quadrature -------------------------
        {
          if (is_tiled || is_sparse) {
            TIME_SCOPE(fock_timing, "Exchange functional (Vx)");
            XC_result_polarized E_x = evaluate_functional_sparse(
                x_info, func_x, basis, grid, nl, k_C_alpha, k_occ_alpha,
                k_C_beta, k_occ_beta);
            E_exchange = E_x.energy; // holds total E_xc if combined functional
            Vx_alpha = kokkos_to_arma(E_x.potential_alpha);
            Vx_beta = kokkos_to_arma(E_x.potential_beta);

            if (has_separate_c) {
              TIME_SCOPE(fock_timing, "Correlation functional (Vc)");
              XC_result_polarized E_c = evaluate_functional_sparse(
                  c_info, func_c, basis, grid, nl, k_C_alpha, k_occ_alpha,
                  k_C_beta, k_occ_beta);
              E_correlation = E_c.energy;
              Vc_alpha = kokkos_to_arma(E_c.potential_alpha);
              Vc_beta = kokkos_to_arma(E_c.potential_beta);
            }

          } else {
            TIME_SCOPE(fock_timing, "Exchange functional (Vx)");
            XC_result_polarized E_x = evaluate_functional(
                x_info, func_x, basis_collocation, basis_collocation_gx,
                basis_collocation_gy, basis_collocation_gz, grid.weights,
                k_C_alpha, k_occ_alpha, k_C_beta, k_occ_beta);
            E_exchange = E_x.energy; // holds total E_xc if combined functional
            Vx_alpha = kokkos_to_arma(E_x.potential_alpha);
            Vx_beta = kokkos_to_arma(E_x.potential_beta);

            if (has_separate_c) {
              TIME_SCOPE(fock_timing, "Correlation functional (Vc)");
              XC_result_polarized E_c = evaluate_functional(
                  c_info, func_c, basis_collocation, basis_collocation_gx,
                  basis_collocation_gy, basis_collocation_gz, grid.weights,
                  k_C_alpha, k_occ_alpha, k_C_beta, k_occ_beta);
              E_correlation = E_c.energy;
              Vc_alpha = kokkos_to_arma(E_c.potential_alpha);
              Vc_beta = kokkos_to_arma(E_c.potential_beta);
            }
          }
        }

        if (a_exx > 0) {
          TIME_SCOPE(fock_timing, "Exact exchange (hybrid) build");
          DeviceView2DLeft K_alpha, K_beta;
          {
            TIME_SCOPE(fock_timing, "  K alpha");
            if (is_tiled) {
              K_alpha = compute_exact_exchange_tiled(space, k_C_alpha, k_occ_alpha, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux), grid, nl,
                  half_inverse_X, tile);
            } else if (is_sparse) {
              K_alpha = compute_exact_exchange_sparse(space, k_C_alpha, k_occ_alpha, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux), grid, nl,
                  half_inverse_X);
            } else {
              K_alpha = compute_exact_exchange(
                  space, k_C_alpha, k_occ_alpha, basis_collocation,
                  basis_aux_collocation, potential_collocation_scaled,
                  half_inverse_X);
            }
          }
          {
            TIME_SCOPE(fock_timing, "  K beta");
            if (is_tiled) {
              K_beta = compute_exact_exchange_tiled(space, k_C_beta, k_occ_beta, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux), grid, nl,
                                                    half_inverse_X, tile);
            } else if (is_sparse) {
              K_beta = compute_exact_exchange_sparse(space, k_C_beta, k_occ_beta, STOPrimitiveEvaluator(basis), STOPotentialEvaluator(basis_aux), grid, nl,
                  half_inverse_X);
            } else {
              K_beta = compute_exact_exchange(
                  space, k_C_beta, k_occ_beta, basis_collocation,
                  basis_aux_collocation, potential_collocation_scaled,
                  half_inverse_X);
            }
          }
          K_alpha_arma = kokkos_to_arma(K_alpha);
          K_beta_arma = kokkos_to_arma(K_beta);
          E_exact_exchange = -0.5 * a_exx *
                             (arma::trace(D_alpha * K_alpha_arma) +
                              arma::trace(D_beta * K_beta_arma));
        }
      }

      double Etot;
      double E_core, E_coulomb;
      {
        TIME_SCOPE(fock_timing, "Energy assembly");
        E_core = arma::trace(D_tot * h_core);
        E_coulomb = 0.5 * arma::trace(D_tot * J_arma);
        Etot = E_nuc + E_core + E_coulomb + E_exchange + E_correlation +
               E_exact_exchange;
      }

      std::cout << "Total energy        : " << std::setprecision(10) << Etot
                << "\n";
      std::cout << "--------------------------------------------- \n";
      std::cout << "Nuclear Repulsion   : " << std::setprecision(10) << E_nuc
                << "\n";
      std::cout << "Electronic   Energy : " << std::setprecision(10)
                << E_core + E_coulomb + E_exchange + E_correlation +
                       E_exact_exchange
                << "\n";
      std::cout << "One Electron Energy : " << std::setprecision(10) << E_core
                << "\n";
      std::cout << "Two Electron Energy : " << std::setprecision(10)
                << E_coulomb + E_exchange + E_correlation + E_exact_exchange
                << "\n\n";

      arma::mat F_alpha_orth, F_beta_orth;
      {
        TIME_SCOPE(fock_timing, "Fock assembly + orthogonalization");
        arma::mat F_alpha, F_beta;
        if (!is_dft) {
          F_alpha = h_core + J_arma - K_alpha_arma;
          F_beta = h_core + J_arma - K_beta_arma;
        } else {
          F_alpha =
              h_core + J_arma - a_exx * K_alpha_arma + Vx_alpha + Vc_alpha;
          F_beta = h_core + J_arma - a_exx * K_beta_arma + Vx_beta + Vc_beta;
        }
        F_alpha_orth = X_arma.t() * F_alpha * X_arma;
        F_beta_orth = X_arma.t() * F_beta * X_arma;
      }

      fock_timing.report("Fock Build #" + std::to_string(*call_count) +
                         " Timing Breakdown");

      // Record this call's real total (sum of its top-level timed sections)
      // into the cumulative registry, so the summary reflects actual
      // per-call cost rather than bookkeeping overhead.
      fock_cumulative_timing.record("Fock build #" +
                                        std::to_string(*call_count),
                                    fock_timing.total_top_level_seconds());

      std::vector<arma::mat> fock_arma;
      fock_arma.push_back(F_alpha_orth);
      fock_arma.push_back(F_beta_orth);

      return std::make_pair(Etot, fock_arma);
    };

    // ---- Construct and run SCF solver
    // ----------------------------------------
    OpenOrbitalOptimizer::SCFSolver<double, double> solver(
        blocks_per_type, max_occupations, number_of_particles, fock_builder,
        block_descriptions);

    solver.verbosity(5);
    solver.initialize_with_fock({F_guess_orth, F_guess_orth});

    Kokkos::Timer scf_timer;
    solver.convergence_threshold(cfg.conv_thr);
    solver.run();
    const double scf_seconds = scf_timer.seconds();

    if (is_dft) {
      xc_func_end(&func_x);
      if (has_separate_c)
        xc_func_end(&func_c);
    }

    // ---- Check SCF converged to a sensible energy
    // -----------------------------
    double E_scf = solver.get_fock_build().first;
    std::cout << "SCF total energy: " << E_scf << " Eh\n";
    std::cout << "SCF wall time   : " << std::fixed << std::setprecision(4)
              << scf_seconds << " s over " << *call_count
              << " Fock build(s) (avg "
              << (*call_count > 0 ? scf_seconds / *call_count : 0.0)
              << " s/build)\n";

    fock_cumulative_timing.report(
        "Cumulative Fock Build Timing (per-call totals)");

    // ---- HOMO-LUMO gap
    // ---------------------------------------------------------
    auto final_fock =
        solver.get_fock_matrix(); // {F_alpha, F_beta}, orthonormal basis
    auto diagonalized =
        solver.compute_orbitals(final_fock); // {orbitals, orbital_energies}
    const auto &orbital_energies =
        diagonalized.second; // vector<arma::vec>, one per block
    const auto &occupations =
        solver.get_orbital_occupations(); // vector<arma::vec>, matching order

    auto find_homo_lumo = [](const arma::vec &energies, const arma::vec &occ,
                             double occ_thresh = 0.5) {
      double homo = -std::numeric_limits<double>::infinity();
      double lumo = std::numeric_limits<double>::infinity();
      for (arma::uword i = 0; i < energies.n_elem; ++i) {
        if (occ(i) > occ_thresh)
          homo = std::max(homo, energies(i));
        else
          lumo = std::min(lumo, energies(i));
      }
      return std::make_pair(homo, lumo);
    };

    auto [homo_a, lumo_a] = find_homo_lumo(orbital_energies[0], occupations[0]);
    auto [homo_b, lumo_b] = find_homo_lumo(orbital_energies[1], occupations[1]);

    double homo = std::max(homo_a, homo_b);
    double lumo = std::min(lumo_a, lumo_b);
    constexpr double Eh_to_eV = 27.211386245988;

    std::cout << "\nHOMO energy : " << std::setprecision(10) << homo << " Eh  ("
              << homo * Eh_to_eV << " eV)\n";
    std::cout << "LUMO energy : " << std::setprecision(10) << lumo << " Eh  ("
              << lumo * Eh_to_eV << " eV)\n";
    std::cout << "HOMO-LUMO gap: " << (lumo - homo) << " Eh  ("
              << (lumo - homo) * Eh_to_eV << " eV)\n";
  }
  Kokkos::finalize();

  std::cout << "\nTotal program wall time: " << std::fixed
            << std::setprecision(4) << program_timer.seconds() << " s\n";
  return 0;
}
