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
#include <nukexc/xc_kernel_session.hpp>

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
  /// Also time the kernel-backed sessions, dense against screened. Off by
  /// default because the dense store holds the whole collocation tower --
  /// gigabytes on benzene at these grid sizes -- and a benchmark should not
  /// surprise a shared machine with that.
  bool sessions = false;
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
                << "  --sessions        Also time the kernel sessions, "
                   "dense against screened\n";
      std::exit(0);
    } else if (p.arg == "--sessions") {
      cfg.sessions = true;
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

// ---------------------------------------------------------------------------
//  The kernel-backed sessions, dense against screened
// ---------------------------------------------------------------------------
//
// Same density matrix, same grid, same functional, same kernels: the two
// differ only in where the collocation lives. What the table measures is
// therefore exactly what that choice costs -- time per Fock build against
// the memory held between builds -- for the three families whose
// collocation stops at the gradient.

struct SessionRow {
  std::string molecule;
  const char *family = "";
  double t_ctor_dense = 0.0, t_ctor_scr = 0.0;
  double t_fock_dense = 0.0, t_fock_scr = 0.0;
  double mem_dense = 0.0, mem_scr = 0.0; ///< MiB held between builds
  double e_dense = 0.0, e_scr = 0.0;
};

/// A symmetric density matrix from the model orbitals, column-major, in the
/// layout the sessions take.
std::vector<double> model_density(const DeviceView2DLeft &C) {
  const int nbf = static_cast<int>(C.extent(0));
  const int nocc = static_cast<int>(C.extent(1));
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, C);
  std::vector<double> P(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int u = 0; u < nbf; ++u)
    for (int v = 0; v < nbf; ++v) {
      double acc = 0.0;
      for (int k = 0; k < nocc; ++k)
        acc += h(u, k) * h(v, k);
      P[static_cast<size_t>(u) + static_cast<size_t>(v) * nbf] = acc;
    }
  return P;
}

/// One Fock build: contract the density, evaluate the functional, assemble.
/// Returns the wall time and the energy.
template <typename Session>
std::pair<double, double> one_fock_build(Session &s, const HostMatrixView &P,
                                         xc_func_type &func, XCFamily fam) {
  const int nbf = s.nbf(), ng = s.npoints();
  std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixRef R{V.data(), nbf, MatrixLayout::ColumnMajor};
  std::vector<double> zk(ng), vrho(ng), vsigma(ng), vlapl(ng), vtau(ng);
  const std::vector<double> lapl_in(ng, 0.0);

  Kokkos::Timer t;
  s.set_density(P);
  if (fam == XCFamily::LDA)
    xc_lda_exc_vxc(&func, ng, s.rho().data(), zk.data(), vrho.data());
  else if (fam == XCFamily::GGA)
    xc_gga_exc_vxc(&func, ng, s.rho().data(), s.sigma().data(), zk.data(),
                   vrho.data(), vsigma.data());
  else
    xc_mgga_exc_vxc(&func, ng, s.rho().data(), s.sigma().data(),
                    lapl_in.data(), s.tau().data(), zk.data(), vrho.data(),
                    vsigma.data(), vlapl.data(), vtau.data());

  XCKernelDerivs d{.vrho = vrho.data()};
  if (fam != XCFamily::LDA)
    d.vsigma = vsigma.data();
  if (fam == XCFamily::MetaGGATau)
    d.vtau = vtau.data();
  const XCOutput out = s.accumulate_fock(d, zk.data(), R);
  Kokkos::fence();
  return {t.seconds(), out.Exc};
}

constexpr double to_mib(size_t doubles) {
  return static_cast<double>(doubles) * sizeof(double) / (1024.0 * 1024.0);
}

void print_session_rows(const std::vector<SessionRow> &rows) {
  std::cout << "\nKernel sessions, dense against screened. Construction is "
               "once per session,\nFock is once per SCF iteration; memory is "
               "what the store holds between builds.\n\n"
            << std::left << std::setw(10) << "Molecule" << std::setw(10)
            << "Family" << std::right << std::setw(11) << "ctor D"
            << std::setw(11) << "ctor S" << std::setw(11) << "Fock D"
            << std::setw(11) << "Fock S" << std::setw(11) << "MiB D"
            << std::setw(11) << "MiB S" << std::setw(16) << "Exc (dense)"
            << std::setw(14) << "dExc\n";
  for (const auto &r : rows)
    std::cout << std::left << std::setw(10) << r.molecule << std::setw(10)
              << r.family << std::right << std::fixed << std::setprecision(4)
              << std::setw(11) << r.t_ctor_dense << std::setw(11)
              << r.t_ctor_scr << std::setw(11) << r.t_fock_dense
              << std::setw(11) << r.t_fock_scr << std::setprecision(1)
              << std::setw(11) << r.mem_dense << std::setw(11) << r.mem_scr
              << std::setprecision(8) << std::setw(16) << r.e_dense
              << std::scientific << std::setprecision(2) << std::setw(14)
              << std::abs(r.e_scr - r.e_dense) << std::defaultfloat << "\n";
  std::cout << std::flush;
}

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

    // Unpolarized handles for the restricted sessions; the screened
    // routines above want the polarized ones.
    xc_func_type lda_u, gga_u, mgga_u;
    if (xc_func_init(&lda_u, XC_LDA_X, XC_UNPOLARIZED) != 0 ||
        xc_func_init(&gga_u, XC_GGA_X_PBE, XC_UNPOLARIZED) != 0 ||
        xc_func_init(&mgga_u, XC_MGGA_X_TPSS, XC_UNPOLARIZED) != 0) {
      std::cerr << "libxc refused to initialize a functional\n";
      Kokkos::finalize();
      return 1;
    }

    std::vector<Row> rows;
    std::vector<SessionRow> srows;
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

        if (!cfg.sessions)
          continue;

        // One density matrix for both stores, and two builds each: on a
        // shared machine a single timing is worth little, so the second is
        // the one reported.
        const auto P = model_density(Ca);
        const HostMatrixView Pv{P.data(), r.nbf, MatrixLayout::ColumnMajor};
        struct Fam {
          XCFamily fam;
          xc_func_type *func;
          const char *name;
        };
        const Fam fams[] = {{XCFamily::LDA, &lda_u, "LDA"},
                            {XCFamily::GGA, &gga_u, "GGA"},
                            {XCFamily::MetaGGATau, &mgga_u, "mGGA-tau"}};
        for (const auto &fm : fams) {
          SessionRow sr;
          sr.molecule = name;
          sr.family = fm.name;

          t.reset();
          XCKernelSession<STOPrimitiveEvaluator> dense(ev, grid, fm.fam);
          Kokkos::fence();
          sr.t_ctor_dense = t.seconds();
          sr.mem_dense = to_mib(dense.collocation_doubles());
          one_fock_build(dense, Pv, *fm.func, fm.fam);
          const auto dres = one_fock_build(dense, Pv, *fm.func, fm.fam);
          sr.t_fock_dense = dres.first;
          sr.e_dense = dres.second;

          t.reset();
          XCScreenedKernelSession<STOPrimitiveEvaluator> scr(ev, grid, nl,
                                                             fm.fam);
          Kokkos::fence();
          sr.t_ctor_scr = t.seconds();
          sr.mem_scr = to_mib(scr.collocation_doubles());
          one_fock_build(scr, Pv, *fm.func, fm.fam);
          const auto sres = one_fock_build(scr, Pv, *fm.func, fm.fam);
          sr.t_fock_scr = sres.first;
          sr.e_scr = sres.second;

          srows.push_back(sr);
        }
      }
      print_rows(rows);
      if (cfg.sessions)
        print_session_rows(srows);
    } catch (const std::exception &e) {
      std::cerr << "benchmark_xc failed: " << e.what() << "\n";
      status = 1;
    }

    xc_func_end(&lda);
    xc_func_end(&gga);
    xc_func_end(&lda_u);
    xc_func_end(&gga_u);
    xc_func_end(&mgga_u);
  }
  Kokkos::finalize();
  return status;
}
