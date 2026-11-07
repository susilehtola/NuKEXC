/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Dump ERKALE's unrestricted exchange-correlation matrices for a fixed pair of
// spin density matrices, so NuKEXC's meta-GGA path can be checked against an
// independent implementation rather than only against itself.
//
// This is NOT part of the NuKEXC build. It links against ERKALE and is run by
// hand; see tests/reference/README.md for the invocation and the ERKALE commit.
//
// WHY A DENSITY MATRIX AND NOT AN SCF
// -----------------------------------
// The quantity under test is the exchange-correlation functional derivative at
// a given density, which needs no self-consistency: any symmetric positive
// semidefinite P will do, and a fixed arithmetic recipe is reproducible across
// both codes in a way converged SCF orbitals are not. The recipe below is
// rank-one plus a positive diagonal shift, so the density it gives is
// nonnegative everywhere -- which matters once it is split between spin
// channels, because Libxc is entitled to misbehave on a negative density.
// It is the same recipe tests/test_xc_kernel_session.cxx uses, but the
// matrices are written into the reference file anyway: reading them from the
// same file as the results is what removes the question of whether the two
// codes were handed the same density.
//
// WHY THE GRID IS NOT DUMPED
// --------------------------
// Both codes integrate on their own grid, converged, and the comparison is
// made at the grid-error level. ERKALE's DFTGrid keeps its points private, so
// dumping them would mean reproducing DFTGrid::construct in here -- a second
// implementation of the thing being trusted, which is worse than a tolerance
// honestly measured. The reference is therefore written at two grid sizes, and
// the test uses the difference between them as the error estimate rather than
// a tolerance someone picked.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include <xc.h>
#include <xc_funcs.h>

#include "basis.h"
#include "basislibrary.h"
#include "dftgrid.h"
#include "settings.h"
#include "xyzutils.h"

// ERKALE's library code reads a global Settings object that its drivers are
// expected to define. Basis construction and the DFT grid both consult it.
Settings settings;

namespace {

/// Rank-one plus a positive diagonal shift: symmetric, positive
/// semidefinite, and a fixed function of the indices alone.
arma::mat make_P(size_t nbf, double freq, double scale) {
  arma::mat P(nbf, nbf, arma::fill::zeros);
  for (size_t i = 0; i < nbf; i++)
    for (size_t j = 0; j < nbf; j++)
      P(i, j) = 0.04 * std::cos(freq * (i + 1)) * std::cos(freq * (j + 1));
  for (size_t i = 0; i < nbf; i++)
    P(i, i) += 0.05;
  return scale * P;
}

void write_block(FILE *f, const char *name, const arma::mat &M) {
  std::fprintf(f, "%s %llu\n", name,
               (unsigned long long)(M.n_rows * M.n_cols));
  for (arma::uword i = 0; i < M.n_rows; i++)
    for (arma::uword j = 0; j < M.n_cols; j++)
      std::fprintf(f, "%.17e\n", M(i, j));
}

void write_scalar(FILE *f, const char *name, double v) {
  std::fprintf(f, "%s 1\n%.17e\n", name, v);
}

struct Functional {
  const char *name;
  int x_func;
  int c_func;
};

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: %s <geometry.xyz> <basis.gbs> <output>\n"
                 "  e.g. %s water.xyz '6-31G(d,p).gbs' water_631gdp_xc.ref\n",
                 argv[0], argv[0]);
    return 1;
  }
  const std::string xyz = argv[1], gbs = argv[2], out = argv[3];

  settings.add_scf_settings();

  std::vector<atom_t> atoms = load_xyz(xyz, true);

  BasisSetLibrary baslib;
  baslib.load_gaussian94(gbs, false);

  BasisSet basis;
  construct_basis(basis, atoms, baslib);
  const size_t nbf = basis.Nbf();

  const arma::mat Pa = make_P(nbf, 0.9, 0.6);
  const arma::mat Pb = make_P(nbf, 1.7, 0.4);

  // LDA and GGA are dumped alongside the meta-GGAs on purpose. They are the
  // cases NuKEXC already validates by other means, so if the two codes'
  // conventions were mismatched -- basis ordering, the factor on the XC
  // matrix, what "unrestricted" means for Exc -- those would fail too, and
  // the meta-GGA numbers would not be left carrying the blame alone.
  const Functional funcs[] = {
      {"LDA_X", XC_LDA_X, 0},
      {"GGA_X_PBE", XC_GGA_X_PBE, 0},
      {"GGA_XC_PBE", XC_GGA_X_PBE, XC_GGA_C_PBE},
      {"MGGA_X_TPSS", XC_MGGA_X_TPSS, 0},
      {"MGGA_XC_TPSS", XC_MGGA_X_TPSS, XC_MGGA_C_TPSS},
      {"MGGA_X_SCAN", XC_MGGA_X_SCAN, 0},
      {"MGGA_X_BR89", XC_MGGA_X_BR89_EXPLICIT, 0},
  };

  // Two grid sizes, so the test can read the convergence off the file
  // instead of being handed a tolerance.
  const struct {
    const char *tag;
    int nrad;
    int lmax;
  } grids[] = {{"coarse", 75, 29}, {"fine", 150, 41}};

  FILE *f = std::fopen(out.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", out.c_str());
    return 1;
  }

  const std::vector<GaussianShell> &shells = basis.shells_ref();
  std::fprintf(f, "# ERKALE unrestricted XC reference\n");
  std::fprintf(f, "# geometry %s\n# basis %s\n", xyz.c_str(), gbs.c_str());
  std::fprintf(f, "nshell %zu\nnbf %zu\n", shells.size(), nbf);

  for (size_t i = 0; i < shells.size(); i++) {
    const GaussianShell &sh = shells[i];
    const std::vector<contr_t> &c = sh.contr_ref();
    const arma::mat &cf = sh.coefs();
    const coords_t o = sh.center();
    std::fprintf(f, "shell %d %d %.17e %.17e %.17e %zu %zu\n", sh.am(),
                 sh.lm_in_use() ? 1 : 0, o.x, o.y, o.z, c.size(), sh.Nctr());
    for (size_t p = 0; p < c.size(); p++)
      std::fprintf(f, "  exp %.17e\n", c[p].z);
    for (size_t k = 0; k < sh.Nctr(); k++)
      for (size_t p = 0; p < c.size(); p++)
        std::fprintf(f, "  coef %.17e\n", cf(p, k));
  }

  write_block(f, "Pa", Pa);
  write_block(f, "Pb", Pb);

  for (const auto &g : grids) {
    std::fprintf(f, "# grid %s: %d radial shells, Lebedev lmax %d\n", g.tag,
                 g.nrad, g.lmax);
    std::fprintf(f, "nrad_%s 1\n%d\n", g.tag, g.nrad);
    std::fprintf(f, "lmax_%s 1\n%d\n", g.tag, g.lmax);

    for (const auto &fn : funcs) {
      DFTGrid grid(&basis, false);
      grid.construct(g.nrad, g.lmax, fn.x_func, fn.c_func);

      arma::mat Ha, Hb;
      double Exc = 0.0, Nel = 0.0;
      grid.eval_Fxc(fn.x_func, fn.c_func, Pa, Pb, Ha, Hb, Exc, Nel);

      const std::string pre = std::string(fn.name) + "_" + g.tag;
      write_scalar(f, (pre + "_Exc").c_str(), Exc);
      write_scalar(f, (pre + "_Nel").c_str(), Nel);
      write_scalar(f, (pre + "_npoints").c_str(),
                   static_cast<double>(grid.Npoints()));
      write_block(f, (pre + "_Ha").c_str(), Ha);
      write_block(f, (pre + "_Hb").c_str(), Hb);

      std::printf("%-14s %-6s  npts %8zu  Exc % .12f  Nel % .10f\n", fn.name,
                  g.tag, grid.Npoints(), Exc, Nel);
      std::fflush(stdout);
    }
  }

  std::fclose(f);
  std::printf("wrote %s: %zu shells, %zu functions\n", out.c_str(),
              shells.size(), nbf);
  return 0;
}
