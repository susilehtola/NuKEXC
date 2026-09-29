/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// Dump an ERKALE basis set and its analytic overlap matrix, so NuKEXC can be
// compared against ERKALE rather than only against its own closed forms.
//
// This is NOT part of the NuKEXC build. It links against ERKALE and is run by
// hand to regenerate the reference files under tests/reference/; see the
// README there for the exact invocation and the ERKALE commit used.
//
// It emits the basis in the same plain-array form NuKEXC's GTOBasisInput
// consumes -- shells, exponents, contraction-major coefficients, centers --
// followed by the overlap. Dumping both together is the point: it removes any
// question of whether the two codes were given the same basis, which is
// otherwise the easiest way to get a meaningless comparison.
//
// Coefficients are written as ERKALE holds them internally, i.e. coefficients
// of UNNORMALIZED primitives after GaussianShell::normalize(), which is what
// NuKEXC reproduces with NormalizationConvention::ERKALE.

#include <cstdio>
#include <string>
#include <vector>

#include "basis.h"
#include "basislibrary.h"
#include "settings.h"
#include "xyzutils.h"

// ERKALE's library code reads a global Settings object that its drivers are
// expected to define. Basis construction consults it (basis.cpp does, for
// the Cartesian/spherical default among other things), so define one and
// populate the keys that path touches.
Settings settings;

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: %s <geometry.xyz> <basis.gbs> <output>\n"
                 "  e.g. %s water.xyz cc-pVDZ.gbs water_ccpvdz.ref\n",
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

  const std::vector<GaussianShell> &shells = basis.shells_ref();
  const arma::mat S = basis.overlap();

  FILE *f = std::fopen(out.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", out.c_str());
    return 1;
  }

  std::fprintf(f, "# ERKALE basis + analytic overlap reference\n");
  std::fprintf(f, "# geometry %s\n# basis %s\n", xyz.c_str(), gbs.c_str());
  std::fprintf(f, "nshell %zu\nnbf %zu\n", shells.size(), basis.Nbf());

  // One record per shell: l, spherical, center, nprim, nctr, then the
  // exponents and the contraction-major coefficient matrix.
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

  std::fprintf(f, "overlap %llu\n",
               (unsigned long long)(S.n_rows * S.n_cols));
  for (arma::uword i = 0; i < S.n_rows; i++)
    for (arma::uword j = 0; j < S.n_cols; j++)
      std::fprintf(f, "%.17e\n", S(i, j));

  std::fclose(f);
  std::printf("wrote %s: %zu shells, %zu functions\n", out.c_str(),
              shells.size(), basis.Nbf());
  return 0;
}
