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

// What the Cartesian detour costs a spherical Gaussian shell.
//
// A spherical Gaussian is R(r) S_lm with R = sum_p c_p exp(-alpha_p r^2).
// nukexc/gto_eval.hpp does not evaluate it that way: it builds the
// cart_size(l) = (l+1)(l+2)/2 Cartesian components x^i y^j z^k exp(-alpha
// r^2), renormalizes each, and then applies a (2l+1) x cart_size(l)
// transform per derivative component, per point. That is strictly more work
// than the product -- more components, plus the transform -- and the
// separability the Cartesian form exists for is not used at all here: the
// kernel computes r^2 once and calls exp once per primitive, exactly as the
// spherical route does.
//
// It is not obvious how much that costs, because both routes pay the same
// exponentials and those are expensive. The ratio therefore depends on l (the
// transform grows as (2l+1) cart_size(l)) and on the contraction length (more
// primitives means more exp, which both pay equally). Hence a measurement
// rather than an operation count.
//
// Cartesian shells are a separate matter: a Cartesian Gaussian is not a
// single R(r) S_lm, so for a basis that genuinely uses them the Cartesian
// path is the only one that can express the functions, and gto_eval.hpp
// already skips the transform for them. This benchmark is about the
// spherical case.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/basis_evaluator.hpp>
#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/radial_shell.hpp>

#include "standards.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

/// One shell of the given l on every atom, with `nprim` primitives.
struct Synthetic {
  std::vector<int> l, nprim, prim_offset, coeff_offset;
  std::vector<double> alpha, coeff, centers;
  int nshell = 0;
};

Synthetic make_shells(const Molecule &mol, int l, int nprim) {
  Synthetic s;
  const int natom = static_cast<int>(mol.natoms);
  auto c_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.atom_centers);
  s.nshell = natom;
  int po = 0;
  for (int a = 0; a < natom; ++a) {
    s.l.push_back(l);
    s.nprim.push_back(nprim);
    s.prim_offset.push_back(po);
    s.coeff_offset.push_back(po);
    for (int p = 0; p < nprim; ++p) {
      // A plausible contraction: exponents spanning two decades, positive
      // coefficients. The timing does not depend on the values.
      s.alpha.push_back(0.4 * std::pow(3.0, p));
      s.coeff.push_back(1.0 / (p + 1));
    }
    po += nprim;
    for (int k = 0; k < 3; ++k)
      s.centers.push_back(c_h(a)[k]);
  }
  s.prim_offset.push_back(po);
  s.coeff_offset.push_back(po);
  return s;
}

GTOBasisDevice cartesian_path(const Synthetic &s) {
  GTOBasisInput in;
  in.nshell = s.nshell;
  in.l = s.l.data();
  in.nprim = s.nprim.data();
  in.prim_offset = s.prim_offset.data();
  in.coeff_offset = s.coeff_offset.data();
  in.alpha = s.alpha.data();
  in.coeff = s.coeff.data();
  in.centers = s.centers.data();
  in.default_type = ShellType::Spherical; // the case under test
  return make_gto_device_basis(make_gto_basis(in));
}

GaussianRadial spherical_path(const Synthetic &s) {
  GaussianRadial rad;
  ShellTable &t = rad.sh;
  t.nshell = s.nshell;
  t.l = Kokkos::View<int *>("l", s.nshell);
  t.first_bf = Kokkos::View<int *>("first bf", s.nshell + 1);
  t.prim_offset = Kokkos::View<int *>("prim offset", s.nshell + 1);
  t.center = Kokkos::View<Point *>("center", s.nshell);
  t.radius = Kokkos::View<double *>("radius", s.nshell);

  auto tl = Kokkos::create_mirror_view(t.l);
  auto tf = Kokkos::create_mirror_view(t.first_bf);
  auto tp = Kokkos::create_mirror_view(t.prim_offset);
  auto tc = Kokkos::create_mirror_view(t.center);
  auto tr = Kokkos::create_mirror_view(t.radius);

  int bf = 0;
  for (int i = 0; i < s.nshell; ++i) {
    tl(i) = s.l[i];
    tf(i) = bf;
    tp(i) = s.prim_offset[i];
    for (int k = 0; k < 3; ++k)
      tc(i)[k] = s.centers[3 * i + k];
    tr(i) = 1e30; // no screening, so both paths do the same work everywhere
    bf += 2 * s.l[i] + 1;
  }
  tf(s.nshell) = bf;
  tp(s.nshell) = s.prim_offset[s.nshell];
  t.nbf = bf;

  Kokkos::deep_copy(t.l, tl);
  Kokkos::deep_copy(t.first_bf, tf);
  Kokkos::deep_copy(t.prim_offset, tp);
  Kokkos::deep_copy(t.center, tc);
  Kokkos::deep_copy(t.radius, tr);

  rad.alpha = Kokkos::View<double *>("alpha", s.alpha.size());
  rad.coeff = Kokkos::View<double *>("coeff", s.coeff.size());
  auto ah = Kokkos::create_mirror_view(rad.alpha);
  auto ch = Kokkos::create_mirror_view(rad.coeff);
  for (size_t i = 0; i < s.alpha.size(); ++i) {
    ah(i) = s.alpha[i];
    ch(i) = s.coeff[i];
  }
  Kokkos::deep_copy(rad.alpha, ah);
  Kokkos::deep_copy(rad.coeff, ch);
  return rad;
}

/// Time a value-only and a value-and-gradient collocation fill.
template <typename Evaluator>
std::pair<double, double> time_fills(const Evaluator &ev, const FlatGrid &grid,
                                     int repeats) {
  const int nbf = ev.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  DeviceView2DLeft v("v", nbf, npts), gx("gx", nbf, npts), gy("gy", nbf, npts),
      gz("gz", nbf, npts);
  ExecSpace space;

  // Warm up once; the first launch of a kernel pays for its instantiation.
  fill_collocation_values(space, ev, grid.quad_points, v);
  Kokkos::fence();

  Kokkos::Timer t;
  for (int i = 0; i < repeats; ++i)
    fill_collocation_values(space, ev, grid.quad_points, v);
  Kokkos::fence();
  const double tv = t.seconds() / repeats;

  fill_collocation_value_grad(space, ev, grid.quad_points, v, gx, gy, gz);
  Kokkos::fence();
  t.reset();
  for (int i = 0; i < repeats; ++i)
    fill_collocation_value_grad(space, ev, grid.quad_points, v, gx, gy, gz);
  Kokkos::fence();
  const double tg = t.seconds() / repeats;
  return {tv, tg};
}

} // namespace

int main(int argc, char *argv[]) {
  int nrad = 50, nang = 20, repeats = 3;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--nrad=", 0) == 0)
      nrad = std::stoi(a.substr(7));
    else if (a.rfind("--nang=", 0) == 0)
      nang = std::stoi(a.substr(7));
    else if (a.rfind("--repeats=", 0) == 0)
      repeats = std::stoi(a.substr(10));
    else {
      std::cout << "Usage: " << argv[0]
                << " [--nrad=N] [--nang=N] [--repeats=N]\n";
      return a == "--help" || a == "-h" ? 0 : 1;
    }
  }

  Kokkos::initialize(argc, argv);
  {
    Molecule mol = make_benzene();
    const auto grid = make_flat_grid<ta_type, ll_type>(mol, nrad, nang);
    const int npts = static_cast<int>(grid.quad_points.extent(0));

    std::cout << "\nSpherical Gaussian collocation: Cartesian-and-transform "
                 "against the direct product\n"
              << "benzene, " << npts << " grid points, " << repeats
              << " repeats, one shell per atom\n\n"
              << std::left << std::setw(4) << "l" << std::setw(7) << "nprim"
              << std::right << std::setw(12) << "cart" << std::setw(12)
              << "value (s)" << std::setw(12) << "direct" << std::setw(9)
              << "ratio" << std::setw(14) << "grad (s)" << std::setw(12)
              << "direct" << std::setw(9) << "ratio" << "\n";

    for (int l = 0; l <= 4; ++l) {
      for (int nprim : {1, 3, 6}) {
        const auto s = make_shells(mol, l, nprim);
        const GTOEvaluator cart(cartesian_path(s));
        const GaussianShellEvaluator sph(spherical_path(s));
        if (cart.nbf() != sph.nbf()) {
          std::cout << "l = " << l << ": function counts differ ("
                    << cart.nbf() << " vs " << sph.nbf() << ")\n";
          continue;
        }
        // Do the two routes agree on the FUNCTIONS, not just the count?
        // If they do, dispatching to the direct product for spherical
        // shells is a drop-in; if they differ by a constant per (l, m),
        // that constant has to be folded in first.
        if (nprim == 3 && l <= 2) {
          const int nf = 2 * l + 1;
          Kokkos::View<double *> a("a", 32), b("b", 32);
          auto ce = cart;
          auto se = sph;
          Kokkos::parallel_for(
              "compare", Kokkos::RangePolicy<ExecSpace>(0, 1),
              KOKKOS_LAMBDA(const int) {
                double bc[64], bs[64];
                for (int i = 0; i < 64; ++i)
                  bc[i] = bs[i] = 0.0;
                ce.template eval_block<BasisDeriv::Value>(ce.load_block(0),
                                                          0.31, -0.22, 0.17,
                                                          bc, 32);
                se.template eval_block<BasisDeriv::Value>(se.load_block(0),
                                                          0.31, -0.22, 0.17,
                                                          bs, 32);
                for (int i = 0; i < 32; ++i) {
                  a(i) = bc[i];
                  b(i) = bs[i];
                }
              });
          Kokkos::fence();
          auto ah = Kokkos::create_mirror_view_and_copy(HostSpace{}, a);
          auto bh = Kokkos::create_mirror_view_and_copy(HostSpace{}, b);
          // The ratio is one scalar per shell, the same for every m: the
          // two routes build the SAME functions in the same order, and
          // differ only in radial normalization -- gto_eval.hpp renormalizes
          // its contractions and folds in a per-Cartesian relnorm, while
          // this GaussianRadial was handed raw coefficients.
          // solid_harmonics_transform.hpp's matrix produces normalized real
          // solid harmonics r^l Y_lm, which is exactly what the hardcoded
          // switch returns, so there is no angular convention to reconcile.
          double r0 = 0.0, spread = 0.0;
          for (int i = 0; i < nf; ++i) {
            const double ri = (bh(i) == 0.0) ? 0.0 : ah(i) / bh(i);
            if (i == 0)
              r0 = ri;
            else
              spread = std::max(spread, std::abs(ri - r0));
          }
          std::cout << "  l = " << l << ": cart/direct = " << std::setprecision(6)
                    << r0 << " for every m (spread " << spread << ")\n";
        }
        const auto c = time_fills(cart, grid, repeats);
        const auto p = time_fills(sph, grid, repeats);
        std::cout << std::left << std::setw(4) << l << std::setw(7) << nprim
                  << std::right << std::setw(12) << ((l + 1) * (l + 2) / 2)
                  << std::fixed << std::setprecision(4) << std::setw(12)
                  << c.first << std::setw(12) << p.first
                  << std::setprecision(2) << std::setw(9)
                  << c.first / p.first << std::setprecision(4)
                  << std::setw(14) << c.second << std::setw(12) << p.second
                  << std::setprecision(2) << std::setw(9)
                  << c.second / p.second << "\n";
      }
    }
    std::cout << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
