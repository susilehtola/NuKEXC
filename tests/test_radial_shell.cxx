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

// One evaluator, several radial families.
//
// WHAT IS BEING TESTED
// --------------------
// nukexc/radial_shell.hpp writes the shared half of a basis evaluation once
// -- the solid harmonics, the shell structure, the m = -l..l ordering, the
// gradient chain rule, the cutoff -- and leaves only R(r) to the family. The
// risk in that arrangement is that the shared half is subtly wrong for one
// family and right for another, which no amount of reading catches. So each
// family is pinned to a reference that does not share code with it:
//
//   Slater    against STOPrimitiveEvaluator through compute_core_hamiltonian.
//             That path is validated against ERKALE, so agreement puts the
//             templated one on the same footing.
//
//   Gaussian  against the product R(r) S_lm written out in this file from
//             the basis parameters. Three lines, no shared machinery, and it
//             checks the one thing the Slater comparison cannot: that the
//             shared half is right for a family whose radial has a different
//             shape.
//
// And both against finite differences for the gradient, which is the only
// reference for a chain rule that cannot share its mistake.

#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/radial_shell.hpp>
#include <nukexc/spherical_harmonics.hpp>
#include <nukexc/stobasis.hpp>

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

/// A shell as plain numbers, so a test can write a basis down.
struct ShellSpec {
  int l = 0;
  Point center{};
  double r_cut = 0.0;
  std::vector<double> exponent; ///< zeta or alpha
  std::vector<double> coeff;
  std::vector<int> n;           ///< Slater only
};

ShellTable make_shell_table(const std::vector<ShellSpec> &shells) {
  const int nshell = static_cast<int>(shells.size());
  ShellTable t;
  t.nshell = nshell;
  t.l = Kokkos::View<int *>("l", nshell);
  t.first_bf = Kokkos::View<int *>("first bf", nshell + 1);
  t.prim_offset = Kokkos::View<int *>("prim offset", nshell + 1);
  t.center = Kokkos::View<Point *>("center", nshell);
  t.radius = Kokkos::View<double *>("radius", nshell);

  auto l_h = Kokkos::create_mirror_view(t.l);
  auto fbf_h = Kokkos::create_mirror_view(t.first_bf);
  auto po_h = Kokkos::create_mirror_view(t.prim_offset);
  auto c_h = Kokkos::create_mirror_view(t.center);
  auto r_h = Kokkos::create_mirror_view(t.radius);

  int bf = 0, prim = 0;
  for (int s = 0; s < nshell; ++s) {
    l_h(s) = shells[s].l;
    c_h(s) = shells[s].center;
    r_h(s) = shells[s].r_cut;
    fbf_h(s) = bf;
    po_h(s) = prim;
    bf += 2 * shells[s].l + 1;
    prim += static_cast<int>(shells[s].exponent.size());
  }
  fbf_h(nshell) = bf;
  po_h(nshell) = prim;
  t.nbf = bf;

  Kokkos::deep_copy(t.l, l_h);
  Kokkos::deep_copy(t.first_bf, fbf_h);
  Kokkos::deep_copy(t.prim_offset, po_h);
  Kokkos::deep_copy(t.center, c_h);
  Kokkos::deep_copy(t.radius, r_h);
  return t;
}

template <typename View> View upload(const std::vector<double> &v,
                                     const char *name) {
  View d(name, v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < v.size(); ++i)
    h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

std::vector<double> flat_exponents(const std::vector<ShellSpec> &shells) {
  std::vector<double> out;
  for (const auto &s : shells)
    out.insert(out.end(), s.exponent.begin(), s.exponent.end());
  return out;
}
std::vector<double> flat_coeffs(const std::vector<ShellSpec> &shells) {
  std::vector<double> out;
  for (const auto &s : shells)
    out.insert(out.end(), s.coeff.begin(), s.coeff.end());
  return out;
}

SlaterRadial make_slater(const std::vector<ShellSpec> &shells) {
  SlaterRadial rad;
  rad.sh = make_shell_table(shells);
  rad.zeta = upload<Kokkos::View<double *>>(flat_exponents(shells), "zeta");
  rad.coeff = upload<Kokkos::View<double *>>(flat_coeffs(shells), "coeff");
  std::vector<int> nn;
  for (const auto &s : shells)
    nn.insert(nn.end(), s.n.begin(), s.n.end());
  rad.n = Kokkos::View<int *>("n", nn.size());
  auto h = Kokkos::create_mirror_view(rad.n);
  for (size_t i = 0; i < nn.size(); ++i)
    h(i) = nn[i];
  Kokkos::deep_copy(rad.n, h);
  return rad;
}

GaussianRadial make_gaussian(const std::vector<ShellSpec> &shells) {
  GaussianRadial rad;
  rad.sh = make_shell_table(shells);
  rad.alpha = upload<Kokkos::View<double *>>(flat_exponents(shells), "alpha");
  rad.coeff = upload<Kokkos::View<double *>>(flat_coeffs(shells), "coeff");
  return rad;
}

/// Evaluate one shell through the device evaluator, on the host.
struct Block {
  std::vector<double> val, gx, gy, gz;
};

template <typename Evaluator>
Block eval_block_host(const Evaluator &ev, int shell, double x, double y,
                      double z) {
  constexpr int nf = Evaluator::max_block_size;
  Kokkos::View<double *> out("out", 4 * nf);
  auto e = ev;
  Kokkos::parallel_for(
      "eval one shell", Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        double buf[4 * nf];
        for (int i = 0; i < 4 * nf; ++i)
          buf[i] = 0.0;
        e.template eval_block<BasisDeriv::Gradient>(e.load_block(shell), x, y,
                                                    z, buf, nf);
        for (int i = 0; i < 4 * nf; ++i)
          out(i) = buf[i];
      });
  Kokkos::fence();
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);
  Block b;
  for (int i = 0; i < nf; ++i) {
    b.val.push_back(h(BF_VAL * nf + i));
    b.gx.push_back(h(BF_DX * nf + i));
    b.gy.push_back(h(BF_DY * nf + i));
    b.gz.push_back(h(BF_DZ * nf + i));
  }
  return b;
}

std::vector<double> matrix_to_host(const DeviceView2DLeft &m) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, m);
  const int n = static_cast<int>(m.extent(0));
  std::vector<double> out(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      out[static_cast<size_t>(i) * n + j] = h(i, j);
  return out;
}

} // namespace

TEST_CASE("The templated Slater path reproduces the Slater path",
          "[radial][sto]") {
  // Two centres, a 1s and a 2p each, with m ascending inside the p shell so
  // the per-function basis and the per-shell one number their functions
  // identically. compute_core_hamiltonian then compares element by element:
  // the overlap, the kinetic energy (which exercises the gradient) and the
  // nuclear attraction.
  const double R = 1.6, z1s = 1.3, z2p = 1.1;
  const std::vector<STOFunc> funcs = {
      {1, 0, 0, z1s, 0.0, 0.0, 0.0},   {2, 1, -1, z2p, 0.0, 0.0, 0.0},
      {2, 1, 0, z2p, 0.0, 0.0, 0.0},   {2, 1, 1, z2p, 0.0, 0.0, 0.0},
      {1, 0, 0, z1s, 0.0, 0.0, R},     {2, 1, -1, z2p, 0.0, 0.0, R},
      {2, 1, 0, z2p, 0.0, 0.0, R},     {2, 1, 1, z2p, 0.0, 0.0, R}};
  const STOBasisSet sto = make_manual_basis(funcs, 1e-12);

  // The same functions, as shells, with the Slater path's own
  // normalizations and cutoff radii so the two are the same basis rather
  // than proportional ones.
  auto norm_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sto.norm);
  auto cut_h =
      Kokkos::create_mirror_view_and_copy(HostSpace{}, sto.cutoff_radii);

  std::vector<ShellSpec> shells;
  for (int centre = 0; centre < 2; ++centre) {
    const int i0 = 4 * centre;
    const double oz = centre * R;
    shells.push_back(ShellSpec{0, Point{0.0, 0.0, oz}, cut_h(i0),
                               {z1s}, {norm_h(i0)}, {1}});
    shells.push_back(ShellSpec{1, Point{0.0, 0.0, oz}, cut_h(i0 + 1),
                               {z2p}, {norm_h(i0 + 1)}, {2}});
  }
  const SlaterShellEvaluator ev(make_slater(shells));
  REQUIRE(ev.nbf() == static_cast<int>(sto.nbf()));
  REQUIRE(ev.nblock() == 4);
  REQUIRE(ev.block_size(1) == 3);

  Molecule mol(std::vector<std::vector<double>>{{0.0, 0.0, 0.0},
                                                {0.0, 0.0, R}},
               std::vector<unsigned>{1u, 1u});
  const auto grid = make_flat_grid<ta_type, ll_type>(mol, 90, 29);

  const auto ref = compute_core_hamiltonian(STOPrimitiveEvaluator(sto), grid);
  const auto got = compute_core_hamiltonian(ev, grid);

  struct Pair {
    const char *name;
    DeviceView2DLeft a, b;
  };
  for (const auto &pr : {Pair{"overlap", got.overlap, ref.overlap},
                         Pair{"kinetic", got.kinetic, ref.kinetic},
                         Pair{"nuclear", got.nuclear, ref.nuclear}}) {
    const auto a = matrix_to_host(pr.a);
    const auto b = matrix_to_host(pr.b);
    double worst = 0.0, scale = 0.0;
    for (size_t k = 0; k < b.size(); ++k) {
      worst = std::max(worst, std::abs(a[k] - b[k]));
      scale = std::max(scale, std::abs(b[k]));
    }
    INFO(pr.name << ": worst |templated - per-function| = " << worst
                 << ", scale " << scale);
    REQUIRE(scale > 1e-3);
    // Same formula, same normalizations, same quadrature: this is
    // arithmetic reordering and nothing else.
    REQUIRE(worst < 1e-12 * scale);
  }
}

TEST_CASE("The templated Gaussian path matches R(r) S_lm written out",
          "[radial][gto]") {
  // The reference here shares no code with the evaluator: the test forms
  // sum_p c_p exp(-alpha_p r^2) itself and multiplies by the solid harmonic.
  // That is what pins the shared half of the template for a second family.
  const std::vector<ShellSpec> shells = {
      // a contracted d shell
      ShellSpec{2, Point{0.2, -0.1, 0.4}, 12.0,
                {3.1, 0.82, 0.21}, {0.17, 0.43, 0.29}, {}},
      // and a primitive p shell elsewhere
      ShellSpec{1, Point{-0.5, 0.3, 0.0}, 12.0, {0.57}, {0.73}, {}}};
  const GaussianShellEvaluator ev(make_gaussian(shells));
  REQUIRE(ev.nbf() == 5 + 3);
  REQUIRE(ev.block_size(0) == 5);
  REQUIRE(ev.block_size(1) == 3);
  REQUIRE(ev.block_first_bf(1) == 5);

  double worst = 0.0, peak = 0.0;
  for (int ip = 0; ip < 30; ++ip) {
    const double t = 0.37 * (ip + 1);
    const double x = 1.4 * std::cos(t), y = 1.1 * std::sin(1.3 * t),
                 z = 0.9 * std::cos(0.7 * t + 0.5);
    for (int s = 0; s < 2; ++s) {
      const auto b = eval_block_host(ev, s, x, y, z);
      const auto &sp = shells[s];
      const double dx = x - sp.center[0], dy = y - sp.center[1],
                   dz = z - sp.center[2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      double Rref = 0.0;
      for (size_t p = 0; p < sp.exponent.size(); ++p)
        Rref += sp.coeff[p] * std::exp(-sp.exponent[p] * r2);
      for (int i = 0; i < 2 * sp.l + 1; ++i) {
        double S = 0.0;
        real_solid_harmonic_cart_precomputed(sp.l, i - sp.l, dx, dy, dz, S);
        const double ref = Rref * S;
        worst = std::max(worst, std::abs(b.val[i] - ref));
        peak = std::max(peak, std::abs(ref));
      }
    }
  }
  INFO("worst |evaluator - written out| = " << worst << " of " << peak);
  REQUIRE(peak > 1e-3);
  REQUIRE(worst < 1e-14 * peak);
}

TEST_CASE("A contraction of one primitive is the primitive",
          "[radial][gto][sto]") {
  // The invariant that lets one policy serve pGTO and cGTO, and pSTO and
  // cSTO: there is no separate uncontracted code path to disagree with.
  const Point O{0.1, 0.2, -0.3};
  const std::vector<ShellSpec> one = {
      ShellSpec{1, O, 10.0, {0.9}, {1.3}, {2}}};
  const std::vector<ShellSpec> padded = {
      ShellSpec{1, O, 10.0, {0.9, 0.9}, {0.5, 0.8}, {2, 2}}};

  // 1.3 against 0.5 + 0.8: the same radial, reached as one term or two.
  for (int family = 0; family < 2; ++family) {
    INFO((family == 0 ? "Gaussian" : "Slater"));
    for (int ip = 0; ip < 10; ++ip) {
      const double t = 0.6 * (ip + 1);
      const double x = 1.2 * std::cos(t), y = 0.7 * std::sin(t), z = 0.4 * t;
      std::vector<double> a, b;
      if (family == 0) {
        a = eval_block_host(GaussianShellEvaluator(make_gaussian(one)), 0, x,
                            y, z)
                .val;
        b = eval_block_host(GaussianShellEvaluator(make_gaussian(padded)), 0,
                            x, y, z)
                .val;
      } else {
        a = eval_block_host(SlaterShellEvaluator(make_slater(one)), 0, x, y, z)
                .val;
        b = eval_block_host(SlaterShellEvaluator(make_slater(padded)), 0, x, y,
                            z)
                .val;
      }
      for (int i = 0; i < 3; ++i) {
        INFO("component " << i << ": " << a[i] << " against " << b[i]);
        REQUIRE_THAT(b[i], Catch::Matchers::WithinRel(a[i], 1e-14));
      }
    }
  }
}

TEST_CASE("Gradients match finite differences, for both families",
          "[radial][gradient]") {
  // The chain rule through dR/dr rhat S + R grad S, for a radial that rises
  // like r^(n-l-1) (Slater) and one that does not (Gaussian), at l = 2 where
  // the angular gradient is not trivial.
  const Point O{0.05, -0.15, 0.25};
  const std::vector<ShellSpec> gto = {
      ShellSpec{2, O, 12.0, {1.7, 0.4}, {0.6, 0.35}, {}}};
  const std::vector<ShellSpec> sto = {
      ShellSpec{2, O, 25.0, {1.15}, {0.9}, {3}}};

  const double h = 1e-5;
  for (int family = 0; family < 2; ++family) {
    INFO((family == 0 ? "Gaussian" : "Slater"));
    double worst = 0.0, peak = 0.0;
    for (int ip = 0; ip < 10; ++ip) {
      const double t = 0.55 * (ip + 1);
      const double p[3] = {1.0 * std::cos(t), 0.8 * std::sin(1.4 * t),
                           0.6 * std::cos(0.8 * t + 0.3)};
      auto value_at = [&](const double q[3], int comp) {
        if (family == 0)
          return eval_block_host(GaussianShellEvaluator(make_gaussian(gto)), 0,
                                 q[0], q[1], q[2])
              .val[comp];
        return eval_block_host(SlaterShellEvaluator(make_slater(sto)), 0, q[0],
                               q[1], q[2])
            .val[comp];
      };
      const auto base =
          family == 0
              ? eval_block_host(GaussianShellEvaluator(make_gaussian(gto)), 0,
                                p[0], p[1], p[2])
              : eval_block_host(SlaterShellEvaluator(make_slater(sto)), 0, p[0],
                                p[1], p[2]);
      for (int d = 0; d < 3; ++d) {
        double pp[3] = {p[0], p[1], p[2]}, pm[3] = {p[0], p[1], p[2]};
        pp[d] += h;
        pm[d] -= h;
        for (int i = 0; i < 5; ++i) {
          const double num = (value_at(pp, i) - value_at(pm, i)) / (2.0 * h);
          const double ana =
              (d == 0) ? base.gx[i] : (d == 1 ? base.gy[i] : base.gz[i]);
          worst = std::max(worst, std::abs(num - ana));
          peak = std::max(peak, std::abs(ana));
        }
      }
    }
    INFO("worst |analytic - numerical| = " << worst << " of " << peak);
    REQUIRE(peak > 1e-3);
    REQUIRE(worst < 1e-8 * peak);
  }
}

TEST_CASE("The cutoff is honoured exactly as a zero", "[radial]") {
  // Whatever the family, past r_cut the shell contributes nothing -- which
  // is what makes the neighbour list and the per-box packing mean the same
  // thing for every basis.
  const std::vector<ShellSpec> shells = {
      ShellSpec{1, Point{0.0, 0.0, 0.0}, 3.0, {0.5}, {1.0}, {}}};
  const GaussianShellEvaluator ev(make_gaussian(shells));

  const auto inside = eval_block_host(ev, 0, 2.99, 0.0, 0.0);
  REQUIRE(std::abs(inside.val[2]) > 0.0);

  for (double r : {3.0, 3.0001, 50.0}) {
    const auto out = eval_block_host(ev, 0, r, 0.0, 0.0);
    INFO("r = " << r);
    for (int i = 0; i < 3; ++i) {
      REQUIRE(out.val[i] == 0.0);
      REQUIRE(out.gx[i] == 0.0);
      REQUIRE(out.gy[i] == 0.0);
      REQUIRE(out.gz[i] == 0.0);
    }
  }
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int result = 0;
  {
    result = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return result;
}
