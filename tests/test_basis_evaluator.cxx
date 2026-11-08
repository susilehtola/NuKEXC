/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The basis-evaluator concept: one templated body, three basis types.
//
// The point of the concept is that downstream code is written once and
// instantiated per evaluator, with no run-time dispatch. So the tests here
// are themselves templated over the evaluator and run for all three:
// primitive STO, contracted STO, and GTO.

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <nukexc/basis_evaluator.hpp>
#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/sto_contracted_basis.hpp>
#include <nukexc/stobasis.hpp>

using namespace Nukexc;

namespace {

Kokkos::View<Point *> make_points(const std::vector<std::array<double, 3>> &p) {
  Kokkos::View<Point *> d("pts", p.size());
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < p.size(); ++i) {
    h(i)[0] = p[i][0];
    h(i)[1] = p[i][1];
    h(i)[2] = p[i][2];
  }
  Kokkos::deep_copy(d, h);
  return d;
}

const std::vector<std::array<double, 3>> kPoints = {
    {0.4, 0.15, -0.35}, {-1.2, 0.8, 0.25}, {0.11, -0.05, 0.62},
    {0.0, 0.0, 0.7},    {-0.3, -0.45, 0.2}};

/// Collocation through the generic path, returned on the host.
template <BasisDeriv D, typename E>
Kokkos::View<double ***, HostSpace> collocate(const E &ev,
                                              Kokkos::View<Point *> pts) {
  constexpr int NC = basis_ncomp(D);
  Kokkos::View<double ***> comp("comp", NC, ev.nbf(), pts.extent(0));
  fill_collocation<D>(ExecSpace{}, ev, pts, comp);
  Kokkos::fence();
  Kokkos::View<double ***, HostSpace> h("comp_h", NC, ev.nbf(),
                                        pts.extent(0));
  Kokkos::deep_copy(h, comp);
  return h;
}

// ---------------------------------------------------------------------------
//  The three evaluators under test
// ---------------------------------------------------------------------------

// COMPLETE shells, m = -l .. +l, which this fixture did not have before the
// Slater evaluator became shell-based: it carried two of the three 2p
// components and one of the five 3d. A list of independent (n, l, m)
// functions was a legitimate STOBasisSet for a per-function evaluator, and
// is not a basis a shell-based one can represent -- a shell is all 2l+1 of
// its components or it is not a shell. make_slater_radial says so with an
// exception rather than grouping whatever it finds, since the alternative is
// a silently permuted basis.
//
// Every reader in the project emits complete shells in this order, and an
// incomplete one is not rotationally invariant, so nothing real is lost.
STOPrimitiveEvaluator make_sto_primitive() {
  return STOPrimitiveEvaluator(make_manual_basis({
      {1, 0, 0, 1.4, 0.0, 0.0, 0.0},
      {2, 1, -1, 0.9, 0.0, 0.0, 0.0},
      {2, 1, 0, 0.9, 0.0, 0.0, 0.0},
      {2, 1, 1, 0.9, 0.0, 0.0, 0.0},
      {3, 2, -2, 1.1, -0.5, 0.3, 0.2},
      {3, 2, -1, 1.1, -0.5, 0.3, 0.2},
      {3, 2, 0, 1.1, -0.5, 0.3, 0.2},
      {3, 2, 1, 1.1, -0.5, 0.3, 0.2},
      {3, 2, 2, 1.1, -0.5, 0.3, 0.2},
  }));
}

// A complete p shell, for the reason the primitive fixture above gives: the
// three m of a shell share one radial, and a contracted basis stores that
// radial once per function, so each of them repeats the same primitive list.
STOContractedEvaluator make_sto_contracted() {
  const int l[4] = {0, 1, 1, 1};
  const int m[4] = {0, -1, 0, 1};
  const double cen[12] = {0.0, 0.0, 0.0, -0.5, 0.3, 0.2,
                          -0.5, 0.3, 0.2, -0.5, 0.3, 0.2};
  const int off[5] = {0, 3, 5, 7, 9};
  const int n[9] = {1, 2, 1, 2, 3, 2, 3, 2, 3};
  const double zeta[9] = {6.1, 2.4, 1.05, 1.9, 0.8, 1.9, 0.8, 1.9, 0.8};
  const double c[9] = {0.31, 0.52, 0.28, 0.66, 0.41,
                       0.66, 0.41, 0.66, 0.41};

  STOContractedBasisInput in;
  in.nbf = 4;
  in.l = l;
  in.m = m;
  in.centers = cen;
  in.prim_offset = off;
  in.n = n;
  in.zeta = zeta;
  in.coeff = c;
  return STOContractedEvaluator(make_sto_contracted_basis(in));
}

GTOEvaluator make_gto() {
  std::vector<GTOShell<double>> sh;
  sh.emplace_back(0, ShellType::Spherical, std::array<double, 3>{0.0, 0.0, 0.0},
                  std::vector<double>{5.4, 1.3, 0.42},
                  std::vector<double>{0.21, 0.55, 0.37, 0.64, -0.12, 0.28}, 2);
  sh.emplace_back(2, ShellType::Spherical,
                  std::array<double, 3>{-0.5, 0.3, 0.2},
                  std::vector<double>{2.7, 0.83},
                  std::vector<double>{0.4, 0.7}, 1);
  return GTOEvaluator(
      make_gto_device_basis(GTOBasisSet<double>(std::move(sh))));
}

// ---------------------------------------------------------------------------
//  Templated bodies -- written once, instantiated per evaluator
// ---------------------------------------------------------------------------

/// Blocks must tile the basis functions exactly once, in order.
template <typename E> void check_block_partition(const E &ev) {
  int expect = 0;
  for (int b = 0; b < ev.nblock(); ++b) {
    INFO("block " << b);
    REQUIRE(ev.block_first_bf(b) == expect);
    REQUIRE(ev.block_size(b) >= 1);
    REQUIRE(ev.block_size(b) <= E::max_block_size);
    expect += ev.block_size(b);
  }
  REQUIRE(expect == ev.nbf());

  // Screening geometry is per block and usable.
  REQUIRE(static_cast<int>(ev.centers().extent(0)) == ev.nblock());
  REQUIRE(static_cast<int>(ev.radii().extent(0)) == ev.nblock());
  auto r = Kokkos::create_mirror_view_and_copy(HostSpace{}, ev.radii());
  for (int b = 0; b < ev.nblock(); ++b) {
    INFO("block " << b);
    REQUIRE(r(b) > 0.0);
  }
}

/// The gradient an evaluator reports must be the gradient of the value it
/// reports -- checked entirely through the generic interface, so it holds for
/// any basis type satisfying the concept.
template <typename E> void check_gradient_consistency(const E &ev) {
  const double h = 1e-5;
  for (const auto &p : kPoints) {
    auto g = collocate<BasisDeriv::Gradient>(ev, make_points({p}));

    for (int d = 0; d < 3; ++d) {
      auto pp = p, pm = p;
      pp[d] += h;
      pm[d] -= h;
      auto vp = collocate<BasisDeriv::Value>(ev, make_points({pp}));
      auto vm = collocate<BasisDeriv::Value>(ev, make_points({pm}));

      for (int i = 0; i < ev.nbf(); ++i) {
        const double fd = (vp(BF_VAL, i, 0) - vm(BF_VAL, i, 0)) / (2 * h);
        const double an = g(BF_DX + d, i, 0);
        INFO("bf " << i << " d/dx" << d << " at (" << p[0] << "," << p[1]
                   << "," << p[2] << ")");
        REQUIRE_THAT(an, Catch::Matchers::WithinAbs(
                             fd, 1e-6 * std::max(1.0, std::abs(an))));
      }
    }
  }
}

/// The value returned alongside a gradient must equal the value-only result.
template <typename E> void check_value_agrees_across_orders(const E &ev) {
  auto pts = make_points(kPoints);
  auto v = collocate<BasisDeriv::Value>(ev, pts);
  auto g = collocate<BasisDeriv::Gradient>(ev, pts);
  for (int i = 0; i < ev.nbf(); ++i)
    for (size_t gp = 0; gp < kPoints.size(); ++gp) {
      INFO("bf " << i << " point " << gp);
      REQUIRE_THAT(g(BF_VAL, i, gp),
                   Catch::Matchers::WithinRel(v(BF_VAL, i, gp), 1e-14));
    }
}

} // namespace

TEST_CASE("Blocks tile the basis for every evaluator", "[concept]") {
  check_block_partition(make_sto_primitive());
  check_block_partition(make_sto_contracted());
  check_block_partition(make_gto());
}

TEST_CASE("Gradients are consistent for every evaluator", "[concept][fd]") {
  check_gradient_consistency(make_sto_primitive());
  check_gradient_consistency(make_sto_contracted());
  check_gradient_consistency(make_gto());
}

TEST_CASE("Value agrees across derivative orders for every evaluator",
          "[concept]") {
  check_value_agrees_across_orders(make_sto_primitive());
  check_value_agrees_across_orders(make_sto_contracted());
  check_value_agrees_across_orders(make_gto());
}

TEST_CASE("The generic path reproduces each basis's native evaluation",
          "[concept]") {
  auto pts = make_points(kPoints);

  SECTION("primitive STO vs fill_collocation on STOBasisSet") {
    const auto ev = make_sto_primitive();
    auto generic = collocate<BasisDeriv::Value>(ev, pts);

    DeviceView2DLeft native("native", ev.nbf(), pts.extent(0));
    ExecSpace space;
    fill_collocation(space, ev.basis(), pts, native);
    Kokkos::fence();
    auto nh = Kokkos::create_mirror_view_and_copy(HostSpace{}, native);

    for (int i = 0; i < ev.nbf(); ++i)
      for (size_t g = 0; g < kPoints.size(); ++g) {
        INFO("bf " << i << " point " << g);
        REQUIRE_THAT(generic(BF_VAL, i, g),
                     Catch::Matchers::WithinAbs(nh(i, g), 1e-14));
      }
  }

  SECTION("GTO vs the shell evaluator called directly") {
    const auto ev = make_gto();
    auto generic = collocate<BasisDeriv::Value>(ev, pts);

    // Same numbers, obtained by driving gto_eval_contraction per block.
    const auto &b = ev.basis();
    Kokkos::View<double **> direct("direct", ev.nbf(), pts.extent(0));
    const int nblk = ev.nblock();
    Kokkos::parallel_for(
        "direct", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                      {0, 0}, {nblk, (int)pts.extent(0)}),
        KOKKOS_LAMBDA(const int ib, const int g) {
          double loc[GTO_MAX_CART];
          gto_eval_contraction<BasisDeriv::Value>(
              b, b.block_shell(ib), b.block_ctr(ib), pts(g)[0], pts(g)[1],
              pts(g)[2], loc, GTO_MAX_CART);
          for (int i = 0; i < b.block_nfunc(ib); ++i)
            direct(b.block_first_bf(ib) + i, g) = loc[i];
        });
    Kokkos::fence();
    auto dh = Kokkos::create_mirror_view_and_copy(HostSpace{}, direct);

    for (int i = 0; i < ev.nbf(); ++i)
      for (size_t g = 0; g < kPoints.size(); ++g) {
        INFO("bf " << i << " point " << g);
        REQUIRE_THAT(generic(BF_VAL, i, g),
                     Catch::Matchers::WithinAbs(dh(i, g), 1e-14));
      }
  }
}

TEST_CASE("Collocation operands match libxckernel's shapes and packing",
          "[concept][libxckernel]") {
  // Phases 4-5 hand these arrays straight to generated kernels, which declare
  // them in their manifests as
  //     chi       (nbf, ng)
  //     dchi      (3, nbf, ng)
  //     hess_chi  (6, nbf, ng)  packed xx, xy, xz, yy, yz, zz
  // The (ncomp, nbf, ng) layout was chosen so these are contiguous slices
  // rather than repacked copies. That is only true if the component ordering
  // agrees, so check it rather than assert it in a comment.
  const auto ev = make_gto();
  auto pts = make_points(kPoints);
  const int nbf = ev.nbf(), ng = static_cast<int>(kPoints.size());

  constexpr int NC = basis_ncomp(BasisDeriv::Hessian);
  Kokkos::View<double ***> comp("comp", NC, nbf, ng);
  fill_collocation<BasisDeriv::Hessian>(ExecSpace{}, ev, pts, comp);
  Kokkos::fence();

  const auto op = collocation_operands<BasisDeriv::Hessian>(comp);
  REQUIRE(op.has_dchi);
  REQUIRE(op.has_hess_chi);

  REQUIRE(static_cast<int>(op.chi.extent(0)) == nbf);
  REQUIRE(static_cast<int>(op.chi.extent(1)) == ng);
  REQUIRE(static_cast<int>(op.dchi.extent(0)) == 3);
  REQUIRE(static_cast<int>(op.dchi.extent(1)) == nbf);
  REQUIRE(static_cast<int>(op.dchi.extent(2)) == ng);
  REQUIRE(static_cast<int>(op.hess_chi.extent(0)) == 6);
  REQUIRE(static_cast<int>(op.hess_chi.extent(1)) == nbf);
  REQUIRE(static_cast<int>(op.hess_chi.extent(2)) == ng);

  // The views must alias the same data, in the packing libxckernel expects.
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, comp);
  Kokkos::View<double **, HostSpace> chi_h("chi_h", nbf, ng);
  Kokkos::deep_copy(chi_h, op.chi);
  Kokkos::View<double ***, HostSpace> d_h("d_h", 3, nbf, ng), he_h("he_h", 6, nbf, ng);
  Kokkos::deep_copy(d_h, op.dchi);
  Kokkos::deep_copy(he_h, op.hess_chi);

  for (int u = 0; u < nbf; ++u)
    for (int g = 0; g < ng; ++g) {
      INFO("bf " << u << " point " << g);
      REQUIRE(chi_h(u, g) == h(BF_VAL, u, g));
      REQUIRE(d_h(0, u, g) == h(BF_DX, u, g));
      REQUIRE(d_h(1, u, g) == h(BF_DY, u, g));
      REQUIRE(d_h(2, u, g) == h(BF_DZ, u, g));
      // xx, xy, xz, yy, yz, zz
      REQUIRE(he_h(0, u, g) == h(BF_DXX, u, g));
      REQUIRE(he_h(1, u, g) == h(BF_DXY, u, g));
      REQUIRE(he_h(2, u, g) == h(BF_DXZ, u, g));
      REQUIRE(he_h(3, u, g) == h(BF_DYY, u, g));
      REQUIRE(he_h(4, u, g) == h(BF_DYZ, u, g));
      REQUIRE(he_h(5, u, g) == h(BF_DZZ, u, g));
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
