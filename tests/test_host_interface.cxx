/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The host-facing entry points: density matrices in, Exc / Nel / Vxc out.
//
// The load-bearing check is that the density-matrix path agrees with the MO
// path it is meant to replace. Those are different algorithms -- one GEMM
// against P and a column dot product, versus a GEMM against the occupied
// coefficients and a sum over orbitals -- so agreement to round-off is real
// evidence, not a tautology.

#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include <xc.h>
#include <xc_funcs.h>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_eval.hpp>
#include <nukexc/host_interface.hpp>
#include <nukexc/molecule.hpp>

#include "sto_basis_reader.hpp"

using namespace Nukexc;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

namespace {

/// A small water STO basis and grid, shared by the cases below.
struct Fixture {
  Molecule mol;
  STOBasisSet basis;
  FlatGrid grid;

  Fixture() {
    read_xyz("input/water.xyz", mol);
    basis = test::load_adf_basis(mol, "input/zorabasis/TZP", 1e-10);
    grid = make_flat_grid<ta_type, ll_type>(mol, 60, 23);
  }
};

/// Deterministic S-orthonormal occupied coefficients, and the density matrix
/// P = sum_i occ_i c_i c_i^T built from them.
struct Orbitals {
  std::vector<double> C;   // nbf x nocc, column-major
  std::vector<double> P;   // nbf x nbf, column-major
  std::vector<double> occ; // nocc
  int nbf = 0, nocc = 0;
};

Orbitals make_orbitals(const std::vector<double> &S, int nbf, int nocc,
                       double occupancy) {
  Orbitals o;
  o.nbf = nbf;
  o.nocc = nocc;
  o.C.assign(static_cast<size_t>(nbf) * nocc, 0.0);
  o.occ.assign(nocc, occupancy);

  auto col = [&](int k) { return o.C.data() + static_cast<size_t>(k) * nbf; };
  auto Sdot = [&](const double *a, const double *b) {
    double acc = 0.0;
    for (int i = 0; i < nbf; ++i)
      for (int j = 0; j < nbf; ++j)
        acc += a[i] * S[static_cast<size_t>(i) * nbf + j] * b[j];
    return acc;
  };

  for (int k = 0; k < nocc; ++k) {
    double *c = col(k);
    for (int i = 0; i < nbf; ++i)
      c[i] = std::sin(0.61 * (k + 1) * (i + 1)) + 0.23 * std::cos(0.17 * i);
    for (int m = 0; m < k; ++m) {
      const double proj = Sdot(col(m), c);
      for (int i = 0; i < nbf; ++i)
        c[i] -= proj * col(m)[i];
    }
    const double nrm = std::sqrt(Sdot(c, c));
    for (int i = 0; i < nbf; ++i)
      c[i] /= nrm;
  }

  o.P.assign(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int k = 0; k < nocc; ++k)
    for (int i = 0; i < nbf; ++i)
      for (int j = 0; j < nbf; ++j)
        o.P[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf] +=
            occupancy * col(k)[i] * col(k)[j];
  return o;
}

/// Overlap by quadrature, used only to orthonormalize the test orbitals.
std::vector<double> quadrature_overlap(const STOPrimitiveEvaluator &ev,
                                       const FlatGrid &grid) {
  const int n = ev.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  DeviceView2DLeft coll("coll", n, npts);
  fill_collocation_values(ExecSpace{}, ev, grid.quad_points, coll);
  Kokkos::View<double **> S("S", n, n);
  auto w = grid.weights;
  Kokkos::parallel_for(
      "S", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, n}),
      KOKKOS_LAMBDA(const int i, const int j) {
        double acc = 0.0;
        for (int g = 0; g < npts; ++g)
          acc += w(g) * coll(i, g) * coll(j, g);
        S(i, j) = acc;
      });
  Kokkos::fence();
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, S);
  std::vector<double> out(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      out[static_cast<size_t>(i) * n + j] = h(i, j);
  return out;
}

} // namespace

TEST_CASE("compute_nel recovers the electron count", "[host][density]") {
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();

  const auto S = quadrature_overlap(ev, f.grid);
  const auto orb = make_orbitals(S, nbf, 5, 2.0);

  HostMatrixView P{orb.P.data(), nbf, MatrixLayout::ColumnMajor};
  const double nel = compute_nel(ev, f.grid, P);

  // Tr(P S) = sum_i occ_i = 10 by construction.
  INFO("integrated " << nel);
  REQUIRE_THAT(nel, Catch::Matchers::WithinAbs(10.0, 1e-8));
}

TEST_CASE("Matrix layout is honoured on upload", "[host][layout]") {
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const auto S = quadrature_overlap(ev, f.grid);
  const auto orb = make_orbitals(S, nbf, 4, 2.0);

  // Note what CANNOT serve as a check here. Every entry point currently
  // exposed consumes a symmetric density matrix, and Tr(A S) = Tr(A^T S)
  // whenever S is symmetric -- so no integrated quantity can distinguish the
  // two layouts, however the flag is implemented. The flag earns its keep on
  // the non-symmetric matrices of later phases (perturbed densities, complex
  // orbitals), so test it where it actually acts: on the upload.
  std::vector<double> A(static_cast<size_t>(nbf) * nbf);
  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j < nbf; ++j)
      A[static_cast<size_t>(i) * nbf + j] = 100.0 * i + j; // deliberately asymmetric

  const auto col = detail::upload_matrix(
      HostMatrixView{A.data(), nbf, MatrixLayout::ColumnMajor}, nbf, "A");
  const auto row = detail::upload_matrix(
      HostMatrixView{A.data(), nbf, MatrixLayout::RowMajor}, nbf, "A");
  auto ch = Kokkos::create_mirror_view_and_copy(HostSpace{}, col);
  auto rh = Kokkos::create_mirror_view_and_copy(HostSpace{}, row);

  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j < nbf; ++j) {
      INFO("element (" << i << "," << j << ")");
      // row-major reads A[i*ld+j]; column-major reads A[i+j*ld], which for
      // this buffer is the transpose.
      REQUIRE(rh(i, j) == A[static_cast<size_t>(i) * nbf + j]);
      REQUIRE(ch(i, j) == A[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf]);
      REQUIRE(ch(i, j) == rh(j, i));
    }

  // And the positive statement that follows from the symmetry: for a real
  // density matrix the two layouts must give the same answer.
  HostMatrixView Pc{orb.P.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixView Pr{orb.P.data(), nbf, MatrixLayout::RowMajor};
  REQUIRE_THAT(compute_nel(ev, f.grid, Pc),
               Catch::Matchers::WithinRel(compute_nel(ev, f.grid, Pr), 1e-14));
}

TEST_CASE("Malformed host input is rejected", "[host]") {
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  std::vector<double> P(static_cast<size_t>(nbf) * nbf, 0.0);

  HostMatrixView nullp{nullptr, nbf, MatrixLayout::ColumnMajor};
  REQUIRE_THROWS_AS(compute_nel(ev, f.grid, nullp), std::invalid_argument);

  HostMatrixView shortld{P.data(), nbf - 1, MatrixLayout::ColumnMajor};
  REQUIRE_THROWS_AS(compute_nel(ev, f.grid, shortld), std::invalid_argument);
}

TEST_CASE("The density-matrix path agrees with the MO path",
          "[host][density][parity]") {
  // Different algorithms: X = P Phi then a column dot product, versus
  // Psi = C^T Phi then a sum over orbitals. Agreement to round-off is
  // evidence, not a tautology.
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const int npts = static_cast<int>(f.grid.quad_points.extent(0));
  const auto S = quadrature_overlap(ev, f.grid);
  const auto orb = make_orbitals(S, nbf, 5, 2.0);

  DeviceView2DLeft coll("coll", nbf, npts);
  fill_collocation_values(ExecSpace{}, ev, f.grid.quad_points, coll);

  DeviceView2DLeft C("C", nbf, orb.nocc);
  auto C_h = Kokkos::create_mirror_view(C);
  for (int i = 0; i < nbf; ++i)
    for (int k = 0; k < orb.nocc; ++k)
      C_h(i, k) = orb.C[static_cast<size_t>(k) * nbf + i];
  Kokkos::deep_copy(C, C_h);

  DeviceView1D occ("occ", orb.nocc);
  auto occ_h = Kokkos::create_mirror_view(occ);
  for (int k = 0; k < orb.nocc; ++k)
    occ_h(k) = orb.occ[k];
  Kokkos::deep_copy(occ, occ_h);

  DeviceView2DLeft P("P", nbf, nbf);
  auto P_h = Kokkos::create_mirror_view(P);
  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j < nbf; ++j)
      P_h(i, j) = orb.P[static_cast<size_t>(i) + static_cast<size_t>(j) * nbf];
  Kokkos::deep_copy(P, P_h);

  const auto rho_mo = compute_density(coll, C, occ);
  const auto rho_dm = compute_density_dm(coll, P);
  Kokkos::fence();

  auto a = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_mo);
  auto b = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_dm);

  double worst = 0.0, peak = 0.0;
  for (int g = 0; g < npts; ++g) {
    worst = std::max(worst, std::abs(a(g) - b(g)));
    peak = std::max(peak, std::abs(a(g)));
  }
  INFO("peak density " << peak << ", worst difference " << worst);
  REQUIRE(worst < 1e-10 * std::max(1.0, peak));
}

TEST_CASE("eval_vxc returns a symmetric potential and a sane energy",
          "[host][xc]") {
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const auto S = quadrature_overlap(ev, f.grid);
  const auto orb = make_orbitals(S, nbf, 5, 2.0);

  xc_func_type func;
  REQUIRE(xc_func_init(&func, XC_LDA_X, XC_UNPOLARIZED) == 0);

  std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixView P{orb.P.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixRef Vxc{V.data(), nbf, MatrixLayout::ColumnMajor};

  const XCOutput out = eval_vxc(ev, f.grid, P, Vxc, func);

  INFO("Exc = " << out.Exc << ", Nel = " << out.Nel
                << ", non-finite points = " << out.nan_points);
  REQUIRE_THAT(out.Nel, Catch::Matchers::WithinAbs(10.0, 1e-8));
  REQUIRE(out.Exc < 0.0); // Slater exchange is negative
  REQUIRE(std::isfinite(out.Exc));
  REQUIRE(out.nan_points == 0);

  // Vxc must come back symmetric.
  double asym = 0.0;
  for (int i = 0; i < nbf; ++i)
    for (int j = 0; j < nbf; ++j)
      asym = std::max(asym, std::abs(V[static_cast<size_t>(i) + j * nbf] -
                                     V[static_cast<size_t>(j) + i * nbf]));
  REQUIRE(asym < 1e-12);

  // Vxc accumulates rather than overwrites, as ERKALE's H does.
  std::vector<double> V2(V);
  HostMatrixRef Vxc2{V2.data(), nbf, MatrixLayout::ColumnMajor};
  eval_vxc(ev, f.grid, P, Vxc2, func);
  for (int i = 0; i < nbf * nbf; ++i) {
    INFO("element " << i);
    REQUIRE_THAT(V2[i], Catch::Matchers::WithinRel(2.0 * V[i], 1e-10) ||
                            Catch::Matchers::WithinAbs(0.0, 1e-14));
  }

  xc_func_end(&func);
}

TEST_CASE("eval_vxc rejects a mismatched functional", "[host][xc]") {
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  std::vector<double> P(static_cast<size_t>(nbf) * nbf, 0.0);
  std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixRef Vr{V.data(), nbf, MatrixLayout::ColumnMajor};

  xc_func_type pol;
  REQUIRE(xc_func_init(&pol, XC_LDA_X, XC_POLARIZED) == 0);
  REQUIRE_THROWS_AS(eval_vxc(ev, f.grid, Pv, Vr, pol), std::runtime_error);
  xc_func_end(&pol);
}

TEST_CASE("Unrestricted with equal spins reproduces the restricted result",
          "[host][xc][polarized]") {
  // The sharpest consistency check available: Pa = Pb = P/2 is the
  // spin-restricted limit, so a POLARIZED functional evaluated there must
  // return exactly what the UNPOLARIZED one returns for P -- same Exc, same
  // Nel, and Vxc_a = Vxc_b = Vxc. The two go through entirely different code:
  // interleaved (a,b) arrays and an opposite-spin sigma_ab term versus the
  // single-channel path.
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const auto S = quadrature_overlap(ev, f.grid);
  const auto orb = make_orbitals(S, nbf, 5, 2.0);

  std::vector<double> Phalf(orb.P);
  for (double &v : Phalf)
    v *= 0.5;

  const int ids[2] = {XC_LDA_X, XC_GGA_X_PBE};
  for (int id : ids) {
    INFO("functional id " << id);

    xc_func_type fr, fp;
    REQUIRE(xc_func_init(&fr, id, XC_UNPOLARIZED) == 0);
    REQUIRE(xc_func_init(&fp, id, XC_POLARIZED) == 0);

    std::vector<double> Vr(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<double> Va(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<double> Vb(static_cast<size_t>(nbf) * nbf, 0.0);

    HostMatrixView P{orb.P.data(), nbf, MatrixLayout::ColumnMajor};
    HostMatrixView Ph{Phalf.data(), nbf, MatrixLayout::ColumnMajor};
    HostMatrixRef Rr{Vr.data(), nbf, MatrixLayout::ColumnMajor};
    HostMatrixRef Ra{Va.data(), nbf, MatrixLayout::ColumnMajor};
    HostMatrixRef Rb{Vb.data(), nbf, MatrixLayout::ColumnMajor};

    const XCOutput r = eval_vxc(ev, f.grid, P, Rr, fr);
    const XCOutputPolarized u = eval_vxc(ev, f.grid, Ph, Ph, Ra, Rb, fp);

    INFO("restricted Exc " << r.Exc << " vs unrestricted " << u.Exc);
    REQUIRE_THAT(u.Exc, Catch::Matchers::WithinRel(r.Exc, 1e-11));
    REQUIRE_THAT(u.Nel, Catch::Matchers::WithinRel(r.Nel, 1e-12));
    REQUIRE_THAT(u.Nel_alpha, Catch::Matchers::WithinRel(0.5 * r.Nel, 1e-12));
    REQUIRE_THAT(u.Nel_beta, Catch::Matchers::WithinRel(0.5 * r.Nel, 1e-12));
    REQUIRE(u.nan_points == 0);

    double worst_ab = 0.0, worst_r = 0.0, scale = 0.0;
    for (size_t k = 0; k < Vr.size(); ++k) {
      worst_ab = std::max(worst_ab, std::abs(Va[k] - Vb[k]));
      worst_r = std::max(worst_r, std::abs(Va[k] - Vr[k]));
      scale = std::max(scale, std::abs(Vr[k]));
    }
    INFO("|Va-Vb| " << worst_ab << ", |Va-Vr| " << worst_r
                    << ", scale " << scale);
    REQUIRE(worst_ab < 1e-12 * std::max(1.0, scale));
    REQUIRE(worst_r < 1e-10 * std::max(1.0, scale));

    xc_func_end(&fr);
    xc_func_end(&fp);
  }
}

TEST_CASE("eval_diag_vxc matches the diagonal of the full matrix",
          "[host][xc][diag]") {
  // The adaptive grid only ever looks at max_u |Delta F_uu|, so the diagonal
  // path skips the GEMM. It must still agree with the matrix it is standing
  // in for.
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const auto S = quadrature_overlap(ev, f.grid);
  const auto orb = make_orbitals(S, nbf, 5, 2.0);

  const int ids[2] = {XC_LDA_X, XC_GGA_X_PBE};
  for (int id : ids) {
    INFO("functional id " << id);
    xc_func_type func;
    REQUIRE(xc_func_init(&func, id, XC_UNPOLARIZED) == 0);

    std::vector<double> V(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<double> d(nbf, 0.0);
    HostMatrixView P{orb.P.data(), nbf, MatrixLayout::ColumnMajor};
    HostMatrixRef R{V.data(), nbf, MatrixLayout::ColumnMajor};

    const XCOutput full = eval_vxc(ev, f.grid, P, R, func);
    const XCOutput diag = eval_diag_vxc(ev, f.grid, P, d.data(), func);

    REQUIRE_THAT(diag.Exc, Catch::Matchers::WithinRel(full.Exc, 1e-12));
    REQUIRE_THAT(diag.Nel, Catch::Matchers::WithinRel(full.Nel, 1e-12));

    double worst = 0.0, scale = 0.0;
    for (int i = 0; i < nbf; ++i) {
      const double ref = V[static_cast<size_t>(i) + static_cast<size_t>(i) * nbf];
      worst = std::max(worst, std::abs(d[i] - ref));
      scale = std::max(scale, std::abs(ref));
    }
    INFO("worst diagonal difference " << worst << ", scale " << scale);
    REQUIRE(worst < 1e-11 * std::max(1.0, scale));

    // accumulates, like the matrix form
    std::vector<double> d2(d);
    eval_diag_vxc(ev, f.grid, P, d2.data(), func);
    for (int i = 0; i < nbf; ++i)
      REQUIRE_THAT(d2[i], Catch::Matchers::WithinRel(2.0 * d[i], 1e-10) ||
                              Catch::Matchers::WithinAbs(0.0, 1e-14));

    xc_func_end(&func);
  }
}

TEST_CASE("Unrestricted rejects a mismatched functional", "[host][xc]") {
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  std::vector<double> P(static_cast<size_t>(nbf) * nbf, 0.0),
      Va(static_cast<size_t>(nbf) * nbf, 0.0),
      Vb(static_cast<size_t>(nbf) * nbf, 0.0);
  HostMatrixView Pv{P.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixRef Ra{Va.data(), nbf, MatrixLayout::ColumnMajor};
  HostMatrixRef Rb{Vb.data(), nbf, MatrixLayout::ColumnMajor};

  xc_func_type unpol;
  REQUIRE(xc_func_init(&unpol, XC_LDA_X, XC_UNPOLARIZED) == 0);
  REQUIRE_THROWS_AS(eval_vxc(ev, f.grid, Pv, Pv, Ra, Rb, unpol),
                    std::runtime_error);
  xc_func_end(&unpol);
}

TEST_CASE("A complex orbital with zero imaginary part matches the real one",
          "[host][pzsic][complex]") {
  // The split Re/Im path must reduce to the real one when Im C = 0, and a
  // global phase must leave everything invariant: rho_i = |psi_i|^2 cannot
  // depend on it. The phase check is the one that would catch a Re/Im mixup,
  // since it moves weight between the two blocks while fixing the answer.
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const auto S = quadrature_overlap(ev, f.grid);
  const auto orb = make_orbitals(S, nbf, 3, 1.0);

  xc_func_type func;
  REQUIRE(xc_func_init(&func, XC_LDA_X, XC_POLARIZED) == 0);

  auto run = [&](const double *re, const double *im) {
    OrbitalsView ov;
    ov.C_re = re;
    ov.C_im = im;
    ov.ld = nbf;
    ov.layout = MatrixLayout::ColumnMajor;
    ov.norb = 3;
    std::vector<std::vector<double>> V(
        3, std::vector<double>(static_cast<size_t>(nbf) * nbf, 0.0));
    std::vector<HostMatrixRef> refs;
    for (auto &v : V)
      refs.push_back(HostMatrixRef{v.data(), nbf, MatrixLayout::ColumnMajor});
    auto out = eval_vxc_orbitals(ev, f.grid, ov, refs, func);
    return std::make_pair(out, V);
  };

  // (a) purely real
  const auto [real_out, real_V] = run(orb.C.data(), nullptr);
  for (int i2 = 0; i2 < 3; ++i2) {
    INFO("orbital " << i2);
    REQUIRE_THAT(real_out.Nel[i2], Catch::Matchers::WithinAbs(1.0, 1e-8));
    REQUIRE(real_out.Exc[i2] < 0.0);
  }
  REQUIRE(real_out.nan_points == 0);

  // (b) same orbitals rotated by a global phase exp(i theta): C -> C cos,
  //     C sin. |psi|^2 is unchanged, so every result must be.
  const double th = 0.7;
  std::vector<double> Cr(orb.C), Ci(orb.C);
  for (auto &v : Cr)
    v *= std::cos(th);
  for (auto &v : Ci)
    v *= std::sin(th);
  const auto [phase_out, phase_V] = run(Cr.data(), Ci.data());

  for (int i2 = 0; i2 < 3; ++i2) {
    INFO("orbital " << i2);
    REQUIRE_THAT(phase_out.Nel[i2],
                 Catch::Matchers::WithinRel(real_out.Nel[i2], 1e-12));
    REQUIRE_THAT(phase_out.Exc[i2],
                 Catch::Matchers::WithinRel(real_out.Exc[i2], 1e-12));
    double worst = 0.0, scale = 0.0;
    for (size_t k = 0; k < real_V[i2].size(); ++k) {
      worst = std::max(worst, std::abs(phase_V[i2][k] - real_V[i2][k]));
      scale = std::max(scale, std::abs(real_V[i2][k]));
    }
    REQUIRE(worst < 1e-11 * std::max(1.0, scale));
  }

  xc_func_end(&func);
}

TEST_CASE("Per-orbital results match single-orbital evaluations",
          "[host][pzsic]") {
  // Evaluating three orbitals together must give exactly what evaluating
  // each alone gives -- the shared collocation is an optimization, not a
  // change of answer.
  Fixture f;
  const STOPrimitiveEvaluator ev(f.basis);
  const int nbf = ev.nbf();
  const auto S = quadrature_overlap(ev, f.grid);
  const auto orb = make_orbitals(S, nbf, 3, 1.0);

  xc_func_type func;
  REQUIRE(xc_func_init(&func, XC_LDA_X, XC_POLARIZED) == 0);

  OrbitalsView all;
  all.C_re = orb.C.data();
  all.ld = nbf;
  all.norb = 3;
  std::vector<std::vector<double>> V(
      3, std::vector<double>(static_cast<size_t>(nbf) * nbf, 0.0));
  std::vector<HostMatrixRef> refs;
  for (auto &v : V)
    refs.push_back(HostMatrixRef{v.data(), nbf, MatrixLayout::ColumnMajor});
  const auto together = eval_vxc_orbitals(ev, f.grid, all, refs, func);

  for (int k = 0; k < 3; ++k) {
    OrbitalsView one;
    one.C_re = orb.C.data() + static_cast<size_t>(k) * nbf;
    one.ld = nbf;
    one.norb = 1;
    std::vector<double> Vk(static_cast<size_t>(nbf) * nbf, 0.0);
    std::vector<HostMatrixRef> r1{
        HostMatrixRef{Vk.data(), nbf, MatrixLayout::ColumnMajor}};
    const auto alone = eval_vxc_orbitals(ev, f.grid, one, r1, func);

    INFO("orbital " << k);
    REQUIRE_THAT(alone.Exc[0],
                 Catch::Matchers::WithinRel(together.Exc[k], 1e-13));
    REQUIRE_THAT(alone.Nel[0],
                 Catch::Matchers::WithinRel(together.Nel[k], 1e-13));
    for (size_t m = 0; m < Vk.size(); ++m)
      REQUIRE_THAT(Vk[m], Catch::Matchers::WithinRel(V[k][m], 1e-12) ||
                              Catch::Matchers::WithinAbs(0.0, 1e-14));
  }

  // Energies-only mode must agree and skip the Fock builds.
  const auto energies = eval_vxc_orbitals(ev, f.grid, all, {}, func);
  for (int k = 0; k < 3; ++k)
    REQUIRE_THAT(energies.Exc[k],
                 Catch::Matchers::WithinRel(together.Exc[k], 1e-13));

  xc_func_end(&func);
}

TEST_CASE("The density path selector picks sensibly", "[host][density]") {
  // Auto should take the orbital route only when the occupied count is a
  // clear minority of the basis.
  REQUIRE(resolve_density_path(DensityPath::Auto, 5, 100) ==
          DensityPath::Orbital);
  REQUIRE(resolve_density_path(DensityPath::Auto, 60, 100) ==
          DensityPath::DensityMatrix);
  // Explicit choices are honoured.
  REQUIRE(resolve_density_path(DensityPath::Orbital, 60, 100) ==
          DensityPath::Orbital);
  REQUIRE(resolve_density_path(DensityPath::DensityMatrix, 5, 100) ==
          DensityPath::DensityMatrix);
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
