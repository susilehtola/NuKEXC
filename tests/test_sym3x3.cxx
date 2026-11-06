/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

// The symmetric 3x3 eigensolver behind the cut-plane grid batching.
//
// Worth testing on its own rather than only through the batching, because a
// wrong principal axis does not produce a wrong answer anywhere -- it
// produces a slightly worse box partition, which no integration test would
// ever notice. The failure mode is silent inefficiency, so the solver has to
// be pinned directly.

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_all.hpp>

#include <nukexc/sym3x3.hpp>

using namespace Nukexc;

namespace {

/// V diag(w) V^T, to check the decomposition reproduces its input.
std::array<std::array<double, 3>, 3> reconstruct(const Sym3Eigen &e) {
  std::array<std::array<double, 3>, 3> out{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k)
        s += e.values[k] * e.vectors[k][i] * e.vectors[k][j];
      out[i][j] = s;
    }
  return out;
}

double dot3(const std::array<double, 3> &a, const std::array<double, 3> &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

} // namespace

TEST_CASE("A diagonal matrix comes back with its diagonal sorted",
          "[sym3x3]") {
  const double A[3][3] = {{3.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 7.0}};
  const auto e = sym3_eigen(A);
  REQUIRE_THAT(e.values[0], Catch::Matchers::WithinAbs(-1.0, 1e-14));
  REQUIRE_THAT(e.values[1], Catch::Matchers::WithinAbs(3.0, 1e-14));
  REQUIRE_THAT(e.values[2], Catch::Matchers::WithinAbs(7.0, 1e-14));

  // The eigenvectors must be the coordinate axes, up to sign.
  REQUIRE_THAT(std::abs(e.vectors[0][1]), Catch::Matchers::WithinAbs(1.0, 1e-14));
  REQUIRE_THAT(std::abs(e.vectors[1][0]), Catch::Matchers::WithinAbs(1.0, 1e-14));
  REQUIRE_THAT(std::abs(e.vectors[2][2]), Catch::Matchers::WithinAbs(1.0, 1e-14));

  const auto ax = principal_axis(A);
  REQUIRE_THAT(std::abs(ax[2]), Catch::Matchers::WithinAbs(1.0, 1e-14));
}

TEST_CASE("A matrix with a degenerate pair still gives the right top axis",
          "[sym3x3]") {
  // [[4,1,1],[1,4,1],[1,1,4]] has eigenvalues 3, 3, 6; the 6 belongs to
  // (1,1,1)/sqrt(3). The degenerate pair is the case the analytic closed
  // form handles badly, which is why this solver is iterative.
  const double A[3][3] = {{4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
  const auto e = sym3_eigen(A);
  REQUIRE_THAT(e.values[0], Catch::Matchers::WithinAbs(3.0, 1e-13));
  REQUIRE_THAT(e.values[1], Catch::Matchers::WithinAbs(3.0, 1e-13));
  REQUIRE_THAT(e.values[2], Catch::Matchers::WithinAbs(6.0, 1e-13));

  const auto ax = principal_axis(A);
  const double s = 1.0 / std::sqrt(3.0);
  // Up to an overall sign.
  const double sign = (ax[0] > 0.0) ? 1.0 : -1.0;
  for (int d = 0; d < 3; ++d)
    REQUIRE_THAT(sign * ax[d], Catch::Matchers::WithinAbs(s, 1e-13));
}

TEST_CASE("Off-diagonals just below a relative threshold still converge",
          "[sym3x3][regression]") {
  // Regression test for a stall. An earlier version of this solver stopped
  // when |a01| + |a02| + |a12| fell below 1e-15 times the matrix scale, and
  // skipped any individual element already below that same threshold. Three
  // elements at 0.4 of the threshold sum to 1.2 of it: the convergence test
  // failed, every rotation was skipped, and the sweep limit ran out on a
  // matrix with nothing wrong with it. It threw a "Jacobi failed to
  // converge" error claiming the input must hold a NaN.
  //
  // Scatter matrices of grid-point batches land here routinely, because the
  // diagonal is the batch's spread -- large -- while the off-diagonal is
  // near zero for a batch that happens to be nearly axis aligned. The bug
  // surfaced as a non-reproducible failure in an unrelated test, since
  // which points share a batch is not fixed run to run.
  for (double diag : {1.0, 1e5, 1e10, 1e20}) {
    // The old threshold was 1e-15 times the LARGEST entry, here 3*diag.
    // Each element must sit just under that while the three of them sum to
    // more than it, which is the whole of the stall.
    const double eps = 0.9 * 1e-15 * (3.0 * diag);
    const double A[3][3] = {{diag, eps, eps},
                            {eps, 2.0 * diag, eps},
                            {eps, eps, 3.0 * diag}};
    INFO("diagonal scale " << diag);
    const auto e = sym3_eigen(A); // must not throw
    REQUIRE(e.values[0] <= e.values[1]);
    REQUIRE(e.values[1] <= e.values[2]);
    REQUIRE_THAT(e.values[0], Catch::Matchers::WithinRel(diag, 1e-12));
    REQUIRE_THAT(e.values[2], Catch::Matchers::WithinRel(3.0 * diag, 1e-12));
    for (int i = 0; i < 3; ++i)
      REQUIRE_THAT(dot3(e.vectors[i], e.vectors[i]),
                   Catch::Matchers::WithinAbs(1.0, 1e-13));
  }
}

TEST_CASE("Ill-conditioned matrices converge too", "[sym3x3]") {
  // Diagonal entries spanning many orders of magnitude, with off-diagonals
  // at every relative size from unity down to underflow. The uniformly
  // random sweep below does not reach these, which is why the stall above
  // survived it.
  std::uint64_t state = 0x9E3779B97F4A7C15ull;
  const auto next = [&state]() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<double>(static_cast<std::int64_t>(state >> 11)) /
               static_cast<double>(1ull << 52) -
           1.0;
  };

  for (int trial = 0; trial < 300; ++trial) {
    double d[3];
    for (int i = 0; i < 3; ++i)
      d[i] = std::pow(10.0, 12.0 * next());
    const double rel = std::pow(10.0, -20.0 * std::abs(next()));

    double A[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j) {
        A[i][j] = (i == j) ? d[i] : rel * std::sqrt(d[i] * d[j]) * next();
        A[j][i] = A[i][j];
      }

    INFO("trial " << trial << ", relative off-diagonal " << rel);
    const auto e = sym3_eigen(A); // must not throw
    REQUIRE(e.values[0] <= e.values[1]);
    REQUIRE(e.values[1] <= e.values[2]);
    for (int i = 0; i < 3; ++i)
      REQUIRE_THAT(dot3(e.vectors[i], e.vectors[i]),
                   Catch::Matchers::WithinAbs(1.0, 1e-13));
    // The trace is invariant under an orthogonal transformation, and is a
    // check the eigenvalues cannot pass by accident.
    REQUIRE_THAT(e.values[0] + e.values[1] + e.values[2],
                 Catch::Matchers::WithinRel(d[0] + d[1] + d[2], 1e-12));
  }
}

TEST_CASE("The decomposition is orthonormal and reproduces its input",
          "[sym3x3]") {
  // A deterministic sweep of matrices rather than one hand-picked case: the
  // generator is a fixed 64-bit LCG so a failure is always reproducible.
  std::uint64_t state = 0x2545F4914F6CDD1Dull;
  const auto next = [&state]() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<double>(static_cast<std::int64_t>(state >> 11)) /
               static_cast<double>(1ull << 52) -
           1.0;
  };

  for (int trial = 0; trial < 500; ++trial) {
    // Scales spanning twenty orders of magnitude, since the convergence
    // threshold is relative and a fixed one would pass only near unity.
    const double scale = std::pow(10.0, 10.0 * next());
    double A[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j) {
        A[i][j] = scale * next();
        A[j][i] = A[i][j];
      }

    const auto e = sym3_eigen(A);

    double norm = 0.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        norm = std::max(norm, std::abs(A[i][j]));

    // Orthonormality.
    for (int i = 0; i < 3; ++i) {
      INFO("trial " << trial << " vector " << i);
      REQUIRE_THAT(dot3(e.vectors[i], e.vectors[i]),
                   Catch::Matchers::WithinAbs(1.0, 1e-13));
      for (int j = i + 1; j < 3; ++j)
        REQUIRE_THAT(dot3(e.vectors[i], e.vectors[j]),
                     Catch::Matchers::WithinAbs(0.0, 1e-13));
    }

    // Ascending eigenvalues.
    REQUIRE(e.values[0] <= e.values[1]);
    REQUIRE(e.values[1] <= e.values[2]);

    // And V diag(w) V^T == A, relative to the matrix scale.
    const auto R = reconstruct(e);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        INFO("trial " << trial << " entry " << i << "," << j << " scale "
                      << norm);
        REQUIRE_THAT(R[i][j] - A[i][j],
                     Catch::Matchers::WithinAbs(0.0, 1e-12 * norm));
      }
  }
}

TEST_CASE("The principal axis of a point cloud is the direction it is "
          "stretched along",
          "[sym3x3]") {
  // This is the use the batching makes of it: build the scatter matrix of a
  // cloud elongated along (1,1,1) and recover that direction.
  const double s = 1.0 / std::sqrt(3.0);
  std::vector<std::array<double, 3>> pts;
  for (int i = -40; i <= 40; ++i) {
    const double t = 0.5 * i; // long axis
    for (int j = -1; j <= 1; ++j)
      // a thin spread across it, along (1,-1,0)/sqrt(2)
      pts.push_back({t * s + 0.02 * j / std::sqrt(2.0),
                     t * s - 0.02 * j / std::sqrt(2.0), t * s});
  }

  double c[3] = {0.0, 0.0, 0.0};
  for (const auto &p : pts)
    for (int d = 0; d < 3; ++d)
      c[d] += p[d] / pts.size();

  double S[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  for (const auto &p : pts)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        S[a][b] += (p[a] - c[a]) * (p[b] - c[b]);

  const auto ax = principal_axis(S);
  const double sign = (ax[0] > 0.0) ? 1.0 : -1.0;
  for (int d = 0; d < 3; ++d)
    REQUIRE_THAT(sign * ax[d], Catch::Matchers::WithinAbs(s, 1e-10));
}

TEST_CASE("A cloud with no spread has no preferred direction", "[sym3x3]") {
  // Every point coincident: the scatter matrix is zero. Any axis is as good
  // as any other, so the solver must return a usable unit vector rather
  // than a NaN, because the batching will project onto whatever it gets.
  const double Z[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  const auto ax = principal_axis(Z);
  REQUIRE_THAT(dot3(ax, ax), Catch::Matchers::WithinAbs(1.0, 1e-14));
}

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
