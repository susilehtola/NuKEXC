/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// Eigendecomposition of a real symmetric 3x3 matrix.
//
// Small enough that a library call would cost more in setup than the solve,
// and needed where no linear-algebra backend is available: the grid batching
// in octree.hpp takes the principal axis of a point cloud's scatter matrix to
// decide which way to cut it. Kept in its own header rather than hidden in a
// detail namespace there, because a 3x3 symmetric solve is the kind of thing a
// second caller wants (inertia tensors, Hessian eigenvalues, anisotropic
// screening radii) and the project's DRY policy says the shared piece lives in
// a common header from the start.
//
// Cyclic Jacobi: unconditionally convergent for symmetric input, and at this
// size the iteration count is a handful of sweeps. The analytic closed form
// via the characteristic cubic is faster but loses accuracy badly for nearly
// degenerate eigenvalues, which is exactly the case a spherically symmetric
// point cloud presents.

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace Nukexc {

/// Eigenvalues and eigenvectors of a symmetric 3x3 matrix.
struct Sym3Eigen {
  /// Eigenvalues, ascending.
  std::array<double, 3> values;
  /// Eigenvectors as columns: vectors[i] is the eigenvector for values[i].
  std::array<std::array<double, 3>, 3> vectors;
};

/**
 *  @brief Eigendecomposition of a real symmetric 3x3 matrix by cyclic Jacobi.
 *
 *  Only the upper triangle of @p A is read; the matrix is assumed symmetric
 *  rather than symmetrized, so an asymmetric argument is a caller error that
 *  this function will not detect.
 *
 *  @param A the matrix
 *  @return eigenvalues ascending, with the matching unit eigenvectors
 *  @throws std::runtime_error if the sweeps do not converge, which for
 *          symmetric input means the matrix held a NaN or an infinity
 */
inline Sym3Eigen sym3_eigen(const double A[3][3]) {
  // Working copy of the upper triangle, and the accumulated rotation.
  double a[3][3];
  double v[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      a[i][j] = A[i][j];
      v[i][j] = (i == j) ? 1.0 : 0.0;
    }

  // The iteration stops only when the off-diagonal is EXACTLY zero, and
  // every pass either annihilates an element by rotating or sets it to zero
  // because it is negligible against the diagonal it sits between. Both
  // branches drive the off-diagonal sum monotonically to zero in finitely
  // many steps, so the loop cannot stall.
  //
  // It is tempting instead to stop when the off-diagonal sum falls below a
  // threshold and to skip elements individually below that same threshold.
  // That does stall: three elements at 0.4 of the threshold each sum to 1.2
  // of it, so the convergence test fails while every rotation is skipped,
  // and the sweep count runs out on a perfectly ordinary matrix. An earlier
  // draft of this function did exactly that, and it surfaced not here but as
  // an intermittent failure in an unrelated integration test, since whether
  // any particular point batch lands in that band depends on which points
  // share a batch. tests/test_sym3x3.cxx pins the case.
  constexpr int MAX_SWEEPS = 50;
  int sweep = 0;
  for (; sweep < MAX_SWEEPS; ++sweep) {
    const double off =
        std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
    if (off == 0.0)
      break;

    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        if (a[p][q] == 0.0)
          continue;

        // Negligible against both diagonal entries, in the sense that
        // adding it changes neither: rotating would be a no-op up to
        // roundoff, so zero it and move on. Deferred for the first few
        // sweeps so that the large elements are cleared first.
        const double g = 100.0 * std::abs(a[p][q]);
        if (sweep > 3 && std::abs(a[p][p]) + g == std::abs(a[p][p]) &&
            std::abs(a[q][q]) + g == std::abs(a[q][q])) {
          a[p][q] = 0.0;
          a[q][p] = 0.0;
          continue;
        }

        // Rotation annihilating a[p][q]. theta = cot(2 phi); the smaller
        // root t = tan(phi) is taken for numerical stability.
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t = (theta >= 0.0)
                             ? 1.0 / (theta + std::sqrt(1.0 + theta * theta))
                             : -1.0 / (-theta + std::sqrt(1.0 + theta * theta));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        const double app = a[p][p], aqq = a[q][q], apq = a[p][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        // The remaining index, rotated against both p and q.
        const int r = 3 - p - q;
        const double arp = (r < p) ? a[r][p] : a[p][r];
        const double arq = (r < q) ? a[r][q] : a[q][r];
        const double np = c * arp - s * arq;
        const double nq = s * arp + c * arq;
        if (r < p)
          a[r][p] = np, a[p][r] = np;
        else
          a[p][r] = np, a[r][p] = np;
        if (r < q)
          a[r][q] = nq, a[q][r] = nq;
        else
          a[q][r] = nq, a[r][q] = nq;

        for (int k = 0; k < 3; ++k) {
          const double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
  }
  if (sweep == MAX_SWEEPS)
    throw std::runtime_error(
        "sym3_eigen: Jacobi failed to converge in " +
        std::to_string(MAX_SWEEPS) +
        " sweeps. Cyclic Jacobi on a finite symmetric 3x3 converges in a "
        "handful, so this means the matrix holds a NaN: every comparison "
        "against one is false, so neither the rotation nor the negligibility "
        "branch can ever retire an element.");

  // Sort ascending by eigenvalue, carrying the columns along.
  int order[3] = {0, 1, 2};
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (a[order[j]][order[j]] < a[order[i]][order[i]])
        std::swap(order[i], order[j]);

  Sym3Eigen out;
  for (int i = 0; i < 3; ++i) {
    const int c = order[i];
    out.values[i] = a[c][c];
    out.vectors[i] = {v[0][c], v[1][c], v[2][c]};
  }
  return out;
}

/**
 *  @brief Unit eigenvector of the largest eigenvalue of a symmetric 3x3.
 *
 *  For a scatter matrix S = sum_i (p_i - c) (p_i - c)^T this is the direction
 *  n maximizing sum_i (n . (p_i - c))^2, i.e. the normal of the plane through
 *  the centroid from which the points are furthest -- the cut direction of
 *  Blum et al., Comput. Phys. Commun. 180 (2009) 2175, step (1).
 *
 *  A matrix with no spread at all (every point coincident) has no preferred
 *  direction; x is returned, since any axis is equally good and refusing
 *  would make the caller handle a case with no consequences.
 *
 *  @param A the matrix
 *  @return a unit vector
 */
inline std::array<double, 3> principal_axis(const double A[3][3]) {
  const auto e = sym3_eigen(A);
  const auto &w = e.vectors[2]; // largest eigenvalue sorts last
  const double n = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  if (!(n > 0.0) || !(e.values[2] > 0.0))
    return {1.0, 0.0, 0.0};
  return {w[0] / n, w[1] / n, w[2] / n};
}

} // namespace Nukexc
