/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// Localized occupied orbitals, for screening the exchange build.
//
// WHY THIS EXISTS
// ---------------
// The RI exchange matrix depends on the occupied orbitals ONLY through the
// density matrix. Writing W = (A|B)^-1,
//
//     K_uv = sum_i occ_i sum_AB (u i|A) W_AB (B|v i)
//          = sum_AB W_AB sum_ls (u l|A) [sum_i occ_i C_li C_si] (s v|B)
//          = sum_AB W_AB sum_ls (u l|A) P_ls (s v|B),
//
// so ANY rank factorization P = L L^T may be substituted for the occupied
// coefficients with unit occupations and the same K comes out. This is not a
// unitary-invariance argument -- it does not require the occupations to be
// equal -- it is just that P is the only thing K sees.
//
// That freedom matters because the screened exchange build
// (compute_exact_exchange_screened) walks boxes inside a loop over occupied
// orbitals, and a box whose orbital amplitude vanishes costs nothing if it
// is skipped. CANONICAL orbitals make that screening worthless: the
// occupied eigenvectors of a Fock matrix are delocalized over the whole
// molecule -- in a long chain they are the standing waves of a band -- so
// every orbital is alive in every box. The density matrix, in contrast,
// decays with the distance between the two functions for any system with a
// gap. A factorization of P inherits that decay; the eigenvectors do not.
//
// WHY PIVOTED CHOLESKY
// --------------------
// The k-th Cholesky vector is the pivot column of the residual density
// matrix, normalized. Its support is therefore the support of a column of P,
// which is the decay length of the density matrix around one function, and
// the pivoting takes the largest remaining diagonal first so the vectors
// come out ordered from the most to the least significant. Aquilante,
// Pedersen and Lindh, J. Chem. Phys. 125 (2006) 174101 named these the
// Cholesky orbitals and showed they are competitive with the iterative
// localizations.
//
// Boys and Pipek-Mezey would localize somewhat better, but both are
// iterative and need integrals this file would otherwise not want -- dipole
// moments for Boys, the overlap and an atomic partition for Pipek-Mezey.
// Pivoted Cholesky needs P and nothing else, is non-iterative, and costs
// nbf^2 * rank, which is nothing next to one exchange build.
//
// HOW WELL IT WORKS, AND WHAT THAT BUYS
// -------------------------------------
// Measured on aligned alkanes in DZ, as a spread in the basis-function
// index -- which for a chain whose functions are numbered along it is a
// faithful proxy for a distance (benchmark_exchange --screen-only):
//
//     chain      C4H10   C8H18  C12H26  C16H34
//     canonical   16.0    30.6    45.9    61.3
//     Cholesky     5.9     6.4     6.9     7.0
//     delocalized 17.3    33.5    49.7    65.8
//
// The canonical set is delocalized to within a few per cent of the limit at
// every length, and the Cholesky factors are SIZE-INDEPENDENT. This part
// works exactly as advertised.
//
// It does not follow that the exchange build gets cheaper in proportion. A
// second moment of seven functions says nothing about the tails, and a
// box-amplitude screen lives or dies on the tails; see the measured table
// in exact_exchange.hpp, which is a fifth of the three-centre work at
// C16H34 rather than the change of scaling one might expect from the
// numbers above. The localization is necessary for any of it and is not
// sufficient for much of it.
//
// A BONUS: THE RANK
// -----------------
// The loop stops when the largest remaining diagonal falls below the
// tolerance, so a density matrix of numerical rank below nocc -- fractional
// occupations, a smeared Fermi level, a linearly dependent basis -- yields
// FEWER vectors than there are orbitals, and the exchange build pays for
// the rank rather than for the orbital count.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "nukexc_config.hpp"

namespace Nukexc {

/// Which occupied factorization an exchange build should use.
///
/// Selectable rather than substituted: the two give the same K, and which
/// one is cheaper depends on how localized the density matrix is and on
/// whether the caller screens at all.
enum class OccupiedFactorization {
  AsGiven,          ///< the orbitals the caller passed, untouched
  CholeskyLocalized ///< pivoted Cholesky vectors of P = C occ C^T
};

/// An occupied factorization in the form the exchange builds take.
struct OccupiedFactors {
  DeviceView2DLeft C; ///< (nbf, rank)
  DeviceView1D occ;   ///< (rank), all ones for a Cholesky factorization
  int rank = 0;
};

/**
 *  @brief The density matrix of an occupied set, P = C diag(occ) C^T.
 *
 *  Occupations must be non-negative; P is then positive semidefinite, which
 *  is what the Cholesky factorization below needs.
 */
inline DeviceView2DLeft occupied_density_matrix(const DeviceView2DLeft &C,
                                                const DeviceView1D &occ) {
  const int n = static_cast<int>(C.extent(0));
  const int nocc = static_cast<int>(C.extent(1));
  if (static_cast<int>(occ.extent(0)) < nocc)
    throw std::invalid_argument(
        "occupied_density_matrix: fewer occupations than orbitals");

  // Scale by sqrt(occ) so the product is a plain symmetric GEMM.
  DeviceView2DLeft Cs("C sqrt(occ)", n, nocc);
  Kokkos::parallel_for(
      "Scale occupied orbitals by sqrt(occ)",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, nocc}),
      KOKKOS_LAMBDA(const int mu, const int i) {
        const double o = occ(i);
        Cs(mu, i) = C(mu, i) * (o > 0.0 ? Kokkos::sqrt(o) : 0.0);
      });
  Kokkos::fence();

  // A negative occupation would have been silently zeroed above, which would
  // give the wrong P rather than an error, so check.
  const auto occ_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, occ);
  for (int i = 0; i < nocc; ++i)
    if (occ_h(i) < 0.0)
      throw std::invalid_argument(
          "occupied_density_matrix: negative occupation number");

  DeviceView2DLeft P("density matrix", n, n);
  KokkosBlas::gemm("N", "T", 1.0, Cs, Cs, 0.0, P);
  Kokkos::fence();
  return P;
}

/**
 *  @brief Pivoted Cholesky factorization P = L L^T, L of rank columns.
 *
 *  @param P    (n, n) symmetric positive semidefinite
 *  @param tol  stop when the largest remaining diagonal falls below
 *              tol times the largest initial diagonal
 *  @param max_rank  cap on the number of vectors, or zero for n
 *
 *  Host-side and serial: the cost is n^2 * rank, which is a rounding error
 *  next to the exchange build this feeds, and the pivot search is a
 *  sequential dependence that a device kernel would serialize anyway.
 */
inline DeviceView2DLeft pivoted_cholesky(const DeviceView2DLeft &P,
                                         const double tol = 1e-12,
                                         const int max_rank = 0) {
  const int n = static_cast<int>(P.extent(0));
  if (static_cast<int>(P.extent(1)) != n)
    throw std::invalid_argument("pivoted_cholesky: P must be square");
  if (!(tol > 0.0))
    throw std::invalid_argument("pivoted_cholesky: tol must be positive");
  const int cap = max_rank > 0 ? std::min(max_rank, n) : n;

  const auto Ph = Kokkos::create_mirror_view_and_copy(HostSpace{}, P);

  std::vector<double> L(static_cast<size_t>(n) * cap, 0.0); // column-major
  std::vector<double> d(n);
  double d0 = 0.0;
  for (int mu = 0; mu < n; ++mu) {
    d[mu] = Ph(mu, mu);
    d0 = std::max(d0, d[mu]);
  }

  int rank = 0;
  if (d0 > 0.0) {
    const double stop = tol * d0;
    for (int k = 0; k < cap; ++k) {
      int p = 0;
      for (int mu = 1; mu < n; ++mu)
        if (d[mu] > d[p])
          p = mu;
      if (!(d[p] > stop))
        break;

      const double piv = std::sqrt(d[p]);
      double *Lk = L.data() + static_cast<size_t>(k) * n;
      for (int mu = 0; mu < n; ++mu) {
        double s = Ph(mu, p);
        for (int j = 0; j < k; ++j)
          s -= L[static_cast<size_t>(j) * n + mu] *
               L[static_cast<size_t>(j) * n + p];
        Lk[mu] = s / piv;
      }
      // The updated diagonal is a difference of positive quantities and
      // drifts negative near the rank; clamping keeps the pivot search from
      // chasing round-off.
      for (int mu = 0; mu < n; ++mu)
        d[mu] = std::max(0.0, d[mu] - Lk[mu] * Lk[mu]);
      ++rank;
    }
  }

  DeviceView2DLeft Ld("Cholesky vectors", n, rank);
  auto Lh = Kokkos::create_mirror_view(Ld);
  for (int k = 0; k < rank; ++k)
    for (int mu = 0; mu < n; ++mu)
      Lh(mu, k) = L[static_cast<size_t>(k) * n + mu];
  Kokkos::deep_copy(Ld, Lh);
  return Ld;
}

/**
 *  @brief Cholesky-localized occupied factors of C diag(occ) C^T.
 *
 *  The returned factors reproduce the same density matrix, hence the same
 *  exchange matrix, with unit occupations. The rank is at most the orbital
 *  count and may be lower; see the header comment.
 */
inline OccupiedFactors cholesky_localized_occupied(const DeviceView2DLeft &C,
                                                   const DeviceView1D &occ,
                                                   const double tol = 1e-12) {
  const int nocc = static_cast<int>(C.extent(1));
  OccupiedFactors f;
  f.C = pivoted_cholesky(occupied_density_matrix(C, occ), tol, nocc);
  f.rank = static_cast<int>(f.C.extent(1));
  f.occ = DeviceView1D("unit occupations", f.rank);
  Kokkos::deep_copy(f.occ, 1.0);
  return f;
}

/**
 *  @brief Either factorization, by name.
 */
inline OccupiedFactors occupied_factors(const OccupiedFactorization how,
                                        const DeviceView2DLeft &C,
                                        const DeviceView1D &occ,
                                        const double tol = 1e-12) {
  if (how == OccupiedFactorization::CholeskyLocalized)
    return cholesky_localized_occupied(C, occ, tol);
  OccupiedFactors f;
  f.C = C;
  f.occ = occ;
  f.rank = static_cast<int>(C.extent(1));
  return f;
}

} // namespace Nukexc
