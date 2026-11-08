/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// ============================================================================
//  The auxiliary basis on the grid: project, solve, expand
// ============================================================================
//
// Density fitting puts the auxiliary basis on the grid twice, with a linear
// solve in between:
//
//     c_alpha = sum_g s(g) V_alpha(g)                 (project)
//     c      <- (A|B)^-1 c                            (solved by the caller)
//     v(g)    = sum_alpha c_alpha V_alpha(g)          (expand)
//
// Both are matrix-vector products against the same (naux, npts) potential
// matrix. That matrix is gigabytes on a real molecule and the solve sits
// between the two uses of it, so it is built one chunk of points at a time
// and consumed by a GEMV, rather than stored whole or reduced with atomics
// into naux accumulators. The evaluation count is one per (auxiliary shell,
// point) per pass, which is the minimum: a thread owns a shell and writes
// every function of it.
//
// WHY THIS IS NOT IN coulomb.hpp
// ------------------------------
// The Coulomb matrix and RI exact exchange both do exactly this, and the
// project/expand pair is also what a fitted XC or a fitted response
// kernel would need. One copy, in a header both can include.
//
// SHELLS, AGAIN
// -------------
// eval_aux_potential() in basis_evaluator.hpp indexes the auxiliary basis by
// FUNCTION and static_asserts that a block holds one of them, which is true
// of every Slater basis and false of every Gaussian one. Nothing here uses
// it: these loops walk blocks and call eval_potential_block, so a Gaussian
// auxiliary basis needs no new code here.

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>

#include "basis_evaluator.hpp"
#include "grid.hpp"
#include "nukexc_config.hpp"

namespace Nukexc {

/**
 *  @brief Points per chunk for the auxiliary potential matrix.
 *
 *  One chunk holds naux * ng doubles. The budget is per chunk, and a chunk
 *  is at least one point wide however large the auxiliary basis is.
 */
inline int aux_chunk_points(const int naux, const int npts,
                            const size_t budget_bytes = (size_t{64} << 20)) {
  if (naux <= 0 || npts <= 0)
    return 0;
  const size_t per_point = static_cast<size_t>(naux) * sizeof(double);
  const int ng = static_cast<int>(std::max<size_t>(1, budget_bytes / per_point));
  return std::min(ng, npts);
}

/**
 *  @brief V_alpha(g) for one chunk of points, as a dense (naux, ng) block.
 *
 *  One thread per (auxiliary shell, point); the shell is evaluated once and
 *  all of its functions are written.
 *
 *  The view's layout is the caller's choice, and the two callers want
 *  different ones: LayoutLeft so that a column range is contiguous and can
 *  be handed to a GEMV, LayoutRight so that the points of one auxiliary
 *  function are contiguous for a hand-rolled dot product.
 */
template <typename PotEvaluator, typename VauxView>
  requires PotentialEvaluator<PotEvaluator>
inline void fill_aux_potential_chunk(const PotEvaluator &aux,
                                     const FlatGrid &grid, const int g0,
                                     const int ng, const VauxView &Vaux) {
  if (ng <= 0)
    return;
  if (static_cast<int>(Vaux.extent(0)) != aux.nbf() ||
      static_cast<int>(Vaux.extent(1)) < ng)
    throw std::invalid_argument(
        "fill_aux_potential_chunk: output must be (naux, >= ng)");

  constexpr int MBS = PotEvaluator::max_block_size;
  const int nblk = aux.nblock();
  auto pts = grid.quad_points;

  Kokkos::parallel_for(
      "fill aux potential chunk",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nblk, ng}),
      KOKKOS_LAMBDA(const int b, const int g) {
        const Point p = pts(g0 + g);
        double local[MBS];
        aux.eval_potential_block(aux.load_block(b), p[0], p[1], p[2], local, 1);
        const int bf0 = aux.block_first_bf(b);
        const int n = aux.block_size(b);
        for (int t = 0; t < n; ++t)
          Vaux(bf0 + t, g) = local[t];
      });
}

/**
 *  @brief coeff_alpha = sum_g s(g) V_alpha(g) over the whole grid.
 *
 *  `s` carries whatever weight the projection needs -- for a Coulomb matrix
 *  that is w(g) rho(g). coeff is overwritten.
 */
template <typename PotEvaluator, typename SView>
  requires PotentialEvaluator<PotEvaluator>
inline void project_on_aux_potential(const PotEvaluator &aux,
                                     const FlatGrid &grid, const SView &s,
                                     const DeviceView1DLeft &coeff) {
  const int naux = aux.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  if (static_cast<int>(coeff.extent(0)) != naux)
    throw std::invalid_argument(
        "project_on_aux_potential: coeff must be naux long");
  if (static_cast<int>(s.extent(0)) != npts)
    throw std::invalid_argument(
        "project_on_aux_potential: s must cover the grid");

  Kokkos::deep_copy(coeff, 0.0);
  const int chunk = aux_chunk_points(naux, npts);
  if (chunk == 0)
    return;

  ExecSpace space;
  DeviceView2DLeft Vaux("aux potential chunk", naux, chunk);
  for (int g0 = 0; g0 < npts; g0 += chunk) {
    const int ng = std::min(chunk, npts - g0);
    fill_aux_potential_chunk(aux, grid, g0, ng, Vaux);
    space.fence();
    // beta = 1: the chunks accumulate into the same coefficients.
    KokkosBlas::gemv(
        space, "N", 1.0,
        Kokkos::subview(Vaux, Kokkos::ALL(), Kokkos::make_pair(0, ng)),
        Kokkos::subview(s, Kokkos::make_pair(g0, g0 + ng)), 1.0, coeff);
    space.fence();
  }
}

/**
 *  @brief v(g) = sum_alpha coeff_alpha V_alpha(g) over the whole grid.
 *
 *  v is overwritten. No quadrature weight is applied: the caller decides
 *  which factor of the integral carries it.
 */
template <typename PotEvaluator, typename VView>
  requires PotentialEvaluator<PotEvaluator>
inline void expand_aux_potential(const PotEvaluator &aux, const FlatGrid &grid,
                                 const DeviceView1DLeft &coeff,
                                 const VView &v) {
  const int naux = aux.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));
  if (static_cast<int>(coeff.extent(0)) != naux)
    throw std::invalid_argument(
        "expand_aux_potential: coeff must be naux long");
  if (static_cast<int>(v.extent(0)) != npts)
    throw std::invalid_argument("expand_aux_potential: v must cover the grid");

  Kokkos::deep_copy(v, 0.0);
  const int chunk = aux_chunk_points(naux, npts);
  if (chunk == 0)
    return;

  ExecSpace space;
  DeviceView2DLeft Vaux("aux potential chunk", naux, chunk);
  for (int g0 = 0; g0 < npts; g0 += chunk) {
    const int ng = std::min(chunk, npts - g0);
    fill_aux_potential_chunk(aux, grid, g0, ng, Vaux);
    space.fence();
    KokkosBlas::gemv(
        space, "T", 1.0,
        Kokkos::subview(Vaux, Kokkos::ALL(), Kokkos::make_pair(0, ng)), coeff,
        0.0, Kokkos::subview(v, Kokkos::make_pair(g0, g0 + ng)));
    space.fence();
  }
}

} // namespace Nukexc
