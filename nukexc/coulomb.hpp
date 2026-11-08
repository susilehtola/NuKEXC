/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Bob Schreiner
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

#pragma once

#include "density.hpp"
#include "grid.hpp"
#include "nukexc/octree.hpp"
#include "aux_potential.hpp"
#include "basis_evaluator.hpp"
#include "nukexc_config.hpp"
#include "packed_blocks.hpp"
#include "stobasis.hpp"
#include "stopotential.hpp"

#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_gemv_impl.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_gemm_impl.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <impl/Kokkos_Profiling.hpp>

namespace Nukexc {

DeviceView2DLeft
compute_coulomb(const ExecSpace space, const DeviceView2DLeft mo_orbitals,
                const DeviceView1D mo_coeff,
                const DeviceView2DLeft basis_collocation,
                const DeviceView2DLeft basis_aux_collocation,
                const DeviceView2DLeft potential_collocation_scaled,
                const DeviceView2DLeft half_inverse_X) {

  Kokkos::Profiling::pushRegion("Compute Coulomb Integral");
  const int N_bf = basis_collocation.extent(0);
  const int N_bf_aux = basis_aux_collocation.extent(0);
  const int N_quad = basis_collocation.extent(1);
  const int K = half_inverse_X.extent(1);

  DeviceView2DLeft basis_collocation_scaled("Basis collocation Scaled", N_bf,
                                            N_quad);
  DeviceView1DLeft expansion_coeff("Expansion coeff", N_bf_aux);
  DeviceView1DLeft potential_on_grid("Expansion coeff scaled", N_quad);
  DeviceView2DLeft result("Coulomb matrix", N_bf, N_bf);
  DeviceView1DLeft density =
      compute_density(basis_collocation, mo_orbitals, mo_coeff);

  DeviceView1DLeft scaling_factor("Scaling factor", K);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Compute expansion coefficients
  KokkosBlas::gemv(space, "N", 1.0, potential_collocation_scaled, density, 0.0,
                   expansion_coeff);

  // Apply (A|B)^{-1}
  KokkosBlas::gemv(space, "T", 1.0, half_inverse_X, expansion_coeff, 0.0,
                   scaling_factor);
  KokkosBlas::gemv(space, "N", 1.0, half_inverse_X, scaling_factor, 0.0,
                   expansion_coeff);

  // Compute potential on grid
  KokkosBlas::gemv(space, "T", 1.0, potential_collocation_scaled,
                   expansion_coeff, 0.0, potential_on_grid);

  // Compute (mu nu | A)
  Kokkos::parallel_for(
      "Scale basis by potential", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double pot_g = potential_on_grid(g);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
              basis_collocation_scaled(i, g) = basis_collocation(i, g) * pot_g;
            });
      });

  KokkosBlas::gemm(space, "N", "T", 1.0, basis_collocation,
                   basis_collocation_scaled, 0.0, result);

  Kokkos::fence();
  Kokkos::Profiling::popRegion();
  return result;
}

// ============================================================================
//  Screened density-fitted Coulomb: pack -> GEMM -> scatter
// ============================================================================
//
// The matrix is
//
//     J_uv = sum_g w_g v_J(g) chi_u(g) chi_v(g),
//     v_J(g) = sum_alpha c_alpha V_alpha(g),
//     c = (A|B)^-1 (sum_g w_g rho(g) V_alpha(g)),
//
// so there are three grid passes and each one is a matrix product: the
// density through the density matrix, the two auxiliary passes through
// aux_potential.hpp's chunked GEMVs, and the matrix itself through the
// per-box collocation blocks of packed_blocks.hpp.
//
// This is the memory-frugal route, not the fast one: the dense path stores
// the collocation and the auxiliary potential and is about twice as quick
// per build wherever those tables fit. packed_blocks.hpp has the numbers.
//
// WHAT THIS REPLACES
// ------------------
// compute_coulomb_sparse and compute_coulomb_tiled, which between them had
// three problems. They gave a thread one pair of neighbour-list entries and
// evaluated both entries at every point, so the cost was
// O(nbe^2 * npts) evaluations where O(nshell * npts) suffices -- the tiled
// variant staged those evaluations through a scratch cache, which bounded
// the memory but not the evaluation count. They indexed the neighbour list
// as though it held basis FUNCTIONS, which it does not: it holds shells, so
// both were restricted to a basis with one function per shell and could not
// even be instantiated for a Gaussian one. And the two differed only in that
// staging, so every fix had to be made twice.
//
// The auxiliary basis is no longer function-indexed either, so a Gaussian
// fitting basis needs no new code here -- only a Gaussian potential
// evaluator, which is a separate matter.

template <typename Evaluator, typename PotEvaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value> &&
           PotentialEvaluator<PotEvaluator>
inline DeviceView2DLeft compute_coulomb_screened(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const Evaluator basis,
    const PotEvaluator basis_aux, const FlatGrid grid, const NeighborList nl,
    const DeviceView2DLeft half_inverse_X) {

  Kokkos::Profiling::pushRegion("Compute Coulomb Integral Screened");
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = static_cast<int>(grid.quad_points.extent(0));
  const int K = static_cast<int>(half_inverse_X.extent(1));

  const PackedBlockLayout L = build_packed_block_layout(basis, nl);

  // rho on the grid, and the projection's integrand w rho.
  DeviceView1D rho("density", N_quad);
  packed_density<BasisDeriv::Value>(
      basis, grid, nl, L,
      density_matrix_from_orbitals(mo_orbitals, mo_coeff), rho);

  DeviceView1DLeft w_rho("w rho", N_quad);
  auto w = grid.weights;
  Kokkos::parallel_for(
      "Weight the density", Kokkos::RangePolicy<ExecSpace>(0, N_quad),
      KOKKOS_LAMBDA(const int g) { w_rho(g) = w(g) * rho(g); });
  space.fence();

  // c_alpha = sum_g w_g rho(g) V_alpha(g), then (A|B)^-1 c.
  DeviceView1DLeft expansion_coeff("Expansion coeff", N_bf_aux);
  project_on_aux_potential(basis_aux, grid, w_rho, expansion_coeff);

  DeviceView1DLeft scaling_factor("Scaling factor", K);
  KokkosBlas::gemv(space, "T", 1.0, half_inverse_X, expansion_coeff, 0.0,
                   scaling_factor);
  KokkosBlas::gemv(space, "N", 1.0, half_inverse_X, scaling_factor, 0.0,
                   expansion_coeff);
  space.fence();

  // v_J on the grid, then the matrix.
  DeviceView1DLeft potential_on_grid("Coulomb potential on grid", N_quad);
  expand_aux_potential(basis_aux, grid, expansion_coeff, potential_on_grid);

  DeviceView2DLeft result("Coulomb matrix", N_bf, N_bf);
  accumulate_collocation_product<BasisDeriv::Value>(
      basis, grid, nl, L, WeightedCollocation{w, potential_on_grid}, result);

  space.fence();
  Kokkos::Profiling::popRegion();
  return result;
}

// ============================================================================
//  The fitting metric (A|B)
// ============================================================================
//
//     (A|B) = sum_g w_g V_A(g) chi_B(g)
//
// which is one GEMM of the auxiliary potential against the auxiliary
// collocation, taken a chunk of points at a time so neither matrix is ever
// stored whole.
//
// WHAT THIS REPLACES, AND WHY IT IS NOT SCREENED
// ----------------------------------------------
// coulomb_overlap_integral_sparse evaluated chi_B once per (auxiliary
// function, shell, point) -- the potential loop was outside the value loop,
// so every chi_B was recomputed for all naux values of A. On water in a TZ2P
// fitting basis that one matrix took 18.0 s of an 18.4 s setup. Its tiled
// twin staged those evaluations through a scratch cache, which again bounded
// the memory and not the count.
//
// Those two screened the VALUE index against the box's neighbour list while
// summing the potential over every auxiliary function, which is the only
// consistent choice there: the Coulomb potential has a 1/r tail and cannot
// be screened by a cutoff radius. The metric is therefore an inherently
// dense (naux, naux) object, it is built once per geometry, and the dense
// path has always built it this way -- so this does not screen either, and
// the two paths now produce the same matrix rather than two slightly
// different ones.

template <typename PotEvaluator>
  requires PotentialEvaluator<PotEvaluator> &&
           BasisEvaluatorWith<PotEvaluator, BasisDeriv::Value>
inline DeviceView2DLeft coulomb_overlap_metric(const ExecSpace space,
                                               const PotEvaluator &basis_aux,
                                               const FlatGrid &grid) {
  Kokkos::Profiling::pushRegion("Coulomb overlap metric");
  const int naux = basis_aux.nbf();
  const int npts = static_cast<int>(grid.quad_points.extent(0));

  DeviceView2DLeft M("Coulomb overlap matrix", naux, naux);
  DeviceView2DLeft sym("Coulomb overlap matrix symmetrized", naux, naux);

  // Two (naux, chunk) buffers are live at once, so the budget is shared.
  const int chunk = aux_chunk_points(2 * naux, npts);
  if (chunk > 0) {
    DeviceView2DLeft Vaux("aux potential chunk", naux, chunk);
    DeviceView2DLeft Phi("aux collocation chunk", naux, chunk);
    auto w = grid.weights;

    for (int g0 = 0; g0 < npts; g0 += chunk) {
      const int ng = std::min(chunk, npts - g0);
      const auto cols = Kokkos::make_pair(0, ng);

      fill_aux_potential_chunk(basis_aux, grid, g0, ng, Vaux);
      fill_collocation_values(
          space, basis_aux,
          Kokkos::subview(grid.quad_points, Kokkos::make_pair(g0, g0 + ng)),
          Phi);
      space.fence();

      Kokkos::parallel_for(
          "Weight the auxiliary potential",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0},
                                                            {naux, ng}),
          KOKKOS_LAMBDA(const int a, const int g) {
            Vaux(a, g) *= w(g0 + g);
          });
      space.fence();

      // beta = 1: the chunks of the grid accumulate into the same matrix.
      KokkosBlas::gemm(space, "N", "T", 1.0,
                       Kokkos::subview(Phi, Kokkos::ALL(), cols),
                       Kokkos::subview(Vaux, Kokkos::ALL(), cols), 1.0, M);
      space.fence();
    }
  }

  // (A|B) is symmetric in exact arithmetic but not in quadrature, and the
  // half-inverse that follows wants a symmetric input.
  Kokkos::parallel_for(
      "Symmetrize Coulomb Overlap Matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0},
                                                        {naux, naux}),
      KOKKOS_LAMBDA(const int i, const int j) {
        sym(i, j) = 0.5 * (M(i, j) + M(j, i));
      });
  space.fence();

  Kokkos::Profiling::popRegion();
  return sym;
}

} // namespace Nukexc
