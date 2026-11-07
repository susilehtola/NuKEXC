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
#include "nukexc/partitioning.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "packed_blocks.hpp"
#include "stobasis.hpp"

#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_gemv_impl.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_gemm_impl.hpp>

#include <KokkosLapack_gesv.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <stdexcept>
#include <xc.h>

#include "basis_evaluator.hpp"

namespace Nukexc {

struct XC_result {
  double energy;
  DeviceView2DLeft potential;
};

struct XC_result_polarized {
  double energy;
  DeviceView2DLeft potential_alpha;
  DeviceView2DLeft potential_beta;
};

inline XC_result compute_lda(const DeviceView2DLeft collocation_values,
                      const DeviceView1D weights,
                      const DeviceView2DLeft mo_orbitals,
                      const DeviceView1D mo_coeff, const xc_func_type func) {

  ExecSpace space;

  // Make sure that the provided xc functional is a LDA
  if (func.info->family != XC_FAMILY_LDA) {
    throw std::runtime_error(
        "Provided funtional is not a part of the LDA family");
  }

  if (func.nspin != XC_UNPOLARIZED) {
    throw std::runtime_error(
        "compute_lda requires a functional initialized with XC_UNPOLARIZED; "
        "use compute_lsda for XC_POLARIZED");
  }

  // Get some constants
  const int N_quad = weights.extent(0);
  const int N_bf = collocation_values.extent(0);

  // Compute densities rho and sigma
  DeviceView1D rho("Rho", N_quad);
  DeviceView1D vrho("VRho", N_quad);
  DeviceView1D exc("Exc", N_quad);

  rho = compute_density(collocation_values, mo_orbitals, mo_coeff);

  // Copy densities rho and sigma to the host device
  auto rho_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);

  auto vrho_h = Kokkos::create_mirror_view(vrho);
  auto exc_h = Kokkos::create_mirror_view(exc);

  Kokkos::deep_copy(vrho_h, 0.0);
  Kokkos::deep_copy(exc_h, 0.0);

  // Evaluate xc functionals on the host device with libxc
  xc_lda_exc_vxc(&func, N_quad, rho_h.data(), exc_h.data(), vrho_h.data());

  // Copy the values back to the execution device
  Kokkos::deep_copy(exc, exc_h);
  Kokkos::deep_copy(vrho, vrho_h);

  DeviceView2DLeft Z("Z LDA", N_bf, N_quad);
  DeviceView2DLeft V("V LDA", N_bf, N_bf);
  DeviceView2DLeft V_result("V LDA Result", N_bf, N_bf);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Compute Z_mu(r)", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double w_g = weights(g);
        const double vrho_g = vrho(g);

        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto Z_subview = Kokkos::subview(Z, Kokkos::ALL(), g);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
              Z_subview(i) = w_g * (0.5 * collocation_subview(i) * vrho_g);
            });
      });

  // Compute gemm
  const double one(1.0);
  const double zero(0.0);
  KokkosBlas::gemm("N", "T", one, collocation_values, Z, zero, V);

  // Symmetrize result
  Kokkos::parallel_for(
      "Symmetrize Result",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
      KOKKOS_LAMBDA(const int i, const int j) {
        V_result(i, j) = V(i, j) + V(j, i);
      });

  double xc_energy = 0.0;

  // Integrate xc_energy density
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &xc_update) {
        xc_update += weights(g) * rho(g) * exc(g);
      },
      xc_energy);

  return XC_result{xc_energy, V_result};
}

// ============================================================================
//  Polarized libxc evaluation, shared by the dense and the screened paths
// ============================================================================
//
// libxc takes host memory in its interleaved spin layout and returns the
// partial derivatives in the same layout. None of that depends on how the
// density reached the grid, so it lives here once. The dense and screened
// quadratures differ in exactly two places -- how rho is built and how V is
// assembled -- and in nothing else, which is why the two used to carry a
// hundred identical lines each.

struct LDAPolarizedKernel {
  DeviceView1D exc;       ///< energy per particle at each point
  DeviceView1D vrho_up;   ///< d(rho e)/d rho_up
  DeviceView1D vrho_down; ///< d(rho e)/d rho_down
};

/// One joint xc_lda_exc_vxc call: exc and both vrho come from the same
/// evaluation at the same (rho_up, rho_down) pair.
inline LDAPolarizedKernel evaluate_lda_polarized(const xc_func_type &func,
                                                 const DeviceView1D &rho_a,
                                                 const DeviceView1D &rho_b) {
  const int N_quad = static_cast<int>(rho_a.extent(0));
  if (static_cast<int>(rho_b.extent(0)) != N_quad)
    throw std::invalid_argument(
        "evaluate_lda_polarized: the two spin densities differ in length");

  auto rho_a_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_a);
  auto rho_b_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_b);

  Kokkos::View<double *, HostSpace> rho_pol_h("rho_pol_h", 2 * N_quad);
  Kokkos::View<double *, HostSpace> vrho_pol_h("vrho_pol_h", 2 * N_quad);
  Kokkos::View<double *, HostSpace> exc_h("exc_h", N_quad);
  for (int g = 0; g < N_quad; ++g) {
    rho_pol_h(2 * g) = rho_a_h(g);
    rho_pol_h(2 * g + 1) = rho_b_h(g);
  }
  Kokkos::deep_copy(vrho_pol_h, 0.0);
  Kokkos::deep_copy(exc_h, 0.0);

  xc_lda_exc_vxc(&func, N_quad, rho_pol_h.data(), exc_h.data(),
                 vrho_pol_h.data());

  LDAPolarizedKernel out;
  out.exc = DeviceView1D("exc", N_quad);
  out.vrho_up = DeviceView1D("vrho up", N_quad);
  out.vrho_down = DeviceView1D("vrho down", N_quad);

  auto exc_hv = Kokkos::create_mirror_view(out.exc);
  auto vu_hv = Kokkos::create_mirror_view(out.vrho_up);
  auto vd_hv = Kokkos::create_mirror_view(out.vrho_down);
  for (int g = 0; g < N_quad; ++g) {
    exc_hv(g) = exc_h(g);
    vu_hv(g) = vrho_pol_h(2 * g);
    vd_hv(g) = vrho_pol_h(2 * g + 1);
  }
  Kokkos::deep_copy(out.exc, exc_hv);
  Kokkos::deep_copy(out.vrho_up, vu_hv);
  Kokkos::deep_copy(out.vrho_down, vd_hv);
  return out;
}

struct GGAPolarizedKernel {
  DeviceView1D exc;
  DeviceView1D vrho_up, vrho_down;
  /// 2 vsigma_uu grad(rho_up) + vsigma_ud grad(rho_down), and the mirror:
  /// the vector the potential's gradient term contracts the basis gradient
  /// against.
  DeviceView1D vecx_up, vecy_up, vecz_up;
  DeviceView1D vecx_dn, vecy_dn, vecz_dn;
};

/// One joint xc_gga_exc_vxc call. The three sigmas are formed here from the
/// two spin gradients, so a caller that has the gradients does not also have
/// to know libxc's (uu, ud, dd) ordering.
inline GGAPolarizedKernel evaluate_gga_polarized(
    const xc_func_type &func, const DeviceView1D &rho_a,
    const DeviceView1D &gxa, const DeviceView1D &gya, const DeviceView1D &gza,
    const DeviceView1D &rho_b, const DeviceView1D &gxb,
    const DeviceView1D &gyb, const DeviceView1D &gzb) {
  const int N_quad = static_cast<int>(rho_a.extent(0));
  if (static_cast<int>(rho_b.extent(0)) != N_quad)
    throw std::invalid_argument(
        "evaluate_gga_polarized: the two spin densities differ in length");

  DeviceView1D sigma_aa("sigma_aa", N_quad), sigma_ud("sigma_ud", N_quad),
      sigma_bb("sigma_bb", N_quad);
  Kokkos::parallel_for(
      "Compute sigmas", Kokkos::RangePolicy<ExecSpace>(0, N_quad),
      KOKKOS_LAMBDA(const int g) {
        sigma_aa(g) = gxa(g) * gxa(g) + gya(g) * gya(g) + gza(g) * gza(g);
        sigma_ud(g) = gxa(g) * gxb(g) + gya(g) * gyb(g) + gza(g) * gzb(g);
        sigma_bb(g) = gxb(g) * gxb(g) + gyb(g) * gyb(g) + gzb(g) * gzb(g);
      });

  auto rho_a_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_a);
  auto rho_b_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_b);
  auto s_uu_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma_aa);
  auto s_ud_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma_ud);
  auto s_dd_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma_bb);

  Kokkos::View<double *, HostSpace> rho_pol_h("rho_pol_h", 2 * N_quad);
  Kokkos::View<double *, HostSpace> sigma_pol_h("sigma_pol_h", 3 * N_quad);
  for (int g = 0; g < N_quad; ++g) {
    rho_pol_h(2 * g) = rho_a_h(g);
    rho_pol_h(2 * g + 1) = rho_b_h(g);
    sigma_pol_h(3 * g) = s_uu_h(g);
    sigma_pol_h(3 * g + 1) = s_ud_h(g);
    sigma_pol_h(3 * g + 2) = s_dd_h(g);
  }

  Kokkos::View<double *, HostSpace> vrho_pol_h("vrho_pol_h", 2 * N_quad);
  Kokkos::View<double *, HostSpace> vsigma_pol_h("vsigma_pol_h", 3 * N_quad);
  Kokkos::View<double *, HostSpace> exc_h("exc_h", N_quad);
  Kokkos::deep_copy(vrho_pol_h, 0.0);
  Kokkos::deep_copy(vsigma_pol_h, 0.0);
  Kokkos::deep_copy(exc_h, 0.0);

  xc_gga_exc_vxc(&func, N_quad, rho_pol_h.data(), sigma_pol_h.data(),
                 exc_h.data(), vrho_pol_h.data(), vsigma_pol_h.data());

  GGAPolarizedKernel out;
  out.exc = DeviceView1D("exc", N_quad);
  out.vrho_up = DeviceView1D("vrho up", N_quad);
  out.vrho_down = DeviceView1D("vrho down", N_quad);
  DeviceView1D vsigma_uu("vsigma_uu", N_quad), vsigma_ud("vsigma_ud", N_quad),
      vsigma_dd("vsigma_dd", N_quad);

  auto exc_hv = Kokkos::create_mirror_view(out.exc);
  auto vu_hv = Kokkos::create_mirror_view(out.vrho_up);
  auto vd_hv = Kokkos::create_mirror_view(out.vrho_down);
  auto vuu_hv = Kokkos::create_mirror_view(vsigma_uu);
  auto vud_hv = Kokkos::create_mirror_view(vsigma_ud);
  auto vdd_hv = Kokkos::create_mirror_view(vsigma_dd);
  for (int g = 0; g < N_quad; ++g) {
    exc_hv(g) = exc_h(g);
    vu_hv(g) = vrho_pol_h(2 * g);
    vd_hv(g) = vrho_pol_h(2 * g + 1);
    vuu_hv(g) = vsigma_pol_h(3 * g);
    vud_hv(g) = vsigma_pol_h(3 * g + 1);
    vdd_hv(g) = vsigma_pol_h(3 * g + 2);
  }
  Kokkos::deep_copy(out.exc, exc_hv);
  Kokkos::deep_copy(out.vrho_up, vu_hv);
  Kokkos::deep_copy(out.vrho_down, vd_hv);
  Kokkos::deep_copy(vsigma_uu, vuu_hv);
  Kokkos::deep_copy(vsigma_ud, vud_hv);
  Kokkos::deep_copy(vsigma_dd, vdd_hv);

  out.vecx_up = DeviceView1D("vecx up", N_quad);
  out.vecy_up = DeviceView1D("vecy up", N_quad);
  out.vecz_up = DeviceView1D("vecz up", N_quad);
  out.vecx_dn = DeviceView1D("vecx dn", N_quad);
  out.vecy_dn = DeviceView1D("vecy dn", N_quad);
  out.vecz_dn = DeviceView1D("vecz dn", N_quad);
  auto vxu = out.vecx_up, vyu = out.vecy_up, vzu = out.vecz_up;
  auto vxd = out.vecx_dn, vyd = out.vecy_dn, vzd = out.vecz_dn;
  Kokkos::parallel_for(
      "Compute vec_up/vec_down", Kokkos::RangePolicy<ExecSpace>(0, N_quad),
      KOKKOS_LAMBDA(const int g) {
        const double uu = vsigma_uu(g), ud = vsigma_ud(g), dd = vsigma_dd(g);
        vxu(g) = 2.0 * uu * gxa(g) + ud * gxb(g);
        vyu(g) = 2.0 * uu * gya(g) + ud * gyb(g);
        vzu(g) = 2.0 * uu * gza(g) + ud * gzb(g);
        vxd(g) = 2.0 * dd * gxb(g) + ud * gxa(g);
        vyd(g) = 2.0 * dd * gyb(g) + ud * gya(g);
        vzd(g) = 2.0 * dd * gzb(g) + ud * gza(g);
      });
  Kokkos::fence();
  return out;
}

inline XC_result_polarized
compute_lsda(const DeviceView2DLeft collocation_values,
             const DeviceView1D weights, const DeviceView2DLeft mo_alpha,
             const DeviceView1D occ_alpha, const DeviceView2DLeft mo_beta,
             const DeviceView1D occ_beta, const xc_func_type func) {
  ExecSpace space;

  if (func.info->family != XC_FAMILY_LDA)
    throw std::runtime_error(
        "Provided functional is not part of the LDA family");
  if (func.nspin != XC_POLARIZED)
    throw std::runtime_error(
        "compute_lsda requires a functional initialized with XC_POLARIZED");

  const int N_quad = weights.extent(0);
  const int N_bf = collocation_values.extent(0);

  DeviceView1D rho_a = compute_density(collocation_values, mo_alpha, occ_alpha);
  DeviceView1D rho_b = compute_density(collocation_values, mo_beta, occ_beta);
  const LDAPolarizedKernel kern = evaluate_lda_polarized(func, rho_a, rho_b);
  const DeviceView1D exc = kern.exc;
  const DeviceView1D vrho_up = kern.vrho_up;
  const DeviceView1D vrho_down = kern.vrho_down;

  auto build_potential = [&](const DeviceView1D &vrho_spin) {
    DeviceView2DLeft Z("Z LSDA", N_bf, N_quad);
    DeviceView2DLeft V("V LSDA", N_bf, N_bf);
    DeviceView2DLeft V_result("V LSDA Result", N_bf, N_bf);
    Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
    using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
    Kokkos::parallel_for(
        "Compute Z_mu(r)", policy,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int g = team_member.league_rank();
          const double w_g = weights(g);
          const double vrho_g = vrho_spin(g);
          auto colloc = Kokkos::subview(collocation_values, Kokkos::ALL(), g);
          auto Zsub = Kokkos::subview(Z, Kokkos::ALL(), g);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, N_bf),
              [=](const int i) { Zsub(i) = w_g * (0.5 * colloc(i) * vrho_g); });
        });
    const double one(1.0), zero(0.0);
    KokkosBlas::gemm("N", "T", one, collocation_values, Z, zero, V);
    Kokkos::parallel_for(
        "Symmetrize Result",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
        KOKKOS_LAMBDA(const int i, const int j) {
          V_result(i, j) = V(i, j) + V(j, i);
        });
    return V_result;
  };

  DeviceView2DLeft V_alpha = build_potential(vrho_up);
  DeviceView2DLeft V_beta = build_potential(vrho_down);

  double xc_energy = 0.0;
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &acc) {
        acc += weights(g) * (rho_a(g) + rho_b(g)) * exc(g);
      },
      xc_energy);

  return XC_result_polarized{xc_energy, V_alpha, V_beta};
}

// ============================================================================
//  Screened XC quadrature: pack -> GEMM -> scatter
// ============================================================================
//
// The potential is the same product the core Hamiltonian is, with the
// quadrature weight carrying the functional derivative instead of the
// nuclear potential:
//
//     V_uv = sum_g w_g v_rho(g) chi_u(g) chi_v(g)
//
// for an LDA, and for a GGA the gradient term on top. Per box that is one
// dense (nbe, npts) collocation block, one GEMM against itself, and a
// scatter through the local-to-global function map -- see
// packed_blocks.hpp.
//
// What this replaces gave each thread one pair of basis FUNCTIONS and
// evaluated both functions' shells at every point of the box, so a shell
// was evaluated once per function pair rather than once per point. It also
// re-evaluated the whole basis once per occupied orbital when forming the
// density. Both factors are gone: the density contracts through the density
// matrix (density.hpp's packed_density) and the potential through the
// packed blocks.

template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Value>
inline XC_result_polarized
compute_lsda_sparse(const Evaluator basis, const FlatGrid grid,
                    const NeighborList nl, const DeviceView2DLeft mo_alpha,
                    const DeviceView1D occ_alpha,
                    const DeviceView2DLeft mo_beta, const DeviceView1D occ_beta,
                    const xc_func_type func) {
  if (func.info->family != XC_FAMILY_LDA)
    throw std::runtime_error(
        "Provided functional is not part of the LDA family");
  if (func.nspin != XC_POLARIZED)
    throw std::runtime_error(
        "compute_lsda requires a functional initialized with XC_POLARIZED");

  const int N_quad = grid.weights.extent(0);
  const int N_bf = basis.nbf();

  const PackedBlockLayout L = build_packed_block_layout(basis, nl);

  DeviceView1D rho_a("rho_a", N_quad), rho_b("rho_b", N_quad);
  packed_density<BasisDeriv::Value>(
      basis, grid, nl, L, density_matrix_from_orbitals(mo_alpha, occ_alpha),
      rho_a);
  packed_density<BasisDeriv::Value>(
      basis, grid, nl, L, density_matrix_from_orbitals(mo_beta, occ_beta),
      rho_b);

  const LDAPolarizedKernel kern = evaluate_lda_polarized(func, rho_a, rho_b);
  const auto w = grid.weights;

  // Z = w v_rho chi makes Z chi^T symmetric, so there is nothing to
  // symmetrize: the dense path's "half the potential plus its transpose" is
  // the same number reached in two steps.
  auto build_potential = [&](const DeviceView1D &vrho) {
    DeviceView2DLeft V("V LSDA", N_bf, N_bf);
    accumulate_collocation_product<BasisDeriv::Value>(
        basis, grid, nl, L, WeightedCollocation{w, vrho}, V);
    return V;
  };

  DeviceView2DLeft V_alpha = build_potential(kern.vrho_up);
  DeviceView2DLeft V_beta = build_potential(kern.vrho_down);

  const auto exc = kern.exc;
  double xc_energy = 0.0;
  Kokkos::parallel_reduce(
      "Compute xc energy", Kokkos::RangePolicy<ExecSpace>(0, N_quad),
      KOKKOS_LAMBDA(const int g, double &acc) {
        acc += w(g) * (rho_a(g) + rho_b(g)) * exc(g);
      },
      xc_energy);

  return XC_result_polarized{xc_energy, V_alpha, V_beta};
}

inline XC_result compute_gga(const DeviceView2DLeft collocation_values,
                      const DeviceView2DLeft collocation_gx,
                      const DeviceView2DLeft collocation_gy,
                      const DeviceView2DLeft collocation_gz,
                      const DeviceView1D weights,
                      const DeviceView2DLeft mo_orbitals,
                      const DeviceView1D mo_coeff, const xc_func_type func) {
  ExecSpace space;

  // Make sure that the provided xc functional is a GGA
  if (func.info->family != XC_FAMILY_GGA &&
      func.info->family != XC_FAMILY_HYB_GGA) {
    throw std::runtime_error(
        "Provided funtional is not a part of the GGA or Hybrid GGA family");
  }

  // Get some constants
  const int N_quad = weights.extent(0);
  const int N_bf = collocation_values.extent(0);

  // Compute densities rho and sigma
  DeviceView1D rho("Rho", N_quad);
  DeviceView1D gx_rho("Grad x Rho", N_quad);
  DeviceView1D gy_rho("Grad y Rho", N_quad);
  DeviceView1D gz_rho("Grad z Rho", N_quad);
  DeviceView1D sigma("Sigma", N_quad);
  DeviceView1D vrho("VRho", N_quad);
  DeviceView1D vsigma("VSigma", N_quad);
  DeviceView1D exc("Exc", N_quad);

  compute_density_and_sigma(collocation_values, collocation_gx, collocation_gy,
                            collocation_gz, mo_orbitals, mo_coeff, rho, gx_rho,
                            gy_rho, gz_rho, sigma);

  // Copy densities rho and sigma to the host device
  auto rho_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);
  auto sigma_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma);

  auto vrho_h = Kokkos::create_mirror_view(vrho);
  auto vsigma_h = Kokkos::create_mirror_view(vsigma);
  auto exc_h = Kokkos::create_mirror_view(exc);

  Kokkos::deep_copy(vrho_h, 0.0);
  Kokkos::deep_copy(vsigma_h, 0.0);
  Kokkos::deep_copy(exc_h, 0.0);

  // Evaluate xc functionals on the host device with libxc
  xc_gga_exc_vxc(&func, N_quad, rho_h.data(), sigma_h.data(), exc_h.data(),
                 vrho_h.data(), vsigma_h.data());

  // Copy the values back to the execution device
  Kokkos::deep_copy(exc, exc_h);
  Kokkos::deep_copy(vrho, vrho_h);
  Kokkos::deep_copy(vsigma, vsigma_h);

  DeviceView2DLeft Z("Z GGA", N_bf, N_quad);
  DeviceView2DLeft V("V GGA", N_bf, N_bf);
  DeviceView2DLeft V_result("V GGA Result", N_bf, N_bf);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Compute Z_mu(r)", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double w_g = weights(g);
        const double vrho_g = vrho(g);
        const double gx_rho_g = gx_rho(g);
        const double gy_rho_g = gy_rho(g);
        const double gz_rho_g = gz_rho(g);
        const double vsigma_g = vsigma(g);

        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto collocation_gx_subview =
            Kokkos::subview(collocation_gx, Kokkos::ALL(), g);
        auto collocation_gy_subview =
            Kokkos::subview(collocation_gy, Kokkos::ALL(), g);
        auto collocation_gz_subview =
            Kokkos::subview(collocation_gz, Kokkos::ALL(), g);
        auto Z_subview = Kokkos::subview(Z, Kokkos::ALL(), g);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
              double local_sum = 0.0;
              local_sum += gx_rho_g * collocation_gx_subview(i);
              local_sum += gy_rho_g * collocation_gy_subview(i);
              local_sum += gz_rho_g * collocation_gz_subview(i);
              Z_subview(i) = w_g * (0.5 * collocation_subview(i) * vrho_g +
                                    2.0 * local_sum * vsigma_g);
            });
      });

  // Compute gemm
  const double one(1.0);
  const double zero(0.0);
  KokkosBlas::gemm("N", "T", one, collocation_values, Z, zero, V);

  // Symmetrize result
  Kokkos::parallel_for(
      "Symmetrize Result",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
      KOKKOS_LAMBDA(const int i, const int j) {
        V_result(i, j) = V(i, j) + V(j, i);
      });

  double xc_energy = 0.0;

  // Integrate xc_energy density
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &xc_update) {
        xc_update += weights(g) * rho(g) * exc(g);
      },
      xc_energy);

  return XC_result{xc_energy, V_result};
}

inline XC_result_polarized
compute_gga_lsda(const DeviceView2DLeft collocation_values,
                 const DeviceView2DLeft collocation_gx,
                 const DeviceView2DLeft collocation_gy,
                 const DeviceView2DLeft collocation_gz,
                 const DeviceView1D weights, const DeviceView2DLeft mo_alpha,
                 const DeviceView1D occ_alpha, const DeviceView2DLeft mo_beta,
                 const DeviceView1D occ_beta, const xc_func_type func) {
  ExecSpace space;

  // Make sure that the provided xc functional is a GGA
  if (func.info->family != XC_FAMILY_GGA &&
      func.info->family != XC_FAMILY_HYB_GGA) {
    throw std::runtime_error(
        "Provided funtional is not a part of the GGA or Hybrid GGA family");
  }

  if (func.nspin != XC_POLARIZED)
    throw std::runtime_error("compute_gga_lsda requires a functional "
                             "initialized with XC_POLARIZED; "
                             "use compute_gga for XC_UNPOLARIZED");

  const int N_quad = weights.extent(0);
  const int N_bf = collocation_values.extent(0);

  // ---- Per-spin densities, gradients, same-spin sigma (sigma_aa/sigma_bb)
  // --
  DeviceView1D rho_a("rho_a", N_quad), gxa("gxa", N_quad), gya("gya", N_quad),
      gza("gza", N_quad), sigma_aa("sigma_aa", N_quad);
  DeviceView1D rho_b("rho_b", N_quad), gxb("gxb", N_quad), gyb("gyb", N_quad),
      gzb("gzb", N_quad), sigma_bb("sigma_bb", N_quad);

  compute_density_and_sigma(collocation_values, collocation_gx, collocation_gy,
                            collocation_gz, mo_alpha, occ_alpha, rho_a, gxa,
                            gya, gza, sigma_aa);
  compute_density_and_sigma(collocation_values, collocation_gx, collocation_gy,
                            collocation_gz, mo_beta, occ_beta, rho_b, gxb, gyb,
                            gzb, sigma_bb);

  const GGAPolarizedKernel kern = evaluate_gga_polarized(
      func, rho_a, gxa, gya, gza, rho_b, gxb, gyb, gzb);
  const DeviceView1D exc = kern.exc;
  const DeviceView1D vrho_up = kern.vrho_up, vrho_down = kern.vrho_down;
  const DeviceView1D vecx_up = kern.vecx_up, vecy_up = kern.vecy_up,
                     vecz_up = kern.vecz_up;
  const DeviceView1D vecx_dn = kern.vecx_dn, vecy_dn = kern.vecy_dn,
                     vecz_dn = kern.vecz_dn;

  // ---- Build Fock contribution for one spin channel
  // -------------------------
  auto build_potential = [&](const DeviceView1D &vrho_spin,
                             const DeviceView1D &vecx, const DeviceView1D &vecy,
                             const DeviceView1D &vecz) {
    DeviceView2DLeft Z("Z GGA pol", N_bf, N_quad);
    DeviceView2DLeft V("V GGA pol", N_bf, N_bf);
    DeviceView2DLeft V_result("V GGA pol Result", N_bf, N_bf);

    Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
    using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
    Kokkos::parallel_for(
        "Compute Z_mu(r)", policy,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int g = team_member.league_rank();
          const double w_g = weights(g);
          const double vrho_g = vrho_spin(g);
          const double vx = vecx(g), vy = vecy(g), vz = vecz(g);

          auto colloc = Kokkos::subview(collocation_values, Kokkos::ALL(), g);
          auto colloc_gx = Kokkos::subview(collocation_gx, Kokkos::ALL(), g);
          auto colloc_gy = Kokkos::subview(collocation_gy, Kokkos::ALL(), g);
          auto colloc_gz = Kokkos::subview(collocation_gz, Kokkos::ALL(), g);
          auto Zsub = Kokkos::subview(Z, Kokkos::ALL(), g);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
                const double grad_dot =
                    vx * colloc_gx(i) + vy * colloc_gy(i) + vz * colloc_gz(i);
                Zsub(i) = w_g * (0.5 * colloc(i) * vrho_g + grad_dot);
              });
        });

    const double one(1.0), zero(0.0);
    KokkosBlas::gemm("N", "T", one, collocation_values, Z, zero, V);
    Kokkos::parallel_for(
        "Symmetrize Result",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
        KOKKOS_LAMBDA(const int i, const int j) {
          V_result(i, j) = V(i, j) + V(j, i);
        });
    return V_result;
  };

  DeviceView2DLeft V_alpha =
      build_potential(vrho_up, vecx_up, vecy_up, vecz_up);
  DeviceView2DLeft V_beta =
      build_potential(vrho_down, vecx_dn, vecy_dn, vecz_dn);

  // ---- Energy: integrate against total density
  // ------------------------------
  double xc_energy = 0.0;
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &acc) {
        acc += weights(g) * (rho_a(g) + rho_b(g)) * exc(g);
      },
      xc_energy);

  return XC_result_polarized{xc_energy, V_alpha, V_beta};
}

/**
 *  @brief Z = w (1/2 v_rho chi + vec . grad chi): the GGA potential's
 *         pointwise weight.
 *
 *  Z chi^T is NOT symmetric here -- the gradient sits on one index only --
 *  so the caller adds the transpose. That is the usual GGA form, not a
 *  shortcut: the two terms of the functional derivative differ by which
 *  basis function carries the derivative.
 */
template <typename WView, typename VView> struct GGAPotentialWeight {
  WView w;
  VView vrho, vx, vy, vz;
  KOKKOS_INLINE_FUNCTION double operator()(const int g, const double chi,
                                           const double dx, const double dy,
                                           const double dz) const {
    return w(g) * (0.5 * chi * vrho(g) + vx(g) * dx + vy(g) * dy +
                   vz(g) * dz);
  }
};

template <typename Evaluator>
  requires BasisEvaluatorWith<Evaluator, BasisDeriv::Gradient>
inline XC_result_polarized
compute_gga_lsda_sparse(const Evaluator basis, const FlatGrid grid,
                        const NeighborList nl, const DeviceView2DLeft mo_alpha,
                        const DeviceView1D occ_alpha,
                        const DeviceView2DLeft mo_beta,
                        const DeviceView1D occ_beta, const xc_func_type func) {
  if (func.info->family != XC_FAMILY_GGA &&
      func.info->family != XC_FAMILY_HYB_GGA)
    throw std::runtime_error(
        "Provided funtional is not a part of the GGA or Hybrid GGA family");
  if (func.nspin != XC_POLARIZED)
    throw std::runtime_error("compute_gga_lsda requires a functional "
                             "initialized with XC_POLARIZED; "
                             "use compute_gga for XC_UNPOLARIZED");

  const int N_quad = grid.weights.extent(0);
  const int N_bf = basis.nbf();

  const PackedBlockLayout L = build_packed_block_layout(basis, nl);

  DeviceView1D rho_a("rho_a", N_quad), gxa("gxa", N_quad), gya("gya", N_quad),
      gza("gza", N_quad);
  DeviceView1D rho_b("rho_b", N_quad), gxb("gxb", N_quad), gyb("gyb", N_quad),
      gzb("gzb", N_quad);
  packed_density<BasisDeriv::Gradient>(
      basis, grid, nl, L, density_matrix_from_orbitals(mo_alpha, occ_alpha),
      rho_a, gxa, gya, gza);
  packed_density<BasisDeriv::Gradient>(
      basis, grid, nl, L, density_matrix_from_orbitals(mo_beta, occ_beta),
      rho_b, gxb, gyb, gzb);

  const GGAPolarizedKernel kern = evaluate_gga_polarized(
      func, rho_a, gxa, gya, gza, rho_b, gxb, gyb, gzb);
  const auto w = grid.weights;

  auto build_potential = [&](const DeviceView1D &vrho, const DeviceView1D &vx,
                             const DeviceView1D &vy, const DeviceView1D &vz) {
    DeviceView2DLeft raw("V GGA pol one sided", N_bf, N_bf);
    accumulate_collocation_product<BasisDeriv::Gradient>(
        basis, grid, nl, L, GGAPotentialWeight{w, vrho, vx, vy, vz}, raw);

    DeviceView2DLeft V("V GGA pol", N_bf, N_bf);
    Kokkos::parallel_for(
        "Symmetrize Result",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                          {N_bf, N_bf}),
        KOKKOS_LAMBDA(const int i, const int j) {
          V(i, j) = raw(i, j) + raw(j, i);
        });
    Kokkos::fence();
    return V;
  };

  DeviceView2DLeft V_alpha =
      build_potential(kern.vrho_up, kern.vecx_up, kern.vecy_up, kern.vecz_up);
  DeviceView2DLeft V_beta =
      build_potential(kern.vrho_down, kern.vecx_dn, kern.vecy_dn,
                      kern.vecz_dn);

  const auto exc = kern.exc;
  double xc_energy = 0.0;
  Kokkos::parallel_reduce(
      "Compute xc energy", Kokkos::RangePolicy<ExecSpace>(0, N_quad),
      KOKKOS_LAMBDA(const int g, double &acc) {
        acc += w(g) * (rho_a(g) + rho_b(g)) * exc(g);
      },
      xc_energy);

  return XC_result_polarized{xc_energy, V_alpha, V_beta};
}

// EXPERIMENTAL -- meta-GGA quadrature. Not on any production path and not
// reached by any test or driver; pairs with compute_density_and_sigma_and_tau()
// in density.hpp. Kept as the starting point for whoever adds mGGA support
// next.
// TODO:: This function has not been tested and probably contains bugs
template <typename Evaluator>
XC_result compute_mgga(const Evaluator basis, const FlatGrid grid,
                       const DeviceView2DLeft mo_orbitals,
                       const DeviceView1D mo_coeff, const xc_func_type func) {
  ExecSpace space;
  // Make sure that the provided xc functional is an MGGA
  if (func.info->family != XC_FAMILY_MGGA) {
    throw std::runtime_error(
        "Provided funtional is not a part of the MGGA family");
  }

  // Get some constants
  const int N_quad = grid.quad_points.extent(0);
  const int N_bf = basis.nbf();

  // Compute densities rho and sigma
  DeviceView2DLeft collocation("Collocation", N_bf, N_quad);
  DeviceView1D rho("Rho", N_quad);
  DeviceView1D gx_rho("Grad x Rho", N_quad);
  DeviceView1D gy_rho("Grad y Rho", N_quad);
  DeviceView1D gz_rho("Grad z Rho", N_quad);
  DeviceView1D sigma("Sigma", N_quad);
  DeviceView1D tau("Tau", N_quad);
  DeviceView1D lapl("Laplacian", N_quad);

  DeviceView1D vrho("VRho", N_quad);
  DeviceView1D vsigma("VSigma", N_quad);
  DeviceView1D vtau("VTau", N_quad);
  DeviceView1D vlapl("VLaplacian", N_quad);
  DeviceView1D exc("Exc", N_quad);

  // Fill the collocation Views
  DeviceView2DLeft collocation_values("collocation values", N_bf, N_quad);
  DeviceView2DLeft collocation_gx("collocation gx", N_bf, N_quad);
  DeviceView2DLeft collocation_gy("collocation gy", N_bf, N_quad);
  DeviceView2DLeft collocation_gz("collocation gz", N_bf, N_quad);

  fill_collocation(space, basis, grid.quad_points, collocation_values);

  fill_grad_collocation(space, basis, grid.quad_points, collocation_gx,
                        collocation_gy, collocation_gz);

  compute_density_and_sigma_and_tau(
      collocation_values, collocation_gx, collocation_gy, collocation_gz,
      mo_orbitals, mo_coeff, rho, gx_rho, gy_rho, gz_rho, sigma, tau);

  // Copy densities rho and sigma to the host device
  auto rho_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);
  auto sigma_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma);
  auto tau_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, tau);
  auto lapl_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, lapl);

  auto vrho_h = Kokkos::create_mirror_view(vrho);
  auto vsigma_h = Kokkos::create_mirror_view(vsigma);
  auto vtau_h = Kokkos::create_mirror_view(vtau);
  auto vlapl_h = Kokkos::create_mirror_view(vlapl);
  auto exc_h = Kokkos::create_mirror_view(exc);

  Kokkos::deep_copy(exc_h, 0.0);
  Kokkos::deep_copy(vrho_h, 0.0);
  Kokkos::deep_copy(vsigma_h, 0.0);
  Kokkos::deep_copy(vtau_h, 0.0);
  Kokkos::deep_copy(vlapl_h, 0.0);

  // Evaluate xc functionals on the host device with libxc
  xc_mgga_exc_vxc(&func, N_quad, rho_h.data(), sigma_h.data(), lapl_h.data(),
                  tau_h.data(), exc_h.data(), vrho_h.data(), vsigma_h.data(),
                  vlapl_h.data(), vtau_h.data());

  // Copy the values back to the execution device
  Kokkos::deep_copy(exc, exc_h);
  Kokkos::deep_copy(vrho, vrho_h);
  Kokkos::deep_copy(vsigma, vsigma_h);
  Kokkos::deep_copy(vtau, vtau_h);
  Kokkos::deep_copy(vlapl, vlapl_h);

  DeviceView2DLeft Z("Z MGGA", N_bf, N_quad);
  DeviceView2DLeft V("V MGGA", N_bf, N_bf);
  DeviceView2DLeft W_x("W_x MGGA", N_bf, N_bf);
  DeviceView2DLeft W_y("W_y MGGA", N_bf, N_bf);
  DeviceView2DLeft W_z("W_x MGGA", N_bf, N_bf);
  DeviceView2DLeft V_result("V MGGA Result", N_bf, N_bf);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Compute Z_mu(r) and W_mu(r)", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double w_g = grid.weights(g);
        const double vrho_g = vrho(g);
        const double gx_rho_g = gx_rho(g);
        const double gy_rho_g = gy_rho(g);
        const double gz_rho_g = gz_rho(g);
        const double vsigma_g = vsigma(g);
        const double vtau_g = vtau(g);

        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto collocation_gx_subview =
            Kokkos::subview(collocation_gx, Kokkos::ALL(), g);
        auto collocation_gy_subview =
            Kokkos::subview(collocation_gy, Kokkos::ALL(), g);
        auto collocation_gz_subview =
            Kokkos::subview(collocation_gz, Kokkos::ALL(), g);
        auto Z_subview = Kokkos::subview(Z, Kokkos::ALL(), g);

        auto W_x_subview = Kokkos::subview(W_x, Kokkos::ALL(), g);
        auto W_y_subview = Kokkos::subview(W_y, Kokkos::ALL(), g);
        auto W_z_subview = Kokkos::subview(W_z, Kokkos::ALL(), g);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
              double local_sum = 0.0;
              local_sum += gx_rho_g * collocation_gx_subview(i);
              local_sum += gy_rho_g * collocation_gy_subview(i);
              local_sum += gz_rho_g * collocation_gz_subview(i);
              Z_subview(i) = w_g * (0.5 * collocation_subview(i) * vrho_g +
                                    2.0 * local_sum * vsigma_g);
              W_x_subview(i) =
                  w_g * (0.25 * vtau_g * collocation_gx_subview(i));
              W_y_subview(i) =
                  w_g * (0.25 * vtau_g * collocation_gy_subview(i));
              W_z_subview(i) =
                  w_g * (0.25 * vtau_g * collocation_gz_subview(i));
            });
      });

  // Compute gemm
  const double one(1.0);
  const double zero(0.0);
  KokkosBlas::gemm("N", "T", one, collocation, Z, zero, V);
  KokkosBlas::gemm("N", "T", one, collocation_gx, W_x, one, V);
  KokkosBlas::gemm("N", "T", one, collocation_gy, W_y, one, V);
  KokkosBlas::gemm("N", "T", one, collocation_gz, W_z, one, V);

  // Symmetrize result
  Kokkos::parallel_for(
      "Symmetrize Result",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
      KOKKOS_LAMBDA(const int i, const int j) {
        V_result(i, j) = V(i, j) + V(j, i);
      });

  double xc_energy = 0.0;

  // Integrate xc_energy density
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &xc_update) {
        xc_update += grid.weights(g) * rho(g) * exc(g);
      },
      xc_energy);

  return XC_result{xc_energy, V_result};
}
} // namespace Nukexc
