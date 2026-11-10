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

// ===========================================================================
// Helpers specific to the standalone SCF driver: timing infrastructure, the
// runtime functional-name lookup/dispatch, and Kokkos<->Armadillo conversions.
//
// These used to sit at the top of standalone.cxx and made the driver hard to
// read. They pull in the (non-inline) XC integral kernels, so this header is
// intended to be included by standalone.cxx only -- keep it out of the shared
// test_io module so the lighter test drivers don't drag in libxc/armadillo.
// ===========================================================================

#pragma once

#include "nukexc/octree.hpp"
#include <nukexc/nukexc_config.hpp>
#include <nukexc/xc_integrals.hpp>
#include <nukexc/xc_screened.hpp>

#include <Kokkos_Core.hpp>
#include <armadillo>
#include <xc.h>
#include <xc_funcs.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


namespace Nukexc {

// ===========================================================================
// Timing infrastructure
// ===========================================================================
struct StackEntry {
  std::string name;
  std::unique_ptr<Kokkos::Timer> timer;
  int depth;
};

class TimingRegistry {
public:
  struct Entry {
    std::string name;
    double seconds;
    int depth;
  };

  void start(const std::string &name) {
    stack_.push_back(
        StackEntry{name, std::make_unique<Kokkos::Timer>(), depth_});
    ++depth_;
  }

  // Record a pre-measured duration directly (e.g. a total merged from
  // another registry) without going through start()/stop().
  void record(const std::string &name, double seconds) {
    entries_.push_back(Entry{name, seconds, depth_});
  }

  void stop() {
    --depth_;
    StackEntry top = std::move(stack_.back());
    stack_.pop_back();
    entries_.push_back(Entry{top.name, top.timer->seconds(), top.depth});
  }

  // Total time of all depth-0 (top level) blocks recorded so far.
  double total_top_level_seconds() const {
    double total = 0.0;
    for (const auto &e : entries_)
      if (e.depth == 0)
        total += e.seconds;
    return total;
  }

  void report(const std::string &title) const {
    std::cout << "\n"
              << title << "\n"
              << std::string(title.size(), '=') << "\n";
    if (entries_.empty()) {
      std::cout << "  (no timed sections recorded)\n\n";
      return;
    }
    constexpr int name_col = 42;
    for (const auto &e : entries_) {
      std::string indented = std::string(e.depth * 2, ' ') + e.name;
      std::cout << "  " << std::left << std::setw(name_col) << indented << ": "
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(10) << e.seconds << " s\n";
    }
    std::cout << "  " << std::string(name_col + 15, '-') << "\n";
    std::cout << "  " << std::left << std::setw(name_col) << "Total (top level)"
              << ": " << std::right << std::fixed << std::setprecision(4)
              << std::setw(10) << total_top_level_seconds() << " s\n\n"
              << std::flush;
  }

  void clear() {
    entries_.clear();
    stack_.clear();
    depth_ = 0;
  }

private:
  std::vector<StackEntry> stack_;
  std::vector<Entry> entries_;
  int depth_ = 0;
};

// RAII scope: starts a named timer on construction, stops it on destruction.
struct ScopedTiming {
  ScopedTiming(TimingRegistry &reg, std::string name) : reg_(reg) {
    reg_.start(name);
  }
  ~ScopedTiming() { reg_.stop(); }
  ScopedTiming(const ScopedTiming &) = delete;
  ScopedTiming &operator=(const ScopedTiming &) = delete;
  TimingRegistry &reg_;
};

// ===========================================================================
// Compile-time string hashing so we can "switch" on a runtime std::string.
// C++ can't switch on std::string directly, but case labels only need to be
// constant expressions -- so we hash the *literal* case labels at compile
// time via this constexpr function, and hash the *runtime* input string with
// the exact same function at runtime, then switch on the resulting uint32_t.
// ===========================================================================
constexpr std::uint32_t fnv1a(const char *s, std::uint32_t h = 2166136261u) {
  return (*s == '\0') ? h
                      : fnv1a(s + 1, (h ^ static_cast<std::uint32_t>(
                                              static_cast<unsigned char>(*s))) *
                                         16777619u);
}
inline std::uint32_t fnv1a_rt(const std::string &s) { return fnv1a(s.c_str()); }

// ---------------------------------------------------------------------------
// Libxc id, resolved from a runtime name via switch.
//
// This carried an XCFamily enum of its own, written by every entry below and
// read by nobody: both evaluators dispatch on func.info->family, which is
// Libxc's answer rather than a second list to keep in step. It also shadowed
// the library's XCFamily, which covers the meta-GGAs this one did not. Gone.
// ---------------------------------------------------------------------------
struct FunctionalInfo {
  int xc_id = -1;
  std::string canonical_name;
};

inline FunctionalInfo lookup_functional(const std::string &name) {
  switch (fnv1a_rt(name)) {
  // ---- LDA exchange -----------------------------------------------------
  case fnv1a("lda_x"):
    return {XC_LDA_X, "lda_x"};

  // ---- LDA correlation ---------------------------------------------------
  case fnv1a("lda_c_pw"):
    return {XC_LDA_C_PW, "lda_c_pw"};
  case fnv1a("lda_c_vwn"):
    return {XC_LDA_C_VWN, "lda_c_vwn"};

  // ---- GGA exchange -------------------------------------------------------
  case fnv1a("gga_x_pbe"):
    return {XC_GGA_X_PBE, "gga_x_pbe"};
  case fnv1a("gga_x_b88"):
    return {XC_GGA_X_B88, "gga_x_b88"};
  case fnv1a("gga_x_pw91"):
    return {XC_GGA_X_PW91, "gga_x_pw91"};
  case fnv1a("gga_xc_b3lyp"):
    return {XC_HYB_GGA_XC_B3LYP, "gga_xc_b3lyp"};
  case fnv1a("gga_xc_b3lyp3"):
    return {XC_HYB_GGA_XC_B3LYP3, "gga_xc_b3lyp3"};
  case fnv1a("gga_xc_b3lyp5"):
    return {XC_HYB_GGA_XC_B3LYP5, "gga_xc_b3lyp5"};
  case fnv1a("gga_xc_b3lyps"):
    return {XC_HYB_GGA_XC_B3LYPS, "gga_xc_b3lyp5"};

  // ---- GGA correlation ------------------------------------------------
  case fnv1a("gga_c_pbe"):
    return {XC_GGA_C_PBE, "gga_c_pbe"};
  case fnv1a("gga_c_lyp"):
    return {XC_GGA_C_LYP, "gga_c_lyp"};
  case fnv1a("gga_c_pw91"):
    return {XC_GGA_C_PW91, "gga_c_pw91"};

  default:
    throw std::runtime_error(
        "Unknown functional name: '" + name +
        "' (see --help or lookup_functional() for supported names)");
  }
}

// ---------------------------------------------------------------------------
// Dispatch a single functional (X or C) to the correct polarized evaluator
// based on its family. Both compute_lsda and compute_gga_lsda must return
// XC_result_polarized { energy, potential_alpha, potential_beta }.
// ---------------------------------------------------------------------------
inline XC_result_polarized evaluate_functional(
    const FunctionalInfo &info, const xc_func_type &func,
    const DeviceView2DLeft &basis_collocation,
    const DeviceView2DLeft &basis_collocation_gx,
    const DeviceView2DLeft &basis_collocation_gy,
    const DeviceView2DLeft &basis_collocation_gz, const DeviceView1D &weights,
    const DeviceView2DLeft &k_C_alpha, const DeviceView1D &k_occ_alpha,
    const DeviceView2DLeft &k_C_beta, const DeviceView1D &k_occ_beta) {
  (void)info;
  switch (func.info->family) {
  case XC_FAMILY_LDA:
    return compute_lsda(basis_collocation, weights, k_C_alpha, k_occ_alpha,
                        k_C_beta, k_occ_beta, func);
  case XC_FAMILY_HYB_LDA:
    return compute_lsda(basis_collocation, weights, k_C_alpha, k_occ_alpha,
                        k_C_beta, k_occ_beta, func);
  case XC_FAMILY_GGA:
    return compute_gga_lsda(basis_collocation, basis_collocation_gx,
                            basis_collocation_gy, basis_collocation_gz, weights,
                            k_C_alpha, k_occ_alpha, k_C_beta, k_occ_beta, func);
  case XC_FAMILY_HYB_GGA:
    return compute_gga_lsda(basis_collocation, basis_collocation_gx,
                            basis_collocation_gy, basis_collocation_gz, weights,
                            k_C_alpha, k_occ_alpha, k_C_beta, k_occ_beta, func);
  default:
    throw std::runtime_error(
        "evaluate_functional: no dense quadrature for this Libxc family");
  }
}

// ---------------------------------------------------------------------------
// The screened polarized evaluator. There is no family switch here: one
// screened quadrature covers every family, since the potential comes from
// the generated kernels and the family only decides which collocation
// order and which Libxc call it needs. On this driver's Slater basis that
// means LDA, GGA and tau meta-GGA -- a Laplacian meta-GGA needs second
// derivatives, which the radials do not provide. The dense counterpart
// above still switches, because its chain rule is written per family.
// ---------------------------------------------------------------------------
inline XC_result_polarized evaluate_functional_sparse(
    const FunctionalInfo &info, const xc_func_type &func,
    const STOBasisSet &basis, const FlatGrid &grid, const NeighborList &nl,
    const DeviceView2DLeft &k_C_alpha, const DeviceView1D &k_occ_alpha,
    const DeviceView2DLeft &k_C_beta, const DeviceView1D &k_occ_beta) {
  (void)info;
  return compute_xc_polarized_sparse(STOPrimitiveEvaluator(basis), grid, nl,
                                     k_C_alpha, k_occ_alpha, k_C_beta,
                                     k_occ_beta, func);
}

// ---------------------------------------------------------------------------
// Helper: Kokkos DeviceView2DLeft → arma::mat (column-major copy)
// OOO expects arma::mat where columns are MOs; NuKEXC stores mo_coeff(nbf, nmo)
// ---------------------------------------------------------------------------
inline arma::mat kokkos_to_arma(const DeviceView2DLeft &v) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  arma::mat out(h.extent(0), h.extent(1));
  for (std::size_t i = 0; i < h.extent(0); ++i)
    for (std::size_t j = 0; j < h.extent(1); ++j)
      out(i, j) = h(i, j);
  return out;
}

// ---------------------------------------------------------------------------
// Helper: arma::mat → Kokkos DeviceView2DLeft
// ---------------------------------------------------------------------------
inline DeviceView2DLeft arma_to_kokkos(const arma::mat &m,
                                       const std::string &label) {
  DeviceView2DLeft v(label, m.n_rows, m.n_cols);
  auto h = Kokkos::create_mirror_view(v);
  for (std::size_t i = 0; i < m.n_rows; ++i)
    for (std::size_t j = 0; j < m.n_cols; ++j)
      h(i, j) = m(i, j);
  Kokkos::deep_copy(v, h);
  return v;
}

// ---------------------------------------------------------------------------
// Helper: arma::vec → Kokkos DeviceView1D
// ---------------------------------------------------------------------------
inline DeviceView1D arma_to_kokkos1d(const arma::vec &v,
                                     const std::string &label) {
  DeviceView1D kv(label, v.n_elem);
  auto h = Kokkos::create_mirror_view(kv);
  for (std::size_t i = 0; i < v.n_elem; ++i)
    h(i) = v(i);
  Kokkos::deep_copy(kv, h);
  return kv;
}

} // namespace Nukexc

// ---------------------------------------------------------------------------
// RAII timing scope macro. Kept at global scope (macros ignore namespaces).
// Usage: { TIME_SCOPE(registry, "Some step"); do_work(); }
// ---------------------------------------------------------------------------
#define NUKEXC_CONCAT_(a, b) a##b
#define NUKEXC_CONCAT(a, b) NUKEXC_CONCAT_(a, b)
#define TIME_SCOPE(reg, name)                                                  \
  Nukexc::ScopedTiming NUKEXC_CONCAT(_scoped_timer_, __LINE__)((reg), (name))
