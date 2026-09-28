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

#pragma once

// ============================================================================
//  Gaussian basis set container (host side)
// ============================================================================
//
// NuKEXC does not read basis-set files. The calling program owns basis-set
// input -- parsing, element lookup, library paths, all of it -- and hands the
// result over as plain arrays through GTOBasisInput. That keeps file formats,
// search paths and element tables out of a numerical-integration library, and
// makes the entry point usable from any language with a C ABI.
//
// (Test fixtures may of course build a basis from a file; that code belongs in
// tests/, not here.)
//
// LAYOUT
// ------
// The input is a CSR-style description of `nshell` generally contracted
// shells. For shell `s`:
//
//   exponents    alpha[ prim_offset[s] .. prim_offset[s+1] )      nprim[s]
//   coefficients coeff[ coeff_offset[s] .. coeff_offset[s+1] )    nprim*nctr
//   center       centers[3*s .. 3*s+3)
//
// Coefficients are contraction-major within a shell: contraction `ictr`
// occupies the contiguous run of `nprim[s]` values starting at
// `coeff_offset[s] + ictr * nprim[s]`. They are the coefficients of
// UNNORMALIZED primitives unless `already_normalized` is set.
//
// Basis functions are laid out shell-major, then contraction, then angular
// component -- matching ERKALE's ordering (see gto_shell.hpp).

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "gto_shell.hpp"

namespace Nukexc {

/**
 *  @brief Caller-supplied description of a Gaussian basis set.
 *
 *  Plain arrays owned by the caller; nothing is retained past the
 *  make_gto_basis() call that consumes it.
 */
struct GTOBasisInput {
  /// Number of shells.
  int nshell = 0;

  /// Angular momentum per shell. [nshell]
  const int *l = nullptr;

  /// Per-shell angular representation: 0 = Cartesian, 1 = spherical.
  /// May be null, in which case `default_type` applies to every shell.
  const int *spherical = nullptr;

  /// Primitives per shell. [nshell]
  const int *nprim = nullptr;

  /// Contractions per shell. May be null, meaning one contraction each.
  const int *nctr = nullptr;

  /// CSR offsets into `alpha`. [nshell + 1]
  const int *prim_offset = nullptr;

  /// CSR offsets into `coeff`. [nshell + 1]
  const int *coeff_offset = nullptr;

  /// Primitive exponents. [prim_offset[nshell]]
  const double *alpha = nullptr;

  /// Contraction coefficients, contraction-major per shell.
  /// [coeff_offset[nshell]]
  const double *coeff = nullptr;

  /// Shell centers, x/y/z interleaved. [3 * nshell]
  const double *centers = nullptr;

  /// Angular representation used where `spherical` is null.
  ShellType default_type = ShellType::Spherical;

  /// Set when the coefficients are already normalized and must be taken
  /// verbatim; otherwise ERKALE normalization is applied.
  bool already_normalized = false;

  /// Basis-function cutoff tolerance used for the per-shell screening radii.
  double cutoff_tolerance = 1e-10;
};

/**
 *  @brief A Gaussian basis set: an ordered list of generally contracted
 *         shells plus the basis-function offsets derived from them.
 */
template <typename F> class GTOBasisSet {
public:
  using shell_type = GTOShell<F>;

  GTOBasisSet() = default;

  explicit GTOBasisSet(std::vector<shell_type> shells)
      : shells_(std::move(shells)) {
    rebuild_offsets();
  }

  // -- shells -------------------------------------------------------------

  int nshells() const { return static_cast<int>(shells_.size()); }
  const shell_type &shell(int i) const { return shells_.at(static_cast<size_t>(i)); }
  const std::vector<shell_type> &shells() const { return shells_; }

  void add_shell(shell_type sh) {
    shells_.push_back(std::move(sh));
    rebuild_offsets();
  }

  // -- basis functions ----------------------------------------------------

  /// Total number of basis functions.
  int nbf() const { return nbf_; }

  /// Total number of Cartesian functions, i.e. what the evaluator produces
  /// before any spherical transform.
  int nbf_cart() const { return nbf_cart_; }

  /// Index of the first basis function of shell `i`. [nshells + 1]
  int shell_offset(int i) const { return offsets_.at(static_cast<size_t>(i)); }
  const std::vector<int> &shell_offsets() const { return offsets_; }

  /// Highest angular momentum present.
  int max_l() const {
    int m = 0;
    for (const auto &s : shells_)
      m = std::max(m, s.l());
    return m;
  }

  /// Largest number of basis functions on any one shell.
  int max_shell_size() const {
    int m = 0;
    for (const auto &s : shells_)
      m = std::max(m, s.size());
    return m;
  }

  /// Largest screening radius over all shells.
  double max_cutoff_radius() const {
    double m = 0.0;
    for (const auto &s : shells_)
      m = std::max(m, s.cutoff_radius());
    return m;
  }

  /// Reset the basis-function cutoff tolerance on every shell, recomputing
  /// the screening radii.
  void set_cutoff_tolerance(double tol) {
    for (auto &s : shells_)
      s.set_tolerance(tol);
  }

  /// Number of basis functions in a subset of shells.
  template <typename It> int nbf_subset(It first, It last) const {
    int n = 0;
    for (auto it = first; it != last; ++it)
      n += shells_.at(static_cast<size_t>(*it)).size();
    return n;
  }

private:
  void rebuild_offsets() {
    offsets_.assign(shells_.size() + 1, 0);
    nbf_cart_ = 0;
    for (size_t i = 0; i < shells_.size(); ++i) {
      offsets_[i + 1] = offsets_[i] + shells_[i].size();
      nbf_cart_ += shells_[i].cart_total();
    }
    nbf_ = offsets_.empty() ? 0 : offsets_.back();
  }

  std::vector<shell_type> shells_;
  std::vector<int> offsets_;
  int nbf_ = 0;
  int nbf_cart_ = 0;
};

namespace detail {

inline void require(bool cond, const std::string &what) {
  if (!cond)
    throw std::invalid_argument("make_gto_basis: " + what);
}

} // namespace detail

/**
 *  @brief Build a basis set from caller-supplied arrays.
 *
 *  Validates the CSR description eagerly and throws std::invalid_argument on
 *  anything inconsistent -- a malformed basis is a caller bug, and silently
 *  skipping shells would produce a wrong answer that looks plausible.
 */
template <typename F = double>
inline GTOBasisSet<F> make_gto_basis(const GTOBasisInput &in) {
  detail::require(in.nshell >= 0, "negative shell count");
  if (in.nshell == 0)
    return GTOBasisSet<F>{};

  detail::require(in.l != nullptr, "l array is null");
  detail::require(in.nprim != nullptr, "nprim array is null");
  detail::require(in.prim_offset != nullptr, "prim_offset array is null");
  detail::require(in.coeff_offset != nullptr, "coeff_offset array is null");
  detail::require(in.alpha != nullptr, "alpha array is null");
  detail::require(in.coeff != nullptr, "coeff array is null");
  detail::require(in.centers != nullptr, "centers array is null");
  detail::require(in.cutoff_tolerance > 0.0, "non-positive cutoff tolerance");

  const NormalizationConvention conv = in.already_normalized
                                           ? NormalizationConvention::None
                                           : NormalizationConvention::ERKALE;

  std::vector<GTOShell<F>> shells;
  shells.reserve(static_cast<size_t>(in.nshell));

  for (int s = 0; s < in.nshell; ++s) {
    const std::string at = "shell " + std::to_string(s) + ": ";

    const int np = in.nprim[s];
    const int nc = in.nctr ? in.nctr[s] : 1;
    detail::require(np > 0, at + "no primitives");
    detail::require(nc > 0, at + "no contractions");

    const int p0 = in.prim_offset[s];
    const int p1 = in.prim_offset[s + 1];
    detail::require(p0 >= 0 && p1 >= p0, at + "prim_offset is not increasing");
    detail::require(p1 - p0 == np,
                    at + "prim_offset spans " + std::to_string(p1 - p0) +
                        " exponents but nprim is " + std::to_string(np));

    const int c0 = in.coeff_offset[s];
    const int c1 = in.coeff_offset[s + 1];
    detail::require(c0 >= 0 && c1 >= c0, at + "coeff_offset is not increasing");
    detail::require(c1 - c0 == np * nc,
                    at + "coeff_offset spans " + std::to_string(c1 - c0) +
                        " coefficients but nprim*nctr is " +
                        std::to_string(np * nc));

    std::vector<F> alpha(in.alpha + p0, in.alpha + p1);
    std::vector<F> coeff(in.coeff + c0, in.coeff + c1);

    const std::array<double, 3> O = {in.centers[3 * s + 0],
                                     in.centers[3 * s + 1],
                                     in.centers[3 * s + 2]};

    const ShellType type =
        in.spherical ? (in.spherical[s] ? ShellType::Spherical
                                        : ShellType::Cartesian)
                     : in.default_type;

    shells.emplace_back(in.l[s], type, O, std::move(alpha), std::move(coeff),
                        nc, conv, in.cutoff_tolerance);
  }

  return GTOBasisSet<F>(std::move(shells));
}

/**
 *  @brief Fold segmented shells that share a center, angular momentum,
 *         representation and primitive exponents into general contractions.
 *
 *  Basis-set libraries usually list e.g. cc-pVTZ as repeated blocks over one
 *  exponent set. Merging them lets the evaluator compute each primitive
 *  exponential once and reuse it across contractions, which is the whole point
 *  of supporting general contractions.
 *
 *  Shell order is otherwise preserved: each merged shell keeps the position of
 *  its first constituent.
 */
template <typename F>
inline GTOBasisSet<F> merge_general_contractions(const GTOBasisSet<F> &in) {
  std::vector<GTOShell<F>> out;
  out.reserve(static_cast<size_t>(in.nshells()));

  for (int s = 0; s < in.nshells(); ++s) {
    const auto &sh = in.shell(s);
    auto it = std::find_if(out.begin(), out.end(), [&](const GTOShell<F> &o) {
      return o.same_primitives(sh);
    });
    if (it != out.end())
      it->merge_contraction(sh);
    else
      out.push_back(sh);
  }

  return GTOBasisSet<F>(std::move(out));
}

} // namespace Nukexc
