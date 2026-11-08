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

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomic_properties.hpp"
#include "molecule.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "basis_evaluator.hpp"
#include "radial_shell.hpp"
#include "spherical_harmonics.hpp"

namespace Nukexc {

struct STOBasisSet {
  Kokkos::View<int *> n;
  Kokkos::View<int *> l;
  Kokkos::View<int *> m;
  Kokkos::View<double *> norm;
  Kokkos::View<double *> zeta;
  Kokkos::View<Point *> O;
  Kokkos::View<double *> cutoff_radii;

  size_t nbf() const { return O.extent(0); };
};

// ============================================================
//  Manual STOBasisSet
// ============================================================
//
// Constructs an STOBasisSet directly from a plain list rather than
// reading from disk.  This decouples integration tests from the
// basis-file format and makes the quantum numbers explicit.
//
// Each entry is { n, l, m, zeta, ox, oy, oz }.

struct STOFunc {
  int n, l, m;
  double zeta;
  double ox, oy, oz;
};
inline double compute_cutoff(STOFunc f, const double norm, const double cutoff_tol) {
  // Preconditions
  KOKKOS_ASSERT(f.n >= 1);
  KOKKOS_ASSERT(f.zeta > 0.0);
  KOKKOS_ASSERT(cutoff_tol > 0.0);

  // If the function is already everywhere below tolerance, cutoff is at origin
  if (norm <= cutoff_tol)
    return 0.0;

  const double ln_tol = Kokkos::log(cutoff_tol);
  const double ln_norm = Kokkos::log(norm);

  // g(r) = ln_norm + (n-1)*ln(r) - zeta*r - ln_tol = 0
  // g'(r) = (n-1)/r - zeta
  // The unique root we want is on the far side of the peak, i.e. r > peak,
  // where g'(r) < 0 so the function is strictly decreasing.
  const double peak = (f.n > 1) ? (f.n - 1) / f.zeta : 0.0;

  // Initial guess: asymptotically correct when the log(r) term is negligible.
  // Guaranteed to be >= peak for reasonable inputs (norm >> tol).
  double r = Kokkos::max(peak + (ln_norm - ln_tol) / f.zeta, peak + 1e-6);

  // Safe interval: we stay in (peak, +inf) where g is monotone decreasing,
  // so there is exactly one root and Newton steps are well-defined.
  // We track an upper bound via the interval [r_lo, r_hi] and fall back to
  // bisection whenever a Newton step would leave (peak, r_hi].
  double r_lo = peak + 1e-10;
  double r_hi = r;

  // Ensure r_hi is actually above the root (g(r_hi) <= 0)
  // If not, double r_hi until it is.
  for (int k = 0; k < 64; ++k) {
    double g_hi =
        ln_norm + (f.n - 1) * Kokkos::log(r_hi) - f.zeta * r_hi - ln_tol;
    if (g_hi <= 0.0)
      break;
    r_hi *= 2.0;
  }
  r = r_hi;

  for (int k = 0; k < 50; ++k) {
    double g = ln_norm + (f.n - 1) * Kokkos::log(r) - f.zeta * r - ln_tol;

    // Update bracket
    if (g > 0.0)
      r_lo = r;
    else
      r_hi = r;

    double g_prime = (f.n - 1) / r - f.zeta;

    // g_prime < 0 everywhere in (peak, +inf); if somehow we're at the peak
    // itself, fall back to bisection immediately.
    double r_next;
    if (g_prime >= 0.0) {
      r_next = 0.5 * (r_lo + r_hi); // bisection fallback
    } else {
      r_next = r - g / g_prime; // Newton step
      // Reject step if it leaves the safe interval
      if (r_next <= r_lo || r_next >= r_hi)
        r_next = 0.5 * (r_lo + r_hi); // bisection fallback
    }

    double delta = Kokkos::abs(r_next - r);
    r = r_next;

    if (delta < 1e-10 * r)
      break;

#ifndef NDEBUG
    // In debug builds, assert we haven't diverged
    KOKKOS_ASSERT(r > 0.0);
#endif
  }

#ifndef NDEBUG
  {
    double g_final = ln_norm + (f.n - 1) * Kokkos::log(r) - f.zeta * r - ln_tol;
    KOKKOS_ASSERT(Kokkos::abs(g_final) < 1e-6);
  }
#endif
  return r;
}

// ============================================================
//  Caller-supplied basis-set input
// ============================================================
//
// NuKEXC does not read basis-set files. The calling program owns basis-set
// input -- file formats, library paths, element tables -- and hands the result
// over as plain arrays through STOBasisInput, which any C-ABI caller can fill.
// (The Gaussian- and ADF-format readers that used to live here now sit in
// tests/sto_basis_reader.hpp, where they serve the vendored test data under
// input/.)
//
// Each entry is one Slater-type basis function
//
//     chi(r) = norm * r^(n-1) * exp(-zeta r) * S_lm(x, y, z)
//
// with S_lm the real solid harmonic of spherical_harmonics.hpp.

/// Caller-supplied description of an STO basis set.
struct STOBasisInput {
  /// Number of basis functions.
  int nbf = 0;

  /// Principal quantum number per function. [nbf]
  const int *n = nullptr;
  /// Angular momentum per function. [nbf]
  const int *l = nullptr;
  /// Magnetic quantum number per function, -l .. +l. [nbf]
  const int *m = nullptr;
  /// Slater exponents. [nbf]
  const double *zeta = nullptr;
  /// Function centers, x/y/z interleaved. [3 * nbf]
  const double *centers = nullptr;

  /// Optional explicit normalization constants. [nbf]
  /// When null, the Slater norm (2 zeta)^(n + 1/2) / sqrt((2n)!) is used.
  const double *norm = nullptr;

  /// Amplitude below which a function is treated as zero; sets the per-function
  /// screening radius handed to the ArborX neighbour lists.
  double cutoff_tolerance = 1e-10;
};

/**
 *  @brief Build an STOBasisSet from caller-supplied arrays.
 *
 *  Validates eagerly and throws std::invalid_argument on malformed input: a
 *  bad basis is a caller bug, and quietly dropping functions would produce a
 *  wrong answer that still looks like a converged calculation.
 */
inline STOBasisSet make_sto_basis(const STOBasisInput &in) {
  auto require = [](bool cond, const std::string &what) {
    if (!cond)
      throw std::invalid_argument("make_sto_basis: " + what);
  };

  require(in.nbf >= 0, "negative basis-function count");
  if (in.nbf == 0)
    return STOBasisSet{};

  require(in.n != nullptr, "n array is null");
  require(in.l != nullptr, "l array is null");
  require(in.m != nullptr, "m array is null");
  require(in.zeta != nullptr, "zeta array is null");
  require(in.centers != nullptr, "centers array is null");
  require(in.cutoff_tolerance > 0.0, "non-positive cutoff tolerance");

  const size_t nbf = static_cast<size_t>(in.nbf);

  STOBasisSet basis;
  basis.n = Kokkos::View<int *>("n", nbf);
  basis.l = Kokkos::View<int *>("l", nbf);
  basis.m = Kokkos::View<int *>("m", nbf);
  basis.zeta = Kokkos::View<double *>("zeta", nbf);
  basis.norm = Kokkos::View<double *>("coeff", nbf);
  basis.O = Kokkos::View<Point *>(" centers ", nbf);
  basis.cutoff_radii = Kokkos::View<double *>("cutoff radii", nbf);

  auto n_h = Kokkos::create_mirror_view(basis.n);
  auto l_h = Kokkos::create_mirror_view(basis.l);
  auto m_h = Kokkos::create_mirror_view(basis.m);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta);
  auto norm_h = Kokkos::create_mirror_view(basis.norm);
  auto O_h = Kokkos::create_mirror_view(basis.O);
  auto cutoff_radii_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  for (size_t i = 0; i < nbf; ++i) {
    const int ni = in.n[i];
    const int li = in.l[i];
    const int mi = in.m[i];
    const double zi = in.zeta[i];
    const std::string at = "function " + std::to_string(i) + ": ";

    require(ni >= 1, at + "principal quantum number must be >= 1");
    require(li >= 0, at + "negative angular momentum");
    require(li < ni, at + "l must be < n");
    require(mi >= -li && mi <= li, at + "|m| > l");
    require(zi > 0.0, at + "non-positive exponent");

    // Slater normalization, matching what the ADF/Thakkar readers produced:
    //   N = (2 zeta)^(n + 1/2) / sqrt((2n)!)
    const double norm =
        in.norm ? in.norm[i]
                : std::pow(2.0 * zi, ni + 0.5) /
                      std::sqrt(static_cast<double>(factorial(2 * ni)));
    require(norm > 0.0, at + "non-positive normalization constant");

    n_h(i) = ni;
    l_h(i) = li;
    m_h(i) = mi;
    zeta_h(i) = zi;
    norm_h(i) = norm;
    O_h(i)[0] = in.centers[3 * i + 0];
    O_h(i)[1] = in.centers[3 * i + 1];
    O_h(i)[2] = in.centers[3 * i + 2];
    cutoff_radii_h(i) =
        compute_cutoff(STOFunc{ni, li, mi, zi, 0.0, 0.0, 0.0}, norm,
                       in.cutoff_tolerance);
  }

  Kokkos::deep_copy(basis.n, n_h);
  Kokkos::deep_copy(basis.l, l_h);
  Kokkos::deep_copy(basis.m, m_h);
  Kokkos::deep_copy(basis.zeta, zeta_h);
  Kokkos::deep_copy(basis.norm, norm_h);
  Kokkos::deep_copy(basis.O, O_h);
  Kokkos::deep_copy(basis.cutoff_radii, cutoff_radii_h);

  return basis;
}

/// Convenience wrapper over make_sto_basis for C++ callers holding a list of
/// STOFunc. Kept because it reads better in tests than filling five arrays.
inline STOBasisSet make_manual_basis(const std::vector<STOFunc> &funcs,
                                     double cutoff_tol = 1e-10) {
  const size_t nbf = funcs.size();
  std::vector<int> n(nbf), l(nbf), m(nbf);
  std::vector<double> zeta(nbf), centers(3 * nbf);
  for (size_t i = 0; i < nbf; ++i) {
    n[i] = funcs[i].n;
    l[i] = funcs[i].l;
    m[i] = funcs[i].m;
    zeta[i] = funcs[i].zeta;
    centers[3 * i + 0] = funcs[i].ox;
    centers[3 * i + 1] = funcs[i].oy;
    centers[3 * i + 2] = funcs[i].oz;
  }

  STOBasisInput in;
  in.nbf = static_cast<int>(nbf);
  in.n = n.data();
  in.l = l.data();
  in.m = m.data();
  in.zeta = zeta.data();
  in.centers = centers.data();
  in.cutoff_tolerance = cutoff_tol;
  return make_sto_basis(in);
}

struct ShellParams {
  int l, m;
  int k; // radial power exponent, precomputed as n - l - 1
  double zeta, norm, ox, oy, oz;
};

KOKKOS_INLINE_FUNCTION
void basis_eval(const STOBasisSet basis, const int basis_idx, const double x,
                const double y, const double z, double &val) {

  const int l_val = basis.l(basis_idx);
  const int m_val = basis.m(basis_idx);

  const double dx = x - basis.O(basis_idx)[0];
  const double dy = y - basis.O(basis_idx)[1];
  const double dz = z - basis.O(basis_idx)[2];

  double radial_part;
  {
    // radial part of the shell
    // radial_part = R_nl(r) = r^(n-1) * C_nl * exp(-⍺ * r))
    const int n_val = basis.n(basis_idx);
    const double zeta = basis.zeta(basis_idx);
    const double norm = basis.norm(basis_idx);
    const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

    radial_part = norm * int_pow(r, n_val - l_val - 1) * Kokkos::exp(-zeta * r);
  }

  // Angular part of the shell
  // https://en.wikipedia.org/wiki/Spherical_harmonics
  double angular_part;
  real_solid_harmonic_cart_precomputed(l_val, m_val, dx, dy, dz, angular_part);

  val = radial_part * angular_part;
}

KOKKOS_INLINE_FUNCTION
ShellParams load_shell(const STOBasisSet &basis, int basis_idx) {
  const int l = basis.l(basis_idx);
  return {l,
          basis.m(basis_idx),
          basis.n(basis_idx) - l - 1, // k = n - l - 1
          basis.zeta(basis_idx),
          basis.norm(basis_idx),
          basis.O(basis_idx)[0],
          basis.O(basis_idx)[1],
          basis.O(basis_idx)[2]};
}

// Non-inlined boundary around the angular polynomial switch. The high-l cases
// of real_solid_harmonic_cart_precomputed materialize up to ~10 degree-7
// monomial temporaries; keeping them in this callee's own frame stops them from
// widening the live-register set of the pairwise integral loops that call
// basis_eval_fast. (The generated harmonics header is left untouched.)
NUKEXC_NOINLINE_FUNCTION double eval_solid_harmonic(const int l, const int m,
                                                    const double x,
                                                    const double y,
                                                    const double z) {
  double angular_part;
  real_solid_harmonic_cart_precomputed(l, m, x, y, z, angular_part);
  return angular_part;
}

KOKKOS_INLINE_FUNCTION
double basis_eval_fast(const ShellParams &sh, double x, double y, double z) {
  x -= sh.ox;
  y -= sh.oy;
  z -= sh.oz;
  double radial;
  {

    const double r = Kokkos::sqrt(x * x + y * y + z * z);
    radial = (sh.k == 0)
                 ? sh.norm * Kokkos::exp(-sh.zeta * r)
                 : sh.norm * int_pow(r, sh.k) * Kokkos::exp(-sh.zeta * r);
  }
  const double angular_part = eval_solid_harmonic(sh.l, sh.m, x, y, z);
  return radial * angular_part;
} // namespace Nukexc

template <typename PointsView, typename ValuesView>
void fill_collocation(
    ExecSpace &space, const STOBasisSet &basis, PointsView collocation_points,
    ValuesView collocation_values) // pre-allocated, written in place
{
  int N = basis.nbf();
  int G = collocation_points.extent(0);
  Kokkos::parallel_for(
      "Fill collocation",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0}, {N, G}),
      KOKKOS_LAMBDA(int i, int j) {
        // same math as before, written directly into col(i,j)
        const int n_val = basis.n(i);
        const int l_val = basis.l(i);
        const int m_val = basis.m(i);
        const double norm = basis.norm(i);
        const double zeta = basis.zeta(i);

        // radial part of the shell
        // radial_part = R_nl(r) = r^(n-1) * C_nl * exp(-⍺ * r))
        double dx = collocation_points(j)[0] - basis.O(i)[0];
        double dy = collocation_points(j)[1] - basis.O(i)[1];
        double dz = collocation_points(j)[2] - basis.O(i)[2];

        double radial_part;
        {
          double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
          radial_part =
              norm * int_pow(r, n_val - l_val - 1) * Kokkos::exp(-zeta * r);
        }
        // Angular part of the shell
        // https://en.wikipedia.org/wiki/Spherical_harmonics
        double angular_part;
        real_solid_harmonic_cart_precomputed(l_val, m_val, dx, dy, dz,
                                             angular_part);

        collocation_values(i, j) = radial_part * angular_part;
      });
}

template <typename PointsView, typename ValuesView>
void fill_grad_collocation(ExecSpace &space, const STOBasisSet &basis_set,
                           PointsView &collocation_points,
                           ValuesView &collocation_gx,
                           ValuesView &collocation_gy,
                           ValuesView &collocation_gz) {

  size_t col_points = collocation_points.extent(0);
  size_t nbasis_functions = basis_set.nbf();

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> md_policy(
      space, {0, 0}, {nbasis_functions, col_points});

  Kokkos::parallel_for(
      "Fill collocation grad", md_policy,
      KOKKOS_LAMBDA(const int &i, const int &g) {
        const int n_val = basis_set.n(i);
        const int l_val = basis_set.l(i);
        const int m_val = basis_set.m(i);
        const double norm = basis_set.norm(i);
        const double zeta = basis_set.zeta(i);

        double dx = collocation_points(g)[0] - basis_set.O(i)[0];
        double dy = collocation_points(g)[1] - basis_set.O(i)[1];
        double dz = collocation_points(g)[2] - basis_set.O(i)[2];
        double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

        // Angular part
        double S_val;
        S_val = real_solid_harmonic_cart(l_val, m_val, dx, dy, dz);

        double dS_dx;
        double dS_dy;
        double dS_dz;

        grad_real_solid_harmonic_cart(l_val, m_val, dx, dy, dz, dS_dx, dS_dy,
                                      dS_dz);

        // Radial part
        double pow_term = safe_pow(r, n_val - l_val - 1);
        double exp_term = Kokkos::exp(-zeta * r);
        double R_pre = norm * pow_term * exp_term;

        double dR_dr = ((n_val - l_val - 1) / r - zeta) * R_pre;
        double common_R = dR_dr / r;

        collocation_gx(i, g) = R_pre * dS_dx + S_val * (dx * common_R);
        collocation_gy(i, g) = R_pre * dS_dy + S_val * (dy * common_R);
        collocation_gz(i, g) = R_pre * dS_dz + S_val * (dz * common_R);
      });
}

struct VG {
  double val; // Value
  double dx;  // d/dx
  double dy;  // d/dy
  double dz;  // d/dz
};

struct ScratchBasisParams {
  double zeta, norm;
  Point O;
  int n, l, m;
};

KOKKOS_INLINE_FUNCTION
void basis_eval_with_grad(const ScratchBasisParams &basis, const Point &p,
                          double &val, double &gradx, double &grady,
                          double &gradz) {

  const int n_val = basis.n;
  const int l_val = basis.l;
  const int m_val = basis.m;
  const double norm = basis.norm;
  const double zeta = basis.zeta;
  const Point origin = basis.O;

  double dx = p[0] - origin[0];
  double dy = p[1] - origin[1];
  double dz = p[2] - origin[2];
  double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

  // Angular part
  double S_val;
  double dS_dx;
  double dS_dy;
  double dS_dz;

  // Completely precomputed coefficients
  real_solid_harmonic_cart_and_grad_precomputed(l_val, m_val, dx, dy, dz, S_val,
                                                dS_dx, dS_dy, dS_dz);

  // Radial part
  double pow_term = int_pow(r, n_val - l_val - 1);
  double exp_term = Kokkos::exp(-zeta * r);
  double R_pre = norm * pow_term * exp_term;

  double dR_dr = ((n_val - l_val - 1) / r - zeta) * R_pre;
  double common_R = dR_dr / r;

  val = R_pre * S_val;
  gradx = R_pre * dS_dx + S_val * (dx * common_R);
  grady = R_pre * dS_dy + S_val * (dy * common_R);
  gradz = R_pre * dS_dz + S_val * (dz * common_R);
}

// ============================================================================
//  BasisEvaluator adapter
// ============================================================================
//
// One basis function per block: each primitive STO is independent, so there
// is no shared work to amortize across a block.

/**
 *  @brief Group an STOBasisSet's per-function layout into shells.
 *
 *  STOBasisSet stores one entry per (n, l, m) function; a shell is the 2l+1
 *  of them that share (n, l, zeta, centre). Every reader in the project
 *  emits those consecutively with m running -l .. +l -- load_adf_basis,
 *  load_thakkar_primitives and the contracted reader all expand with
 *  `for (int m = -l; m <= l; ++m)` -- so the grouping preserves the basis
 *  FUNCTION ORDER exactly, and matrices from the shell-based evaluator are
 *  comparable element by element with everything recorded before it.
 *
 *  That is a property of the readers rather than of the format, so it is
 *  checked here rather than assumed: a group that is not exactly 2l+1
 *  consecutive functions with ascending m throws, because the alternative
 *  is a silently permuted basis.
 */
inline SlaterRadial make_slater_radial(const STOBasisSet &b) {
  const int nbf = static_cast<int>(b.nbf());
  if (nbf == 0)
    throw std::invalid_argument("make_slater_radial: empty basis");

  auto n_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.n);
  auto l_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.l);
  auto m_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.m);
  auto z_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.zeta);
  auto norm_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.norm);
  auto o_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.O);
  auto cut_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.cutoff_radii);

  std::vector<int> sh_l, sh_n;
  std::vector<Point> sh_center;
  std::vector<double> sh_radius, sh_zeta, sh_coeff;

  int i = 0;
  while (i < nbf) {
    const int l = l_h(i);
    const int nf = 2 * l + 1;
    if (i + nf > nbf)
      throw std::runtime_error(
          "make_slater_radial: basis function " + std::to_string(i) +
          " starts an l = " + std::to_string(l) + " shell with only " +
          std::to_string(nbf - i) + " functions left");
    double radius = 0.0;
    for (int k = 0; k < nf; ++k) {
      const int j = i + k;
      const bool same = l_h(j) == l && n_h(j) == n_h(i) &&
                        z_h(j) == z_h(i) && norm_h(j) == norm_h(i) &&
                        o_h(j)[0] == o_h(i)[0] && o_h(j)[1] == o_h(i)[1] &&
                        o_h(j)[2] == o_h(i)[2];
      if (!same || m_h(j) != -l + k)
        throw std::runtime_error(
            "make_slater_radial: functions " + std::to_string(i) + ".." +
            std::to_string(i + nf - 1) +
            " do not form one shell with m = -l..+l in order; this basis is "
            "not laid out shell by shell and grouping it would permute it");
      radius = Kokkos::max(radius, cut_h(j));
    }
    sh_l.push_back(l);
    sh_n.push_back(n_h(i));
    sh_zeta.push_back(z_h(i));
    sh_coeff.push_back(norm_h(i));
    sh_center.push_back(o_h(i));
    sh_radius.push_back(radius);
    i += nf;
  }

  const int nshell = static_cast<int>(sh_l.size());
  SlaterRadial rad;
  ShellTable &t = rad.sh;
  t.nshell = nshell;
  t.nbf = nbf;
  t.l = Kokkos::View<int *>("sto shell l", nshell);
  t.first_bf = Kokkos::View<int *>("sto shell first bf", nshell + 1);
  t.prim_offset = Kokkos::View<int *>("sto shell prim offset", nshell + 1);
  t.center = Kokkos::View<Point *>("sto shell center", nshell);
  t.radius = Kokkos::View<double *>("sto shell radius", nshell);
  rad.zeta = Kokkos::View<double *>("sto zeta", nshell);
  rad.coeff = Kokkos::View<double *>("sto coeff", nshell);
  rad.n = Kokkos::View<int *>("sto n", nshell);

  auto tl = Kokkos::create_mirror_view(t.l);
  auto tf = Kokkos::create_mirror_view(t.first_bf);
  auto tp = Kokkos::create_mirror_view(t.prim_offset);
  auto tc = Kokkos::create_mirror_view(t.center);
  auto tr = Kokkos::create_mirror_view(t.radius);
  auto rz = Kokkos::create_mirror_view(rad.zeta);
  auto rc = Kokkos::create_mirror_view(rad.coeff);
  auto rn = Kokkos::create_mirror_view(rad.n);

  int bf = 0;
  for (int s = 0; s < nshell; ++s) {
    tl(s) = sh_l[s];
    tf(s) = bf;
    tp(s) = s; // one primitive per shell
    tc(s) = sh_center[s];
    tr(s) = sh_radius[s];
    rz(s) = sh_zeta[s];
    rc(s) = sh_coeff[s];
    rn(s) = sh_n[s];
    bf += 2 * sh_l[s] + 1;
  }
  tf(nshell) = bf;
  tp(nshell) = nshell;

  Kokkos::deep_copy(t.l, tl);
  Kokkos::deep_copy(t.first_bf, tf);
  Kokkos::deep_copy(t.prim_offset, tp);
  Kokkos::deep_copy(t.center, tc);
  Kokkos::deep_copy(t.radius, tr);
  Kokkos::deep_copy(rad.zeta, rz);
  Kokkos::deep_copy(rad.coeff, rc);
  Kokkos::deep_copy(rad.n, rn);
  return rad;
}

/**
 *  @brief The primitive Slater evaluator: shells of 2l+1, radial once each.
 *
 *  A thin layer over SolidHarmonicEvaluator<SlaterRadial> that keeps the
 *  constructor every call site uses and the STOBasisSet accessor some tests
 *  want. The evaluation itself is the generic one -- which is the point:
 *  this family differs from the Gaussian one in R(r) alone.
 *
 *  It used to hand out one basis FUNCTION per block, which meant evaluating
 *  r^(n-l-1) e^(-zeta r) once per (function, point) rather than once per
 *  (shell, point): 2l+1 times over for a p, d or f shell.
 */
class STOPrimitiveEvaluator : public SolidHarmonicEvaluator<SlaterRadial> {
public:
  using base = SolidHarmonicEvaluator<SlaterRadial>;
  using block_params = base::block_params;

  STOPrimitiveEvaluator() = default;
  explicit STOPrimitiveEvaluator(STOBasisSet b)
      : base(make_slater_radial(b)), b_(std::move(b)) {}

  /// The per-function basis this was built from, for callers that still
  /// index by function -- the dense collocation fills, mostly.
  const STOBasisSet &basis() const { return b_; }

private:
  STOBasisSet b_;
};

// The per-function Slater evaluator that used to live here is gone: it gave
// out one basis FUNCTION per block, so a p, d or f shell evaluated its
// radial 2l+1 times per point instead of once. It survives as a reference
// implementation in tests/test_radial_shell.cxx, where it pins the
// shell-based one element by element -- the Slater path has no external
// oracle of that kind otherwise, since the ERKALE comparisons use Gaussian
// bases.

static_assert(BasisEvaluator<STOPrimitiveEvaluator>);

} // namespace Nukexc
