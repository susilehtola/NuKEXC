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
//  Contracted Slater-type orbital basis sets
// ============================================================================
//
// The second of NuKEXC's two STO paths.
//
//   * stobasis.hpp          -- PRIMITIVE sets: one Slater primitive per basis
//                              function, e.g. the ADF/ZORA uncontracted sets.
//   * this header           -- CONTRACTED sets: one basis function is a fixed
//                              linear combination of primitives, e.g. Thakkar's
//                              Roothaan-Hartree-Fock orbitals.
//
// The distinction is not cosmetic. A contracted STO shares one angular factor
// across primitives that carry *different principal quantum numbers*, which the
// primitive layout cannot express. A Thakkar oxygen 1s orbital is
//
//     2S   24.975698   -0.0004756
//     1S   12.554759   -0.1852813
//     2S   10.357096   -0.1195096
//     1S    7.071190   -0.6980503
//     ...
//
// i.e. n alternates between 1 and 2 down one contraction. Loading such a file
// into the primitive path drops the coefficients and yields the uncontracted
// parent set -- a legitimate basis, but not the one the file describes.
//
// FUNCTIONAL FORM
// ---------------
//     chi(r) = S_lm(x, y, z) * sum_p c_p N_p r^(n_p - l - 1) exp(-zeta_p r)
//
// with S_lm the real solid harmonic r^l Y_lm of spherical_harmonics.hpp and
// N_n(zeta) = (2 zeta)^(n + 1/2) / sqrt((2n)!) the unit-norm Slater primitive.
// Pulling S_lm out front means the stored radial powers are n_p - l - 1, which
// is exactly the convention the primitive path already uses -- so the device
// evaluators below mirror basis_eval_fast/basis_eval_with_grad.
//
// STORAGE
// -------
// Struct of Views, CSR over primitives: basis function `i` owns the primitive
// range [prim_offset(i), prim_offset(i+1)). This mirrors the general
// contraction layout of gto_basisset.hpp.

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "spherical_harmonics.hpp"
#include "stobasis.hpp"

namespace Nukexc {

/// Device-resident contracted STO basis set.
struct STOContractedBasisSet {
  // -- per basis function, length nbf -------------------------------------
  Kokkos::View<int *> l;
  Kokkos::View<int *> m;
  Kokkos::View<Point *> O;
  Kokkos::View<double *> cutoff_radii;
  /// CSR row pointers into the primitive arrays. Length nbf + 1.
  Kokkos::View<int *> prim_offset;

  // -- per primitive, length nprim ----------------------------------------
  Kokkos::View<int *> n;
  Kokkos::View<double *> zeta;
  /// Contraction coefficient with the primitive normalization folded in,
  /// i.e. c_p * N_p. The evaluators use it directly.
  Kokkos::View<double *> coeff;

  size_t nbf() const { return l.extent(0); }
  size_t nprim() const { return zeta.extent(0); }
};

/// Caller-supplied description of a contracted STO basis set.
///
/// As everywhere else in NuKEXC, the calling program owns basis-set input and
/// hands it over as plain arrays; nothing here reads files.
struct STOContractedBasisInput {
  /// Number of basis functions.
  int nbf = 0;

  /// Angular momentum per function. [nbf]
  const int *l = nullptr;
  /// Magnetic quantum number per function, -l .. +l. [nbf]
  const int *m = nullptr;
  /// Function centers, x/y/z interleaved. [3 * nbf]
  const double *centers = nullptr;
  /// CSR offsets into the primitive arrays. [nbf + 1]
  const int *prim_offset = nullptr;

  /// Principal quantum number per primitive. [prim_offset[nbf]]
  const int *n = nullptr;
  /// Slater exponent per primitive. [prim_offset[nbf]]
  const double *zeta = nullptr;
  /// Contraction coefficient per primitive. [prim_offset[nbf]]
  const double *coeff = nullptr;

  /// Whether `coeff` multiplies unit-normalized primitives (the usual
  /// tabulated convention, and what Thakkar's files give) or raw
  /// r^(n-1) exp(-zeta r).
  bool coefficients_for_normalized_primitives = true;

  /// Rescale every contracted function to unit self-overlap. Tabulated
  /// coefficient sets are normalized already, so this is a safety net that
  /// costs one O(nprim^2) pass per function; turn it off to take the
  /// coefficients verbatim.
  bool normalize = true;

  /// Amplitude below which a function is treated as zero; sets the screening
  /// radius handed to the ArborX neighbour lists.
  double cutoff_tolerance = 1e-10;
};

namespace detail {

/// Self-overlap of a contracted STO whose angular part is a unit-normalized
/// real spherical harmonic:
///
///     <chi|chi> = sum_pq a_p a_q (n_p + n_q)! / (zeta_p + zeta_q)^(n_p+n_q+1)
///
/// with a_p = c_p N_p the folded coefficients.
inline double sto_contracted_self_overlap(int np, const int *n,
                                          const double *zeta,
                                          const double *a) {
  double acc = 0.0;
  for (int p = 0; p < np; ++p)
    for (int q = 0; q < np; ++q) {
      const int npq = n[p] + n[q];
      acc += a[p] * a[q] * static_cast<double>(factorial(npq)) /
             std::pow(zeta[p] + zeta[q], npq + 1);
    }
  return acc;
}

/// Radial envelope sum_p |a_p| r^(n_p - 1) exp(-zeta_p r), used for screening.
inline double sto_contracted_envelope(int np, const int *n, const double *zeta,
                                      const double *a, double r) {
  double acc = 0.0;
  for (int p = 0; p < np; ++p)
    acc += std::abs(a[p]) * std::pow(r, n[p] - 1) * std::exp(-zeta[p] * r);
  return acc;
}

/// Smallest r beyond which the envelope stays below `tol`. The envelope is
/// eventually monotonically decreasing, so bracket by doubling and bisect --
/// same approach as GTOShell::compute_cutoff_radius.
inline double sto_contracted_cutoff(int np, const int *n, const double *zeta,
                                    const double *a, double tol) {
  // Start past the outermost primitive peak, at r = (n-1)/zeta.
  double lo = 0.0;
  for (int p = 0; p < np; ++p)
    if (n[p] > 1)
      lo = std::max(lo, (n[p] - 1) / zeta[p]);

  if (sto_contracted_envelope(np, n, zeta, a, lo) <= tol)
    return lo;

  double hi = std::max(lo, 1.0);
  int doublings = 0;
  while (sto_contracted_envelope(np, n, zeta, a, hi) > tol) {
    lo = hi;
    hi *= 2.0;
    if (++doublings > 200)
      throw std::runtime_error(
          "make_sto_contracted_basis: cutoff radius search failed to bracket");
  }

  for (int it = 0; it < 60 && (hi - lo) > 1e-12 * std::max(1.0, hi); ++it) {
    const double mid = 0.5 * (lo + hi);
    if (sto_contracted_envelope(np, n, zeta, a, mid) > tol)
      lo = mid;
    else
      hi = mid;
  }
  return hi;
}

} // namespace detail

/**
 *  @brief Build a contracted STO basis set from caller-supplied arrays.
 *
 *  Validates eagerly and throws std::invalid_argument on malformed input.
 */
inline STOContractedBasisSet
make_sto_contracted_basis(const STOContractedBasisInput &in) {
  auto require = [](bool cond, const std::string &what) {
    if (!cond)
      throw std::invalid_argument("make_sto_contracted_basis: " + what);
  };

  require(in.nbf >= 0, "negative basis-function count");
  if (in.nbf == 0)
    return STOContractedBasisSet{};

  require(in.l != nullptr, "l array is null");
  require(in.m != nullptr, "m array is null");
  require(in.centers != nullptr, "centers array is null");
  require(in.prim_offset != nullptr, "prim_offset array is null");
  require(in.n != nullptr, "n array is null");
  require(in.zeta != nullptr, "zeta array is null");
  require(in.coeff != nullptr, "coeff array is null");
  require(in.cutoff_tolerance > 0.0, "non-positive cutoff tolerance");
  require(in.prim_offset[0] == 0, "prim_offset must start at 0");

  const int nbf = in.nbf;
  const int nprim_total = in.prim_offset[nbf];
  require(nprim_total > 0, "basis set has no primitives");

  // Host staging.
  std::vector<int> h_l(nbf), h_m(nbf), h_off(nbf + 1);
  std::vector<Point> h_O(nbf);
  std::vector<double> h_cut(nbf);
  std::vector<int> h_n(nprim_total);
  std::vector<double> h_zeta(nprim_total), h_a(nprim_total);

  for (int i = 0; i <= nbf; ++i)
    h_off[i] = in.prim_offset[i];

  for (int i = 0; i < nbf; ++i) {
    const std::string at = "function " + std::to_string(i) + ": ";
    const int p0 = in.prim_offset[i];
    const int p1 = in.prim_offset[i + 1];
    require(p1 > p0, at + "empty contraction");
    require(p0 >= 0 && p1 <= nprim_total, at + "prim_offset out of range");

    const int li = in.l[i];
    const int mi = in.m[i];
    require(li >= 0, at + "negative angular momentum");
    require(mi >= -li && mi <= li, at + "|m| > l");

    const int np = p1 - p0;

    // Fold the primitive normalization into the coefficient, so the device
    // evaluator only ever multiplies and accumulates.
    for (int p = p0; p < p1; ++p) {
      const int npn = in.n[p];
      const double z = in.zeta[p];
      require(npn >= 1, at + "principal quantum number must be >= 1");
      require(npn > li, at + "primitive n must be > l");
      require(z > 0.0, at + "non-positive exponent");

      const double Np =
          in.coefficients_for_normalized_primitives
              ? std::pow(2.0 * z, npn + 0.5) /
                    std::sqrt(static_cast<double>(factorial(2 * npn)))
              : 1.0;
      h_n[p] = npn;
      h_zeta[p] = z;
      h_a[p] = in.coeff[p] * Np;
    }

    if (in.normalize) {
      const double s = detail::sto_contracted_self_overlap(
          np, h_n.data() + p0, h_zeta.data() + p0, h_a.data() + p0);
      require(s > 0.0, at + "non-positive self-overlap; check the coefficients");
      const double scale = 1.0 / std::sqrt(s);
      for (int p = p0; p < p1; ++p)
        h_a[p] *= scale;
    }

    h_l[i] = li;
    h_m[i] = mi;
    h_O[i][0] = in.centers[3 * i + 0];
    h_O[i][1] = in.centers[3 * i + 1];
    h_O[i][2] = in.centers[3 * i + 2];
    h_cut[i] = detail::sto_contracted_cutoff(np, h_n.data() + p0,
                                             h_zeta.data() + p0,
                                             h_a.data() + p0,
                                             in.cutoff_tolerance);
  }

  STOContractedBasisSet basis;
  basis.l = Kokkos::View<int *>("l", nbf);
  basis.m = Kokkos::View<int *>("m", nbf);
  basis.O = Kokkos::View<Point *>("centers", nbf);
  basis.cutoff_radii = Kokkos::View<double *>("cutoff radii", nbf);
  basis.prim_offset = Kokkos::View<int *>("prim offset", nbf + 1);
  basis.n = Kokkos::View<int *>("n", nprim_total);
  basis.zeta = Kokkos::View<double *>("zeta", nprim_total);
  basis.coeff = Kokkos::View<double *>("coeff", nprim_total);

  auto copy_in = [](auto view, const auto &src) {
    using view_t = decltype(view);
    using value_t = typename view_t::non_const_value_type;
    Kokkos::View<const value_t *, HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h(src.data(), src.size());
    Kokkos::deep_copy(view, h);
  };

  copy_in(basis.l, h_l);
  copy_in(basis.m, h_m);
  copy_in(basis.O, h_O);
  copy_in(basis.cutoff_radii, h_cut);
  copy_in(basis.prim_offset, h_off);
  copy_in(basis.n, h_n);
  copy_in(basis.zeta, h_zeta);
  copy_in(basis.coeff, h_a);

  return basis;
}

// ============================================================================
//  Device evaluation
// ============================================================================

/// Everything needed to evaluate one contracted function, loaded once so the
/// inner primitive loop reads only from the flat arrays.
struct ContractedShellParams {
  int l, m;
  int prim_begin, prim_end;
  double ox, oy, oz;
};

KOKKOS_INLINE_FUNCTION
ContractedShellParams load_contracted_shell(const STOContractedBasisSet &basis,
                                            int i) {
  return {basis.l(i),
          basis.m(i),
          basis.prim_offset(i),
          basis.prim_offset(i + 1),
          basis.O(i)[0],
          basis.O(i)[1],
          basis.O(i)[2]};
}

/// chi(r) for one contracted function.
KOKKOS_INLINE_FUNCTION
double contracted_basis_eval(const STOContractedBasisSet &basis,
                             const ContractedShellParams &sh, double x,
                             double y, double z) {
  x -= sh.ox;
  y -= sh.oy;
  z -= sh.oz;
  const double r = Kokkos::sqrt(x * x + y * y + z * z);

  double radial = 0.0;
  for (int p = sh.prim_begin; p < sh.prim_end; ++p)
    radial += basis.coeff(p) *
              int_pow(r, basis.n(p) - sh.l - 1) *
              Kokkos::exp(-basis.zeta(p) * r);

  return radial * eval_solid_harmonic(sh.l, sh.m, x, y, z);
}

/// chi and its Cartesian gradient for one contracted function.
///
/// With  chi = R(r) S_lm(x,y,z),  R(r) = sum_p a_p r^(k_p) e^(-z_p r),
/// k_p = n_p - l - 1, the gradient is
///
///     grad chi = R grad S + S (dR/dr) r_hat
///
/// and dR/dr = sum_p a_p (k_p / r - z_p) r^(k_p) e^(-z_p r), so the per-primitive
/// term is reused between the two sums.
KOKKOS_INLINE_FUNCTION
void contracted_basis_eval_with_grad(const STOContractedBasisSet &basis,
                                     const ContractedShellParams &sh, double x,
                                     double y, double z, double &val,
                                     double &gx, double &gy, double &gz) {
  x -= sh.ox;
  y -= sh.oy;
  z -= sh.oz;
  const double r = Kokkos::sqrt(x * x + y * y + z * z);

  double R = 0.0;
  double dR = 0.0;
  for (int p = sh.prim_begin; p < sh.prim_end; ++p) {
    const int k = basis.n(p) - sh.l - 1;
    const double zp = basis.zeta(p);
    const double term =
        basis.coeff(p) * int_pow(r, k) * Kokkos::exp(-zp * r);
    R += term;
    // k/r is finite for k >= 1; the k == 0 primitives contribute -z_p * term,
    // and r == 0 is excluded from every atomic grid.
    dR += (k == 0) ? -zp * term : (k / r - zp) * term;
  }

  double S, dSx, dSy, dSz;
  real_solid_harmonic_cart_and_grad_precomputed(sh.l, sh.m, x, y, z, S, dSx,
                                                dSy, dSz);

  const double common = (r > 0.0) ? dR / r : 0.0;
  val = R * S;
  gx = R * dSx + S * x * common;
  gy = R * dSy + S * y * common;
  gz = R * dSz + S * z * common;
}

} // namespace Nukexc
