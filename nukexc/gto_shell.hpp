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
//  Generally contracted Gaussian-type orbital shells (host side)
// ============================================================================
//
// This is the host-side data layer for Gaussian basis sets. It is deliberately
// free of Kokkos: shells are built, normalized and inspected on the host, and
// a separate device-side struct-of-Views (see gto_basisset.hpp) is produced
// from them for the integration kernels.
//
// CONVENTIONS
// -----------
// A shell holds `nprim` shared primitive exponents and an (nprim x nctr)
// coefficient matrix -- i.e. it is *generally* contracted. A segmented shell
// is the nctr == 1 special case. Basis functions run contraction-slowest:
//
//     [ctr 0: components 0..ncomp-1][ctr 1: components 0..ncomp-1]...
//
// which is the order ERKALE's GaussianShell::function_norms() uses.
//
// Cartesian components are ordered as in ERKALE (src/basis.cpp, the
// GaussianShell constructor):
//
//     n = 0
//     for i in 0..l:  nx = l - i
//       for j in 0..i:  ny = i - j;  nz = j;  component[n++] = (nx, ny, nz)
//
// so l = 2 gives xx, xy, xz, yy, yz, zz. The flat index of (nx, ny, nz) is
// cart_index() below, matching ERKALE's getind() in src/mathf.cpp.
//
// Spherical components run m = -l .. +l and are obtained from the Cartesians
// by ylm_transmat(l), the normalized real solid harmonic transform (see
// solid_harmonics_transform.hpp).
//
// NORMALIZATION (NormalizationConvention::ERKALE, the default)
// ------------------------------------------------------------
// Reproduces ERKALE's GaussianShell::normalize(), which works with the
// coefficients of *unnormalized* primitives:
//
//   1. Each contraction column is scaled so that the first Cartesian function
//      of that contraction (the x^l component) has unit self-overlap:
//
//        angfac = pi^(3/2) (2l-1)!! / 2^l
//        fact   = angfac * sum_ij c_i c_j / (z_i + z_j)^(l + 3/2)
//        c     /= sqrt(fact)
//
//   2. Cartesian shells then carry per-component relative normalization
//
//        relnorm[c] = sqrt( (2l-1)!! / ((2nx-1)!! (2ny-1)!! (2nz-1)!!) )
//
//      i.e. every Cartesian is normalized *relative to* the x^l component.
//
//   3. Spherical shells instead divide the (unit) relnorm of every Cartesian
//      by sqrt(S(0)), where S(0) is the self-overlap of the first spherical
//      function of the first contraction.
//
// NormalizationConvention::None leaves the input coefficients untouched, for
// hosts that hand over already-normalized data.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "solid_harmonics_transform.hpp"

namespace Nukexc {

/// Angular representation of a shell's components.
enum class ShellType { Cartesian, Spherical };

/// Which normalization to impose when a shell is constructed.
enum class NormalizationConvention {
  ERKALE, ///< ERKALE's GaussianShell::normalize() (see header comment)
  None    ///< leave caller-supplied coefficients untouched
};

/// One Cartesian component x^nx y^ny z^nz of a shell, with the relative
/// normalization factor that multiplies it.
struct CartComponent {
  int nx, ny, nz;
  double relnorm;
};

namespace detail {

/// (2n-1)!! for n >= 0, with the convention (-1)!! = 1. Exact in double up to
/// well past the l <= 7 the library needs (and the l <= 14 ERKALE allows).
inline double double_factorial_2nm1(int n) {
  double result = 1.0;
  for (int k = 2 * n - 1; k > 1; k -= 2)
    result *= static_cast<double>(k);
  return result;
}

} // namespace detail

/// Number of Cartesian components of a shell of angular momentum l.
inline constexpr int cart_size(int l) { return (l + 1) * (l + 2) / 2; }

/// Number of spherical components of a shell of angular momentum l.
inline constexpr int pure_size(int l) { return 2 * l + 1; }

/// Flat index of the Cartesian component (nx, ny, nz) within its shell.
/// Matches ERKALE's getind() (src/mathf.cpp): with i = ny + nz and j = nz,
/// the index is i(i+1)/2 + j.
inline constexpr int cart_index(int nx, int ny, int nz) {
  (void)nx;
  const int i = ny + nz;
  return i * (i + 1) / 2 + nz;
}

/// The Cartesian components of a shell of angular momentum l, in ERKALE order,
/// each carrying the ERKALE relative normalization factor
/// sqrt((2l-1)!! / ((2nx-1)!!(2ny-1)!!(2nz-1)!!)).
inline std::vector<CartComponent> cartesian_components(int l) {
  if (l < 0)
    throw std::invalid_argument("cartesian_components: negative l");

  std::vector<CartComponent> out(static_cast<size_t>(cart_size(l)));
  const double dfl = detail::double_factorial_2nm1(l);

  for (int i = 0; i <= l; ++i) {
    const int nx = l - i;
    for (int j = 0; j <= i; ++j) {
      const int ny = i - j;
      const int nz = j;
      const double denom = detail::double_factorial_2nm1(nx) *
                           detail::double_factorial_2nm1(ny) *
                           detail::double_factorial_2nm1(nz);
      out[static_cast<size_t>(cart_index(nx, ny, nz))] =
          CartComponent{nx, ny, nz, std::sqrt(dfl / denom)};
    }
  }
  return out;
}

// ============================================================================
//  GTOShell
// ============================================================================

/**
 *  @brief A generally contracted shell of Gaussian basis functions.
 *
 *  Holds `nprim` shared primitive exponents and an (nprim x nctr) coefficient
 *  matrix stored contraction-major, so that contraction `ictr` occupies the
 *  contiguous range [ictr*nprim, (ictr+1)*nprim).
 *
 *  @tparam F floating-point type of the exponents and coefficients
 */
template <typename F> class GTOShell {
public:
  using value_type = F;

  GTOShell() = default;

  /**
   *  @param l        angular momentum
   *  @param type     Cartesian or spherical components
   *  @param O        shell center
   *  @param alpha    primitive exponents, length nprim
   *  @param coeff    coefficients of UNNORMALIZED primitives, contraction-major,
   *                  length nprim * nctr
   *  @param nctr     number of contractions sharing these primitives
   *  @param conv     normalization to impose
   *  @param tol      basis-function cutoff tolerance for the screening radius
   */
  GTOShell(int l, ShellType type, const std::array<double, 3> &O,
           std::vector<F> alpha, std::vector<F> coeff, int nctr = 1,
           NormalizationConvention conv = NormalizationConvention::ERKALE,
           double tol = 1e-10)
      : l_(l), type_(type), O_(O), alpha_(std::move(alpha)),
        coeff_(std::move(coeff)), nctr_(nctr), tol_(tol) {

    if (l_ < 0)
      throw std::invalid_argument("GTOShell: negative angular momentum");
    if (nctr_ < 1)
      throw std::invalid_argument("GTOShell: need at least one contraction");
    if (alpha_.empty())
      throw std::invalid_argument("GTOShell: no primitives");
    if (coeff_.size() != alpha_.size() * static_cast<size_t>(nctr_))
      throw std::invalid_argument(
          "GTOShell: coefficient matrix is " + std::to_string(coeff_.size()) +
          " long, expected nprim*nctr = " +
          std::to_string(alpha_.size() * static_cast<size_t>(nctr_)));
    for (auto a : alpha_)
      if (!(a > F{0}))
        throw std::invalid_argument("GTOShell: non-positive exponent");
    if (!(tol_ > 0.0))
      throw std::invalid_argument("GTOShell: non-positive cutoff tolerance");

    cart_ = cartesian_components(l_);
    if (conv == NormalizationConvention::ERKALE)
      normalize_erkale();
    compute_cutoff_radius();
  }

  // -- geometry and shape -------------------------------------------------

  int l() const { return l_; }
  ShellType type() const { return type_; }
  bool is_spherical() const { return type_ == ShellType::Spherical; }
  const std::array<double, 3> &O() const { return O_; }
  void set_center(const std::array<double, 3> &O) { O_ = O; }

  int nprim() const { return static_cast<int>(alpha_.size()); }
  int nctr() const { return nctr_; }

  /// Components per contraction (Cartesian or spherical, per type()).
  int ncomp() const {
    return is_spherical() ? pure_size(l_) : cart_size(l_);
  }
  /// Total basis functions on the shell: nctr * ncomp.
  int size() const { return nctr_ * ncomp(); }
  /// Total Cartesian functions on the shell: nctr * cart_size(l).
  int cart_total() const { return nctr_ * cart_size(l_); }

  // -- data ---------------------------------------------------------------

  const std::vector<F> &alpha() const { return alpha_; }
  const std::vector<F> &coeff() const { return coeff_; }
  /// Coefficient of primitive `iprim` in contraction `ictr`.
  F coeff(int iprim, int ictr) const {
    return coeff_[static_cast<size_t>(ictr) * alpha_.size() +
                  static_cast<size_t>(iprim)];
  }
  const std::vector<CartComponent> &cart() const { return cart_; }

  /// Screening radius: beyond this the shell's largest radial amplitude falls
  /// below the cutoff tolerance.
  double cutoff_radius() const { return cutoff_radius_; }

  double tolerance() const { return tol_; }
  void set_tolerance(double tol) {
    if (!(tol > 0.0))
      throw std::invalid_argument("GTOShell: non-positive cutoff tolerance");
    if (tol != tol_) {
      tol_ = tol;
      compute_cutoff_radius();
    }
  }

  /// May `rhs` be merged in as an extra contraction? Requires the same center,
  /// angular momentum, angular representation and primitive exponents.
  bool same_primitives(const GTOShell &rhs) const {
    if (l_ != rhs.l_ || type_ != rhs.type_ || O_ != rhs.O_)
      return false;
    if (alpha_.size() != rhs.alpha_.size())
      return false;
    for (size_t i = 0; i < alpha_.size(); ++i)
      if (alpha_[i] != rhs.alpha_[i])
        return false;
    return true;
  }

  /// Append `rhs`'s contraction columns after this shell's, forming a more
  /// generally contracted shell. This is how a segmented basis-set file with
  /// repeated exponent blocks is folded into general contractions.
  void merge_contraction(const GTOShell &rhs) {
    if (!same_primitives(rhs))
      throw std::invalid_argument(
          "GTOShell::merge_contraction: shells do not share primitives");
    coeff_.insert(coeff_.end(), rhs.coeff_.begin(), rhs.coeff_.end());
    nctr_ += rhs.nctr_;
    compute_cutoff_radius();
  }

  /// Self-overlap of every function on the shell, contraction-slowest.
  /// Mirrors ERKALE's GaussianShell::function_norms(): both functions sit on
  /// the same center, so the integral factorizes over x, y and z and vanishes
  /// unless every direction carries an even total power.
  std::vector<double> function_norms() const {
    const int ncart = cart_size(l_);
    const int nout = ncomp();
    const size_t np = alpha_.size();
    std::vector<double> norms(static_cast<size_t>(nctr_ * nout), 0.0);

    std::vector<double> transmat;
    if (is_spherical())
      transmat = ylm_transmat(l_);

    std::vector<double> S(static_cast<size_t>(ncart) * ncart);
    for (int ictr = 0; ictr < nctr_; ++ictr) {
      std::fill(S.begin(), S.end(), 0.0);

      for (int ic = 0; ic < ncart; ++ic)
        for (int jc = 0; jc < ncart; ++jc) {
          const int L[3] = {cart_[ic].nx + cart_[jc].nx,
                            cart_[ic].ny + cart_[jc].ny,
                            cart_[ic].nz + cart_[jc].nz};
          if (L[0] % 2 || L[1] % 2 || L[2] % 2)
            continue;

          double val = 0.0;
          for (size_t ip = 0; ip < np; ++ip)
            for (size_t jp = 0; jp < np; ++jp) {
              const double zeta =
                  static_cast<double>(alpha_[ip]) + static_cast<double>(alpha_[jp]);
              double term = static_cast<double>(coeff(static_cast<int>(ip), ictr)) *
                            static_cast<double>(coeff(static_cast<int>(jp), ictr)) *
                            std::pow(M_PI / zeta, 1.5);
              for (int ix = 0; ix < 3; ++ix)
                term *= detail::double_factorial_2nm1(L[ix] / 2) /
                        std::pow(2.0 * zeta, L[ix] / 2);
              val += term;
            }
          S[static_cast<size_t>(ic) * ncart + jc] =
              cart_[ic].relnorm * cart_[jc].relnorm * val;
        }

      if (is_spherical()) {
        // diag( T S T^T )
        for (int m = 0; m < nout; ++m) {
          double acc = 0.0;
          for (int ic = 0; ic < ncart; ++ic) {
            const double tim = transmat[static_cast<size_t>(m) * ncart + ic];
            if (tim == 0.0)
              continue;
            for (int jc = 0; jc < ncart; ++jc)
              acc += tim * S[static_cast<size_t>(ic) * ncart + jc] *
                     transmat[static_cast<size_t>(m) * ncart + jc];
          }
          norms[static_cast<size_t>(ictr) * nout + m] = acc;
        }
      } else {
        for (int ic = 0; ic < nout; ++ic)
          norms[static_cast<size_t>(ictr) * nout + ic] =
              S[static_cast<size_t>(ic) * ncart + ic];
      }
    }
    return norms;
  }

private:
  /// ERKALE's GaussianShell::normalize(). See the header comment.
  void normalize_erkale() {
    const size_t np = alpha_.size();
    const double angfac = std::pow(M_PI, 1.5) *
                          detail::double_factorial_2nm1(l_) /
                          std::pow(2.0, l_);

    // 1. unit self-overlap of the x^l function, per contraction
    for (int ictr = 0; ictr < nctr_; ++ictr) {
      double fact = 0.0;
      for (size_t i = 0; i < np; ++i)
        for (size_t j = 0; j < np; ++j)
          fact += static_cast<double>(coeff(static_cast<int>(i), ictr)) *
                  static_cast<double>(coeff(static_cast<int>(j), ictr)) /
                  std::pow(static_cast<double>(alpha_[i]) +
                               static_cast<double>(alpha_[j]),
                           l_ + 1.5);
      fact *= angfac;
      if (!(fact > 0.0))
        throw std::runtime_error(
            "GTOShell: contraction " + std::to_string(ictr) +
            " has non-positive self-overlap; check the coefficients");
      const F scale = static_cast<F>(1.0 / std::sqrt(fact));
      for (size_t i = 0; i < np; ++i)
        coeff_[static_cast<size_t>(ictr) * np + i] *= scale;
    }

    // 2/3. relative component normalization
    if (is_spherical()) {
      // cartesian_components() already filled the Cartesian relnorms; the
      // spherical convention starts from unity instead and rescales by the
      // norm of the first spherical function of the first contraction.
      for (auto &c : cart_)
        c.relnorm = 1.0;
      const std::vector<double> S = function_norms();
      if (!(S[0] > 0.0))
        throw std::runtime_error(
            "GTOShell: first spherical function has non-positive norm");
      const double inv = 1.0 / std::sqrt(S[0]);
      for (auto &c : cart_)
        c.relnorm *= inv;
    }
    // Cartesian shells keep the relnorms from cartesian_components().
  }

  /// Largest radial amplitude over contractions, r^l * sum_i |c_i| exp(-a_i r^2).
  double radial_envelope(double r) const {
    const size_t np = alpha_.size();
    const double r2 = r * r;
    double best = 0.0;
    for (int ictr = 0; ictr < nctr_; ++ictr) {
      double acc = 0.0;
      for (size_t i = 0; i < np; ++i)
        acc += std::abs(static_cast<double>(coeff(static_cast<int>(i), ictr))) *
               std::exp(-static_cast<double>(alpha_[i]) * r2);
      best = std::max(best, acc);
    }
    return std::pow(r, static_cast<double>(l_)) * best;
  }

  /// Smallest r beyond which radial_envelope(r) stays below the tolerance.
  ///
  /// The envelope is eventually monotonically decreasing, so we bracket the
  /// crossing by doubling and then bisect. This replaces the fixed 0.01-step
  /// linear walk of the original gau_rad_cutoff(), which cost thousands of
  /// evaluations for diffuse shells and mislabelled its own search direction.
  void compute_cutoff_radius() {
    // Start past the peak of r^l exp(-a r^2), which sits at r = sqrt(l/2a).
    const double amin = static_cast<double>(
        *std::min_element(alpha_.begin(), alpha_.end()));
    double lo = (l_ > 0) ? std::sqrt(l_ / (2.0 * amin)) : 0.0;

    if (radial_envelope(lo) <= tol_) {
      // Already below tolerance at the peak: nothing of this shell matters.
      cutoff_radius_ = lo;
      return;
    }

    double hi = std::max(lo, 1.0);
    int doublings = 0;
    while (radial_envelope(hi) > tol_) {
      lo = hi;
      hi *= 2.0;
      if (++doublings > 200)
        throw std::runtime_error(
            "GTOShell: cutoff radius search failed to bracket");
    }

    // 60 bisections takes the bracket below any meaningful resolution.
    for (int it = 0; it < 60 && (hi - lo) > 1e-12 * std::max(1.0, hi); ++it) {
      const double mid = 0.5 * (lo + hi);
      if (radial_envelope(mid) > tol_)
        lo = mid;
      else
        hi = mid;
    }
    cutoff_radius_ = hi;
  }

  int l_ = 0;
  ShellType type_ = ShellType::Spherical;
  std::array<double, 3> O_ = {0.0, 0.0, 0.0};
  std::vector<F> alpha_;
  std::vector<F> coeff_; ///< contraction-major, nprim * nctr
  int nctr_ = 1;
  double tol_ = 1e-10;
  double cutoff_radius_ = 0.0;
  std::vector<CartComponent> cart_;
};

} // namespace Nukexc
