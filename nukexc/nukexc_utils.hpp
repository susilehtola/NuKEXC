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

#include "nukexc/nukexc_config.hpp"
#include <Kokkos_Core.hpp>

#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_gemm_impl.hpp>

#include <KokkosLapack_svd.hpp>
#include <stdexcept>

namespace Nukexc {

// Plain multiplication rather than pow(x, 2): these two are called inside the
// partitioning kernel's natoms^2 loop and inside every screening test in the
// library, and a call to pow for an exponent known at compile time is not
// something to rely on the optimizer to undo (issue #5).
KOKKOS_INLINE_FUNCTION
double dist_squared(const Point a, const Point b) {
  const double dx = a[0] - b[0];
  const double dy = a[1] - b[1];
  const double dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

KOKKOS_INLINE_FUNCTION
double dist(const Point a, const Point b) {
  return Kokkos::sqrt(dist_squared(a, b));
}

KOKKOS_INLINE_FUNCTION
double double_factorial(const int n) {
  if (n <= 0)
    return 1.0;

  double result = 1.0;
  for (int i = n; i > 0; i -= 2) {
    result *= i;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
double factorial(const int n) {

  if (n <= 0)
    return 1.0;

  double result = 1.0;
  for (int i = 1; i <= n; ++i) {
    result *= i;
  }
  return result;
}
KOKKOS_INLINE_FUNCTION
long int binomial(const int n, const int k) {
  long int result = factorial(n) / (factorial(k) * factorial(n - k));
  return result;
}

KOKKOS_INLINE_FUNCTION
double log_factorial_ratio(const int top, const int bottom) {
  // Computes log(top! / bottom!) = log(bottom+1) + ... + log(top)
  // Assumes top >= bottom >= 0
  double result = 0.0;
  for (int i = bottom + 1; i <= top; ++i)
    result += Kokkos::log((double)i);
  return result;
}

/// Forward declaration: safe_pow delegates to int_pow, which is defined
/// below because it is the longer of the two.
KOKKOS_INLINE_FUNCTION double int_pow(const double r, const int k);

KOKKOS_INLINE_FUNCTION
double safe_pow(const double base, const int exp) {
  if (exp == 0)
    return 1.0; // always 1 regardless of base
  if (base == 0.0)
    return 0.0; // 0^positive = 0
  // int_pow, not pow: the value path already uses int_pow for the same
  // radial factor, so going through libm here made the STO gradient
  // disagree with the STO value in the last bits for no reason.
  return int_pow(base, exp);
}

KOKKOS_INLINE_FUNCTION
double int_pow(const double r, const int k) {

  double result = 0.;
  switch (k) {
  case 0:
    result = 1.;
    break;
  case 1:
    result = r;
    break;
  case 2:
    result = r * r;
    break;
  case 3:
    result = r * r * r;
    break;
  case 4:
    result = r * r * r * r;
    break;
  case 5:
    result = r * r * r * r * r;
    break;
  case 6:
    result = r * r * r * r * r * r;
    break;
  case 7:
    result = r * r * r * r * r * r * r;
    break;
  default: {
    // Binary exponentiation for the rare large exponent -- n + l + 2 reaches
    // eleven for a 6f Slater function. Still multiplication, and still not a
    // libm call: pow() with an integer exponent is NOT folded by the
    // compiler at -O3 (errno semantics block even the exact square), and
    // this function is called per grid point.
    double base = r;
    int e = k;
    if (k < 0) {
      base = 1.0 / r;
      e = -k;
    }
    double acc = 1.0;
    while (e > 0) {
      if (e & 1)
        acc *= base;
      base *= base;
      e >>= 1;
    }
    result = acc;
    break;
  }
  }
  return result;
}

/// Upper incomplete gamma, with exp(-x) supplied by the caller.
///
/// The STO potential needs both incomplete gammas at the same argument, and
/// each of them used to call exp(-x) itself. One libm call per grid point
/// per auxiliary function is worth the extra parameter; the two-argument
/// form below keeps the plain interface.
KOKKOS_INLINE_FUNCTION
double upper_gamma(const int n, const double x, const double exp_minus_x) {
  if (n <= 0)
    return 0.0;

  // This iterative Horner-like approach preserves precision perfectly.
  double sum = 1.0;
  double term = 1.0;
  for (int k = 1; k < n; ++k) {
    term *= x / k;
    sum += term;
  }

  double fact = factorial(n - 1);
  return fact * exp_minus_x * sum;
}

KOKKOS_INLINE_FUNCTION
double upper_gamma(const int n, const double x) {
  return upper_gamma(n, x, Kokkos::exp(-x));
}

/// Lower incomplete gamma, with exp(-x) supplied by the caller.
KOKKOS_INLINE_FUNCTION
double lower_gamma(const int n, const double x, const double exp_minus_x) {
  if (n <= 0)
    return 0.0;

  // For small/moderate x, use the stable ascending series:
  // gamma(n, x) = x^n * e^-x * \sum_{k=0}^\infty \frac{x^k}{n(n+1)...(n+k)}
  // This avoids the catastrophic "1.0 - result" cancellation entirely.
  if (x < n) {
    double sum = 1.0 / n;
    double term = 1.0 / n;

    // Loop to a safe convergence limit or a fixed precision bound
    for (int k = 1; k < 1000; ++k) {
      term *= x / (n + k);
      sum += term;
      if (term < 1e-16 * sum)
        break; // Converged to machine precision
    }

    // x^n e^-x directly, rather than exp(n log x - x). This branch has
    // x < n, so x^n is at most n^n and cannot overflow for any n this is
    // called with -- the guard below keeps that true for an n no caller has
    // yet -- and the direct product is both one libm call cheaper and more
    // accurate than composing a log with an exp.
    if (n <= 64)
      return int_pow(x, n) * exp_minus_x * sum;
    const double log_prefactor = n * Kokkos::log(x) - x;
    return Kokkos::exp(log_prefactor) * sum;

  } else {
    double Gamma_n = factorial(n - 1);
    return Gamma_n - upper_gamma(n, x, exp_minus_x);
  }
}

KOKKOS_INLINE_FUNCTION
double lower_gamma(const int n, const double x) {
  return lower_gamma(n, x, Kokkos::exp(-x));
}

DeviceView2DLeft compute_half_inverse(const DeviceView2DLeft &overlap_matrix,
                                      const double lin_dep_threshold = 1e-5) {

  const int N = overlap_matrix.extent(0);
  DeviceView2DLeft Us("U", N, N);
  DeviceView2DLeft VTs("VTs", N, N);
  DeviceView1D sigma("sigma", N);
  DeviceView2DLeft A("matrix A", N, N);
  DeviceView1D D_N("Diagonal Preconditioner matrix", N);

  Kokkos::parallel_for(
      "Compute preconditioner matrix", N, KOKKOS_LAMBDA(const int i) {
        D_N(i) = Kokkos::sqrt(overlap_matrix(i, i));
      });

  Kokkos::parallel_for(
      "Apply preconditioner matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        A(i, j) = overlap_matrix(i, j) / (D_N(i) * D_N(j));
      });

  // SVD of S to handle potential singularity
  KokkosLapack::svd("S", "S", A, sigma, Us, VTs);

  int negative_sigma = 0;
  Kokkos::parallel_reduce(
      "SwitchSigns", N,
      KOKKOS_LAMBDA(const int j, int &local_count) {
        if (-sigma(j) > lin_dep_threshold) {
          sigma(j) = -sigma(j);
          Kokkos::printf("Negative sigma %d : %.3e \n", j, sigma(j));
          local_count += 1;
        }
      },
      negative_sigma);

  // TODO:: Find a way to deal with negative sigma
  // For now just throw a runtime error for safety
  if (negative_sigma > 0)
    throw std::runtime_error(
        "Negative singular values in core Hamiltonian detected");

  auto sigma_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sigma);

  int K = 0;
  std::vector<int> kept;
  kept.reserve(N);
  for (int i = 0; i < N; ++i) {
    if (sigma_h(i) > lin_dep_threshold) {
      kept.push_back(i);
      ++K;
    }
  }

  // Upload the index map so the kernel can use it on device
  Kokkos::View<int *, Kokkos::HostSpace> kept_h(kept.data(), K);
  Kokkos::View<int *> kept_d("kept_d", K);
  Kokkos::deep_copy(kept_d, kept_h);

  DeviceView2DLeft X("Half inverse", N, K);

  Kokkos::parallel_for(
      "Invert overlap", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, K}),
      KOKKOS_LAMBDA(const int i, const int k) {
        const int src = kept_d(k);
        X(i, k) = Us(i, src) / Kokkos::sqrt(sigma(src));
        X(i, k) /= D_N(i);
      });

  return X;
}
} // namespace Nukexc
