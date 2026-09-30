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
//  Koster's parameter-free radial mapping
// ============================================================================
//
// The radial scheme ERKALE uses (src/chebyshev.cpp):
//
//     r(x) = ln(2 / (1 - x)) / ln 2      dr/dx = 1 / (ln 2 (1 - x))
//
// applied to a modified Gauss-Chebyshev quadrature of the second kind. Its
// distinguishing feature is that it carries NO atomic size parameter: every
// element gets the same mapping, so there is no per-element table to
// tabulate, get wrong, or silently fall back to 1.0 for elements it does not
// cover -- which is what the Treutler-Ahlrichs xi table in grid.hpp does
// today beyond Z = 36.
//
// WHAT IS AND IS NOT IMPLEMENTED HERE
// -----------------------------------
// The quadrature primitive is NOT reimplemented. IntegratorXX already
// provides the modified Gauss-Chebyshev rule as GaussChebyshev2Modified, and
// its nodes and weights are the same expressions ERKALE uses -- Eqs. 32 and
// 33 of Perez-Jorda, San-Fabian and Moscardo, J. Chem. Phys. 100 (1994)
// 6520, doi:10.1063/1.467061. Only the MAPPING is missing upstream, so only
// the mapping is added, as a RadialTraits implementation that plugs into
// IntegratorXX's RadialTransformQuadrature exactly as Becke, Mura-Knowles,
// Murray-Handy-Laming and Treutler-Ahlrichs do.
//
// THREE UPSTREAM GAPS, found while doing this
// -------------------------------------------
//   1. GaussChebyshev2Modified's quadrature_traits does not declare
//      `bound_inclusive`, which RadialTransformQuadrature requires. Of the
//      seven primitives only gausschebyshev2 and uniform declare it, so the
//      modified rule cannot currently be composed with ANY radial transform
//      through IntegratorXX's own machinery.
//   2. There is no logarithmic / Koster radial transform.
//   3. Adding one to the runtime RadialQuad spec used by SphericalGridFactory
//      would be a third change.
//
// Because of (1) the quadrature is assembled here directly from the
// primitive rather than through RadialTransformQuadrature. KosterRadialTraits
// is nevertheless written to IntegratorXX's RadialTraits interface, so it is
// ready to be contributed upstream once (1) is fixed -- at which point the
// generator below collapses to a type alias.

#include <cmath>
#include <memory>
#include <vector>

#include <integratorxx/quadratures/primitive/gausschebyshev2modified.hpp>
#include <integratorxx/quadratures/radial/radial_transform.hpp>

namespace Nukexc {

/**
 *  @brief Koster's parameter-free logarithmic radial mapping.
 *
 *  r(x) = R ln(2/(1-x)) / ln 2 on x in [-1, 1), mapping the Chebyshev
 *  interval onto [0, infinity).
 *
 *  R defaults to 1 and exists only so a caller can rescale deliberately;
 *  ERKALE uses R = 1 for every element, and the absence of a size parameter
 *  is the point of the scheme rather than an omission.
 */
class KosterRadialTraits : public IntegratorXX::RadialTraits {
  size_t npts_;
  double R_;

public:
  explicit KosterRadialTraits(size_t npts, double R = 1.0)
      : npts_(npts), R_(R) {}

  size_t npts() const noexcept override { return npts_; }

  std::unique_ptr<RadialTraits> clone() const override {
    return std::make_unique<KosterRadialTraits>(*this);
  }

  bool compare(const RadialTraits &other) const noexcept override {
    auto ptr = dynamic_cast<const KosterRadialTraits *>(&other);
    return ptr ? *this == *ptr : false;
  }

  bool operator==(const KosterRadialTraits &other) const noexcept {
    return npts_ == other.npts_ && R_ == other.R_;
  }

  /// r(x) = R ln(2/(1-x)) / ln 2
  template <typename PointType>
  inline auto radial_transform(PointType x) const noexcept {
    return R_ * std::log(2.0 / (1.0 - x)) / M_LN2;
  }

  /// dr/dx = R / (ln 2 (1 - x))
  template <typename PointType>
  inline auto radial_jacobian(PointType x) const noexcept {
    return R_ / (M_LN2 * (1.0 - x));
  }
};

/// A radial grid: radii and the matching weights.
struct RadialGrid {
  std::vector<double> r;
  std::vector<double> w;
  size_t size() const { return r.size(); }
};

/**
 *  @brief The radial quadrature ERKALE uses: modified Gauss-Chebyshev of the
 *         second kind under Koster's mapping.
 *
 *  The weights EXCLUDE the r^2 volume element, matching IntegratorXX's
 *  convention for its other radial quadratures and ERKALE's
 *  radial_chebyshev (as opposed to its radial_chebyshev_jac).
 *
 *  Nodes come out in increasing radius with no reindexing. IntegratorXX's
 *  abscissae are the exact negation of ERKALE's -- verified numerically, the
 *  weights being identical and symmetric -- so IntegratorXX's natural order
 *  is already the one ERKALE reaches via its reversed `ixc` index. Walking
 *  it backwards, as an incautious port does, silently produces DECREASING
 *  radii, which pairs every radius with the wrong weight; the ordering is
 *  therefore asserted element-by-element in the tests rather than assumed.
 *
 *  Non-finite nodes are dropped, as ERKALE does: x -> 1 maps to r -> inf.
 */
inline RadialGrid koster_chebyshev_radial(size_t npts, double R = 1.0) {
  using Prim = IntegratorXX::GaussChebyshev2Modified<double, double>;
  const auto [x, wx] = IntegratorXX::quadrature_traits<Prim>::generate(npts);

  const KosterRadialTraits traits(npts, R);

  RadialGrid g;
  g.r.reserve(npts);
  g.w.reserve(npts);
  for (size_t i = 0; i < x.size(); ++i) {
    const double r = traits.radial_transform(x[i]);
    const double w = wx[i] * traits.radial_jacobian(x[i]);
    if (std::isfinite(r) && std::isfinite(w)) {
      g.r.push_back(r);
      g.w.push_back(w);
    }
  }
  return g;
}

} // namespace Nukexc
