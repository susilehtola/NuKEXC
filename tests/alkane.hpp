/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// Linear alkanes, the canonical screening test system.
//
// Water does not exhibit screening at all: at three atoms every basis
// function overlaps every grid box, so a screened path and a dense one do
// the same work and a timing comparison between them measures only
// overhead. A quantity that is supposed to scale linearly needs a system
// with a length to scale in, and the standard one is a linear alkane
// C_n H_(2n+2) with n as the variable.
//
// Two orientations, and the difference between them is the point:
//
//   Aligned   the chain runs along x. Grid boxes built from an
//             axis-aligned subdivision line up with the molecule, so each
//             box is small and overlaps few shells.
//
//   Diagonal  the chain runs along (1,1,1)/sqrt(3). The points now occupy
//             a thin tube through a large cube, so an AXIS-ALIGNED box
//             around any run of consecutive points along that tube is far
//             bigger than the points in it, overlaps many more shells, and
//             the neighbour lists grow. This is the case that separates an
//             implementation which orients its boxes from one which does
//             not, and the reason the diagonal variant is a unit test and
//             not just a benchmark.
//
// Geometry: all-anti backbone, C-C 1.526 A, C-C-C 112.0 deg, C-H 1.09 A,
// H-C-H 107.0 deg, converted to bohr. Interior hydrogens are placed in the
// plane bisecting the C-C-C angle and perpendicular to it; terminal
// carbons get three hydrogens staggered about the single C-C bond.
// tests/test_alkane_scaling.cxx checks the bond lengths and the chain
// direction that come out, rather than trusting the construction.

#include <array>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>

namespace Nukexc {
namespace test {

/// Which way the chain runs.
enum class ChainAxis {
  Aligned, ///< along x
  Diagonal ///< along (1,1,1)/sqrt(3)
};

namespace detail {

constexpr double ANGSTROM = 1.8897261254578281; // bohr per angstrom

using V3 = std::array<double, 3>;

inline V3 add(const V3 &a, const V3 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline V3 sub(const V3 &a, const V3 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline V3 scale(const V3 &a, double s) {
  return {a[0] * s, a[1] * s, a[2] * s};
}
inline double dot(const V3 &a, const V3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline V3 cross(const V3 &a, const V3 &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
inline V3 unit(const V3 &a) {
  const double n = std::sqrt(dot(a, a));
  if (!(n > 0.0))
    throw std::runtime_error("alkane: degenerate direction");
  return scale(a, 1.0 / n);
}

/// Rotation carrying x onto (1,1,1)/sqrt(3), as a right-handed frame.
inline V3 to_diagonal(const V3 &v) {
  // Columns: the chain axis, and two orthogonal completions.
  const double s3 = 1.0 / std::sqrt(3.0);
  const double s2 = 1.0 / std::sqrt(2.0);
  const double s6 = 1.0 / std::sqrt(6.0);
  const V3 e1{s3, s3, s3};
  const V3 e2{s2, -s2, 0.0};
  const V3 e3{s6, s6, -2.0 * s6};
  return {v[0] * e1[0] + v[1] * e2[0] + v[2] * e3[0],
          v[0] * e1[1] + v[1] * e2[1] + v[2] * e3[1],
          v[0] * e1[2] + v[1] * e2[2] + v[2] * e3[2]};
}

} // namespace detail

/// An alkane as plain coordinates, so callers can inspect or perturb it.
struct AlkaneGeometry {
  std::vector<unsigned> Z;
  std::vector<std::array<double, 3>> xyz; ///< bohr
  size_t size() const { return Z.size(); }
};

/**
 *  @brief C_n H_(2n+2), all-anti, in bohr.
 *
 *  @param ncarbon chain length, at least 1
 *  @param axis    orientation (see ChainAxis)
 */
inline AlkaneGeometry alkane_geometry(int ncarbon, ChainAxis axis) {
  using namespace detail;
  if (ncarbon < 1)
    throw std::invalid_argument("alkane_geometry: need at least one carbon");

  const double dcc = 1.526 * ANGSTROM;
  const double dch = 1.090 * ANGSTROM;
  const double ccc = 112.0 * M_PI / 180.0;
  const double hch = 107.0 * M_PI / 180.0;

  // Backbone: a planar zigzag in the xy plane. Step dcc*sin(ccc/2) along x
  // and alternate +-dcc*cos(ccc/2) in y gives bonds of length dcc and the
  // requested angle between consecutive bonds.
  const double sx = dcc * std::sin(ccc / 2.0);
  const double sy = dcc * std::cos(ccc / 2.0);

  std::vector<V3> C(static_cast<size_t>(ncarbon));
  for (int i = 0; i < ncarbon; ++i)
    C[static_cast<size_t>(i)] = {i * sx, (i % 2) ? sy : 0.0, 0.0};

  AlkaneGeometry g;
  auto push = [&](unsigned z, const V3 &p) {
    g.Z.push_back(z);
    g.xyz.push_back(axis == ChainAxis::Diagonal ? to_diagonal(p) : p);
  };

  for (int i = 0; i < ncarbon; ++i) {
    const V3 &c = C[static_cast<size_t>(i)];
    push(6u, c);

    // Directions to the bonded carbons.
    std::vector<V3> nb;
    if (i > 0)
      nb.push_back(unit(sub(C[static_cast<size_t>(i - 1)], c)));
    if (i + 1 < ncarbon)
      nb.push_back(unit(sub(C[static_cast<size_t>(i + 1)], c)));

    if (nb.size() == 2) {
      // CH2: hydrogens in the plane bisecting C-C-C, perpendicular to it.
      const V3 bis = unit(scale(add(nb[0], nb[1]), -1.0));
      const V3 nrm = unit(cross(nb[0], nb[1]));
      const double a = hch / 2.0;
      push(1u, add(c, scale(add(scale(bis, std::cos(a)),
                                scale(nrm, std::sin(a))),
                            dch)));
      push(1u, add(c, scale(sub(scale(bis, std::cos(a)),
                                scale(nrm, std::sin(a))),
                            dch)));
    } else if (nb.size() == 1) {
      // CH3: three hydrogens on a cone about the single C-C bond, at the
      // tetrahedral angle, staggered 120 degrees apart.
      const V3 ax = nb[0];
      // any vector not parallel to ax
      V3 tmp{0.0, 0.0, 1.0};
      if (std::abs(dot(tmp, ax)) > 0.9)
        tmp = {0.0, 1.0, 0.0};
      const V3 u = unit(cross(ax, tmp));
      const V3 v = unit(cross(ax, u));
      const double tet = 109.471 * M_PI / 180.0;
      for (int k = 0; k < 3; ++k) {
        const double phi = 2.0 * M_PI * k / 3.0;
        const V3 dir = add(scale(ax, std::cos(tet)),
                           scale(add(scale(u, std::cos(phi)),
                                     scale(v, std::sin(phi))),
                                 std::sin(tet)));
        push(1u, add(c, scale(unit(dir), dch)));
      }
    } else {
      // CH4: a lone carbon. Four hydrogens at the tetrahedral vertices.
      const double q = dch / std::sqrt(3.0);
      push(1u, add(c, V3{q, q, q}));
      push(1u, add(c, V3{q, -q, -q}));
      push(1u, add(c, V3{-q, q, -q}));
      push(1u, add(c, V3{-q, -q, q}));
    }
  }
  return g;
}

/// The same, as a device-resident Molecule.
inline Molecule alkane(int ncarbon, ChainAxis axis) {
  const auto g = alkane_geometry(ncarbon, axis);
  const size_t n = g.size();

  Molecule mol;
  mol.natoms = static_cast<unsigned>(n);
  mol.atom_centers = Kokkos::View<Point *>("atom centers", n);
  mol.Z = Kokkos::View<unsigned *>("Z", n);
  auto ch = Kokkos::create_mirror_view(mol.atom_centers);
  auto zh = Kokkos::create_mirror_view(mol.Z);
  for (size_t i = 0; i < n; ++i) {
    ch(i) = Point{g.xyz[i][0], g.xyz[i][1], g.xyz[i][2]};
    zh(i) = g.Z[i];
  }
  Kokkos::deep_copy(mol.atom_centers, ch);
  Kokkos::deep_copy(mol.Z, zh);

  // read_xyz fills these too; anything downstream that reads them (basis
  // construction, SAP tables) would silently see an empty element list.
  mol.Z_total = 0;
  for (size_t i = 0; i < n; ++i) {
    mol.element_list.insert(g.Z[i]);
    mol.Z_total += g.Z[i];
  }
  return mol;
}

} // namespace test
} // namespace Nukexc
