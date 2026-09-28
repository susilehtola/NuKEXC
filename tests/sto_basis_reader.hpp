/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// ============================================================================
//  Basis-set file readers -- TEST SUPPORT ONLY
// ============================================================================
//
// These do not belong in the library. NuKEXC takes basis sets as caller-
// supplied arrays (Nukexc::STOBasisInput / Nukexc::GTOBasisInput); file
// formats, library paths and element tables are the calling program's problem.
//
// They live here so the vendored basis sets under input/ -- which exist as
// test data, not as a shipped basis library -- can be fed to the tests,
// benchmarks and convergence studies.
//
// Unlike the versions that used to sit in nukexc/stobasis.hpp, these take the
// data directory as a required argument and throw when a basis file is
// missing, instead of defaulting to a hard-coded relative path and silently
// producing a smaller basis than the caller asked for.

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nukexc/atomic_properties.hpp"
#include "nukexc/molecule.hpp"
#include "nukexc/stobasis.hpp"

namespace Nukexc {
namespace test {

/// Map a spectroscopic shell label to its angular momentum, -1 if unknown.
inline int label_to_l(char label) {
  label = std::toupper(label);
  static const std::string order =
      "SPDFGHI"; // Standard spectroscopic notation (skipping J)
  size_t pos = order.find(label);
  return (pos != std::string::npos) ? static_cast<int>(pos) : -1;
}

inline STOBasisSet load_adf_basis(const Molecule &mol,
                                  const std::string &data_dir,
                                  const double cutoff_tol = 1e-10,
                                  const bool fit = false) {

  std::vector<STOFunc> temp_basis;
  auto Z_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.Z);

  for (size_t i = 0; i < mol.natoms; ++i) {
    std::string element_symbol = detail::symbols[Z_h(i)];
    element_symbol[0] = std::toupper(element_symbol[0]);

    std::string filename = data_dir + "/" + element_symbol;

    std::ifstream file(filename);
    if (!file.is_open())
      throw std::runtime_error("could not open basis file " + filename);

    std::string line;

    // Skip all lines until we find the basis keyword
    std::string basis_keyword = fit ? "FIT" : "BASIS";
    while (std::getline(file, line)) {
      if (line.find(basis_keyword) != std::string::npos)
        break;
    }

    // Read line by line until we find END keyword
    while (std::getline(file, line)) {

      if (line.find("END") != std::string::npos) {
        break;

        // Skip the line if it is empty
      } else if (line.empty()) {
        continue;

      } else {

        // momentum labels This handles S, P, D, F, G, H, I, K (l=0 to 7)

        //  Parse Basis rows: Label (e.g. 4F), zeta
        std::stringstream ss(line);
        std::string label;
        double zeta;

        if (ss >> label >> zeta) {

          int n = std::stoi(label.substr(0, 1));
          int l = label_to_l(label[1]);

          auto atom_centers_h = Kokkos::create_mirror_view_and_copy(
              HostSpace{}, mol.atom_centers);

          // 4. Expand for each m component (-l to +l)
          // This accounts for the degeneracy of higher l states
          for (int m = -l; m <= l; ++m) {
            temp_basis.push_back({n, l, m, zeta, atom_centers_h(i)[0],
                                  atom_centers_h(i)[1], atom_centers_h(i)[2]});
          }
        }
      }
    }
  }
  return make_manual_basis(temp_basis, cutoff_tol);
}

// Read a Thakkar orbital file as the underlying PRIMITIVE set.
//
// Thakkar files list, per shell block, an exponent followed by that
// primitive's contraction coefficient in each orbital:
//
//     2S       24.975698     -0.0004756     -0.0001236
//
// STOBasisSet has no contraction dimension -- one basis function is one
// primitive -- so those coefficients are dropped and every primitive becomes
// an independent function. The result is the uncontracted parent set, which
// spans the contracted set, so it is a legitimate basis; it is simply not the
// basis the file describes. Hence the name.
//
// Use the contracted STO path for the basis the file actually specifies.
inline STOBasisSet
load_thakkar_primitives(const Molecule &mol,
                        const std::string &data_dir,
                        const double cutoff_tol = 1e-10) {

  std::vector<STOFunc> temp_basis;

  auto Z_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.Z);
  for (size_t i = 0; i < mol.natoms; ++i) {
    std::string element_symbol = detail::symbols[Z_h(i)];
    element_symbol[0] = std::tolower(element_symbol[0]);

    std::string filename = data_dir + "/" + element_symbol;

    std::ifstream file(filename);
    if (!file.is_open())
      throw std::runtime_error("could not open basis file " + filename);

    std::string line;
    int current_l = 0;

    while (std::getline(file, line)) {
      if (line.empty())
        continue;

      // 1. Peek at the first word of the line
      std::stringstream check_ss(line);
      std::string first_word;
      check_ss >> first_word;

      // 2. If the first word is a single letter (S, P, D, etc.)
      if (first_word.length() == 1) {
        int found_l = label_to_l(first_word[0]);
        if (found_l != -1) {
          current_l = found_l;
          // Don't 'continue' here!
          // The line might also contain the first basis function.
        }
      }

      // Detect any shell block by checking for single-letter angular
      // momentum labels This handles S, P, D, F, G, H, I, K (l=0 to 7)

      // Skip headers/metadata
      if (line.find("BASIS") != std::string::npos ||
          line.find("CUSP") != std::string::npos)
        continue;

      //  Parse Basis rows: Label (e.g. 4F), zeta, then coefficients
      std::stringstream ss(line);
      std::string label;
      double zeta;

      if (ss >> label >> zeta) {
        // Ensure the label starts with a digit (the n quantum number)
        if (!std::isdigit(label[0]))
          continue;

        int n = std::stoi(label.substr(0, 1));
        std::vector<double> coeffs;
        double c;
        while (ss >> c)
          coeffs.push_back(c);

        auto atom_centers_h =
            Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.atom_centers);
        // 4. Expand for each m component (-l to +l)
        // This accounts for the degeneracy of higher l states
        for (int m = -current_l; m <= current_l; ++m) {
          temp_basis.push_back({n, current_l, m, zeta, atom_centers_h(i)[0],
                                atom_centers_h(i)[1], atom_centers_h(i)[2]});
        }
      }
    }
  }

  return make_manual_basis(temp_basis, cutoff_tol);
}

} // namespace test
} // namespace Nukexc
