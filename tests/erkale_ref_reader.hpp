/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// ============================================================================
//  Reader for the ERKALE reference files under tests/reference
// ============================================================================
//
// Every such file carries the basis set together with the quantities computed
// from it, so that no test has to assume the two codes were handed the same
// basis -- which is otherwise the easiest way to make a cross-code comparison
// meaningless. See tests/reference/README.md for the format and provenance.
//
// The parser lives here rather than in one test because there is more than
// one consumer: test_erkale_overlap.cxx wants the overlap, test_erkale_xc.cxx
// wants density matrices and exchange-correlation matrices. Everything beyond
// the shells is read as a NAMED BLOCK of doubles, so a new reference quantity
// needs no parser change at all -- only a dumper that writes
//
//     <name> <count>
//     <value>            x count
//
// which is also why the lookups throw on a missing name: a test asking for a
// block the file does not have has either been pointed at a stale reference
// or has a typo, and both should say so rather than quietly compare zeros.

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nukexc/gto_basisset.hpp>
#include <nukexc/gto_shell.hpp>

namespace test {

/// An ERKALE reference file: the basis, and the named blocks computed from it.
struct ErkaleReference {
  Nukexc::GTOBasisSet<double> basis;
  int nbf = 0;
  int nshell = 0;
  std::map<std::string, std::vector<double>> blocks;

  /// A named block. Throws if the file does not have it.
  const std::vector<double> &block(const std::string &name) const {
    const auto it = blocks.find(name);
    if (it == blocks.end()) {
      std::string have;
      for (const auto &kv : blocks)
        have += " " + kv.first;
      throw std::runtime_error("reference has no block '" + name +
                               "'; it has:" + have);
    }
    return it->second;
  }

  /// A named block of exactly one value.
  double scalar(const std::string &name) const {
    const auto &b = block(name);
    if (b.size() != 1)
      throw std::runtime_error("reference block '" + name + "' has " +
                               std::to_string(b.size()) +
                               " values, expected one");
    return b[0];
  }

  /// A named block that must be nbf x nbf, row-major.
  const std::vector<double> &matrix(const std::string &name) const {
    const auto &b = block(name);
    const size_t want = static_cast<size_t>(nbf) * nbf;
    if (b.size() != want)
      throw std::runtime_error("reference block '" + name + "' has " +
                               std::to_string(b.size()) + " values, expected " +
                               std::to_string(want) + " for nbf = " +
                               std::to_string(nbf));
    return b;
  }
};

/**
 *  @brief Parse an ERKALE reference file.
 *
 *  @param path  file to read, relative to the test working directory
 *  @throws std::runtime_error if it cannot be opened, if a block is short,
 *          or if the shells do not give the nbf the file declares
 */
inline ErkaleReference load_erkale_reference(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open reference " + path +
                             " (see tests/reference/README.md)");

  std::vector<Nukexc::GTOShell<double>> shells;
  std::map<std::string, std::vector<double>> blocks;
  int nbf = 0, nshell = 0;
  std::string line;

  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag.empty())
      continue;

    if (tag == "nbf") {
      ss >> nbf;
    } else if (tag == "nshell") {
      ss >> nshell;
    } else if (tag == "shell") {
      int l, sph, nprim, nctr;
      double ox, oy, oz;
      ss >> l >> sph >> ox >> oy >> oz >> nprim >> nctr;

      std::vector<double> alpha(nprim),
          coeff(static_cast<size_t>(nprim) * nctr);
      for (int p = 0; p < nprim; ++p) {
        std::getline(in, line);
        std::istringstream es(line);
        std::string t;
        es >> t >> alpha[p];
      }
      for (size_t c = 0; c < coeff.size(); ++c) {
        std::getline(in, line);
        std::istringstream cs(line);
        std::string t;
        cs >> t >> coeff[c];
      }

      // ERKALE writes coefficients of UNNORMALIZED primitives, already
      // normalized -- exactly NormalizationConvention::ERKALE's output. Ask
      // for that convention rather than None so the round trip also checks
      // that renormalizing an already-normalized shell is idempotent.
      shells.emplace_back(
          l, sph ? Nukexc::ShellType::Spherical : Nukexc::ShellType::Cartesian,
          std::array<double, 3>{ox, oy, oz}, std::move(alpha), std::move(coeff),
          nctr, Nukexc::NormalizationConvention::ERKALE);
    } else {
      // Any other tag is a named block: a count, then that many values.
      size_t n = 0;
      if (!(ss >> n))
        throw std::runtime_error("reference " + path + ": tag '" + tag +
                                 "' is not followed by a count");
      std::vector<double> v(n);
      for (size_t i = 0; i < n; ++i)
        if (!(in >> v[i]))
          throw std::runtime_error("reference " + path + ": block '" + tag +
                                   "' ended after " + std::to_string(i) +
                                   " of " + std::to_string(n) + " values");
      blocks.emplace(tag, std::move(v));
    }
  }

  ErkaleReference r{Nukexc::GTOBasisSet<double>(std::move(shells)), nbf, nshell,
                    std::move(blocks)};
  if (r.basis.nbf() != nbf)
    throw std::runtime_error("reference " + path + " declares nbf " +
                             std::to_string(nbf) + " but the shells give " +
                             std::to_string(r.basis.nbf()));
  return r;
}

} // namespace test
