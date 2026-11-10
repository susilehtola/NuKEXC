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
// Shared command-line I/O helpers for the test / benchmark drivers.
//
// Every driver that takes command-line arguments used to carry its own copy of
// the same three argument-parsing lambdas, a `repeat()` helper and a hand-drawn
// "config box". Each driver still owns its own Config struct (the fields differ
// from test to test), but the boilerplate machinery now lives here.
// ===========================================================================

#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// One argv token, with the "--prefix=<value>" matching logic each driver used
// to duplicate as local lambdas. A driver's parse_args() constructs one
// ArgParser per token and chains the *_opt() calls, e.g.:
//
//   for (int i = 1; i < argc; ++i) {
//     ArgParser p{argv[i]};
//     if (p.arg == "--help") { ...; std::exit(0); }
//     else if (!p.string_opt("--xyz=", cfg.xyz_file) &&
//              !p.int_opt("--nrad=", cfg.nrad) &&
//              ...) {
//       throw std::runtime_error("Unknown argument: " + p.arg);
//     }
//   }
// ---------------------------------------------------------------------------
struct ArgParser {
  std::string arg;

  // If `arg` starts with `prefix`, parse the remainder into `out` and return
  // true; otherwise leave `out` untouched and return false.
  bool string_opt(const std::string &prefix, std::string &out) const;
  bool int_opt(const std::string &prefix, int &out) const;
  bool double_opt(const std::string &prefix, double &out) const;

  // A valueless switch: if `arg` is exactly `name`, set `out` to true and
  // return true. Takes the WHOLE token, not a prefix, so "--foo" does not
  // also match "--foobar".
  bool bool_opt(const std::string &name, bool &out) const;
};

// Concatenate `n` copies of `s` (used to draw the box frame lines).
std::string repeat(const std::string &s, int n);

// Render any streamable value the way std::cout would, so it can be dropped
// into a print_config_box() cell. Keeps scientific notation (e.g. "1e-06")
// instead of std::to_string's "0.000001".
template <typename T> std::string cfg_val(const T &v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

// Print a titled two-column configuration box:
//
//   ┌────────────────────────────────┐
//   │           <title>              │
//   ├──────────────────┬─────────────┤
//   │ <label>          │ <value>     │
//   └──────────────────┴─────────────┘
//
// Column widths are derived from the longest label/value.
void print_config_box(
    const std::string &title,
    const std::vector<std::pair<std::string, std::string>> &rows);
