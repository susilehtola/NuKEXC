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

#include "test_io.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>

bool ArgParser::string_opt(const std::string &prefix, std::string &out) const {
  if (arg.rfind(prefix, 0) == 0) {
    out = arg.substr(prefix.size());
    return true;
  }
  return false;
}

bool ArgParser::int_opt(const std::string &prefix, int &out) const {
  if (arg.rfind(prefix, 0) == 0) {
    out = std::stoi(arg.substr(prefix.size()));
    return true;
  }
  return false;
}

bool ArgParser::double_opt(const std::string &prefix, double &out) const {
  if (arg.rfind(prefix, 0) == 0) {
    out = std::stod(arg.substr(prefix.size()));
    return true;
  }
  return false;
}

bool ArgParser::bool_opt(const std::string &name, bool &out) const {
  if (arg == name) {
    out = true;
    return true;
  }
  return false;
}

std::string repeat(const std::string &s, int n) {
  std::string r;
  for (int i = 0; i < n; ++i)
    r += s;
  return r;
}

void print_config_box(
    const std::string &title,
    const std::vector<std::pair<std::string, std::string>> &rows) {
  std::size_t label_w = 0;
  std::size_t value_w = 0;
  for (const auto &r : rows) {
    label_w = std::max(label_w, r.first.size());
    value_w = std::max(value_w, r.second.size());
  }
  value_w = std::max<std::size_t>(value_w, 20);

  const int label_cell = static_cast<int>(label_w) + 2; // one space each side
  const int value_cell = static_cast<int>(value_w) + 2;
  const int full = label_cell + 1 + value_cell; // + column divider

  // Centre the title across the full inner width.
  const int title_len = static_cast<int>(title.size());
  const int pad = full > title_len ? full - title_len : 0;
  const int pad_l = pad / 2;
  const int pad_r = pad - pad_l;

  std::cout << "\n";
  std::cout << "┌" << repeat("─", full) << "┐\n";
  std::cout << "│" << repeat(" ", pad_l) << title << repeat(" ", pad_r)
            << "│\n";
  std::cout << "├" << repeat("─", label_cell) << "┬" << repeat("─", value_cell)
            << "┤\n";
  for (const auto &r : rows) {
    std::cout << "│ " << std::left << std::setw(static_cast<int>(label_w))
              << r.first << " │ " << std::left
              << std::setw(static_cast<int>(value_w)) << r.second << " │\n";
  }
  std::cout << "└" << repeat("─", label_cell) << "┴" << repeat("─", value_cell)
            << "┘\n\n";
  std::cout << std::flush;
}
