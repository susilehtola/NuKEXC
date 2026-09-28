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

#include "standards.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "sto_basis_reader.hpp"

using namespace Nukexc;

TEST_CASE("NeighborList: benzene regression",
          "[octree][benzene][regression]") {
  // Screening must actually discard basis functions: the neighbor list has to
  // come out strictly smaller than the dense num_boxes * N product. Benzene at
  // a 1e-6 cutoff currently drops ~18% of the pairs, which is too close to the
  // noise to pin to a fixed percentage -- so the assertion is just "screening
  // removed something" and the measured figure is printed for tracking. The
  // taxol case below carries the hard sparsity floor.
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  Molecule mol = make_benzene();
  STOBasisSet basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-6);
  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 30, 30);

  const int points_per_box = 8;

  auto bb = create_bounding_boxes(grid, points_per_box);

  NeighborList nl;
  build_neighbor_list(basis, bb, points_per_box, grid.quad_points.extent(0),
                      nl, true, "benzene_histogram");

  auto offsets_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.offsets);

  const int64_t num_boxes = static_cast<int64_t>(bb.extent(0));
  const int64_t N = static_cast<int64_t>(basis.nbf());
  const int64_t dense = num_boxes * N;
  const int64_t screened = offsets_h(num_boxes);
  const double sparsity = 1.0 - static_cast<double>(screened) / dense;

  std::cout << "[benzene regression]\n"
            << "  boxes    : " << num_boxes << '\n'
            << "  basis    : " << N << '\n'
            << "  dense    : " << dense << '\n'
            << "  screened : " << screened << '\n'
            << "  sparsity : " << std::fixed << std::setprecision(1)
            << sparsity * 100 << "%\n";

  // Screening removed work -- the property this regression test exists for.
  CHECK(screened < dense);

  // nl.offsets is a CSR prefix array, so it must be non-decreasing.
  for (int b = 0; b < num_boxes; ++b)
    CHECK(offsets_h(b + 1) >= offsets_h(b));
}

TEST_CASE("NeighborList: taxol regression",
          "[octree][neighborlist][regression]") {
  // Same screening regression as the benzene case above, on a molecule large
  // enough for a hard sparsity floor to be meaningful.
  //
  // NOTE: GPU-only. On a CPU build the whole body is preprocessed away and this
  // test case passes without exercising anything.
#if (defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA))
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  Molecule mol = make_taxol();
  STOBasisSet basis = test::load_adf_basis(mol, "input/zorabasis/TZP", 1e-6);
  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 30, 30);

  const int points_per_box = 8;

  auto bb = create_bounding_boxes(grid, points_per_box);
  NeighborList nl;
  build_neighbor_list(basis, bb, points_per_box, grid.quad_points.extent(0),
                      nl, true, "taxol_histogram");

  auto offsets_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.offsets);

  const int64_t num_boxes = static_cast<int64_t>(bb.extent(0));
  const int64_t N = static_cast<int64_t>(basis.nbf());
  const int64_t dense = num_boxes * N;
  const int64_t screened = offsets_h(num_boxes);
  const double sparsity = 1.0 - static_cast<double>(screened) / dense;

  std::cout << "[taxol regression]\n"
            << "  boxes    : " << num_boxes << '\n'
            << "  basis    : " << N << '\n'
            << "  dense    : " << dense << '\n'
            << "  screened : " << screened << '\n'
            << "  sparsity : " << std::fixed << std::setprecision(1)
            << sparsity * 100 << "%\n";

  // Taxol on TZP at a 1e-6 cutoff: expect at least 30% of the dense pairs gone.
  CHECK(sparsity > 0.3);

  // nl.offsets is a CSR prefix array, so it must be non-decreasing.
  for (int b = 0; b < num_boxes; ++b)
    CHECK(offsets_h(b + 1) >= offsets_h(b));
#endif
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int result = 0;
  {
    result = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return result;
}
