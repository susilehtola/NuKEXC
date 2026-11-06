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

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string_view>

#include <catch2/catch_all.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "nukexc/grid.hpp"
#include "standards.hpp"

using namespace Nukexc;

using bk_type = IntegratorXX::Becke<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;

using ah_type = IntegratorXX::AhrensBeylkin<double>;
using de_type = IntegratorXX::Delley<double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using wo_type = IntegratorXX::Womersley<double>;

using sph_test_types =
    std::tuple<std::tuple<bk_type, ah_type>, std::tuple<bk_type, de_type>,
               std::tuple<ta_type, ll_type>, std::tuple<ta_type, wo_type>>;

TEMPLATE_LIST_TEST_CASE("Unpruned", "[sph-gen]", sph_test_types) {

  using namespace IntegratorXX;
  using radial_type =
      std::decay_t<decltype(std::get<0>(std::declval<TestType>()))>;
  using angular_type =
      std::decay_t<decltype(std::get<1>(std::declval<TestType>()))>;
  using angular_traits = quadrature_traits<angular_type>;

  using spherical_type = SphericalQuadrature<radial_type, angular_type>;

  size_t nrad = 10;
  size_t nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(
          3)); // Smallest possible angular grid

  // Generate the quadrature manually
  radial_type rq(nrad, 1.0);
  angular_type aq(nang);
  spherical_type sph_ref(rq, aq);

  // Generate via runtime API
  auto rad_spec = radial_from_type<radial_type>();
  auto rad_traits = make_radial_traits(rad_spec, nrad, 1.0);
  UnprunedSphericalGridSpecification unp(
      rad_spec, *rad_traits, angular_from_type<angular_type>(), nang);

  auto sph = SphericalGridFactory::generate_grid(unp);

  // Check that they're the same
  REQUIRE(sph->npts() == sph_ref.npts());

  const auto npts = sph->npts();
  for (auto i = 0; i < npts; ++i) {
    auto pt = sph->points()[i];
    auto pt_ref = sph_ref.points()[i];
    REQUIRE_THAT(pt[0], Catch::Matchers::WithinAbs(pt_ref[0], 1e-15));
    REQUIRE_THAT(pt[1], Catch::Matchers::WithinAbs(pt_ref[1], 1e-15));
    REQUIRE_THAT(pt[2], Catch::Matchers::WithinAbs(pt_ref[2], 1e-15));

    auto w = sph->weights()[i];
    auto w_ref = sph_ref.weights()[i];
    REQUIRE_THAT(w, Catch::Matchers::WithinAbs(w_ref, 1e-15));
  }
}

void visualize_points_with_tiles(const FlatGrid &grid) {

  Kokkos::View<Point *, ExecSpace> points_dev = grid.quad_points;
  auto pts =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_dev);
  const int total = static_cast<int>(pts.extent(0));

  const std::string path = "grid/grid_points.vtk";

  std::filesystem::path p(path);
  std::filesystem::create_directories(p.parent_path());

  std::ofstream out(path);
  if (!out) {
    std::cerr << "[visualize_points_with_tiles] ERROR: cannot open " << path
              << '\n';
    return;
  }

  out << "# vtk DataFile Version 3.0\n"
      << "Grid Points\n"
      << "ASCII\n"
      << "DATASET UNSTRUCTURED_GRID\n\n";

  out << "POINTS " << total << " double\n";
  for (int i = 0; i < total; ++i)
    out << pts(i)[0] << ' ' << pts(i)[1] << ' ' << pts(i)[2] << '\n';

  // Each point is its own VTK_VERTEX cell (type 1)
  out << "\nCELLS " << total << ' ' << total * 2 << '\n';
  for (int i = 0; i < total; ++i)
    out << "1 " << i << '\n';

  out << "\nCELL_TYPES " << total << '\n';
  for (int i = 0; i < total; ++i)
    out << "1\n";

  out << "\nCELL_DATA " << total << '\n'
      << "SCALARS tile_id int 1\n"
      << "LOOKUP_TABLE default\n";

  // Colour each surviving point by its owning atom. This always matches the
  // point count (even for compacted / irregular grids), unlike the old
  // nrad*nang shell indexing.
  auto owners =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid.point_owner);
  for (int i = 0; i < total; ++i)
    out << owners(i) << '\n';

  out.flush();
  std::cout << "[visualize_points_with_tiles]\n"
            << "  Points : " << total << '\n'
            << "  Output : " << path << '\n'
            << "  Tip    : load alongside bounding_boxes.vtk in ParaView;\n"
            << "           matching tile_id colours confirm points are\n"
            << "           correctly contained within their boxes.\n\n";
}

TEST_CASE("The same molecule gives the same grid, in the same order",
          "[grid][determinism]") {
  // make_flat_grid prunes points whose partition weight is negligible and
  // compacts what is left. It used to assign each survivor its slot with an
  // atomic counter, which hands out slots in thread-arrival order: the point
  // SET was reproducible but the point ORDER was not, so two grids built
  // from the same molecule in the same process came out permuted with
  // respect to each other. That makes every quadrature sum accumulate in a
  // different order -- results moving in their last bits run to run -- and
  // it makes anything derived from the point index, including the grid
  // batching, unreproducible. It also turns a latent bug anywhere downstream
  // into an intermittent one, which is how this was found.
  Molecule mol(
      std::vector<std::vector<double>>{{0., 0., 0.}, {0., 0., 1.4}},
      std::vector<unsigned>{1u, 8u});

  const auto a = make_flat_grid<ta_type, ll_type>(mol, 40, 20);
  const auto b = make_flat_grid<ta_type, ll_type>(mol, 40, 20);

  REQUIRE(a.quad_points.extent(0) == b.quad_points.extent(0));
  REQUIRE(a.quad_points.extent(0) > 1000); // or there is nothing to permute

  auto pa = Kokkos::create_mirror_view_and_copy(HostSpace{}, a.quad_points);
  auto pb = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.quad_points);
  auto wa = Kokkos::create_mirror_view_and_copy(HostSpace{}, a.weights);
  auto wb = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.weights);
  auto oa = Kokkos::create_mirror_view_and_copy(HostSpace{}, a.point_owner);
  auto ob = Kokkos::create_mirror_view_and_copy(HostSpace{}, b.point_owner);

  // Bit for bit, index by index: the two runs did the same arithmetic, so
  // anything short of equality is a reordering.
  for (size_t i = 0; i < pa.extent(0); ++i) {
    INFO("point " << i);
    REQUIRE(pa(i)[0] == pb(i)[0]);
    REQUIRE(pa(i)[1] == pb(i)[1]);
    REQUIRE(pa(i)[2] == pb(i)[2]);
    REQUIRE(wa(i) == wb(i));
    REQUIRE(oa(i) == ob(i));
  }
}

int main() {

  Kokkos::initialize();

  int result = Catch::Session().run();
  {
    Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}},
                 std::vector<unsigned>{1u});
    auto grid = make_flat_grid<ta_type, ll_type>(mol, 20, 40);
    visualize_points_with_tiles(grid);
  } // grid and mol destroyed here, while Kokkos is still alive
  Kokkos::finalize();
  return result;
}
