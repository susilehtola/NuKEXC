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

#include "nukexc/nukexc_config.hpp"
#include "nukexc/octree.hpp"
#include <Kokkos_Core.hpp>

#include <Kokkos_Core_fwd.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <iomanip>
#include <iostream>

#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/stobasis.hpp>

#include <catch2/catch_all.hpp>
#include <catch2/catch_assertion_info.hpp>
#include <vector>

#include "test_io.hpp"

#include "sto_basis_reader.hpp"

using namespace Nukexc;

struct Config {
  int benchmark = 0;
  std::string algorithm = "fasts";
  int nrad = 50;
  int nang = 20;
  double screening_tol = 1e-6;
  int max_points_per_box = 32;
  int tile = 32; // neighbor-tile size for the "tiled" algorithm
};

Config parse_args(int argc, char *argv[]) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    ArgParser p{argv[i]};

    if (p.arg == "--help" || p.arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "  --benchmark=<int> Benchmark? (0=No, 1=Yes) (default: "
          << cfg.benchmark << ")\n"

          << "  --alg=<string>    "
             "Algorithm(fasts,fastv,slows,slowv,cached,tiled) (default: "
          << cfg.algorithm << ")\n"
          << "  --nrad=<int>              Radial grid points        (default: "
          << cfg.nrad << ")\n"
          << "  --nang=<int>              Angular grid points       (default: "
          << cfg.nang << ")\n"
          << "  --tol=<float>             Screening tolerance       (default: "
          << cfg.screening_tol << ")\n"
          << "  --box-size=<int>          Max points per box        (default: "
          << cfg.max_points_per_box << ")\n"
          << "  --tile=<int>              Neighbor tile (alg=tiled) (default: "
          << cfg.tile << ")\n";
      std::exit(0);
    } else if (!p.int_opt("--benchmark=", cfg.benchmark) &&
               !p.string_opt("--alg=", cfg.algorithm) &&
               !p.int_opt("--nrad=", cfg.nrad) &&
               !p.int_opt("--nang=", cfg.nang) &&
               !p.double_opt("--tol=", cfg.screening_tol) &&
               !p.int_opt("--box-size=", cfg.max_points_per_box) &&
               !p.int_opt("--tile=", cfg.tile)) {
      throw std::runtime_error("Unknown argument: " + p.arg + " (try --help)");
    }
  }
  if (cfg.nrad <= 0 || cfg.nang <= 0)
    throw std::runtime_error("--nrad and --nang must be positive");
  return cfg;
}

void print_config(const Config &cfg) {
  print_config_box("SCF Benchmark Configuration",
                   {
                       {"Benchmark", cfg_val(cfg.benchmark)},
                       {"Algorithm", cfg.algorithm},
                       {"Radial points", cfg_val(cfg.nrad)},
                       {"Angular order", cfg_val(cfg.nang)},
                       {"Screening tolerance", cfg_val(cfg.screening_tol)},
                       {"Max points per box", cfg_val(cfg.max_points_per_box)},
                   });
}
TEST_CASE("H2O_thakkar", "[h20_thakkar]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = test::load_thakkar_primitives(mol, "input/k99light/neutral");
};

TEST_CASE("H2O_adf_regular", "[h20][adf]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = test::load_adf_basis(mol, "input/zorabasis/TZP");
};

TEST_CASE("H2O_adf_QZ4P", "[h20][adf]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P");
};

TEST_CASE("H2O_adf_QZ4P_AUX", "[h20][adf][fit]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const bool fit = true;
  const double tol = 1e-10;
  STOBasisSet basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P", tol, fit);
};

TEST_CASE("H2O_adf_QZ4P_AUX_cholesky", "[h20][adf][fit][cholesky]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const bool fit = true;
  const double tol = 1e-10;

  STOBasisSet basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P", tol);
  STOBasisSet basis_cholesky =
      test::load_adf_basis(mol, "input/zorabasis_cholesky/QZ4P.cholesky", tol);
  STOBasisSet basis_aux = test::load_adf_basis(mol, "input/zorabasis/QZ4P", tol, fit);
  STOBasisSet basis_aux_cholesky =
      test::load_adf_basis(mol, "input/zorabasis_cholesky/QZ4P.cholesky", tol, fit);

  // Check that both regular basis sets are the same
  REQUIRE(basis.nbf() == basis_cholesky.nbf());

  // Check that the auxiliary basis set created with auto Aux is larger than
  // the regular ADF auxiliary basis
  REQUIRE(basis_aux.nbf() < basis_aux_cholesky.nbf());
};

TEST_CASE("Basis Cutoff", "[h20][cutoff]") {

  using bk_type = IntegratorXX::Becke<double, double>;
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  double cutoff_tol = 1e-15;
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P", cutoff_tol);

  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 100, 50);

  int N = basis.nbf();
  int G = grid.quad_points.extent(0);

  Kokkos::View<double **> grid_values("Values", N, G);
  Kokkos::View<double **> grid_r("Radii", N, G);

  Kokkos::parallel_for(
      "Evaluate basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, G}),
      KOKKOS_LAMBDA(const int i, const int g) {
        double r = dist(basis.O(i), grid.quad_points(g));
        basis_eval(basis, i, grid.quad_points(g)[0], grid.quad_points(g)[1],
                   grid.quad_points(g)[2], grid_values(i, g));
        grid_r(i, g) = r;
      });

  auto grid_values_h = Kokkos::create_mirror_view(grid_values);
  auto grid_r_h = Kokkos::create_mirror_view(grid_r);
  auto cutoff_radii_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::fence();
  Kokkos::deep_copy(grid_values_h, grid_values);
  Kokkos::deep_copy(grid_r_h, grid_r);
  Kokkos::deep_copy(cutoff_radii_h, basis.cutoff_radii);

  int count = 0;
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int g = 0; g < G; ++g) {
      if (cutoff_radii_h(i) < grid_r_h(i, g)) {
        ++count;
        REQUIRE_THAT(grid_values_h(i, g),
                     Catch::Matchers::WithinAbs(0., cutoff_tol));
      }
    }
  }
};

void benchmark_collocation(Config cfg) {

  using bk_type = IntegratorXX::Becke<double, double>;
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  double cutoff_tol = cfg.screening_tol;
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = test::load_adf_basis(mol, "input/zorabasis/QZ4P", cutoff_tol);

  FlatGrid grid = make_flat_grid<bk_type, ll_type>(mol, cfg.nrad, cfg.nang);
  int N_bf = basis.nbf();
  int N_quad = grid.quad_points.extent(0);

  const int max_points_per_box = cfg.max_points_per_box;
  NeighborList nl;
  auto bounding_boxes = create_bounding_boxes(grid, max_points_per_box);
  build_neighbor_list(STOPrimitiveEvaluator(basis), bounding_boxes, max_points_per_box, grid.quad_points, nl);

  const int num_boxes = nl.offsets.extent(0) - 1;
  Kokkos::View<double **> S("overlap", N_bf,
                            N_bf); // per box, or global-indexed
  ExecSpace space;

  Kokkos::TeamPolicy<ExecSpace> policy_boxes(space, num_boxes, Kokkos::AUTO(),
                                             Kokkos::AUTO());

  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
  typedef ExecSpace::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;

  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  int scratch_size_team = shared_view_double::shmem_size(max_points_per_box) +
                          shared_view_points::shmem_size(max_points_per_box);

  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_size_team));

  Kokkos::Timer time_vector, time_serial;
  time_vector.reset();
  time_serial.reset();

  if (cfg.algorithm == "fastv") {
    double start_vector = time_vector.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Thread->Vector] [basis_eval_fast]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                        num_neighbors),
              [=](const int local_i, const int local_j) {
                const int global_i = nl.neighbors(start_neighbors + local_i);
                const int global_j = nl.neighbors(start_neighbors + local_j);

                const ShellParams shell_i = load_shell(basis, global_i);
                const ShellParams shell_j = load_shell(basis, global_j);

                double sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team_member, num_points),
                    [=](const int local_g, double &local_sum) {
                      local_sum +=
                          basis_eval_fast(shell_i, points_scratch(local_g)[0],
                                          points_scratch(local_g)[1],
                                          points_scratch(local_g)[2]) *
                          basis_eval_fast(shell_j, points_scratch(local_g)[0],
                                          points_scratch(local_g)[1],
                                          points_scratch(local_g)[2]);
                    },
                    sum);
                S(local_i, local_j) = sum;
              });
        });

    Kokkos::fence();
    double end_vector = time_vector.seconds();
    Kokkos::printf(
        "Benchmark [Team->Thread->Vector] [basis_eval_fast] took %fs \n",
        end_vector - start_vector);
  } else if (cfg.algorithm == "fasts") {

    double start_serial = time_serial.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Thread->Serial] [basis_eval_fast]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                        num_neighbors),
              [=](const int local_i, const int local_j) {
                const int global_i = nl.neighbors(start_neighbors + local_i);
                const int global_j = nl.neighbors(start_neighbors + local_j);

                const ShellParams shell_i = load_shell(basis, global_i);
                const ShellParams shell_j = load_shell(basis, global_j);

                double sum = 0;
                for (int local_g = 0; local_g < num_points; ++local_g) {
                  sum += basis_eval_fast(shell_i, points_scratch(local_g)[0],
                                         points_scratch(local_g)[1],
                                         points_scratch(local_g)[2]) *
                         basis_eval_fast(shell_j, points_scratch(local_g)[0],
                                         points_scratch(local_g)[1],
                                         points_scratch(local_g)[2]);
                }

                S(local_i, local_j) = sum;
              });
        });
    Kokkos::fence();

    double end_serial = time_serial.seconds();
    Kokkos::printf(
        "Benchmark [Team->Thread->Serial] [basis_eval_fast] took %fs \n",
        end_serial - start_serial);

  } else if (cfg.algorithm == "slowv") {
    double start_vector_slow = time_vector.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Thread->Vector] [basis_eval]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                        num_neighbors),
              [=](const int local_i, const int local_j) {
                const int global_i = nl.neighbors(start_neighbors + local_i);
                const int global_j = nl.neighbors(start_neighbors + local_j);

                double sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team_member, num_points),
                    [=](const int local_g, double &local_sum) {
                      double phi_i;
                      double phi_j;

                      basis_eval(basis, global_i, points_scratch(local_g)[0],
                                 points_scratch(local_g)[1],
                                 points_scratch(local_g)[2], phi_i);

                      basis_eval(basis, global_j, points_scratch(local_g)[0],
                                 points_scratch(local_g)[1],
                                 points_scratch(local_g)[2], phi_j);
                      local_sum += phi_i * phi_j;
                    },
                    sum);

                S(local_i, local_j) = sum;
              });
        });

    Kokkos::fence();
    double end_vector_slow = time_vector.seconds();
    Kokkos::printf("Benchmark [Team->Thread->Vector] [basis_eval] took %fs \n",
                   end_vector_slow - start_vector_slow);
  } else if (cfg.algorithm == "slows") {
    double start_serial_slow = time_serial.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Thread->Serial] [basis_eval]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                        num_neighbors),
              [=](const int local_i, const int local_j) {
                const int global_i = nl.neighbors(start_neighbors + local_i);
                const int global_j = nl.neighbors(start_neighbors + local_j);

                double sum = 0;
                for (int local_g = 0; local_g < num_points; ++local_g) {
                  double phi_i;
                  double phi_j;

                  basis_eval(basis, global_i, points_scratch(local_g)[0],
                             points_scratch(local_g)[1],
                             points_scratch(local_g)[2], phi_i);

                  basis_eval(basis, global_j, points_scratch(local_g)[0],
                             points_scratch(local_g)[1],
                             points_scratch(local_g)[2], phi_j);
                  sum += phi_i * phi_j;
                }
                S(local_i, local_j) = sum;
              });
        });

    Kokkos::fence();
    double end_serial_slow = time_serial.seconds();
    Kokkos::printf("Benchmark [Team->Thread->Serial] [basis_eval] took %fs \n",
                   end_serial_slow - start_serial_slow);
  } else if (cfg.algorithm == "cached") {
    // Precompute phi(neighbor, point) once into team scratch, then form the
    // pairwise S(i,j) by reading the cache. This removes the O(neighbors^2 *
    // points) redundant basis_eval_fast calls (and the two live ShellParams)
    // from the hot reduction loop -- the register-pressure fix intended for
    // the screened Coulomb and exchange kernels. The Coulomb one now packs
    // the collocation per box instead (see nukexc/packed_blocks.hpp), which
    // removes the redundancy rather than caching it; the exchange one has
    // not been converted yet.

    // Upper bound on neighbors per box, needed to size the phi cache scratch.
    auto offsets_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.offsets);
    int max_neighbors = 0;
    for (int b = 0; b + 1 < static_cast<int>(offsets_h.extent(0)); ++b) {
      const int nb = offsets_h(b + 1) - offsets_h(b);
      if (nb > max_neighbors)
        max_neighbors = nb;
    }

    const int scratch_cached =
        shared_view_double::shmem_size(max_points_per_box) +           // weights
        shared_view_points::shmem_size(max_points_per_box) +           // points
        shared_view_double::shmem_size(max_neighbors *                 // phi cache
                                       max_points_per_box);
    policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_cached));

    Kokkos::Timer time_cached;
    double start_cached = time_cached.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Cache->Thread] [basis_eval_fast cached]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          // phi_cache is laid out [neighbor][point] with stride num_points.
          shared_view_double phi_cache(team_member.team_scratch(0),
                                       num_neighbors * num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });

          team_member.team_barrier();

          // Evaluate each basis function on every point exactly once.
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, num_neighbors),
              [=](const int local_i) {
                const int global_i = nl.neighbors(start_neighbors + local_i);
                const ShellParams shell_i = load_shell(basis, global_i);
                const int base_i = local_i * num_points;
                for (int local_g = 0; local_g < num_points; ++local_g) {
                  phi_cache(base_i + local_g) = basis_eval_fast(
                      shell_i, points_scratch(local_g)[0],
                      points_scratch(local_g)[1], points_scratch(local_g)[2]);
                }
              });

          team_member.team_barrier();

          // Pairwise contraction reads only the cache: no ShellParams, no
          // harmonic switch, no int_pow in the inner loop.
          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                        num_neighbors),
              [=](const int local_i, const int local_j) {
                const int base_i = local_i * num_points;
                const int base_j = local_j * num_points;
                double sum = 0;
                for (int local_g = 0; local_g < num_points; ++local_g) {
                  sum +=
                      phi_cache(base_i + local_g) * phi_cache(base_j + local_g);
                }
                S(local_i, local_j) = sum;
              });
        });

    Kokkos::fence();
    double end_cached = time_cached.seconds();
    Kokkos::printf(
        "Benchmark [Team->Cache->Thread] [basis_eval_fast cached] took %fs \n",
        end_cached - start_cached);
  } else if (cfg.algorithm == "tiled") {
    // Neighbor-tiled cache. Same idea as "cached" (evaluate phi once, then
    // contract from scratch) but the phi buffers hold only a fixed-size TILE of
    // neighbors at a time, so the shared-memory footprint depends on tile_size
    // and max_points_per_box ONLY -- never on num_neighbors. This is what makes
    // it usable for large molecules where the full [num_neighbors x points]
    // cache overflows scratch.
    //
    // The pairwise S = Phi^T Phi is done as a blocked, lower-triangular product:
    // tile ti is held in phi_a, tile tj (<= ti) in phi_b, and the ilen x jlen
    // block is contracted from the two buffers. A tile's phi is re-evaluated
    // once per outer tile it pairs with -- eval work is ~num_neighbors^2/(2*tile)
    // vs num_neighbors^2 for the recompute kernels, i.e. ~2*tile fewer evals,
    // for a bounded 2*tile*points scratch cost.
    const int tile_size = cfg.tile;

    const int scratch_tiled =
        shared_view_double::shmem_size(max_points_per_box) +           // weights
        shared_view_points::shmem_size(max_points_per_box) +           // points
        shared_view_double::shmem_size(tile_size * max_points_per_box) + // phi_a
        shared_view_double::shmem_size(tile_size * max_points_per_box);  // phi_b
    policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_tiled));

    Kokkos::Timer time_tiled;
    double start_tiled = time_tiled.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Tile->Thread] [basis_eval_fast tiled]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);
          shared_view_points points_scratch(team_member.team_scratch(0),
                                             num_points);

          // Fixed-size neighbor tiles: footprint independent of num_neighbors.
          shared_view_double phi_a(team_member.team_scratch(0),
                                   tile_size * num_points);
          shared_view_double phi_b(team_member.team_scratch(0),
                                   tile_size * num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });
          team_member.team_barrier();

          const int num_tiles = (num_neighbors + tile_size - 1) / tile_size;

          for (int ti = 0; ti < num_tiles; ++ti) {
            const int i0 = ti * tile_size;
            const int ilen = Kokkos::min(tile_size, num_neighbors - i0);

            // Evaluate the outer tile's phi into phi_a (once per outer tile).
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, ilen), [=](const int li) {
                  const int global_i = nl.neighbors(start_neighbors + i0 + li);
                  const ShellParams shell_i = load_shell(basis, global_i);
                  const int base = li * num_points;
                  for (int g = 0; g < num_points; ++g)
                    phi_a(base + g) = basis_eval_fast(
                        shell_i, points_scratch(g)[0], points_scratch(g)[1],
                        points_scratch(g)[2]);
                });
            team_member.team_barrier();

            for (int tj = 0; tj <= ti; ++tj) {
              const int j0 = tj * tile_size;
              const int jlen = Kokkos::min(tile_size, num_neighbors - j0);

              // Diagonal tile reuses phi_a; off-diagonal is evaluated into phi_b.
              if (tj != ti) {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, jlen),
                    [=](const int lj) {
                      const int global_j =
                          nl.neighbors(start_neighbors + j0 + lj);
                      const ShellParams shell_j = load_shell(basis, global_j);
                      const int base = lj * num_points;
                      for (int g = 0; g < num_points; ++g)
                        phi_b(base + g) = basis_eval_fast(
                            shell_j, points_scratch(g)[0], points_scratch(g)[1],
                            points_scratch(g)[2]);
                    });
                team_member.team_barrier();
              }

              const double *pj = (tj == ti) ? phi_a.data() : phi_b.data();

              Kokkos::parallel_for(
                  Kokkos::TeamThreadMDRange(team_member, ilen, jlen),
                  [=](const int li, const int lj) {
                    const int base_i = li * num_points;
                    const int base_j = lj * num_points;
                    double sum = 0;
                    for (int g = 0; g < num_points; ++g)
                      sum += phi_a(base_i + g) * pj[base_j + g];
                    S(i0 + li, j0 + lj) = sum;
                  });
              team_member.team_barrier();
            }
          }
        });

    Kokkos::fence();
    double end_tiled = time_tiled.seconds();
    Kokkos::printf(
        "Benchmark [Team->Tile->Thread] [basis_eval_fast tiled] took %fs \n",
        end_tiled - start_tiled);
  }
};

int main(int argc, char *argv[]) {

  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  Kokkos::initialize();
  int result = 0;
  {
    if (cfg.benchmark) {
      print_config(cfg);
      benchmark_collocation(cfg);
    } else {
      result = Catch::Session().run();
    }
  }
  Kokkos::finalize();
  return result;
}
