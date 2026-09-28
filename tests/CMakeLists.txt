# Dependencies (Catch2, Armadillo, OpenOrbitalOptimizer) and the shared
# NUKEXC_TEST_SUPPORT_DIR are set up once by the top-level CMakeLists, since
# benchmarking/ and convergence_studies/ need them too.

add_executable(test_grid standards.cxx test_grid.cxx )
add_executable(test_partition standards.cxx test_partition.cxx )
add_executable(test_spherical_harmonics test_spherical_harmonics.cxx)
add_executable(test_molecule test_molecule.cxx)
add_executable(test_basisset test_basisset.cxx test_io.cxx)
add_executable(test_integration test_integration.cxx standards.cxx)
add_executable(test_hydrogen test_hydrogen.cxx)
add_executable(test_arborx test_arborx.cxx standards.cxx)
add_executable(test_neighborlist test_neighborlist.cxx standards.cxx)
add_executable(test_potential test_potential.cxx)
add_executable(test_density test_density.cxx)
add_executable(test_libxc test_libxc.cxx)
add_executable(test_helium test_helium.cxx)
add_executable(test_gto_basis test_gto_basis.cxx)
add_executable(test_sto_contracted test_sto_contracted.cxx)
add_executable(test_gto_eval test_gto_eval.cxx)
add_executable(test_gto_overlap test_gto_overlap.cxx)

add_executable(standalone standards.cxx standalone.cxx test_io.cxx)
add_executable(minimal_example standards.cxx minimal_example.cxx)

add_custom_target(copy_test_inputs
    COMMAND  ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/input" "${CMAKE_CURRENT_BINARY_DIR}/input"
    COMMENT "Ensuring input directory is available in the build folder"
    )

# Create a list of all your test targets
set(ALL_TEST_TARGETS
    test_grid
    test_partition
    test_spherical_harmonics
    test_molecule
    test_basisset
    test_integration
    test_hydrogen
    test_arborx
    test_neighborlist
    test_potential
    test_density
    test_libxc
    test_helium
    test_gto_basis
    test_sto_contracted
    test_gto_eval
    test_gto_overlap

    standalone
    minimal_example

)

# Apply linking to all of them at once
foreach(tgt IN LISTS ALL_TEST_TARGETS)
    target_link_libraries(${tgt} PRIVATE libnukexc Catch2::Catch2)

    add_dependencies(${tgt} copy_test_inputs)
endforeach()

nukexc_ptxas_verbose(${ALL_TEST_TARGETS})

add_test( NAME GRID_TEST COMMAND $<TARGET_FILE:test_grid> )
add_test( NAME PARTITION_TEST COMMAND $<TARGET_FILE:test_partition> )
add_test( NAME HARMONICS_TEST COMMAND $<TARGET_FILE:test_spherical_harmonics> )
add_test( NAME MOLECULE_TEST COMMAND $<TARGET_FILE:test_molecule> )
add_test( NAME BASISSET_TEST COMMAND $<TARGET_FILE:test_basisset> )
add_test( NAME INTEGRATION_TEST COMMAND $<TARGET_FILE:test_integration> )
add_test( NAME HYDROGEN_TEST COMMAND $<TARGET_FILE:test_hydrogen> )
add_test( NAME NEIGHBORLIST_TEST COMMAND $<TARGET_FILE:test_neighborlist> )
add_test( NAME POTENTIAL_TEST COMMAND $<TARGET_FILE:test_potential> )
add_test( NAME DENSITY_TEST COMMAND $<TARGET_FILE:test_density> )
add_test( NAME LIBXC_TEST COMMAND $<TARGET_FILE:test_libxc> )
add_test( NAME HELIUM_TEST COMMAND $<TARGET_FILE:test_helium> )
add_test( NAME GTO_BASIS_TEST COMMAND $<TARGET_FILE:test_gto_basis> )
add_test( NAME STO_CONTRACTED_TEST COMMAND $<TARGET_FILE:test_sto_contracted> )
add_test( NAME GTO_EVAL_TEST COMMAND $<TARGET_FILE:test_gto_eval> )
add_test( NAME GTO_OVERLAP_TEST COMMAND $<TARGET_FILE:test_gto_overlap> )
# test_arborx was built but never registered, so nothing ran it.
add_test( NAME ARBORX_TEST COMMAND $<TARGET_FILE:test_arborx> )
