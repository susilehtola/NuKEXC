# Performance benchmarks. These are drivers, not pass/fail tests, so nothing
# here is registered with add_test() -- run the binaries by hand from
# ${CMAKE_BINARY_DIR}/benchmarking (see benchmarking/README.md).
#
# Catch2/Armadillo/OOO and NUKEXC_TEST_SUPPORT_DIR come from the top-level
# CMakeLists; the shared driver helpers (standards.*, test_io.*) still live in
# tests/ and are compiled in from there.

set(SUPPORT ${NUKEXC_TEST_SUPPORT_DIR})

add_executable(benchmark_partition ${SUPPORT}/standards.cxx benchmark_partition.cxx)
add_executable(benchmark_coreH     ${SUPPORT}/standards.cxx benchmark_coreH.cxx  ${SUPPORT}/test_io.cxx)
add_executable(benchmark_cutoff    ${SUPPORT}/standards.cxx benchmark_cutoff.cxx ${SUPPORT}/test_io.cxx)
add_executable(benchmark_xc        ${SUPPORT}/standards.cxx benchmark_xc.cxx     ${SUPPORT}/test_io.cxx)
add_executable(benchmark_collocation ${SUPPORT}/standards.cxx benchmark_collocation.cxx)

set(ALL_BENCHMARK_TARGETS
    benchmark_partition
    benchmark_coreH
    benchmark_cutoff
    benchmark_xc
    benchmark_collocation
)

# The benchmarks resolve their data with paths relative to the working
# directory ("input/zorabasis/TZP", ...), so each build directory that holds
# benchmark binaries needs its own copy of input/.
add_custom_target(copy_benchmark_inputs
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/input" "${CMAKE_CURRENT_BINARY_DIR}/input"
    COMMENT "Ensuring input directory is available in the benchmarking build folder"
    )

foreach(tgt IN LISTS ALL_BENCHMARK_TARGETS)
    target_link_libraries(${tgt} PRIVATE libnukexc Catch2::Catch2)
    target_include_directories(${tgt} PRIVATE ${SUPPORT})
    add_dependencies(${tgt} copy_benchmark_inputs)
endforeach()

nukexc_ptxas_verbose(${ALL_BENCHMARK_TARGETS})
