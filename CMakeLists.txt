cmake_minimum_required( VERSION 3.20 FATAL_ERROR )
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
set( FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FC Updates" )

project( NuKEXC 
	VERSION 0.1.0 
	DESCRIPTION "A library for integrating numerical atomic orbitals on GPUs with Kokkos"
	LANGUAGES C CXX )

include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Place local modules in the path
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# Guard some options settings to only default when not a subproject
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Populate BUILD_TESTING prior to dependencies to avoid clash
  include(CTest)

  # Default the built type
  if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
    set( CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build" FORCE )
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  endif()
endif()

set (SOVERSION 0.1.0)

set (nxc NuKEXC) # Namespace 

# ===== Options ====================================================================
# there are paths that work for both windows and unix but they repeat ${nxc} :-(
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    set(${nxc}_INSTALL_CMAKEDIR "${nxc}/CMake"
        CACHE STRING "Directory to which CMake files are installed")
else()
    set(${nxc}_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${nxc}"
        CACHE STRING "Directory to which CMake files are installed")
endif()
message(STATUS "Showing option ${nxc}_INSTALL_CMAKEDIR: ${${nxc}_INSTALL_CMAKEDIR}")

option(${nxc}_BUILD_TESTING "Build test-suite" ${PROJECT_IS_TOP_LEVEL})
message(STATUS "Showing option ${nxc}_BUILD_TESTING: ${${nxc}_BUILD_TESTING}")

# ==== Build =======================================================================

add_library(libnukexc INTERFACE)

target_compile_features(
	libnukexc
	INTERFACE
	cxx_std_20
)
target_include_directories(
  libnukexc
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  # Vendored, header-only libxckernel. Its headers include each other as
  # "xckernel/...", so external/ is the include root rather than
  # external/xckernel/. Regenerate with python/generate_xc_kernels.py;
  # see external/README.md for the version and command.
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(${nxc}::NuKEXC ALIAS libnukexc)

include( nukexc-integratorxx ) # finds package integratorxx
include( nukexc-kokkos) # finds package Kokkos 
include( nukexc-kokkos-kernels) # finds package Kokkos-kernels
include( nukexc-arborx) # finds package arborx
include( nukexc-libxc ) # finds libxc


set(export_properties
  "${nxc}_VERSION"
  )

set_target_properties (
  libnukexc
  PROPERTIES
    EXPORT_NAME "${nxc}"
    ${nxc}_VERSION ${${nxc}_VERSION}
  )

set_property(TARGET libnukexc APPEND PROPERTY EXPORT_PROPERTIES "${export_properties}")


# Guard the tests
if(${nxc}_BUILD_TESTING)
    # Test-only dependencies, resolved once: tests/, benchmarking/ and
    # convergence_studies/ all link against Catch2 / Armadillo / OOO.
    include( nukexc-catch2 )
    include( nukexc-armadillo )
    include( nukexc-openorbitaloptimizer )
    include( nukexc-ptxas ) # defines nukexc_ptxas_verbose()

    # standards.*, test_io.* and standalone_helpers.hpp are shared driver
    # support that lives in tests/; the benchmarking and convergence-study
    # directories compile/include them from there rather than duplicating them.
    set(NUKEXC_TEST_SUPPORT_DIR ${PROJECT_SOURCE_DIR}/tests)

    add_subdirectory(tests)
    add_subdirectory(benchmarking)
    add_subdirectory(convergence_studies)
endif()


# ====  Export & Install  =======================================================
install(
  DIRECTORY
    nukexc
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.hpp*"
  )

install(
  TARGETS
    libnukexc 
  EXPORT
    ho_interface
  RUNTIME DESTINATION
    ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION
    ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION
    ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  EXPORT
    ho_interface
  FILE
    "${PROJECT_NAME}Targets.cmake"
  NAMESPACE
    "${PROJECT_NAME}::"
  DESTINATION
    ${${nxc}_INSTALL_CMAKEDIR}
  )

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "cmake/${nxc}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${nxc}Config.cmake"
  INSTALL_DESTINATION ${${nxc}_INSTALL_CMAKEDIR}
  )

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${nxc}ConfigVersion.cmake"
  VERSION ${${nxc}_VERSION}
  COMPATIBILITY SameMinorVersion
  )

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${nxc}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${nxc}ConfigVersion.cmake
  DESTINATION
    ${${nxc}_INSTALL_CMAKEDIR}
  )
