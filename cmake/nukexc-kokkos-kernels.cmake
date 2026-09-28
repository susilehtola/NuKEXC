# KokkosKernels provides the BLAS (GEMM/GEMV/TRSM), LAPACK (SVD/GESV/POTRF)
# and Batched routines NuKEXC's integral kernels are built on.
#
# Prefer an external installation: a site build can enable the vendor TPLs
# (cuBLAS/rocBLAS, MKL) and is shared between projects. Fall back to
# FetchContent so a bare clone still configures, the way IntegratorXX, ArborX
# and Catch2 already do.

find_package(KokkosKernels QUIET CONFIG)

if(KokkosKernels_FOUND)

  message(STATUS "Found Kokkos-kernels: ${KokkosKernels_DIR} (version \"${KokkosKernels_VERSION}\")")

else()

  include(nukexc-dep-versions)
  include(FetchContent)

  # KokkosKernels releases track Kokkos version-for-version and a mismatched
  # pair does not build, so default the tag to whatever Kokkos we just found
  # (nukexc-kokkos is included before this module). NUKEXC_KOKKOSKERNELS_REVISION
  # overrides, and covers the case where Kokkos came from a source tree with no
  # version set.
  if(NUKEXC_KOKKOSKERNELS_REVISION)
    set(_nxc_kk_tag ${NUKEXC_KOKKOSKERNELS_REVISION})
  elseif(Kokkos_VERSION)
    set(_nxc_kk_tag ${Kokkos_VERSION})
  else()
    set(_nxc_kk_tag ${NUKEXC_KOKKOSKERNELS_FALLBACK_REVISION})
  endif()

  message(STATUS "Kokkos-kernels not found externally. Fetching ${_nxc_kk_tag} via FetchContent.")
  message(STATUS "  (install Kokkos-kernels yourself to enable vendor BLAS/LAPACK TPLs and avoid this build)")

  # Only the components NuKEXC includes. SPARSE and GRAPH alone account for the
  # bulk of KokkosKernels' explicit template instantiation, and nothing in
  # nukexc/ includes them -- leaving them off turns a very long build into a
  # tolerable one.
  #
  # ENABLE_ALL_COMPONENTS defaults to ON and force-overrides every individual
  # component flag (cmake/kokkoskernels_components.cmake), so it has to go off
  # first or the settings below are silently discarded.
  set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_BLAS    ON  CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_LAPACK  ON  CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON  CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_SPARSE  OFF CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_GRAPH   OFF CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_ODE     OFF CACHE BOOL "" FORCE)

  # NuKEXC is double precision throughout and mixes both layouts (LayoutLeft for
  # the LAPACK-facing matrices, LayoutRight elsewhere).
  set(KokkosKernels_INST_DOUBLE      ON CACHE BOOL "" FORCE)
  set(KokkosKernels_INST_LAYOUTLEFT  ON CACHE BOOL "" FORCE)
  set(KokkosKernels_INST_LAYOUTRIGHT ON CACHE BOOL "" FORCE)

  set(KokkosKernels_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)

  # Without a host BLAS/LAPACK TPL, KokkosKernels falls back to reference
  # implementations that are orders of magnitude slower (see README). Wire the
  # host libraries up when CMake can find them; a GPU build should still prefer
  # an external installation with the vendor TPLs enabled.
  # KokkosKernels' FindTPLBLAS/FindTPLLAPACK hand whatever is in BLAS_LIBRARIES
  # straight to find_library() as a list of *names*. CMake's FindBLAS happily
  # returns things that are not names -- on Fedora it yields
  # "/usr/lib64/libflexiblas.so;-lm;-ldl" -- and the "-lm" entry then fails the
  # lookup and aborts the whole configure. So reduce the result to a single
  # (directory, bare name) pair, which is the form KokkosKernels can consume,
  # and only enable the TPLs when that succeeds.
  find_package(BLAS QUIET)
  find_package(LAPACK QUIET)

  set(_nxc_blas_lib "")
  set(_nxc_blas_dir "")
  foreach(_cand IN LISTS BLAS_LIBRARIES)
    if(IS_ABSOLUTE "${_cand}" AND EXISTS "${_cand}")
      get_filename_component(_nxc_blas_dir "${_cand}" DIRECTORY)
      get_filename_component(_nxc_blas_lib "${_cand}" NAME_WE)
      string(REGEX REPLACE "^lib" "" _nxc_blas_lib "${_nxc_blas_lib}")
      break()
    endif()
  endforeach()

  if(BLAS_FOUND AND LAPACK_FOUND AND _nxc_blas_lib)
    # FindTPLBLAS takes the (LIBRARIES, LIBRARY_PATHS) branch when both are set.
    set(BLAS_LIBRARIES      ${_nxc_blas_lib})
    set(BLAS_LIBRARY_DIRS   ${_nxc_blas_dir})
    set(LAPACK_LIBRARIES    ${_nxc_blas_lib})
    set(LAPACK_LIBRARY_DIRS ${_nxc_blas_dir})
    set(KokkosKernels_ENABLE_TPL_BLAS   ON CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_LAPACK ON CACHE BOOL "" FORCE)
    message(STATUS "  host BLAS/LAPACK: -l${_nxc_blas_lib} in ${_nxc_blas_dir} "
                   "-- enabling KokkosKernels BLAS/LAPACK TPLs")
  else()
    set(KokkosKernels_ENABLE_TPL_BLAS   OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_LAPACK OFF CACHE BOOL "" FORCE)
    message(WARNING
      "No usable host BLAS/LAPACK for KokkosKernels. It will fall back to its "
      "reference implementations, which are orders of magnitude slower. "
      "Install KokkosKernels yourself against your BLAS to avoid this.")
  endif()

  unset(_nxc_blas_lib)
  unset(_nxc_blas_dir)
  unset(_cand)

  FetchContent_Declare(
    kokkoskernels
    GIT_REPOSITORY ${NUKEXC_KOKKOSKERNELS_REPOSITORY}
    GIT_TAG        ${_nxc_kk_tag}
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(kokkoskernels)

  unset(_nxc_kk_tag)

endif()

target_link_libraries(libnukexc INTERFACE Kokkos::kokkoskernels)
