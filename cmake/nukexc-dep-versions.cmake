set( NUKEXC_INTEGRATORXX_REPOSITORY https://github.com/wavefunction91/IntegratorXX.git )
set( NUKEXC_INTEGRATORXX_REVISION   1369be58d7a3235dac36d75dd964fef058830622 )

set( NUKEXC_HIGHFIVE_REPOSITORY https://github.com/highfive-devs/HighFive.git )
set( NUKEXC_HIGHFIVE_REVISION 805f0e13d09b47c4b01d40682621904aa3b31bb8 )

set( NUKEXC_LIBXC_REPOSITORY https://gitlab.com/libxc/libxc.git )
set( NUKEXC_LIBXC_REVISION   7bd5bb41415968db94c499a2f093309c9a2dcf53)

set( NUKEXC_OPENORBITALOPTIMIZER_REPOSITORY
     "https://github.com/susilehtola/OpenOrbitalOptimizer.git" )
set( NUKEXC_OPENORBITALOPTIMIZER_REVISION   9e2119026ef43fbd90fb7b4b224a6edb3f30a7da ) # v0.3.0, 2026-05-21

set( NUKEXC_KOKKOSKERNELS_REPOSITORY https://github.com/kokkos/kokkos-kernels.git )
# Left empty on purpose: nukexc-kokkos-kernels defaults the tag to the Kokkos
# version it found, since the two must match. Set this to pin explicitly.
set( NUKEXC_KOKKOSKERNELS_REVISION   "" CACHE STRING "KokkosKernels git tag (default: match Kokkos_VERSION)" )
# Used only when Kokkos reports no version (e.g. an in-tree Kokkos build).
set( NUKEXC_KOKKOSKERNELS_FALLBACK_REVISION 4.6.02 )
