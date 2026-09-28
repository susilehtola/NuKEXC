find_package(Armadillo REQUIRED)

if(NOT Armadillo_FOUND)
  message(FATAL_ERROR "Armadillo not found externally. Please install Armadillo first.")
endif()

# CMake ships its own FindArmadillo.cmake, so MODULE mode wins by default and
# defines only ARMADILLO_INCLUDE_DIRS / ARMADILLO_LIBRARIES -- no imported
# target and no Armadillo_DIR. Reporting "Found Armadillo: ${Armadillo_DIR}"
# is therefore misleading: that variable is empty in MODULE mode, or holds
# whatever a later CONFIG-mode lookup (OpenOrbitalOptimizer does one) left in
# the cache. Print the paths actually in use instead.
message(STATUS "Found Armadillo")
if(ARMADILLO_INCLUDE_DIRS)
  message(STATUS "  include dirs: ${ARMADILLO_INCLUDE_DIRS}")
endif()
if(ARMADILLO_LIBRARIES)
  message(STATUS "  libraries   : ${ARMADILLO_LIBRARIES}")
endif()
if(Armadillo_DIR)
  message(STATUS "  package dir : ${Armadillo_DIR}")
endif()

if(TARGET armadillo)
  # Already correct
elseif(TARGET Armadillo::Armadillo)
  add_library(armadillo ALIAS Armadillo::Armadillo)
elseif(DEFINED ARMADILLO_LIBRARIES)
  add_library(armadillo INTERFACE IMPORTED)
  target_link_libraries(armadillo INTERFACE ${ARMADILLO_LIBRARIES})
  if(DEFINED ARMADILLO_INCLUDE_DIRS)
    target_include_directories(armadillo INTERFACE ${ARMADILLO_INCLUDE_DIRS})
  endif()
else()
  message(FATAL_ERROR
    "Armadillo was found but no usable target or variable "
    "(armadillo, Armadillo::Armadillo, ARMADILLO_LIBRARIES) could be identified. "
    "Check your Armadillo installation.")
endif()

# Make sure the -I for Armadillo actually reaches the compiler.
#
# CMake suppresses -I for any directory in CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES,
# i.e. one the compiler is believed to search by itself. That list is probed
# once, when a build tree first detects the compiler, and it picks up CPATH /
# CPLUS_INCLUDE_PATH from the environment of that single run. Configure once
# from a shell exporting CPATH=/opt/homebrew/include (a common Homebrew setup)
# and the Armadillo prefix is baked into the cached compiler description; build
# later from a shell without CPATH and CMake still drops the -I while the
# compiler no longer searches there. Configure then reports success and every
# translation unit including <armadillo> dies on "'armadillo' file not found".
#
# Removing the Armadillo prefix from the implicit list forces the -I to be
# emitted. It is a no-op when the directory genuinely is on the compiler's
# default search path, so this stays correct on the clusters too.
#
# Note this file must be include()d from the top-level CMakeLists before any
# add_subdirectory() that builds Armadillo users -- subdirectories take their
# copy of this variable at add_subdirectory() time.
#
# The standard C system directories are exempt. A distribution package puts
# Armadillo straight into /usr/include, and forcing -I/usr/include there is
# actively harmful: it hoists /usr/include ahead of the C++ library headers,
# so the `#include_next <stdlib.h>` in /usr/include/c++/NN/cstdlib skips the
# directory it was meant to land in and the build dies with
# "fatal error: stdlib.h: No such file or directory" in every translation
# unit. Those directories are on the default search path by definition, so
# the workaround is never needed for them anyway.
set(_nxc_system_inc_dirs /usr/include /usr/local/include)

foreach(_arma_inc IN LISTS ARMADILLO_INCLUDE_DIRS)
  if(_arma_inc IN_LIST _nxc_system_inc_dirs)
    message(STATUS "  ${_arma_inc} is a standard system include dir "
                   "-- leaving the implicit list alone")
  elseif(_arma_inc IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "${_arma_inc}")
    message(STATUS "  forcing -I${_arma_inc} "
                   "(suppressed as an implicit compiler include dir)")
  endif()
endforeach()

unset(_nxc_system_inc_dirs)

target_link_libraries(libnukexc INTERFACE armadillo)
