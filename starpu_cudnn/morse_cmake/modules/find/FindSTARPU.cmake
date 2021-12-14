###
#
# @copyright (c) 2012-2020 Inria. All rights reserved.
# @copyright (c) 2012-2016 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
#
# Copyright 2012-2019 Inria
# Copyright 2012-2013 Emmanuel Agullo
# Copyright 2012-2013 Mathieu Faverge
# Copyright 2012      Cedric Castagnede
# Copyright 2013-2020 Florent Pruvost
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file MORSE-Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of Morse, substitute the full
#  License text for the above reference.)
###
#
# - Find STARPU include dirs and libraries  using pkg-config file.
# Use this module by invoking find_package with the form:
#  find_package(STARPU
#               [version] [EXACT]      # Minimum or EXACT version e.g. 1.1
#               [REQUIRED]             # Fail with error if starpu is not found
#              )
#
#   Optional dependencies
#   - BLAS
#   - CUDA
#   - FORTRAN interface
#   - FXT
#   - HDF5
#   - MAGMA
#   - MPI
#   - OPENCL
#   - SIMGRID
#
#  STARPU_FOUND_WITH_PKGCONFIG - True if found with pkg-config
#  if found with pkg-config the following variables are set
#  <PREFIX>  = STARPU
#  <PREFIX>_PREFIX          ... installation path of the lib found
#  <PREFIX>_VERSION         ... version of the lib
#  <XPREFIX> = <PREFIX>        for common case
#  <XPREFIX> = <PREFIX>_STATIC for static linking
#  <XPREFIX>_FOUND          ... set to 1 if module(s) exist
#  <XPREFIX>_LIBRARIES      ... only the libraries (w/o the '-l')
#  <XPREFIX>_LIBRARY_DIRS   ... the paths of the libraries (w/o the '-L')
#  <XPREFIX>_LDFLAGS        ... all required linker flags
#  <XPREFIX>_LDFLAGS_OTHER  ... all other linker flags
#  <XPREFIX>_INCLUDE_DIRS   ... the '-I' preprocessor flags (w/o the '-I')
#  <XPREFIX>_CFLAGS         ... all required cflags
#  <XPREFIX>_CFLAGS_OTHER   ... the other compiler flags
#
#  STARPU_FORTRAN_MOD            - Points to the StarPU Fortran interface module fstarpu_mod.f90
#
# Set STARPU_STATIC to 1 to force using static libraries if exist.
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``MORSE::STARPU``
#   The headers and libraries to use for STARPU if found.
#
#=============================================================================

# Common macros to use in finds
include(FindMorseInit)

if (NOT STARPU_FIND_QUIETLY)
  message(STATUS "FindSTARPU needs pkg-config program and PKG_CONFIG_PATH set with starpu-x.y.pc (x.y the version) file path.")
endif()

if (NOT STARPU_FIND_VERSION)
  set(STARPU_FIND_VERSION "1.0")
  message(STATUS "FindSTARPU needs a version to check minimal API requirement. As it is not given we set to 1.0 by default.")
endif()

# use pkg-config to detect include/library dirs (if pkg-config is available)
# --------------------------------------------------------------------------
if (PKG_CONFIG_EXECUTABLE)
  unset(STARPU_FOUND CACHE)
  pkg_search_module(STARPU starpumpi-${STARPU_FIND_VERSION})
  if (NOT STARPU_FOUND)
    pkg_search_module(STARPU starpu-${STARPU_FIND_VERSION})
  endif()

  if (NOT STARPU_FIND_QUIETLY)
    if (STARPU_FOUND AND STARPU_LIBRARIES)
      message(STATUS "Looking for STARPU - found using PkgConfig")
    else()
      message(STATUS "${Magenta}Looking for STARPU - not found using PkgConfig."
        "\n   Perhaps you should add the directory containing libstarpumpi.pc, libstarpu.pc to"
        "\n   the PKG_CONFIG_PATH environment variable.${ColourReset}")
    endif()
  endif()
  if (STARPU_FOUND AND STARPU_LIBRARIES)
    if (NOT STARPU_INCLUDE_DIRS)
      pkg_get_variable(STARPU_INCLUDE_DIRS libstarpu includedir)
    endif()
    set(STARPU_FOUND_WITH_PKGCONFIG "TRUE")
    morse_find_pkgconfig_libraries_absolute_path(STARPU)
  else()
    set(STARPU_FOUND_WITH_PKGCONFIG "FALSE")
  endif()

  if (STARPU_STATIC AND STARPU_STATIC_LIBRARIES)
    set (STARPU_DEPENDENCIES ${STARPU_STATIC_LIBRARIES})
    list (REMOVE_ITEM STARPU_DEPENDENCIES "starpu")
    list (REMOVE_ITEM STARPU_DEPENDENCIES "starpumpi")
    list (APPEND STARPU_LIBRARIES ${STARPU_DEPENDENCIES})
    set(STARPU_CFLAGS_OTHER ${STARPU_STATIC_CFLAGS_OTHER})
    set(STARPU_LDFLAGS_OTHER ${STARPU_STATIC_LDFLAGS_OTHER})
    if (NOT STARPU_FIND_QUIETLY)
      message(STATUS "STARPU_STATIC set to 1 by user, STARPU_LIBRARIES: ${STARPU_LIBRARIES}.")
    endif()
  endif()
endif()

# check a function to validate the find
if(STARPU_FOUND AND STARPU_LIBRARIES)

  # check if static or dynamic lib
  morse_check_static_or_dynamic(STARPU STARPU_LIBRARIES)
  if(STARPU_STATIC)
    set(STATIC "_STATIC")
  else()
    set(STATIC "")
  endif()

  # set required libraries for link
  morse_set_required_test_lib_link(STARPU)

  # test link
  unset(STARPU_WORKS CACHE)
  include(CheckFunctionExists)
  check_function_exists(starpu_init STARPU_WORKS)
  mark_as_advanced(STARPU_WORKS)

  if(STARPU_WORKS)
    unset(FSTARPU_WORKS CACHE)
    check_function_exists(fstarpu_task_insert FSTARPU_WORKS)
    if(FSTARPU_WORKS)
      set(STARPU_FMOD_DIRS "STARPU_FMOD_DIRS-NOTFOUND")
      find_path(STARPU_FMOD_DIRS
        NAMES "fstarpu_mod.f90"
        HINTS ${STARPU_INCLUDE_DIRS})
      if(STARPU_FMOD_DIRS)
        set(STARPU_FORTRAN_MOD "${STARPU_FMOD_DIRS}/fstarpu_mod.f90")
        mark_as_advanced(STARPU_FORTRAN_MOD)
        if (NOT STARPU_FIND_QUIETLY)
          message(STATUS "Looking for starpu fstarpu_mod.f90: found")
        endif()
      else(STARPU_FMOD_DIRS)
        set(FSTARPU_WORKS "FSTARPU_WORKS-NOTFOUND")
      endif(STARPU_FMOD_DIRS)
    endif(FSTARPU_WORKS)

    if(NOT FSTARPU_WORKS)
      if(NOT STARPU_FIND_REQUIRED_FORTRAN)
        if (NOT STARPU_FIND_QUIETLY)
          message(STATUS "Looking for starpu Fortran interface: not found (not required)")
        endif (NOT STARPU_FIND_QUIETLY)
      else()
        if (NOT STARPU_FIND_QUIETLY)
          message(STATUS "Looking for starpu Fortran interface: not found")
        endif (NOT STARPU_FIND_QUIETLY)
        set(STARPU_WORKS FALSE)
        set(STARPU_FOUND FALSE)
      endif(NOT STARPU_FIND_REQUIRED_FORTRAN)
    else()
      if (NOT STARPU_FIND_QUIETLY)
        message(STATUS "Looking for starpu Fortran interface: found")
      endif (NOT STARPU_FIND_QUIETLY)
    endif(NOT FSTARPU_WORKS)
  endif(STARPU_WORKS)

  if(NOT STARPU_WORKS)
    if(NOT STARPU_FIND_QUIETLY)
      message(STATUS "Looking for starpu : test of starpu_init fails")
      message(STATUS "CMAKE_REQUIRED_LIBRARIES: ${CMAKE_REQUIRED_LIBRARIES}")
      message(STATUS "CMAKE_REQUIRED_INCLUDES: ${CMAKE_REQUIRED_INCLUDES}")
      message(STATUS "CMAKE_REQUIRED_FLAGS: ${CMAKE_REQUIRED_FLAGS}")
      message(STATUS "Check in CMakeFiles/CMakeError.log to figure out why it fails")
      message(STATUS "Maybe STARPU is linked with specific libraries. "
        "Have you tried with COMPONENTS (HWLOC, CUDA, MPI, BLAS, MAGMA, FXT, SIMGRID, FORTRAN)? "
        "See the explanation in FindSTARPU.cmake.")
    endif()
  endif()
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_LIBRARIES)

endif(STARPU_FOUND AND STARPU_LIBRARIES)

# check that STARPU has been found
# --------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(STARPU
  REQUIRED_VARS STARPU_LIBRARIES
                STARPU_WORKS
  VERSION_VAR   STARPU_VERSION)

 # Add imported targe
if (STARPU_FOUND)
  morse_create_imported_target(STARPU)
endif()