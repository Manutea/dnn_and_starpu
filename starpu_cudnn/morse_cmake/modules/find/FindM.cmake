###
#
# @copyright (c) 2012-2020 Inria. All rights reserved.
# @copyright (c) 2012-2014 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
#
# Copyright 2020 Florent Pruvost
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
#
###
#
# - Find M (Math library) include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(M
#               [REQUIRED]) # Fail with error if m is not found
#
# The following variables are set if found:
#
#   M_INCLUDE_DIRS gives the path to headers
#   M_LIBRARIES gives the library m
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``MORSE::M``
#   The headers and libraries to use for M, if found.
#
#=============================================================================

include(FindPackageHandleStandardArgs)

# check that we can call math directly with the compiler
include(CheckCSourceCompiles)
set(LIBM_TEST_SOURCE "#include<math.h>\nfloat f; int main(){sqrt(f);return
0;}")
check_c_source_compiles("${LIBM_TEST_SOURCE}" HAVE_MATH)

# if works with the compiler we do not need anything else, variables are empty
if(HAVE_MATH)

  set(M_INCLUDE_DIRS)
  set(M_LIBRARIES)

  if(NOT TARGET MORSE::M)
    add_library(MORSE::M INTERFACE IMPORTED)
  endif()

  find_package_handle_standard_args(M DEFAULT_MSG)

else()

  # look for header math.h to get the path to headers
  find_path(M_INCLUDE_DIRS NAMES math.h)

  # look for libm
  find_library(M_LIBRARIES m)

  # check call to math
  set(CMAKE_REQUIRED_LIBRARIES ${M_LIBRARIES})
  check_c_source_compiles("${LIBM_TEST_SOURCE}" LIBM_MATH_WORKS)
  unset(CMAKE_REQUIRED_LIBRARIES)

  # check and set M_FOUND
  find_package_handle_standard_args(M DEFAULT_MSG M_LIBRARIES M_INCLUDE_DIRS LIBM_MATH_WORKS)
  mark_as_advanced(M_INCLUDE_DIRS M_LIBRARIES LIBM_MATH_WORKS)

  # add imported target
  if(M_FOUND)
    if(NOT TARGET MORSE::M)
      add_library(MORSE::M INTERFACE IMPORTED)
      set_target_properties(MORSE::M PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${M_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${M_LIBRARIES}")
    endif()
  endif()

endif()