# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.23)

project(qugar_nanobind)

if(WIN32)
  # Windows requires all symbols to be manually exported. This flag exports all
  # symbols automatically, as in Unix.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif(WIN32)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

if (DEFINED qugar_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${qugar_DIR}")
endif()
find_package(qugar CONFIG REQUIRED)

if(qugar_FOUND)
  message(STATUS "Found qugar at ${qugar_DIR}")
endif()

# include(../cmake/CPM.cmake)
# if(NOT algoim::algoim)
#   CPMAddPackage("gh:pantolin/algoim@1.0.3")
#   if(NOT TARGET algoim::algoim)
#     if(TARGET algoim)
#       add_library(algoim::algoim ALIAS algoim)
#     elseif(DEFINED algoim_SOURCE_DIR)
#       add_library(algoim::algoim INTERFACE IMPORTED)
#       target_include_directories(algoim::algoim INTERFACE "${algoim_SOURCE_DIR}")
#     endif()
#   endif()
# endif()
#
# include(../cmake/lapacke.cmake)
# qugar_find_lapacke()

# TODO: Seems better than copying code, and shouldn't add any overhead, since we
# are neither building docs nor tests.
include(../cmake/Dependencies.cmake)
qugar_setup_dependencies()

# Create the binding library nanobind handles its own calls to
# target_link_libraries
nanobind_add_module(
  cpp
  NOMINSIZE
  MODULE
  nanobind_wrappers/qugar.cpp
  nanobind_wrappers/common.cpp
  nanobind_wrappers/quad.cpp
  nanobind_wrappers/cut_quad.cpp
  nanobind_wrappers/unf_domain.cpp
  nanobind_wrappers/reparam.cpp
  nanobind_wrappers/impl_functions.cpp
)
target_compile_definitions(cpp PRIVATE cxx_std_20)

# Add QUGAR libraries
# target_link_libraries(cpp PRIVATE qugar::qugar qugar::qugar_options qugar::qugar_warnings qugar::algoim algoim::algoim)
target_link_libraries(cpp PRIVATE qugar::qugar algoim::algoim ${LAPACKE_LIBRARIES})

target_include_directories(cpp SYSTEM PRIVATE
  ${Python_INCLUDE_DIRS}
  ${NB_DIR}/include
)

# target_include_directories(cpp PRIVATE QUGAR::qugar)

set_target_properties(cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cpp DESTINATION qugar)
