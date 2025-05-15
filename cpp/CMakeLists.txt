# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

# Adding sources
add_subdirectory(qugar)

# Adding the tests:
include(CTest)

if(BUILD_TESTING)
  # message(AUTHOR_WARNING "Building Tests. Be sure to check out test/constexpr_tests.cpp for constexpr testing")
  add_subdirectory(test)
endif()


# If MSVC is being used, and ASAN is enabled, we need to set the debugger environment
# so that it behaves well with MSVC's debugger, and we can run the target from visual studio
if(MSVC)
  get_all_installable_targets(all_targets)
  message("all_targets=${all_targets}")
  set_target_properties(${all_targets} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=$(VC_ExecutablePath_x64);%PATH%")
endif()

if(qugar_BUILD_DOC)
  add_subdirectory(doc)
endif()

# # set the startup project for the "play" button in MSVC
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT demo_reparam_gygroid)
if(qugar_WITH_DEMOS)
  add_subdirectory(demo)
endif()

qugar_install_project()