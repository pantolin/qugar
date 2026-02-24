include(../cmake/CPM.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/modules")

macro(qugar_setup_dependencies)

  if(NOT TARGET algoim::algoim)
    if(DEFINED algoim_SOURCE_DIR)
      # Use local version
      add_library(algoim::algoim INTERFACE IMPORTED)
      target_include_directories(algoim::algoim INTERFACE "${algoim_SOURCE_DIR}")
      message(STATUS "Using local algoim from ${algoim_SOURCE_DIR}")
    else()
      # Fetch from GitHub
      # CPMAddPackage("gh:pantolin/algoim@1.0.3")
      # TODO: Switch to the fork until the multiple-definition issue fix is merged
      CPMAddPackage("gh:dyc0/algoim@1.0.0")
      if(TARGET algoim)
        add_library(algoim::algoim ALIAS algoim)
      elseif(DEFINED algoim_SOURCE_DIR)
        add_library(algoim::algoim INTERFACE IMPORTED)
        target_include_directories(algoim::algoim INTERFACE "${algoim_SOURCE_DIR}")
      endif()
    endif()
  endif()

  # Algoim polynomial funcionalities require to solve
  # eigenvalue problems. We use LAPACK for this.
  # Much slower, and less robust, eigenvalue solvers are
  # included in algoim, but we prefer to use LAPACK if
  # available.

  include(../cmake/lapacke.cmake)
  qugar_find_lapacke()

  if(qugar_BUILD_DOC)
    find_package(Doxygen REQUIRED OPTIONAL_COMPONENTS dot)
  endif()


  if(BUILD_TESTING)
    include(../cmake/Catch2.cmake)
    qugar_find_Catch2()
  endif()
endmacro()
