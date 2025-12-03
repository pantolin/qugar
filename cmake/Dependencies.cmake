include(../cmake/CPM.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/modules")

macro(qugar_setup_dependencies)

  if(NOT algoim::algoim)
    cpmaddpackage("gh:pantolin/algoim@1.0.3")
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