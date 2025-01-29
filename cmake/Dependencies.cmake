include(../cmake/CPM.cmake)

macro(qugar_setup_dependencies)

  if(NOT algoim::algoim)
    cpmaddpackage("gh:pantolin/algoim@1.0.0")
  endif()

  # Algoim polynomial funcionalities require to solve
  # eigenvalue problems. We use LAPACK for this.
  # Much slower, and less robust, eigenvalue solvers are
  # included in algoim, but we prefer to use LAPACK if
  # available.

  include(../cmake/lapacke.cmake)
  qugar_find_lapacke()


  if(qugar_BUILD_PYTHON OR qugar_BUILD_DOC)
    include(../cmake/Python.cmake)
    qugar_find_Python()
  endif()

  if(qugar_BUILD_DOC)
    find_package(Doxygen REQUIRED OPTIONAL_COMPONENTS dot)
  endif()


  if(BUILD_TESTING)
    include(../cmake/Catch2.cmake)
    qugar_find_Catch2()
  endif()
endmacro()