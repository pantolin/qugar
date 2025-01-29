macro(qugar_find_lapacke)

  # cmake_policy(SET CMP0069 NEW) 
  # set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
  # cpmaddpackage("gh:Reference-LAPACK/lapack@3.12.0")

  set(LAPACKE_VERSION 3.9)
  if (LAPACKE_DIR)
    find_package(LAPACKE ${LAPACKE_VERSION} HINTS ${LAPACKE_DIR})
  endif()

  if (NOT ${LAPACKE_FOUND})
    find_package(LAPACKE ${LAPACKE_VERSION})
  endif()

  if (${LAPACKE_FOUND})
    message(STATUS "LAPACKE version ${LAPACKE_VERSION} found in: ${LAPACKE_DIR}")
    add_compile_definitions(WITH_LAPACK)
    include_directories(${LAPACKE_INCLUDE_DIRS})
    # message(STATUS "LAPACKE_LIBRARIES: ${LAPACKE_LIBRARIES}")
    # message(STATUS "LAPACKE_INCLUDE_DIRS: ${LAPACKE_INCLUDE_DIRS}")
  else()
    message(STATUS "LAPACKE not found. Algoim's eigenvalue solvers will be used.")
  endif()

endmacro()
