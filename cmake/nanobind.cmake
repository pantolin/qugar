macro(qugar_find_nanobind)
  find_package(nanobind 2.4 CONFIG)

  if(nanobind_FOUND)
    message(STATUS "Found nanobind version ${nanobind_VERSION} in: ${nanobind_DIR}")
  else()
    message(STATUS "nanobind NOT FOUND: version 2.4.0 wil be built from sources.")
    CPMAddPackage("gh:wjakob/nanobind@2.4.0")
  endif()

  # HACK: The only purpose of creating this dummy module is to trigger
  # the generation of the nanobind library target using the current
  # compilation options (and not the ones of the project).
  nanobind_add_module(dummy_nanobind

    # Target the stable ABI for Python 3.12+, which reduces
    # the number of binary wheels that must be built. This
    # does nothing on older Python versions
    STABLE_ABI
    NOMINSIZE

    python/qugar/wrappers/nanobind_dummy.cpp)
endmacro()
