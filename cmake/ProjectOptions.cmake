include(../cmake/SystemLink.cmake)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

macro(qugar_supports_sanitizers)
  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND NOT WIN32)
    set(SUPPORTS_UBSAN ON)
  else()
    set(SUPPORTS_UBSAN OFF)
  endif()

  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND WIN32)
    set(SUPPORTS_ASAN OFF)
  else()
    set(SUPPORTS_ASAN ON)
  endif()
endmacro()

macro(qugar_setup_options)
  option(qugar_ENABLE_HARDENING "Enable hardening" ON)
  option(qugar_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  cmake_dependent_option(
    qugar_ENABLE_GLOBAL_HARDENING
    "Attempt to push hardening options to built dependencies"
    ON
    qugar_ENABLE_HARDENING
    OFF)

  qugar_supports_sanitizers()

  if(NOT PROJECT_IS_TOP_LEVEL OR NOT qugar_DEVELOPER_MODE)
    option(qugar_ENABLE_IPO "Enable IPO/LTO" OFF)
    option(qugar_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
    option(qugar_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
    option(qugar_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
    option(qugar_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(qugar_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
    option(qugar_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(qugar_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(qugar_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(qugar_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(qugar_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
    option(qugar_ENABLE_PCH "Enable precompiled headers" OFF)
    option(qugar_ENABLE_CACHE "Enable ccache" OFF)
  else()
    option(qugar_ENABLE_IPO "Enable IPO/LTO" ON)
    option(qugar_WARNINGS_AS_ERRORS "Treat Warnings As Errors" ON)
    option(qugar_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
    option(qugar_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(qugar_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ON)
    option(qugar_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ON)

    if(qugar_ENABLE_SANITIZER_ADDRESS AND SUPPORTS_ASAN)
      option(qugar_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ON)
    endif()

    if(qugar_ENABLE_SANITIZER_UNDEFINED AND SUPPORTS_UBSAN)
      option(qugar_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ON)
    endif()

    option(qugar_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(qugar_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(qugar_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(qugar_ENABLE_CLANG_TIDY "Enable clang-tidy" ON)
    option(qugar_ENABLE_CPPCHECK "Enable cpp-check analysis" ON)
    option(qugar_ENABLE_PCH "Enable precompiled headers" OFF)
    option(qugar_ENABLE_CACHE "Enable ccache" ON)
  endif()

  if(NOT PROJECT_IS_TOP_LEVEL)
    mark_as_advanced(
      qugar_ENABLE_IPO
      qugar_WARNINGS_AS_ERRORS
      qugar_ENABLE_USER_LINKER
      qugar_ENABLE_SANITIZER_ADDRESS
      qugar_ENABLE_SANITIZER_LEAK
      qugar_ENABLE_SANITIZER_UNDEFINED
      qugar_ENABLE_SANITIZER_THREAD
      qugar_ENABLE_SANITIZER_MEMORY
      qugar_ENABLE_UNITY_BUILD
      qugar_ENABLE_CLANG_TIDY
      qugar_ENABLE_CPPCHECK
      qugar_ENABLE_COVERAGE
      qugar_ENABLE_PCH
      qugar_ENABLE_CACHE)
  endif()

endmacro()

macro(qugar_global_options)
  if(qugar_ENABLE_IPO)
    include(../cmake/InterproceduralOptimization.cmake)
    qugar_enable_ipo()
  endif()

  qugar_supports_sanitizers()

  if(qugar_ENABLE_HARDENING AND qugar_ENABLE_GLOBAL_HARDENING)
    include(../cmake/Hardening.cmake)

    if(NOT SUPPORTS_UBSAN
      OR qugar_ENABLE_SANITIZER_UNDEFINED
      OR qugar_ENABLE_SANITIZER_ADDRESS
      OR qugar_ENABLE_SANITIZER_THREAD
      OR qugar_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      if (qugar_DEVELOPER_MODE)
        set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
      else()
        set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
      endif()
    endif()

    # message("${qugar_ENABLE_HARDENING} ${ENABLE_UBSAN_MINIMAL_RUNTIME} ${qugar_ENABLE_SANITIZER_UNDEFINED}")
    qugar_enable_hardening(qugar_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()
endmacro()

macro(qugar_local_options)
  if(PROJECT_IS_TOP_LEVEL)
    include(../cmake/StandardProjectSettings.cmake)
  endif()

  add_library(qugar_warnings INTERFACE)
  add_library(qugar_options INTERFACE)

  target_compile_features(qugar_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})

  add_library(qugar::qugar_options ALIAS qugar_options)
  add_library(qugar::qugar_warnings ALIAS qugar_warnings)

  # add_library(qugar::qugar_options INTERFACE IMPORTED)
  # add_library(qugar::qugar_warnings INTERFACE IMPORTED)

  include(../cmake/CompilerWarnings.cmake)
  qugar_set_project_warnings(
    qugar_warnings
    ${qugar_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
    "")

  if(qugar_ENABLE_USER_LINKER)
    include(../cmake/Linker.cmake)
    qugar_configure_linker(qugar_options)
  endif()

  include(../cmake/Sanitizers.cmake)
  qugar_enable_sanitizers(
    qugar_options
    ${qugar_ENABLE_SANITIZER_ADDRESS}
    ${qugar_ENABLE_SANITIZER_LEAK}
    ${qugar_ENABLE_SANITIZER_UNDEFINED}
    ${qugar_ENABLE_SANITIZER_THREAD}
    ${qugar_ENABLE_SANITIZER_MEMORY})

  set_target_properties(qugar_options PROPERTIES UNITY_BUILD ${qugar_ENABLE_UNITY_BUILD})

  if(qugar_ENABLE_PCH)
    target_precompile_headers(
      qugar_options
      INTERFACE
      <vector>
      <string>
      <utility>)
  endif()

  if(qugar_ENABLE_CACHE)
    include(../cmake/Cache.cmake)
    qugar_enable_cache()
  endif()

  include(../cmake/StaticAnalyzers.cmake)

  if(qugar_ENABLE_CLANG_TIDY)
    qugar_enable_clang_tidy(qugar_options ${qugar_WARNINGS_AS_ERRORS})
  endif()

  if(qugar_ENABLE_CPPCHECK)
    qugar_enable_cppcheck(${qugar_WARNINGS_AS_ERRORS} "" # override cppcheck options
    )
  endif()

  if(qugar_ENABLE_COVERAGE)
    include(../cmake/Tests.cmake)
    qugar_enable_coverage(qugar_options)
  endif()

  if(qugar_WARNINGS_AS_ERRORS)
    check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)

    if(LINKER_FATAL_WARNINGS)
      # This is not working consistently, so disabling for now
      # target_link_options(qugar_options INTERFACE -Wl,--fatal-warnings)
    endif()
  endif()

  if(qugar_ENABLE_HARDENING AND NOT qugar_ENABLE_GLOBAL_HARDENING)
    include(../cmake/Hardening.cmake)

    if(NOT SUPPORTS_UBSAN
      OR qugar_ENABLE_SANITIZER_UNDEFINED
      OR qugar_ENABLE_SANITIZER_ADDRESS
      OR qugar_ENABLE_SANITIZER_THREAD
      OR qugar_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()

    qugar_enable_hardening(qugar_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()

endmacro()

macro(qugar_setup_config_options)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set (_IS_DEBUG_BUILD ON)
  else()
    set (_IS_DEBUG_BUILD OFF)
  endif()

  option(qugar_DEVELOPER_MODE "Enable developer mode (sanitizers and many other checks are enabled)." ${_IS_DEBUG_BUILD})
  option(BUILD_TESTING "Enable testing" OFF)
  option(qugar_BUILD_DOC "Build documentation" OFF)
  option(qugar_WITH_DEMOS "Build demos" OFF)
  option(qugar_BUILD_CATCH2 "Enforce the build of Catch2 from sources instead of using existing installation" OFF)

  unset(_IS_DEBUG_BUILD)
endmacro()