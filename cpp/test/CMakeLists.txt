cmake_minimum_required(VERSION 3.15...3.23)

project(CmakeConfigPackageTests LANGUAGES CXX)

# ---- Test as standalone project the exported config package ----
if(PROJECT_IS_TOP_LEVEL OR TEST_INSTALLED_VERSION)
  enable_testing()
  find_package(qugar CONFIG REQUIRED) # for demo_reparam_gygroid, project_options, ...
endif()

# ---- Dependencies ----
include(Catch)


# # Provide a simple smoke test to make sure that the CLI works and can display a --help message
# add_test(NAME cli.has_help COMMAND intro --help)

# # Provide a test to verify that the version being reported from the application
# # matches the version given to CMake. This will be important once you package
# # your program. Real world shows that this is the kind of simple mistake that is easy
# # to make, but also easy to test for.
# add_test(NAME cli.version_matches COMMAND intro --version)
# set_tests_properties(cli.version_matches PROPERTIES PASS_REGULAR_EXPRESSION "${PROJECT_VERSION}")
set(test_list)

macro(add_new_test test_name test_source)
  list(APPEND test_list ${test_name})

  add_executable(${test_name} ${test_source})

  target_link_libraries(
    ${test_name}
    PRIVATE
    qugar_warnings
    qugar_options
    qugar
    algoim::algoim
    Catch2::Catch2WithMain
    ${LAPACKE_LIBRARIES}
  )

  if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET ${test_name}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:tests> $<TARGET_FILE_DIR:tests>
      COMMAND_EXPAND_LISTS)
  endif()

  # automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
  # to whatever you want, or use different for different binaries
  catch_discover_tests(
    ${test_name}
    TEST_PREFIX "unittests."
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    REPORTER XML
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    OUTPUT_PREFIX "unittests."
    OUTPUT_SUFFIX .xml
  )
endmacro(add_new_test)

enable_testing()
add_new_test(test_bbox_0 test_bbox_0.cpp)

add_new_test(test_impl_general_quad_0 test_impl_general_quad_0.cpp)
add_new_test(test_impl_general_quad_1 test_impl_general_quad_1.cpp)
add_new_test(test_impl_general_quad_2 test_impl_general_quad_2.cpp)
add_new_test(test_impl_general_quad_3 test_impl_general_quad_3.cpp)
add_new_test(test_impl_general_quad_4 test_impl_general_quad_4.cpp)
add_new_test(test_impl_general_quad_5 test_impl_general_quad_5.cpp)
add_new_test(test_impl_poly_quad_0 test_impl_poly_quad_0.cpp)
add_new_test(test_impl_poly_quad_1 test_impl_poly_quad_1.cpp)
add_new_test(test_impl_poly_quad_2 test_impl_poly_quad_2.cpp)
add_new_test(test_impl_poly_quad_3 test_impl_poly_quad_3.cpp)
add_new_test(test_impl_poly_quad_4 test_impl_poly_quad_4.cpp)
add_new_test(test_impl_poly_quad_5 test_impl_poly_quad_5.cpp)
add_new_test(test_impl_poly_quad_6 test_impl_poly_quad_6.cpp)
add_new_test(test_impl_bezier_tp_0 test_impl_bezier_tp_0.cpp)
add_new_test(test_impl_bezier_tp_1 test_impl_bezier_tp_1.cpp)
add_new_test(test_impl_bezier_tp_2 test_impl_bezier_tp_2.cpp)
add_new_test(test_impl_bezier_tp_3 test_impl_bezier_tp_3.cpp)
add_new_test(test_impl_bezier_tp_4 test_impl_bezier_tp_4.cpp)
add_new_test(test_impl_bezier_tp_5 test_impl_bezier_tp_5.cpp)
add_new_test(test_impl_bezier_tp_6 test_impl_bezier_tp_6.cpp)
add_new_test(test_impl_monomials_tp_0 test_impl_monomials_tp_0.cpp)
add_new_test(test_impl_monomials_tp_1 test_impl_monomials_tp_1.cpp)
add_new_test(test_impl_monomials_tp_2 test_impl_monomials_tp_2.cpp)
add_new_test(test_impl_monomials_tp_3 test_impl_monomials_tp_3.cpp)
add_new_test(test_impl_poly_reparam_0 test_impl_poly_reparam_0.cpp)
add_new_test(test_impl_poly_reparam_1 test_impl_poly_reparam_1.cpp)
add_new_test(test_impl_general_reparam_0 test_impl_general_reparam_0.cpp)
add_new_test(test_impl_general_reparam_1 test_impl_general_reparam_1.cpp)

# execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory "${qugar_SOURCE_DIR}/data" ${CMAKE_CURRENT_BINARY_DIR}/data)

# add_custom_command(
# TARGET test_bbox_0
# PRE_BUILD
# COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:tests> $<TARGET_FILE_DIR:tests>
# COMMAND_EXPAND_LISTS)
# endif()

# # Add a file containing a set of constexpr tests
# add_executable(constexpr_tests constexpr_tests.cpp)
# target_link_libraries(
# constexpr_tests
# PRIVATE
# qugar_warnings
# qugar_options
# qugar
# Catch2::Catch2WithMain)

# catch_discover_tests(
# constexpr_tests
# TEST_PREFIX
# "constexpr."
# REPORTER
# XML
# OUTPUT_DIR
# .
# OUTPUT_PREFIX
# "constexpr."
# OUTPUT_SUFFIX
# .xml)

# # Disable the constexpr portion of the test, and build again this allows us to have an executable that we can debug when
# # things go wrong with the constexpr testing
# add_executable(relaxed_constexpr_tests constexpr_tests.cpp)
# target_link_libraries(
# relaxed_constexpr_tests
# PRIVATE
# qugar_warnings
# qugar_options
# qugar
# Catch2::Catch2WithMain)
# target_compile_definitions(relaxed_constexpr_tests PRIVATE -DCATCH_CONFIG_RUNTIME_STATIC_REQUIRE)

# catch_discover_tests(
# relaxed_constexpr_tests
# TEST_PREFIX
# "relaxed_constexpr."
# REPORTER
# XML
# OUTPUT_DIR
# .
# OUTPUT_PREFIX
# "relaxed_constexpr."
# OUTPUT_SUFFIX
# .xml)
