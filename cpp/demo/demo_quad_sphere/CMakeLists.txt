add_executable(demo_quad_sphere main.cpp)

target_link_libraries(
  demo_quad_sphere
  PRIVATE
  qugar
  algoim::algoim
  # qugar_options
  qugar_warnings)

target_compile_features(demo_quad_sphere PUBLIC cxx_std_20)

# target_link_system_libraries(
#   demo_quad_sphere
#   PRIVATE
#   qugar_warnings
#   qugar_options
#   qugar

#   algoim::algoim

#   ${LAPACKE_LIBRARIES}
# )

# target_include_directories(demo_quad_sphere PRIVATE "${CMAKE_BINARY_DIR}/include")

