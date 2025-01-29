add_executable(demo_reparam_gyroid main.cpp)

target_link_libraries(
  demo_reparam_gyroid
  PRIVATE
  qugar
  algoim::algoim
  # qugar_options
  qugar_warnings)

target_compile_features(demo_reparam_gyroid PUBLIC cxx_std_20)

# target_link_system_libraries(
#   demo_reparam_gyroid
#   PRIVATE
#   qugar_warnings
#   qugar_options
#   qugar

#   algoim::algoim

#   ${LAPACKE_LIBRARIES}
# )

# target_include_directories(demo_reparam_gyroid PRIVATE "${CMAKE_BINARY_DIR}/include")

