include(GenerateExportHeader)

# add_subdirectory(impl)

add_library(qugar
  qugar.cpp
  affine_transf.cpp
  bbox.cpp
  bezier_tp.cpp
  bezier_tp_utils.cpp
  bezier_tp_utils_composition.cpp
  cart_grid_tp.cpp
  cut_quadrature.cpp
  impl_unfitted_domain.cpp
  impl_funcs_lib.cpp
  impl_utils.cpp
  impl_quadrature.cpp
  impl_reparam_mesh.cpp
  impl_reparam.cpp
  impl_reparam_bezier.cpp
  impl_reparam_general.cpp
  lagrange_tp_utils.cpp
  monomials_tp.cpp
  point.cpp
  polynomial_tp.cpp
  primitive_funcs_lib.cpp
  quadrature.cpp
  quadrature_lib.cpp
  ref_system.cpp
  reparam_mesh.cpp
  tensor_index_tp.cpp
  tolerance.cpp
  tpms_lib.cpp
  unfitted_domain.cpp
  unfitted_domain_kd_tree.cpp
)

add_library(qugar::qugar ALIAS qugar)

target_link_libraries(qugar
  PRIVATE
  qugar_options qugar_warnings
  algoim::algoim
  ${LAPACKE_LIBRARIES}
)

# Below, using CMAKE_INSTALL_INCLUDEDIR causes problems in
# the installation.
target_include_directories(qugar ${WARNING_GUARD} PUBLIC
 $<BUILD_INTERFACE:${qugar_INCLUDE_DIR}>
 $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
#  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
 $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
 )

target_compile_features(qugar PUBLIC cxx_std_20)

set_target_properties(
  qugar
  PROPERTIES VERSION ${PROJECT_VERSION}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON # Needed for linking with nanobind generated shared library
  )



generate_export_header(qugar EXPORT_FILE_NAME ${qugar_INCLUDE_DIR}/qugar/qugar_export.hpp)
configure_file(${qugar_INCLUDE_DIR}/qugar/version.hpp.in ${qugar_INCLUDE_DIR}/qugar/version.hpp @ONLY)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(qugar PUBLIC qugar_LIBRARY_STATIC_DEFINE)
endif()
