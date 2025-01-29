macro(qugar_enable_ipo)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result OUTPUT output)

  if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

    if(APPLE AND(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang"))
      # See https://stackoverflow.com/a/72490406
      add_link_options("-Wl,-object_path_lto,$<TARGET_PROPERTY:NAME>_lto.o")
      add_link_options("-Wl,-cache_path_lto,${CMAKE_CURRENT_BINARY_DIR}/LTOCache")
    endif()
  else()
    message(SEND_ERROR "IPO is not supported: ${output}")
  endif()
endmacro()
