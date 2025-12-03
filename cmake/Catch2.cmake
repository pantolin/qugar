macro(qugar_find_Catch2)
  set(_Catch2_VERSION "3.3.2")

  if (TARGET Catch2::Catch2WithMain)
    message(STATUS "Catch2: previous configuration found.")
  else()
    if (NOT qugar_BUILD_CATCH2)

      find_package(Catch2 ${_Catch2_VERSION})

      if(Catch2_FOUND)
        list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
        message(STATUS "Found Catch2 version ${_Catch2_VERSION} in ${Catch2_DIR}")
      else()
        message(STATUS "Catch2 NOT FOUND: version ${_Catch2_VERSION}.")
      endif()

    endif()

    if (NOT Catch2_FOUND)
      message(STATUS "Catch2 version ${_Catch2_VERSION} wil be built from sources.")
      CPMAddPackage("gh:catchorg/Catch2@${_Catch2_VERSION}")
      list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    endif()

  endif()
endmacro()
