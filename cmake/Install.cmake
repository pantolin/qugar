function(qugar_install_project)

  include(../cmake/PackageProject.cmake)

# if(qugar_WITH_DEMOS)
  if(CMAKE_SKIP_INSTALL_RULES)
    return()
  endif()


  # Add other targets that you want installed here, by default we just package the one executable
  # we know we want to ship

  qugar_package_project(
    VERSION ${CMAKE_PROJECT_VERSION}
    TARGETS
    qugar
    qugar_options
    qugar_warnings

    PUBLIC_INCLUDES
    ${qugar_INCLUDE_DIR}

    # FIXME: this does not work! CK
    # PRIVATE_DEPENDENCIES_CONFIGURED qugar_options qugar_warnings

    # See ../cmake/PackageProject.cmake for all the options
    # PUBLIC_DEPENDENCIES_CONFIGURED
    #   xxx
    # PUBLIC_DEPENDENCIES
    #   xxx
    # PRIVATE_DEPENDENCIES_CONFIGURED
    #   xxx
    # PRIVATE_DEPENDENCIES
    #   xxx
  )

  # Experience shows that explicit package naming can help make it easier to sort
  # out potential ABI related issues before they start, while helping you
  # track a build to a specific GIT SHA
  set(CPACK_PACKAGE_FILE_NAME
    "${CMAKE_PROJECT_NAME}-${CMAKE_PROJECT_VERSION}-${GIT_COMMIT_HASH}-${CMAKE_SYSTEM_NAME}-${CMAKE_BUILD_TYPE}-${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
  )

  include(CPack)
# endif()
endfunction()