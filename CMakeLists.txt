cmake_minimum_required(VERSION 3.23)

cmake_minimum_required(VERSION 3.23)

# Set the project name and language
set(qugar_VERSION "0.0.4")
project(
  qugar 
  VERSION ${qugar_VERSION}
  DESCRIPTION ""
  HOMEPAGE_URL "https://github.com/pantolin/qugar"
  )

include(cmake/GitConfig.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/ProjectOptions.cmake)
include(cmake/Dependencies.cmake)
include(cmake/Install.cmake)

set(qugar_SOURCE_DIR ${PROJECT_SOURCE_DIR}/cpp)
set(qugar_BINARY_DIR ${PROJECT_BINARY_DIR}/cpp)

set(qugar_INCLUDE_DIR ${qugar_SOURCE_DIR}/include)
set(qugar_TEST_DIR ${qugar_SOURCE_DIR}/test)
set(qugar_DOC_DIR ${qugar_SOURCE_DIR}/doc)

# set(cpp_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)


git_config()


# This template attempts to be "fetch_content"-able
# so that it works well with tools like CPM or other
# manual dependency management

# Only set the cxx_standard if it is not set by someone else
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

# strongly encouraged to enable this globally to avoid conflicts between
# -Wpedantic being enabled and -std=c++20 and -std=gnu++20 for example
# when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS ON)

qugar_setup_config_options()
qugar_setup_options()

qugar_global_options()
qugar_setup_dependencies()

qugar_local_options()

# don't know if this should be set globally from here or not...
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

add_subdirectory(${qugar_SOURCE_DIR})