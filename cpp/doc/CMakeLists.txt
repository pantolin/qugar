# Enable doxygen doc builds of source

set(DOXYGEN_THEME "awesome-sidebar")

# set better defaults for doxygen
is_verbose(_is_verbose)

if(NOT ${_is_verbose})
  set(DOXYGEN_QUIET YES)
endif()

set(DOXYGEN_CALLER_GRAPH YES)
set(DOXYGEN_CALL_GRAPH YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_EXTRACT_PRIV_VIRTUAL YES)
set(DOXYGEN_GENERATE_TREEVIEW YES)

# svg files are much smaller than jpeg and png, and yet they have higher quality
set(DOXYGEN_DOT_IMAGE_FORMAT svg)
set(DOXYGEN_DOT_TRANSPARENT YES)

# If not specified, exclude the vcpkg files and the files CMake downloads under _deps (like project_options)
if(NOT DOXYGEN_EXCLUDE_PATTERNS)
  set(DOXYGEN_EXCLUDE_PATTERNS "*/vcpkg_installed/*" "*/_deps/*" "*/dependencies/*" "quadrature_gauss.hpp")
endif()

if("${DOXYGEN_THEME}" STREQUAL "")
  set(DOXYGEN_THEME "awesome-sidebar")
endif()

if("${DOXYGEN_THEME}" STREQUAL "awesome" OR "${DOXYGEN_THEME}" STREQUAL "awesome-sidebar")
  # use a modern doxygen theme
  # https://github.com/jothepro/doxygen-awesome-css v1.6.1
  FetchContent_Declare(_doxygen_theme
    URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v1.6.1.zip)
  FetchContent_MakeAvailable(_doxygen_theme)

  if("${DOXYGEN_THEME}" STREQUAL "awesome" OR "${DOXYGEN_THEME}" STREQUAL "awesome-sidebar")
    set(DOXYGEN_HTML_EXTRA_STYLESHEET "${_doxygen_theme_SOURCE_DIR}/doxygen-awesome.css")
  endif()

  if("${DOXYGEN_THEME}" STREQUAL "awesome-sidebar")
    set(DOXYGEN_HTML_EXTRA_STYLESHEET ${DOXYGEN_HTML_EXTRA_STYLESHEET}
      "${_doxygen_theme_SOURCE_DIR}/doxygen-awesome-sidebar-only.css")
  endif()
else()
  # use the original doxygen theme
endif()

# add docs target
message(STATUS "Adding `docs` target that builds the documentation.")

set(DOC_FOLDERS "${qugar_SOURCE_DIR}/include")

if(qugar_WITH_DEMOS)
  list(APPEND DOC_FOLDERS "${qugar_SOURCE_DIR}/demo")
endif()
list(APPEND DOC_FOLDERS "${qugar_DOC_DIR}")

doxygen_add_docs(docs ALL ${DOC_FOLDERS}
  COMMENT "Generating documentation - entry file: ${CMAKE_CURRENT_BINARY_DIR}/html/index.html")