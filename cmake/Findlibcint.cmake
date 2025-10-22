# Try to locate libcint headers and library (works with conda)
if(DEFINED ENV{CONDA_PREFIX})
    set(_libcint_prefix "$ENV{CONDA_PREFIX}")
endif()

find_path(LIBCINT_INCLUDE_DIR
    NAMES cint.h
    HINTS ${_libcint_prefix}/include /usr/local/include /usr/include
)

find_library(LIBCINT_LIBRARY
    NAMES cint libcint
    HINTS ${_libcint_prefix}/lib /usr/local/lib /usr/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libcint
    REQUIRED_VARS LIBCINT_LIBRARY LIBCINT_INCLUDE_DIR
)

if(libcint_FOUND)
    set(LIBCINT_LIBRARIES ${LIBCINT_LIBRARY})
    set(LIBCINT_INCLUDE_DIRS ${LIBCINT_INCLUDE_DIR})
endif()
