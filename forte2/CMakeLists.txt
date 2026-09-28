if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

include(FetchContent)

# Option to enable/disable libcint integration
option(USE_LIBCINT "Enable libcint integration for two-center integrals" OFF)

if (USE_LIBCINT)
  # Fetch libcint directly from GitHub
  FetchContent_Declare(
    libcint
    GIT_REPOSITORY https://github.com/sunqm/libcint.git
    GIT_TAG v6.1.3    # "master", or specific commit/tag, e.g. v6.1.3
  )

  FetchContent_Populate(libcint)     # produces ${libcint_SOURCE_DIR} and ${libcint_BINARY_DIR}

  # Set option in cache *before* add_subdirectory so the subproject sees it.
  set(PYPZPX ON CACHE BOOL "Enable PYPZPX in libcint" FORCE)

  # configure/build in an isolated build dir:
  add_subdirectory(${libcint_SOURCE_DIR} ${libcint_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import libint2
find_package(Libint2 2.8.1 CONFIG)
if (TARGET Libint2::cxx)
    get_target_property(_loc Libint2::int2 LOCATION)
    get_target_property(Libint2_CONFIGURATION Libint2::int2 Libint2_CONFIGURATION)
    message(STATUS "${Cyan}Found Libint2 ${Libint2_MAX_AM_ERI}${ColourReset}: ${_loc} (found version ${Libint2_VERSION})")
else()
    message(FATAL_ERROR "Suitable Libint2 could not be externally located as user insists")
endif()

# Import Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if (TARGET Eigen3::Eigen)
    message(STATUS "${Cyan}Found Eigen3${ColourReset}: ${Eigen3_DIR}")
else()
    message(FATAL_ERROR "Suitable Eigen3 could not be externally located as user insists")
endif()  

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# Find BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  lib

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # By default nanobind_add_module appends "-Os" (optimize for size)
  # This can override earlier -O flags. Passing this option stops
  # nanobind from appending -Os
  NOMINSIZE

  # Source code goes here
  api/forte2_api.cc
  api/cpp_helpers_api.cc
  api/ints_api.cc
  api/ci_helpers_api.cc
  api/det_api.cc
  api/dsrg_utils_api.cc
  api/sparse_ops_api.cc
  api/rdms_api.cc
  helpers/combinatorial.cc
  helpers/string_algorithms.cc
  integrals/basis.cc
  integrals/nuclear_repulsion.cc
  integrals/one_electron.cc
  integrals/one_electron_deriv.cc
  integrals/two_electron.cc
  integrals/two_electron_deriv.cc
  integrals/value_at_points.cc
  ci/ci_sigma_builder.cc
  ci/ci_sigma_builder_1rdm.cc
  ci/ci_sigma_builder_1trdm.cc
  ci/ci_sigma_builder_2rdm.cc
  ci/ci_sigma_builder_3rdm.cc
  ci/ci_sigma_builder_harrison_zarrabian.cc
  ci/ci_sigma_builder_knowles_handy.cc
  determinant/ci_spin_adapter.cc
  ci/ci_strings.cc
  ci/ci_strings_makers.cc
  ci/ci_string_address.cc
  ci/ci_string_class.cc
  ci/ci_occupation.cc
  determinant/determinant_helpers.cc
  ci/rel_ci_sigma_builder.cc
  ci/rel_ci_sigma_builder_rdm.cc
  ci/rel_ci_sigma_builder_harrison_zarrabian.cc
  determinant/slater_rules.cc
  determinant/rel_slater_rules.cc
  dsrg/dsrg_utils.cc
  sci/sci_helper.cc
  sci/sci_helper_1rdm.cc
  sci/sci_helper_1trdm.cc
  sci/sci_helper_2rdm.cc
  sci/sci_helper_select_hbci.cc
  sci/sci_helper_sigma_build.cc
  sci/sci_helper_spin.cc
  sci/rel_sci_helper.cc
  sci/rel_sci_helper_rdm.cc
  sci/rel_sci_helper_select_hbci.cc
  sci/rel_sci_helper_sigma_build.cc
  sci/sci_strings.cc
  sparse/sparse_exp.cc
  sparse/sparse_operator.cc
  sparse/sparse_operator_hamiltonian.cc
  sparse/sparse_state_functions.cc
  sparse/sparse_state.cc
  sparse/sq_operator_string.cc
)

target_link_libraries(lib PRIVATE Libint2::int2)
target_link_libraries(lib PRIVATE Eigen3::Eigen)
target_link_libraries(lib PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Conditionally link libcint and expose macro to C++
if (USE_LIBCINT)
  target_link_libraries(lib PRIVATE cint)
  target_compile_definitions(lib PRIVATE FORTE2_USE_LIBCINT=1)
else()
  target_compile_definitions(lib PRIVATE FORTE2_USE_LIBCINT=0)
endif()

target_include_directories(lib
  PUBLIC
    .
)

if (USE_LIBCINT)
  # Rename the cint library output, so we don't conflict with PySCF's libcint, if any
  set_target_properties(cint PROPERTIES
      OUTPUT_NAME "cint_forte2"  # This makes it libcint_forte2.dylib
  )
endif()

# Set rpath so the installed extension looks for companion libraries next to itself
set_target_properties(lib PROPERTIES
  INSTALL_RPATH "$<$<PLATFORM_ID:Darwin>:@loader_path>$<$<NOT:$<PLATFORM_ID:Darwin>>:$ORIGIN>"
)

# Install directive for scikit-build-core
install(TARGETS lib LIBRARY DESTINATION forte2)

if (USE_LIBCINT)
  # Install "libcint_forte2.<version>.dylib/so" (and shortened symlinks)
  install(TARGETS cint 
      LIBRARY DESTINATION forte2
  )
endif()

# ---------------------------------------------------------------------------
# Optional DMRG support: build block2 from source (forte2.dmrg).
#
# block2 is configured as an isolated CMake project rather than through
# add_subdirectory: its CMakeLists overrides CMAKE_CXX_COMPILER and declares
# cmake_minimum_required(VERSION 3.0), both of which would leak into this build.
#
# Building it ourselves is what keeps DMRG usable: the upstream wheel vendors
# its own MKL and libgomp, which then service forte2's BLAS calls and change
# its numerics (or fail outright on CPUs whose MKL kernel the wheel omits).
# Built here, block2 links the same BLAS and OpenMP runtime as forte2 itself.
# ---------------------------------------------------------------------------
option(USE_BLOCK2 "Build block2 from source to enable forte2.dmrg" OFF)
set(BLOCK2_GIT_TAG "v0.5.3" CACHE STRING "block2 version to build")

if (USE_BLOCK2)
  include(ExternalProject)
  include(ProcessorCount)

  # block2 needs roughly 1.2 GB of RAM per compile job, so cap the default
  # rather than saturating a large machine and running it out of memory.
  ProcessorCount(_block2_nproc)
  if (_block2_nproc EQUAL 0)
    set(_block2_nproc 2)
  elseif (_block2_nproc GREATER 8)
    set(_block2_nproc 8)
  endif()
  set(BLOCK2_BUILD_JOBS ${_block2_nproc}
      CACHE STRING "Parallel jobs for the block2 build (needs ~1.2 GB each)")

  # block2 v0.5.3 writes `FCIDUMP<FL>::template set_orb_sym(...)` with no
  # template argument list (src/core/integral_compressed.hpp). GCC accepts it;
  # Clang 16+ rejects it by default, not via block2's -Werror. Upstream fixed it
  # after v0.5.3, which is still the newest tag, so suppress it until a release
  # carries the fix. -Wno-unknown-warning-option keeps older Clang quiet about
  # the second flag.
  set(_block2_cxx_flags "")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_block2_cxx_flags
        "-Wno-unknown-warning-option -Wno-missing-template-arg-list-after-template-kw")
  endif()

  # Pin the exact BLAS/LAPACK forte2 itself resolved. Naming a vendor is not
  # enough: block2 does SET(BLA_VENDOR Apple) whenever the compiler is
  # AppleClang, overriding anything passed in, and would link Accelerate
  # alongside forte2's OpenBLAS. FindBLAS/FindLAPACK guard every vendor branch
  # with `if(NOT BLAS_LIBRARIES)`, so presetting the cache skips the search.
  string(REPLACE ";" "|" _block2_blas "${BLAS_LIBRARIES}")
  string(REPLACE ";" "|" _block2_lapack "${LAPACK_LIBRARIES}")

  # Download only; the build itself runs as a separate CMake invocation below.
  FetchContent_Declare(
    block2
    GIT_REPOSITORY https://github.com/block-hczhai/block2-preview.git
    GIT_TAG ${BLOCK2_GIT_TAG}
    GIT_SHALLOW TRUE
  )
  FetchContent_Populate(block2)

  set(BLOCK2_BINARY_DIR "${block2_BINARY_DIR}")

  ExternalProject_Add(block2_build
    SOURCE_DIR       "${block2_SOURCE_DIR}"
    BINARY_DIR       "${BLOCK2_BINARY_DIR}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND   ""
    INSTALL_COMMAND  ""
    LIST_SEPARATOR   |
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      "-DCMAKE_CXX_FLAGS=${_block2_cxx_flags}"
      # Everything block2 knows about Python keys off PYTHON_EXECUTABLE: its
      # include dirs, its extension suffix, and where it looks for pybind11. Its
      # own PYTHON_EXECUTABLE_HINT check is dead code (it tests a path string for
      # truth and compares paths with EQUAL), and the FindPythonInterp it falls
      # back to picks the newest python on PATH, not this environment's.
      -DPYTHON_EXECUTABLE=${Python_EXECUTABLE}
      # Link the same BLAS as forte2 rather than a vendored MKL.
      -DUSE_MKL=OFF
      -DUSE_MKL64=OFF
      -DBLAS_LIBRARIES=${_block2_blas}
      -DLAPACK_LIBRARIES=${_block2_lapack}
      # Required with USE_MKL=OFF. OpenBLAS exports only the trailing-underscore
      # Fortran names, so without this block2 links but fails at import with
      # "undefined symbol: zheev".
      -DF77UNDERSCORE=ON
      -DBUILD_LIB=ON
      -DBUILD_EXE=OFF
      -DLARGE_BOND=ON
      -DUSE_KSYMM=OFF
      # forte2 uses SymmetryTypes.SU2 and SGF|CPX only; SAny and single
      # precision roughly double the build for symmetries nothing here asks for.
      -DUSE_SU2SZ=ON
      -DUSE_SG=ON
      -DUSE_COMPLEX=ON
      -DUSE_SANY=OFF
      -DUSE_SINGLE_PREC=OFF
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=${BLOCK2_BINARY_DIR}/lib
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${BLOCK2_BUILD_JOBS}
  )

  add_dependencies(lib block2_build)

  # Collect the extension and the pure-Python pyblock2 package side by side, the
  # layout forte2/dmrg/__init__.py puts on sys.path.
  set(BLOCK2_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_block2")
  ExternalProject_Add_Step(block2_build stage_for_forte2
    COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/lib ${BLOCK2_STAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/pyblock2 ${BLOCK2_STAGE_DIR}/pyblock2
    DEPENDEES build
  )

  # An editable install imports forte2 from the source tree, so the vendored
  # copy has to be there; a wheel install takes it from the staging dir instead.
  if (SKBUILD_STATE STREQUAL "editable")
    ExternalProject_Add_Step(block2_build stage_in_source_tree
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              ${BLOCK2_STAGE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/_block2
      DEPENDEES stage_for_forte2
    )
  endif()

  # Vendored under forte2/_block2 rather than installed as top-level `block2`
  # and `pyblock2`: those names belong to the upstream distribution, and a later
  # `pip install block2` would overwrite this build with the MKL wheel and
  # silently reintroduce the duplicate-BLAS problem.
  install(DIRECTORY ${BLOCK2_STAGE_DIR}/ DESTINATION forte2/_block2)
endif()

# Enable Release build with high optimization flags if not already set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Optional: explicitly define optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
