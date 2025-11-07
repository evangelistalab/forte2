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
  _forte2

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

  # Source code goes here
  api/forte2_api.cc
  api/ci_api.cc
  api/determinant_api.cc
  api/determinant_helpers_api.cc
  api/helpers_api.cc
  api/integrals_api.cc
  api/logging_api.cc
  api/slater_rules_api.cc
  api/sparse_exp_api.cc
  api/sparse_operator_api.cc
  api/sparse_state_api.cc
  api/sq_operator_string_api.cc
  helpers/combinatorial.cc
  helpers/string_algorithms.cc
  integrals/basis.cc
  integrals/fock_builder.cc
  integrals/nuclear_repulsion.cc
  integrals/one_electron.cc
  integrals/two_electron.cc
  integrals/value_at_points.cc
  ci/ci_sigma_builder.cc
  ci/ci_sigma_builder_1rdm.cc
  ci/ci_sigma_builder_2rdm.cc
  ci/ci_sigma_builder_3rdm.cc
  ci/ci_sigma_builder_test_rdms.cc
  ci/ci_sigma_builder_harrison_zarrabian.cc
  ci/ci_sigma_builder_knowles_handy.cc
  ci/ci_spin_adapter.cc
  ci/ci_strings.cc
  ci/ci_strings_makers.cc
  ci/ci_string_address.cc
  ci/ci_string_class.cc
  ci/ci_occupation.cc
  ci/determinant_helpers.cc
  ci/rel_ci_sigma_builder.cc
  ci/rel_ci_sigma_builder_rdm.cc
  ci/rel_ci_sigma_builder_rdm_debug.cc
  ci/rel_ci_sigma_builder_harrison_zarrabian.cc
  ci/slater_rules.cc
  sparse/sparse_exp.cc
  sparse/sparse_operator.cc
  sparse/sparse_operator_hamiltonian.cc
  sparse/sparse_state_functions.cc
  sparse/sparse_state.cc
  sparse/sq_operator_string.cc
)

target_link_libraries(_forte2 PRIVATE Libint2::int2)
target_link_libraries(_forte2 PRIVATE Eigen3::Eigen)
target_link_libraries(_forte2 PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Conditionally link libcint and expose macro to C++
if (USE_LIBCINT)
  target_link_libraries(_forte2 PRIVATE cint)
  target_compile_definitions(_forte2 PRIVATE FORTE2_USE_LIBCINT=1)
else()
  target_compile_definitions(_forte2 PRIVATE FORTE2_USE_LIBCINT=0)
endif()

target_include_directories(_forte2
  PUBLIC
    .
)

if (USE_LIBCINT)
  # Rename the cint library output, so we don't conflict with PySCF's libcint, if any
  set_target_properties(cint PROPERTIES
      OUTPUT_NAME "cint_forte2"  # This makes it libcint_forte2.dylib
  )
endif()

# Set rpath to use @loader_path so it finds the dylib in the same directory
set_target_properties(_forte2 PROPERTIES
    INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install directive for scikit-build-core
install(TARGETS _forte2 LIBRARY DESTINATION forte2)

if (USE_LIBCINT)
  # Install "libcint_forte2.<version>.dylib/so" (and shortened symlinks)
  install(TARGETS cint 
      LIBRARY DESTINATION "$ENV{CONDA_PREFIX}/lib"
  )
endif()

# Enable Release build with high optimization flags if not already set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Optional: explicitly define optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
