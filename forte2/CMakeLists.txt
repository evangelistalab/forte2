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
  api/integrals_api.cc
  api/determinant_api.cc
  api/determinant_helpers_api.cc
  api/sparse_operator_api.cc
  api/sparse_state_api.cc
  api/sparse_exp_api.cc
  helpers/string_algorithms.cc
  helpers/combinatorial.cc
  ints/basis.cc
  ints/fock_builder.cc
  ints/nuclear_repulsion.cc
  ints/one_electron.cc
  ints/two_electron.cc
  ints/value_at_points.cc
  ci/determinant_helpers.cc
  ci/sparse_exp.cc
  ci/sparse_operator.cc
  ci/sparse_state_functions.cc
  ci/sparse_state.cc
  ci/sq_operator_string.cc
)

target_link_libraries(_forte2 PRIVATE Libint2::int2)
target_link_libraries(_forte2 PRIVATE Eigen3::Eigen)

target_include_directories(_forte2
  PUBLIC
    .
)

# Install directive for scikit-build-core
install(TARGETS _forte2 LIBRARY DESTINATION forte2)
