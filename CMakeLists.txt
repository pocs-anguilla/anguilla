cmake_minimum_required(VERSION 3.18...3.19)

message(STATUS "Using CMake version ${CMAKE_VERSION}")

project(anguilla_cxx VERSION "0.0.19" LANGUAGES CXX)

find_program(CCACHE_EXE ccache)
if(CCACHE_EXE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_EXE}")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

set(CMAKE_CXX_STANDARD_REQUIRED ON FORCE)
set(CMAKE_CXX_EXTENSIONS OFF FORCE)
message(STATUS "C++ standard version: ${CMAKE_CXX_STANDARD}")

##########
# Includes

include(CMakeToolsHelpers OPTIONAL)
include(CMakeDependentOption)
include(FeatureSummary)

################
# Define options

option(PORTABLE_BUILD "Build the C extensions to be portable (e.g., for wheels)" ON)
cmake_dependent_option(WITH_XSIMD "Build using SIMD instructions" ON "NOT PORTABLE_BUILD" OFF)

###############
# Find Python 3

set(Python3_FIND_IMPLEMENTATIONS CPython)

if(SKBUILD)
  if(DEFINED $ENV{CONDA_BUILD})
    set(Python3_EXECUTABLE $ENV{PYTHON})
  endif()

  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

  # Using skbuild's as CMake's FindPython3 isn't working consistently
  # across different platforms on CI builds.
  find_package(NumPy REQUIRED)
  set(Python3_NumPy_INCLUDE_DIRS ${NumPy_INCLUDE_DIRS})
  message("Found NumPy: ${Python3_NumPy_INCLUDE_DIRS}")
else()
  find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
endif()

# Add site-packages to the search path
execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

message(STATUS "Found Python: ${Python3_INTERPRETER_ID}")

execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "import numpy as np; print(np.show_config())"
  OUTPUT_VARIABLE _numpy_config_info
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
message(STATUS "Numpy configuration:")
message(STATUS "${_numpy_config_info}")

###############
# Find Pybind11

find_package(pybind11 CONFIG REQUIRED)

########################
# Find Asan (Debug only)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  find_library(Asan_LIBRARY asan REQUIRED)
endif()

###############################
# Set general compilation flags

if(NOT MSVC)
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -g -fdiagnostics-color -Wstrict-aliasing")
    set(ASAN_FLAGS "-fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=address -fsanitize-address-use-after-scope")
    #set(UBSAN_FLAGS "-fsanitize=return -fsanitize=implicit-conversion -fsanitize=float-cast-overflow -fsanitize=float-divide-by-zero \
    #-fsanitize=implicit-unsigned-integer-truncation -fsanitize=implicit-signed-integer-truncation -fsanitize=unsigned-integer-overflow")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${ASAN_FLAGS} ${UBSAN_FLAGS}")
    #set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${ASAN_FLAGS} -shared-libasan -shared-libubsan")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${ASAN_FLAGS} -shared-libasan")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fno-trapping-math -fno-math-errno")
  endif()
  if(MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_hypot=hypot")
  endif()
  if(WITH_XSIMD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64")
  endif()
else()
  # See: https://cibuildwheel.readthedocs.io/en/stable/faq/#importerror-dll-load-failed-the-specific-module-could-not-be-found-error-on-windows
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /d2FH4-")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W4 /EHsc /MP /bigobj") # required by xtensor
  set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
  if(WITH_XSIMD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
  endif()
endif()

####################
# Add vendor modules

## Boost
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/assert/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/config/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/move/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/static_assert/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/intrusive/include")

## BTree
set(BTREE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/btree/include")

## Spatial
set(SPATIAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/spatial/include")

## Xtensor
add_subdirectory(vendor/cxx/xtl EXCLUDE_FROM_ALL)
add_subdirectory(vendor/cxx/xtensor EXCLUDE_FROM_ALL)

### Xtensor SIMD
if(WITH_XSIMD)
  add_definitions(-DXTENSOR_USE_XSIMD=1)
  add_subdirectory(vendor/cxx/xsimd EXCLUDE_FROM_ALL)
endif()

### Xtensor BLAS
if(NOT PORTABLE_BUILD)
  find_package(BLAS REQUIRED)
  if(BLAS_FOUND)
    add_definitions(-DHAVE_CBLAS=1)
    find_package(LAPACK REQUIRED)
    message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
    message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
  endif()
else()
  message(STATUS "No BLAS found, will use FLENS + CXXBLAS.")
  add_definitions(-DXTENSOR_USE_FLENS_BLAS=1)
endif()
add_subdirectory(vendor/cxx/xtensor-blas EXCLUDE_FROM_ALL)

### Xtensor Python bindings
#add_subdirectory(vendor/cxx/xtensor-python)
add_subdirectory(vendor/cxx/pyxtensor EXCLUDE_FROM_ALL)

### {fmt}
add_subdirectory(vendor/cxx/fmt EXCLUDE_FROM_ALL)

### Anguilla CXX
add_subdirectory(anguilla/cxx)

feature_summary(WHAT ALL)
