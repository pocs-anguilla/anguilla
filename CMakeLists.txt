cmake_minimum_required(VERSION 3.18)
message(STATUS "Using CMake version ${CMAKE_VERSION}")

project(anguilla_cxx VERSION "0.0.19" LANGUAGES CXX)

find_program(CCACHE_EXE ccache)
if(CCACHE_EXE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_EXE}")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(CMakeToolsHelpers OPTIONAL)
include(FeatureSummary)

set(CMAKE_CXX_STANDARD_REQUIRED ON FORCE)
set(CMAKE_CXX_EXTENSIONS OFF FORCE)
message(STATUS "Using C++ standard version: ${CMAKE_CXX_STANDARD}")

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W4")
  # See: https://cibuildwheel.readthedocs.io/en/stable/faq/#importerror-dll-load-failed-the-specific-module-could-not-be-found-error-on-windows
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /d2FH4-")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall")
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
endif()

# Find Python 3
if(DEFINED $ENV{CONDA_BUILD})
  set(Python3_EXECUTABLE $ENV{PYTHON})
else()
  if(SKBUILD)
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
    set(Python3_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
    set(Python3_LIBRARY "${PYTHON_LIBRARY}")
    set(_DUMMY "${PYTHON_VERSION_STRING}")  # silence warning
  endif()
endif()

set(Python3_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "Found Python: ${Python3_INTERPRETER_ID}")

# Add site-packages to the search path
execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "import numpy as np; print(np.show_config())"
  OUTPUT_VARIABLE _numpy_config_info
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
message(STATUS "Numpy configuration:")
message(STATUS "${_numpy_config_info}")

# Find Pybind11
find_package(pybind11 CONFIG REQUIRED)

# Define options
option(WITH_SHARK_BINDINGS "Build bindings for Shark's HV implementation" OFF)

# Add the different modules

## Vendor

### Boost
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/assert/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/config/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/move/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/static_assert/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/intrusive/include")

### BTree
set(BTREE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/btree/include")

### Spatial
set(SPATIAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/spatial/include")

## Xtensor
add_subdirectory(vendor/cxx/xtl)
add_subdirectory(vendor/cxx/xtensor)
#add_subdirectory(vendor/cxx/xtensor-python)
add_subdirectory(vendor/cxx/pyxtensor)

## Xtensor BLAS

### Find BLAS
add_definitions(-DHAVE_CBLAS=1)
find_package(BLAS)
if(BLAS_FOUND)
  find_package(LAPACK REQUIRED)
  message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
  message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
else()
  message(STATUS "No BLAS found, will use fall-back provided by xtensor.")
  add_definitions(-DXTENSOR_USE_FLENS_BLAS=1)
endif()

add_subdirectory(vendor/cxx/xtensor-blas)

## {fmt}
add_subdirectory(vendor/cxx/fmt)

## Anguilla CXX
add_subdirectory(anguilla/cxx)

feature_summary(WHAT ALL)
