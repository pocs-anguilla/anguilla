cmake_minimum_required(VERSION 3.18)

project(anguilla_cxx VERSION "0.0.1" LANGUAGES CXX)

include(CMakeToolsHelpers OPTIONAL)
include(FeatureSummary)

set(CMAKE_CXX_STANDARD_REQUIRED ON FORCE)
set(CMAKE_CXX_EXTENSIONS OFF FORCE)
message(STATUS "Using C++ standard version: ${CMAKE_CXX_STANDARD}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall")
if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O -g")
  set(ASAN_FLAGS "-fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=address -fsanitize-address-use-after-scope")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${ASAN_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${ASAN_FLAGS} -shared-libasan")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
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
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Add site-packages to the search path
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Find Pybind11
find_package(pybind11 CONFIG REQUIRED)

# Define options
option(WITH_SHARK_BINDINGS "Build bindings for Shark's HV implementation" OFF)

# Add the different modules

## BTree
set(BTREE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/anguilla/cxx/btree/include")

## Hypervolume
add_subdirectory(anguilla/cxx/hypervolume)

## Bindings for Shark's Hypervolume (optional)
if(WITH_SHARK_BINDINGS)
  add_subdirectory(anguilla/cxx/shark_hypervolume)
endif()

feature_summary(WHAT ALL)
