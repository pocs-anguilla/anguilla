cmake_minimum_required(VERSION 3.15...3.18)

project(anguilla_cxx VERSION "0.0.1" LANGUAGES CXX)

include(CMakeToolsHelpers OPTIONAL)
include(FeatureSummary)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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
if(SKBUILD)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARY "${PYTHON_LIBRARY}")
  set(_DUMMY "${PYTHON_VERSION_STRING}")  # silence warning
endif()

set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
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

# Add the different modules

## BTree
set(BTREE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/anguilla/cxx/btree/include")

## Hypervolume
add_subdirectory(anguilla/cxx/hypervolume)

## Bindings for Shark's Hypervolume (optional)
find_package(Shark)
if(Shark_FOUND)
  find_package(Boost COMPONENTS regex system filesystem serialization)
  if(Boost_FOUND)
    message(STATUS "Will build bindings for Shark's HV implementation")
    include(${SHARK_USE_FILE})
    add_subdirectory(anguilla/cxx/shark_hypervolume)
  endif()
endif()

feature_summary(WHAT ALL)
