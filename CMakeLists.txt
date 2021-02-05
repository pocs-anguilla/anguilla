cmake_minimum_required(VERSION 3.18)

project(anguilla_cxx VERSION "0.0.14" LANGUAGES CXX)

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
find_package(Python3 COMPONENTS Interpreter Development)
#find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Add site-packages to the search path
execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Find Pybind11
find_package(pybind11 CONFIG REQUIRED)

# Define options
option(WITH_SHARK_BINDINGS "Build bindings for Shark's HV implementation" OFF)

# Add the different modules

## Vendor

### BTree
set(BTREE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/btree/include")

### Boost
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/assert/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/config/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/move/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/static_assert/include")
list(APPEND BOOST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/boost/intrusive/include")

### Spatial
set(SPATIAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cxx/spatial/include")

## Anguilla

### Common
set(COMMON_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/anguilla/cxx/common/include")

### Hypervolume
add_subdirectory(anguilla/cxx/hypervolume)

### Dominance
add_subdirectory(anguilla/cxx/dominance)

### Archive
add_subdirectory(anguilla/cxx/archive)
set(ANGUILLA_ARCHIVE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/anguilla/cxx/archive/include")

### Bindings for Shark's Hypervolume (optional)
if(WITH_SHARK_BINDINGS)
  add_subdirectory(anguilla/cxx/shark_hypervolume)
endif()

feature_summary(WHAT ALL)
