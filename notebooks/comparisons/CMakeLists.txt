cmake_minimum_required(VERSION 3.10)

project(comparison_experiments)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")


# Find Shark
find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})

# Find Python 3
find_package(Python3 COMPONENTS Interpreter Development NumPy)

# Find Pybind11
find_package(pybind11 REQUIRED)

# Find Boost
find_package(Boost REQUIRED COMPONENTS 
    regex system filesystem serialization)

# Add some include directories (devcontainer-specific)
include_directories(/opt/conda/envs/development/include)
include_directories(/opt/conda/envs/development/lib/python3.8/site-packages/numpy/core/include)
include_directories(/opt/conda/envs/development/lib/python3.8/site-packages/pybind11/include)

# Hypervolume comparison
pybind11_add_module(hv_comparison src/hv_comparison.cpp)
target_include_directories(hv_comparison PRIVATE include)
target_link_libraries(hv_comparison PRIVATE ${SHARK_LIBRARIES})
target_link_libraries(hv_comparison PRIVATE ${Boost_LIBRARIES})
