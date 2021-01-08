# Adapted from: https://github.com/ruslo/polly

find_program(CMAKE_C_COMPILER clang)
find_program(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_COMPILER
    "${CMAKE_C_COMPILER}"
    CACHE STRING "C compiler" FORCE)

set(CMAKE_CXX_COMPILER
    "${CMAKE_CXX_COMPILER}"
    CACHE STRING "C++ compiler" FORCE)

set(CMAKE_CXX_FLAGS "-stdlib=libstdc++" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ version selection" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON FORCE)
set(CMAKE_CXX_EXTENSIONS OFF FORCE)
message(STATUS "Using C++ standard version: ${CMAKE_CXX_STANDARD}")
