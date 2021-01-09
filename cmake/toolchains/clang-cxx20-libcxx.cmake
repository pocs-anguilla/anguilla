# Adapted from: https://github.com/ruslo/polly

find_program(CMAKE_C_COMPILER clang)
find_program(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_COMPILER
    "${CMAKE_C_COMPILER}"
    CACHE STRING "C compiler" FORCE)

set(CMAKE_CXX_COMPILER
    "${CMAKE_CXX_COMPILER}"
    CACHE STRING "C++ compiler" FORCE)

set(CMAKE_CXX_FLAGS "-stdlib=libc++" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection" FORCE)
