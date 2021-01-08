cmake_minimum_required(VERSION 3.10)

project(experiments)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python3
set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Find Shark
find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})

# Find Boost
find_package(Boost REQUIRED COMPONENTS 
    regex system filesystem serialization)

## Project sources
set(EXP1_SRC
  src/experiment1.cpp
)
set(MOCMA_SRC
  src/mocma.cpp
)

set(FITNESS_SRC
  src/fitness.cpp
)

## Project executable
add_executable(experiment1 ${EXP1_SRC})
target_link_libraries(experiment1 PRIVATE ${SHARK_LIBRARIES})
target_link_libraries(experiment1 PRIVATE ${Boost_LIBRARIES})

add_executable(mocma ${MOCMA_SRC})
target_link_libraries(mocma PRIVATE ${SHARK_LIBRARIES})
target_link_libraries(mocma PRIVATE ${Boost_LIBRARIES})
target_link_libraries(mocma PRIVATE ${Python3_LIBRARIES})
target_include_directories(mocma PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(mocma PRIVATE ${Python3_NumPy_INCLUDE_DIRS})

add_executable(fitness ${FITNESS_SRC})
target_link_libraries(fitness PRIVATE ${SHARK_LIBRARIES})
target_link_libraries(fitness PRIVATE ${Boost_LIBRARIES})
