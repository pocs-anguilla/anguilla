cmake_minimum_required(VERSION 3.10)

project(experiments)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")

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

add_executable(fitness ${FITNESS_SRC})
target_link_libraries(fitness PRIVATE ${SHARK_LIBRARIES})
target_link_libraries(fitness PRIVATE ${Boost_LIBRARIES})
