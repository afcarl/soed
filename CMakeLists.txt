cmake_minimum_required(VERSION 2.8)
project(oed)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Werror -Wno-unused -Wno-unused-parameter -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")

INCLUDE_DIRECTORIES ( "src" "extern/eigen" )

set(SOURCES
    src/Driver.cpp
    src/DPSolver.cpp
    src/RandomGenerator.cpp
    src/ValueFunction.cpp)

add_executable(Driver  ${SOURCES})
