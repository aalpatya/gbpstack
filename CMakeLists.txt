# /**************************************************************************************/
# // Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
# // This code is licensed (see LICENSE for details)
# /**************************************************************************************/
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
# Set default build type to release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Name of Project
project(gbp_cpp_project)
add_compile_definitions(EIGEN_DONT_PARALLELIZE)
find_package(OpenMP REQUIRED)
add_subdirectory(thirdparty/raylib)

#################################################################

add_library(gbp_lib STATIC
src/utils.cpp 
src/Variable.cpp 
src/Factorgraph.cpp 
src/Factor.cpp 
src/globals.cpp
src/Robot.cpp
src/Simulator.cpp
)
target_compile_definitions(gbp_lib PUBLIC PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")
target_include_directories(gbp_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
target_link_libraries(gbp_lib PUBLIC OpenMP::OpenMP_CXX raylib)

add_executable(gbp_cpp 
    src/main.cpp
)
target_link_libraries(gbp_cpp PUBLIC gbp_lib)

#################################################################

