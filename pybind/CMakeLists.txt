cmake_minimum_required(VERSION 3.15)
project(mycpp_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set PYBIND11_FINDPYTHON ON to avoid the policy warning
set(PYBIND11_FINDPYTHON ON)

# Find dependencies
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Set Bartels paths
set(bartels_ROOT ${CMAKE_SOURCE_DIR}/../Bartels)

# Set LIBIGL_DIR to point to the correct location before including bartels.cmake
set(LIBIGL_DIR ${bartels_ROOT}/extern/libigl)

# Add Bartels cmake modules to module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${bartels_ROOT}/cmake)

# Include Bartels CMake configuration
include(${bartels_ROOT}/cmake/bartels.cmake)

# Add Python module
pybind11_add_module(pyBartels arap.cpp)

# Set include directories for the Python module
target_include_directories(pyBartels 
    PRIVATE 
        ${EIGEN3_INCLUDE_DIR}
)

# Link against Bartels and Eigen
target_link_libraries(pyBartels 
    PRIVATE 
        bartels
        Eigen3::Eigen
)