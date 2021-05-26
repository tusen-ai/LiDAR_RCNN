cmake_minimum_required(VERSION 2.8.12)
project(lidar_bbox_tools_c)

if (NOT TARGET pybind11)
    find_package(pybind11 REQUIRED)
endif ()
set(PYBIND11_CPP_STANDARD -std=c++11)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3 -Wno-unused-result -Wno-logical-op-parentheses -fext-numeric-literals")
set(PYTHON_EXECUTABLE /usr/bin/python)

find_package(Eigen3 REQUIRED)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)


# add_subdirectory(pybind11)
pybind11_add_module(lidar_bbox_tools_c SHARED THIN_LTO src/calc_conf.cpp include/calc_conf.h 
                                    src/main.cpp src/overlap.cpp include/overlap.h 
                                    src/polygon_overlap.cpp include/polygon_overlap.h
                                    src/extract_points.cpp include/extract_points.h)