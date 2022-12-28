cmake_minimum_required(VERSION 3.0)
project(test)

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14  -O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++14  -O0 -fopenmp -pthread")
# set(CMAKE_CXX_FLAGS_Release "-std=c++14 -O3 -fopenmp -pthread")


#Eigen
#include_directories("/usr/include/eigen3")
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_Include_DIRS})

#OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_Include_DIRS})


## 设置源文件路径、头文件路径

add_executable(RANSAC RANSAC.cpp)
target_link_libraries(test ${OpenCV_LIBS})

add_executable(test test.cpp)
target_link_libraries(test ${OpenCV_LIBS})




