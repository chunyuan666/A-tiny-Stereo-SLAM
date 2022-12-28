cmake_minimum_required(VERSION 3.0)
project(myslam)

# add_compile_options(-fPIC)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -g -fopenmp -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++14  -O3 -fopenmp -pthread")
set(CMAKE_CXX_FLAGS_Release "-std=c++14 -O3 -fopenmp -pthread")

# 第三方编译库
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
message(STATUS ${PROJECT_SOURCE_DIR})

# 设置可执行文件路径、库文件路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
include_directories(SYSTEM "/foo/bar")

#Eigen
#include_directories("/usr/include/eigen3")
find_package(Eigen3 3.1.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_Include_DIRS})

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# glog: 实现应用级别的logging
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# caffe (default path: ~/caffe)
add_definitions( -DCPU_ONLY=1 )
if (NOT Caffe_ROOT_DIR)
	set(Caffe_ROOT_DIR $ENV{HOME}/caffe)
endif()
MESSAGE("${GREEN}Using Caffe_ROOT_DIR: ${Caffe_ROOT_DIR}${ENDL}")
set(Caffe_LIBS ${Caffe_ROOT_DIR}/build/install/lib/libcaffe.so)
set(Caffe_INCLUDE_DIRS ${Caffe_ROOT_DIR}/build/install/include)
include_directories(${Caffe_INCLUDE_DIRS})
MESSAGE("Caffe_LIBS: ${Caffe_LIBS}")
MESSAGE("Caffe_INCLUDE_DIRS: ${Caffe_INCLUDE_DIRS}")



# cuda (optional)
find_package(CUDA QUIET)
if (${CUDA_FOUND})
 	include_directories(${CUDA_INCLUDE_DIRS})
	MESSAGE("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
endif()

# 设置第三方库
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        fmt
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        ${CSPARSE_LIBRARY}
        pthread
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        ${Caffe_LIBS}
        )

## 设置源文件路径、头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/[A-z]*.cpp")
add_library(${PROJECT_NAME} SHARED
        ${SOURCES}
)

# add_library(${PROJECT_NAME} SHARED
#         src/System.cpp
#         src/Camera.cpp
#         src/Frame.cpp
#         src/Frontend.cpp
#         src/ORBExtractor.cpp
#         src/MapPoints.cpp
#         src/Map.cpp
#         src/KeyFrame.cpp
#         src/Optimizer.cpp
#         src/Backend.cpp
#         src/Viewer.cpp
#         src/LoopClosing.cpp
#         src/DeepLCD.cpp)

enable_testing()

add_executable(stereo_kitti app/stereo_kitti.cpp)
target_link_libraries(stereo_kitti ${PROJECT_NAME} ${THIRD_PARTY_LIBS})



