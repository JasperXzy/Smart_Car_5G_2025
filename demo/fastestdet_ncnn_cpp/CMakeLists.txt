cmake_minimum_required(VERSION 3.10)

project(fastestdet_ncnn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# NCNN
set(NCNN_DIR "${CMAKE_SOURCE_DIR}/../../ncnn" CACHE PATH "NCNN source directory")

find_package(ncnn QUIET)

if(ncnn_FOUND)
    message(STATUS "Found installed NCNN")
else()
    message(STATUS "System NCNN not found, trying source directory...")
    
    # 尝试从源码安装目录查找
    if(EXISTS "${NCNN_DIR}/build/install/lib/cmake/ncnn")
        set(ncnn_DIR "${NCNN_DIR}/build/install/lib/cmake/ncnn")
        find_package(ncnn QUIET)
        if(ncnn_FOUND)
            message(STATUS "Found NCNN in source install directory")
        endif()
    endif()
    
    # 尝试直接使用源码构建目录
    if(NOT ncnn_FOUND)
        if(EXISTS "${NCNN_DIR}/build/src/libncnn.a")
            message(STATUS "Using NCNN from source build directory")
            include_directories(${NCNN_DIR}/src)
            link_directories(${NCNN_DIR}/build/src)
            set(NCNN_LIBS ncnn)
        else()
            message(FATAL_ERROR "NCNN not found! Please install NCNN or build it from source.")
        endif()
    endif()
endif()

# 添加可执行文件
add_executable(infer infer.cpp)

# 链接库
if(ncnn_FOUND)
    target_link_libraries(infer ncnn ${OpenCV_LIBS})
else()
    target_link_libraries(infer ${NCNN_LIBS} ${OpenCV_LIBS})
endif()

# 添加OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(infer OpenMP::OpenMP_CXX)
endif()

# 输出配置信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
if(ncnn_FOUND)
    message(STATUS "NCNN: Found (using find_package)")
else()
    message(STATUS "NCNN: Source directory")
    message(STATUS "NCNN path: ${NCNN_DIR}")
endif()
