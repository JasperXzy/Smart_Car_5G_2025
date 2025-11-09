cmake_minimum_required(VERSION 3.10)

project(fastestdet_ncnn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# macOS OpenMP 支持
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    include_directories("/opt/homebrew/opt/libomp/include")
    link_directories("/opt/homebrew/opt/libomp/lib")
    
    # 设置 OpenMP 为已找到，避免 NCNN 查找失败
    set(OpenMP_C_FOUND TRUE)
    set(OpenMP_CXX_FOUND TRUE)
endif()

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
    
    if(EXISTS "${NCNN_DIR}/build/install/lib/cmake/ncnn")
        set(ncnn_DIR "${NCNN_DIR}/build/install/lib/cmake/ncnn")
        find_package(ncnn QUIET)
        if(ncnn_FOUND)
            message(STATUS "Found NCNN in source install directory")
        endif()
    endif()
    
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
if(APPLE)
    target_link_libraries(infer omp)
else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(infer OpenMP::OpenMP_CXX)
    endif()
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
