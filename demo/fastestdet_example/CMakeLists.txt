cmake_minimum_required(VERSION 3.10)
project(FastestDetExample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# macOS OpenMP支持
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    include_directories("/opt/homebrew/opt/libomp/include")
    link_directories("/opt/homebrew/opt/libomp/lib")
    
    # 设置OpenMP为已找到
    set(OpenMP_C_FOUND TRUE)
    set(OpenMP_CXX_FOUND TRUE)
endif()

# 查找OpenCV
find_package(OpenCV REQUIRED)

# NCNN路径配置
find_package(ncnn REQUIRED)

if(NOT DEFINED NCNN_INCLUDE_DIR)
    set(NCNN_INCLUDE_DIR "/opt/homebrew/include/ncnn" CACHE PATH "NCNN include directory")
endif()
if(NOT DEFINED NCNN_LIB_DIR)
    set(NCNN_LIB_DIR "/opt/homebrew/lib" CACHE PATH "NCNN library directory")
endif()

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/../../inc
    ${OpenCV_INCLUDE_DIRS}
    ${NCNN_INCLUDE_DIR}
)

# 链接目录
link_directories(
    ${NCNN_LIB_DIR}
)

# 添加可执行文件
add_executable(fastestdet_example 
    fastestdet_example.cpp
    ../../src/fastestdet.cpp
)

# 链接库
target_link_libraries(fastestdet_example
    ${OpenCV_LIBS}
    ncnn
    pthread
)

# 添加OpenMP
if(APPLE)
    target_link_libraries(fastestdet_example omp)
else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fastestdet_example OpenMP::OpenMP_CXX)
    endif()
endif()

# 打印配置信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "NCNN include: ${NCNN_INCLUDE_DIR}")
message(STATUS "NCNN lib dir: ${NCNN_LIB_DIR}")
