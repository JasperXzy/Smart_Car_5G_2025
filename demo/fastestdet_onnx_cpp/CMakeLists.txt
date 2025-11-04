cmake_minimum_required(VERSION 3.10)
project(FastestDetONNX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# 查找OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# ONNX Runtime 路径
set(ONNXRUNTIME_ROOT_PATH "" CACHE PATH "Path to ONNX Runtime installation")

if(ONNXRUNTIME_ROOT_PATH)
    # 手动指定路径
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
    set(ONNXRUNTIME_LIB_DIRS ${ONNXRUNTIME_ROOT_PATH}/lib)
    
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_LIB_DIRS}
        NO_DEFAULT_PATH
    )
else()
    # 尝试自动查找
    find_path(ONNXRUNTIME_INCLUDE_DIRS
        NAMES onnxruntime_cxx_api.h
        PATHS
        /usr/include
        /usr/local/include
        /usr/include/onnxruntime
        /usr/local/include/onnxruntime
    )
    
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS
        /usr/lib
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu
    )
endif()

if(ONNXRUNTIME_INCLUDE_DIRS AND ONNXRUNTIME_LIBRARY)
    message(STATUS "ONNX Runtime include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
else()
    message(FATAL_ERROR "ONNX Runtime not found! Please set ONNXRUNTIME_ROOT_PATH or install ONNX Runtime.")
endif()

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(fastestdet_onnx infer.cpp)

# 链接库
target_link_libraries(fastestdet_onnx
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

# 安装规则
install(TARGETS fastestdet_onnx
    RUNTIME DESTINATION bin
)
