cmake_minimum_required(VERSION 3.10)

project(SmartCar_5G_2025)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

# 头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    /usr/include
    /usr/local/include  
)

# 库文件搜索路径
link_directories(
    /lib
    /usr/lib
    /usr/local/lib
)

# 查找依赖库
find_package(OpenMV REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(ncnn REQUIRED)
find_package(yaml-cpp REQUIRED)

# pigpio 库路径
set(PIGPIO_LIBRARY "/lib/libpigpio.so")

# 源文件
set(SMARTCAR_SOURCES
    src/Config.cpp
    src/Hardware.cpp
    src/ImageProcess.cpp
    src/AIDetector.cpp
    src/ColorDetector.cpp
    src/Navigation.cpp
    src/RaceController.cpp
)

# 主程序
add_executable(SmartCar
    main.cpp
    ${SMARTCAR_SOURCES}
)

# 输出目录为项目根目录
set_target_properties(SmartCar PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# 链接库
target_link_libraries(SmartCar
    ncnn
    ${OpenCV_LIBS}
    yaml-cpp
    ${CMAKE_THREAD_LIBS_INIT}
    pthread
)

# 添加 OpenMP 链接
if(OpenMP_CXX_FOUND)
    target_link_libraries(SmartCar OpenMP::OpenMP_CXX)
endif()

# 添加 pigpio 链接
if(PIGPIO_FOUND)
    target_link_libraries(SmartCar ${PIGPIO_LIBRARY})
    target_compile_definitions(SmartCar PRIVATE ENABLE_PIGPIO)
endif()

# 打印配置信息
message(STATUS "======================================")
message(STATUS "Smart Car 5G")
message(STATUS "======================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "NCNN found: ${ncnn_FOUND}")
message(STATUS "yaml-cpp found: ${yaml-cpp_FOUND}")
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
message(STATUS "pigpio found: ${PIGPIO_FOUND}")
message(STATUS "======================================")

# 安装配置
install(TARGETS SmartCar DESTINATION bin)
install(DIRECTORY config/ DESTINATION config)
install(DIRECTORY models/ DESTINATION models)
