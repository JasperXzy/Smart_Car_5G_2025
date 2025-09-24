cmake_minimum_required(VERSION 3.12)
project(FastestDetONNX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ONNX Runtime
find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME REQUIRED onnxruntime)

message(STATUS "Found ONNX Runtime via pkg-config")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime libs: ${ONNXRUNTIME_LIBRARIES}")

# Add executable
add_executable(fastestdet_onnx fastestdet_onnx.cpp)

# Include header directories
target_include_directories(fastestdet_onnx PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(fastestdet_onnx 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
)

# Enable CUDA support
target_compile_definitions(fastestdet_onnx PRIVATE USE_CUDA)

# Install
install(TARGETS fastestdet_onnx DESTINATION bin)
