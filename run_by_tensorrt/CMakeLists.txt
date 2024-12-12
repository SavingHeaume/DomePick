cmake_minimum_required(VERSION 3.18)
project(DomePick)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(/w)

# TensorRT paths                                                               
set(TENSORRT_ROOT "D:\\Tools\\TensorRT")
set(CUDA_ROOT "D:\\Tools\\CudaToolkit")

find_package(OpenCV REQUIRED)
find_package(jsoncpp REQUIRED)

# Include directories
include_directories(
    ${TENSORRT_ROOT}/include
    ${CUDA_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Library directories
link_directories(
    ${TENSORRT_ROOT}/lib
    ${CUDA_ROOT}/lib/x64
)

# Source files
set(SRC_FILES
    src/face_detector.cpp
    src/age_gender.cpp
    src/decode_face.cpp
    src/run.cpp
    src/json_converter.cpp
    src/main.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
  nvinfer_10
  nvonnxparser_10
  nvinfer_plugin_10
  cudart
  ${OpenCV_LIBS}
  jsoncpp_lib
)

# Set CUDA runtime library (static or shared)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# Ensure CUDA and TensorRT libraries are found at runtime
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
