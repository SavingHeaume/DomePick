cmake_minimum_required(VERSION 3.18)
project(DomePick)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-w)

# TensorRT paths
set(TENSORRT_ROOT "D:\\Tools\\TensorRT")
set(CUDA_ROOT "D:\\Tools\\CudaToolkit")
set(CV_ROOT "D:\\Tools\\OpenCV")

# Include directories
include_directories(
    ${TENSORRT_ROOT}/include
    ${CUDA_ROOT}/include
    ${CV_ROOT}/include
)

# Library directories
link_directories(
    ${TENSORRT_ROOT}/lib
    ${CUDA_ROOT}/lib/x64
    ${CV_ROOT}/x64/vc16/lib
)

# Source files
set(SRC_FILES
    src/face_detector.cpp
    src/age_gender.cpp
    src/main.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
  nvinfer
  nvonnxparser
  nvinfer_plugin
  cudart
  opencv_core 
  opencv_imgproc 
  opencv_highgui 
  opencv_imgcodecs
)

# Set CUDA runtime library (static or shared)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# Ensure CUDA and TensorRT libraries are found at runtime
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
