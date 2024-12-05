#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "age_gender.h"
#include "utils.h"

class FaceDetector {
 private:
  // 使用唯一指针管理TensorRT资源
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  // 输入输出缓存
  void* deviceInputBuffer = nullptr;
  void* deviceOutputBuffer1 = nullptr;
  void* deviceOutputBuffer2 = nullptr;
  void* deviceOutputBuffer3 = nullptr;

  // 输入输出维度
  nvinfer1::Dims inputDims;
  nvinfer1::Dims outputDims1, outputDims2, outputDims3;

  // CUDA流
  cudaStream_t stream;

  // 日志回调类
  Logger logger;

 public:
  FaceDetector(const std::string& onnxModelPath);

  std::vector<std::vector<float>> detect(cv::Mat image);

  ~FaceDetector();

  std::vector<std::pair<std::vector<float>, AgeGenderDetector::DetectionResult>>
  detectWithAgeGender(const cv::Mat& image,
                      AgeGenderDetector& ageGenderDetector);
};