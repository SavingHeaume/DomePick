#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#include "utils.h"

class AgeGenderDetector {
 private:
  // TensorRT组件
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  // 输入输出缓存
  void* deviceInputBuffer = nullptr;
  void* deviceOutputBuffer1 = nullptr;  // 性别
  void* deviceOutputBuffer2 = nullptr;  // 年龄

  // 输入输出维度
  nvinfer1::Dims inputDims;
  nvinfer1::Dims outputDims1, outputDims2;

  // CUDA流
  cudaStream_t stream;

  // 日志回调类
  Logger logger;

 public:
  // 年龄性别识别结果结构体
  struct DetectionResult {
    int gender;
    std::string age;
  };

  AgeGenderDetector(const std::string& onnxModelPath);
  // 从人脸检测网络的bbox中裁剪和处理图像
  DetectionResult detect(const cv::Mat& image);

  ~AgeGenderDetector();
};