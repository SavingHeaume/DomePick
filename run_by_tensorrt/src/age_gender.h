#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#include "utils.h"


// 年龄组映射
const std::unordered_map<int, std::string> AGE_GROUPS = {
    {0, "(0, 2)"},   {1, "(4, 6)"},   {2, "(8, 13)"},  {3, "(15, 20)"},
    {4, "(25, 32)"}, {5, "(38, 43)"}, {6, "(48, 53)"}, {7, "(60, )"}};

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
    std::vector<float> gender_probs;           // 性别概率 [female, male]
    std::vector<float> age_probs;              // 年龄组概率
    std::vector<int> predicted_ages;           // 预测的年龄组索引
    std::vector<std::string> age_group_names;  // 年龄组名称
  };

  AgeGenderDetector(const std::string& onnxModelPath);
  // 从人脸检测网络的bbox中裁剪和处理图像
  DetectionResult detect(const cv::Mat& image,
                         const std::vector<float>& bbox_regressions);

  ~AgeGenderDetector();
};