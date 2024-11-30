#include "age_gender.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

AgeGenderDetector::AgeGenderDetector(const std::string& onnxModelPath) {
  // 初始化TensorRT运行时
  runtime.reset(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    throw std::runtime_error("创建TensorRT运行时失败");
  }

  // 解析ONNX模型
  auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(
      nvinfer1::createInferBuilder(logger));
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(
      builder->createNetworkV2(0));
  auto parser = std::unique_ptr<nvonnxparser::IParser, InferDeleter>(
      nvonnxparser::createParser(*network, logger));

  // 解析ONNX模型文件
  if (!parser->parseFromFile(onnxModelPath.c_str(), 2)) {
    throw std::runtime_error("解析ONNX模型失败");
  }

  // 配置TensorRT构建
  auto config = std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter>(
      builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);

  // 构建引擎
  engine.reset(builder->buildEngineWithConfig(*network, *config));
  if (!engine) {
    throw std::runtime_error("构建TensorRT引擎失败");
  }

  // 创建执行上下文
  context.reset(engine->createExecutionContext());
  if (!context) {
    throw std::runtime_error("创建执行上下文失败");
  }

  // 获取输入输出维度
  inputDims = network->getInput(0)->getDimensions();
  outputDims1 = network->getOutput(0)->getDimensions();
  outputDims2 = network->getOutput(1)->getDimensions();

  // 创建CUDA流
  cudaStreamCreate(&stream);

  // 分配GPU内存
  size_t inputSize =
      inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
  size_t outputSize1 = outputDims1.d[1] * outputDims1.d[2] * sizeof(float);
  size_t outputSize2 = outputDims2.d[1] * outputDims2.d[2] * sizeof(float);

  cudaMalloc(&deviceInputBuffer, inputSize);
  cudaMalloc(&deviceOutputBuffer1, outputSize1);
  cudaMalloc(&deviceOutputBuffer2, outputSize2);
}

// 从人脸检测网络的bbox中裁剪和处理图像
AgeGenderDetector::DetectionResult AgeGenderDetector::detect(
    const cv::Mat& image, const std::vector<float>& bbox_regressions) {
  // 假设bbox_regressions的格式为 [x, y, width, height]
  // 从bbox中裁剪人脸区域
  cv::Rect face_roi(bbox_regressions[0], bbox_regressions[1],
                    bbox_regressions[2], bbox_regressions[3]);

  // 确保ROI在图像范围内
  face_roi.x = std::max(0, face_roi.x);
  face_roi.y = std::max(0, face_roi.y);
  face_roi.width = std::min(face_roi.width, image.cols - face_roi.x);
  face_roi.height = std::min(face_roi.height, image.rows - face_roi.y);

  // 裁剪人脸区域
  cv::Mat face_image = image(face_roi);

  // 调整大小为网络输入尺寸
  cv::Mat resizedImage;
  cv::resize(face_image, resizedImage, cv::Size(240, 240));
  resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

  // 将图像拷贝到GPU
  cudaMemcpyAsync(deviceInputBuffer, resizedImage.ptr<float>(),
                  240 * 240 * 3 * sizeof(float), cudaMemcpyHostToDevice,
                  stream);

  // 绑定输入输出缓存
  void* bindings[] = {
      deviceInputBuffer,
      deviceOutputBuffer1,  // 性别
      deviceOutputBuffer2   // 年龄
  };

  // 执行推理
  context->executeV2(bindings);

  // 分配主机内存存储结果
  std::vector<float> gender_probs(outputDims1.d[1]);
  std::vector<float> age_probs(outputDims2.d[1]);

  // 从GPU拷贝结果
  cudaMemcpyAsync(gender_probs.data(), deviceOutputBuffer1,
                  gender_probs.size() * sizeof(float), cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(age_probs.data(), deviceOutputBuffer2,
                  age_probs.size() * sizeof(float), cudaMemcpyDeviceToHost,
                  stream);

  // 同步等待
  cudaStreamSynchronize(stream);

  // 后处理
  DetectionResult result;
  result.gender_probs = gender_probs;
  result.age_probs = age_probs;

  // 找到年龄组最大概率索引
  auto max_age_it = std::max_element(age_probs.begin(), age_probs.end());
  int max_age_index = std::distance(age_probs.begin(), max_age_it);

  result.predicted_ages.push_back(max_age_index);
  result.age_group_names.push_back(AGE_GROUPS.at(max_age_index));

  return result;
}

AgeGenderDetector::~AgeGenderDetector() {
  // 资源清理
  cudaFree(deviceInputBuffer);
  cudaFree(deviceOutputBuffer1);
  cudaFree(deviceOutputBuffer2);
  cudaStreamDestroy(stream);
}