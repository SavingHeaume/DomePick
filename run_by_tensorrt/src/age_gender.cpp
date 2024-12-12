#include "age_gender.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

std::unordered_map<int, std::string> AGE_GROUPS = {
    {0, "(0, 2)"},   {1, "(4, 6)"},   {2, "(8, 13)"},  {3, "(15, 20)"},
    {4, "(25, 32)"}, {5, "(38, 43)"}, {6, "(48, 53)"}, {7, "(60, )"} };

AgeGenderDetector::AgeGenderDetector(const std::string& onnxModelPath) {
  // 初始化TensorRT运行时
  runtime.reset(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }

  // 解析ONNX模型
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(logger));
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(0));
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));

  // 解析ONNX模型文件
  if (!parser->parseFromFile(onnxModelPath.c_str(), 2)) {
    throw std::runtime_error("Failed to parse ONNX model");
  }

  // 配置TensorRT构建
  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
      builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);

  // 定义优化配置文件
  auto profile = builder->createOptimizationProfile();
  auto input = network->getInput(0);
  if (!input) {
    throw std::runtime_error("Failed to get network input");
  }

  const std::string inputName = input->getName();
  auto inputDims = input->getDimensions();

  if (inputDims.nbDims != 4) {
    throw std::runtime_error("Unexpected input dimensions");
  }

  // 设置输入的最小、最优和最大值范围
  profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 224, 224));
  profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 224, 224));
  profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 224, 224));

  config->addOptimizationProfile(profile);

  // 构建引擎
  engine.reset(builder->buildEngineWithConfig(*network, *config));
  if (!engine) {
    throw std::runtime_error("Failed to build TensorRT engine");
  }

  // 创建执行上下文
  context.reset(engine->createExecutionContext());
  if (!context) {
    throw std::runtime_error("Failed to create execution context");
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
  size_t outputSize1 = outputDims1.d[1] * sizeof(float);
  size_t outputSize2 = outputDims2.d[1] * sizeof(float);

  cudaMalloc(&deviceInputBuffer, inputSize);
  cudaMalloc(&deviceOutputBuffer1, outputSize1);
  cudaMalloc(&deviceOutputBuffer2, outputSize2);
}

// 从人脸检测网络的bbox中裁剪和处理图像
AgeGenderDetector::DetectionResult AgeGenderDetector::detect(
    const cv::Mat& image) {

  // 标准化和格式转换
  cv::Mat blob = cv::dnn::blobFromImage(
    image,
    1.0 / 255.0,                  // 缩放因子
    cv::Size(224, 224),           // 目标尺寸
    cv::Scalar(0.485, 0.456, 0.406), // 均值 (ImageNet)
    true,                         // 交换通道 BGR -> RGB
    false,                        // 不裁剪
    CV_32F                        // 数据类型
  );

  // 标准化：除以标准差
  std::vector<float> std = { 0.229, 0.224, 0.225 };
  for (int i = 0; i < 3; ++i) {
    cv::Mat channel(blob.size[2], blob.size[3], CV_32F, blob.ptr(0, i));
    channel /= std[i];
  }

  // 将图像拷贝到GPU
  cudaMemcpyAsync(deviceInputBuffer, blob.ptr<float>(),
                  3 * 240 * 240 * sizeof(float), cudaMemcpyHostToDevice,
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

  // std::max_element: 返回指向范围[vec.begin(), vec.end()) 中最大值的迭代器。
  // std::distance: 用于计算从 vec.begin() 到 maxElementIter 的距离，这个距离就是索引。
  auto max_age_it = std::max_element(age_probs.begin(), age_probs.end());
  int max_age_index = std::distance(age_probs.begin(), max_age_it);
  result.age = AGE_GROUPS.at(max_age_index);

  if (gender_probs[0] > gender_probs[1]) {
    result.gender = 0;
  }
  else {
    result.gender = 1;
  }

  return result;
}

AgeGenderDetector::~AgeGenderDetector() {
  // 资源清理
  cudaFree(deviceInputBuffer);
  cudaFree(deviceOutputBuffer1);
  cudaFree(deviceOutputBuffer2);
  cudaStreamDestroy(stream);
}