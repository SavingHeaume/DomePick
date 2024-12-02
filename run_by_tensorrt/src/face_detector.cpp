#include "face_detector.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

FaceDetector::FaceDetector(const std::string& onnxModelPath) {
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
  profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 640, 640));
  profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 640, 640));
  profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 640, 640));

  //if (!config->addOptimizationProfile(profile)) {
  //  throw std::runtime_error("Failed to add optimization profile");
  //}
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
  outputDims3 = network->getOutput(2)->getDimensions();

  // 创建CUDA流
  cudaStreamCreate(&stream);

  // 分配GPU内存
  size_t inputSize =
      inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
  size_t outputSize1 = outputDims1.d[1] * outputDims1.d[2] * sizeof(float);
  size_t outputSize2 = outputDims2.d[1] * outputDims2.d[2] * sizeof(float);
  size_t outputSize3 = outputDims3.d[1] * outputDims3.d[2] * sizeof(float);

  cudaMalloc(&deviceInputBuffer, inputSize);
  cudaMalloc(&deviceOutputBuffer1, outputSize1);
  cudaMalloc(&deviceOutputBuffer2, outputSize2);
  cudaMalloc(&deviceOutputBuffer3, outputSize3);
}

std::vector<std::vector<float>> FaceDetector::detect(const cv::Mat& image) {
  // 预处理：调整图像大小并归一化
  cv::Mat resizedImage;
  cv::resize(image, resizedImage, cv::Size(640, 640));
  resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

  // 将图像拷贝到GPU
  cudaMemcpyAsync(deviceInputBuffer, resizedImage.ptr<float>(),
                  640 * 640 * 3 * sizeof(float), cudaMemcpyHostToDevice,
                  stream);

  // 绑定输入输出缓存
  void* bindings[] = {deviceInputBuffer, deviceOutputBuffer1,
                      deviceOutputBuffer2, deviceOutputBuffer3};

  // 执行推理 (注意：只传入bindings数组)
  context->executeV2(bindings);

  // 分配主机内存存储结果
  std::vector<float> bbox_regressions(outputDims1.d[1] * outputDims1.d[2]);
  std::vector<float> classifications(outputDims2.d[1] * outputDims2.d[2]);
  std::vector<float> ldm_regressions(outputDims3.d[1] * outputDims3.d[2]);

  // 从GPU拷贝结果
  cudaMemcpyAsync(bbox_regressions.data(), deviceOutputBuffer1,
                  bbox_regressions.size() * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(classifications.data(), deviceOutputBuffer2,
                  classifications.size() * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(ldm_regressions.data(), deviceOutputBuffer3,
                  ldm_regressions.size() * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);

  // 同步等待
  cudaStreamSynchronize(stream);

  // 后处理：这里仅为示例，实际需要根据具体模型实现非极大值抑制等
  return {bbox_regressions, classifications, ldm_regressions};
}

FaceDetector::~FaceDetector() {
  // 资源清理
  cudaFree(deviceInputBuffer);
  cudaFree(deviceOutputBuffer1);
  cudaFree(deviceOutputBuffer2);
  cudaFree(deviceOutputBuffer3);
  cudaStreamDestroy(stream);

  // 智能指针会自动调用删除器
}

std::vector<std::pair<std::vector<float>, AgeGenderDetector::DetectionResult>>
FaceDetector::detectWithAgeGender(const cv::Mat& image,
                                  AgeGenderDetector& ageGenderDetector) {
  // 执行人脸检测
  auto detection_results = detect(image);

  std::vector<std::pair<std::vector<float>, AgeGenderDetector::DetectionResult>>
      combined_results;

  // 对每个检测到的人脸进行年龄性别识别
  //for (const auto& bbox : detection_results[0]) {
  int size = detection_results[1].size() / 2;
  for (int i = 0; i < detection_results[0].size(); i += 4) {
    // bbox_regressions每4个元素代表一个人脸的边界框
    float bbox = detection_results[0][i];
    std::vector<float> face_bbox = {bbox, bbox + 1, bbox + 2, bbox + 3};

    // 进行年龄性别识别
    auto age_gender_result = ageGenderDetector.detect(image, face_bbox);

    combined_results.emplace_back(face_bbox, age_gender_result);
  }

  return combined_results;
}