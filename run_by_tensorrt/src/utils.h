#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <iostream>

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity != Severity::kINFO) std::cout << msg << std::endl;
  }
};


struct PriorBoxConfig {
  std::vector<std::vector<int>> min_sizes;
  std::vector<int> steps;
  std::vector<float> variance;
  bool clip;

  PriorBoxConfig(
    std::vector<std::vector<int>> min_sizes,
    std::vector<int> steps,
    std::vector<float> variance,
    bool clip
    ) : min_sizes(min_sizes), steps(steps), 
    variance(variance), clip(clip) {}
};
