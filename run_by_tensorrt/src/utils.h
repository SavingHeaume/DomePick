#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <iostream>


// 自定义删除器
struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      delete obj;
    }
  }
};

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity != Severity::kINFO) std::cout << msg << std::endl;
  }
};
