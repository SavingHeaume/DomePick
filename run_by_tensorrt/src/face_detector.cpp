#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

// 自定义删除器
struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

class FaceDetector {
private:
    // 使用唯一指针管理TensorRT资源
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context;

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
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity != Severity::kINFO)
                std::cout << msg << std::endl;
        }
    } logger;

public:
    FaceDetector(const std::string& onnxModelPath) {
        // 初始化TensorRT运行时
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("创建TensorRT运行时失败");
        }

        // 解析ONNX模型
        auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(
            nvinfer1::createInferBuilder(logger)
        );
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(
            builder->createNetworkV2(0)
        );
        auto parser = std::unique_ptr<nvonnxparser::IParser, InferDeleter>(
            nvonnxparser::createParser(*network, logger)
        );

        // 解析ONNX模型文件
        if (!parser->parseFromFile(onnxModelPath.c_str(), 2)) {
            throw std::runtime_error("解析ONNX模型失败");
        }

        // 配置TensorRT构建
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter>(
            builder->createBuilderConfig()
        );
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
        outputDims3 = network->getOutput(2)->getDimensions();

        // 创建CUDA流
        cudaStreamCreate(&stream);

        // 分配GPU内存
        size_t inputSize = inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
        size_t outputSize1 = outputDims1.d[1] * outputDims1.d[2] * sizeof(float);
        size_t outputSize2 = outputDims2.d[1] * outputDims2.d[2] * sizeof(float);
        size_t outputSize3 = outputDims3.d[1] * outputDims3.d[2] * sizeof(float);

        cudaMalloc(&deviceInputBuffer, inputSize);
        cudaMalloc(&deviceOutputBuffer1, outputSize1);
        cudaMalloc(&deviceOutputBuffer2, outputSize2);
        cudaMalloc(&deviceOutputBuffer3, outputSize3);
    }

    std::vector<std::vector<float>> detect(const cv::Mat& image) {
        // 预处理：调整图像大小并归一化
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(640, 640));
        resizedImage.convertTo(resizedImage, CV_32F, 1.0/255.0);

        // 将图像拷贝到GPU
        cudaMemcpyAsync(deviceInputBuffer, resizedImage.ptr<float>(), 
                        640 * 640 * 3 * sizeof(float), 
                        cudaMemcpyHostToDevice, stream);

        // 绑定输入输出缓存
        void* bindings[] = {
            deviceInputBuffer, 
            deviceOutputBuffer1, 
            deviceOutputBuffer2, 
            deviceOutputBuffer3
        };

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

    ~FaceDetector() {
        // 资源清理
        cudaFree(deviceInputBuffer);
        cudaFree(deviceOutputBuffer1);
        cudaFree(deviceOutputBuffer2);
        cudaFree(deviceOutputBuffer3);
        cudaStreamDestroy(stream);

        // 智能指针会自动调用删除器
    }
};

int main() {
    try {
        // 初始化人脸检测器
        FaceDetector detector("face_detection.onnx");

        // 加载测试图像
        cv::Mat testImage = cv::imread("test_image.jpg");

        // 执行检测
        auto results = detector.detect(testImage);

        // 打印结果（这里仅为示例）
        std::cout << "检测结果：" << std::endl;
        std::cout << "边界框回归数量: " << results[0].size() << std::endl;
        std::cout << "分类结果数量: " << results[1].size() << std::endl;
        std::cout << "关键点数量: " << results[2].size() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}