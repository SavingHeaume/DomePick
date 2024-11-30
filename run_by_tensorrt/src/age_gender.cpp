#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <unordered_map>

// 年龄组映射
const std::unordered_map<int, std::string> AGE_GROUPS = {
    {0, "(0, 2)"},
    {1, "(4, 6)"},
    {2, "(8, 13)"},
    {3, "(15, 20)"},
    {4, "(25, 32)"},
    {5, "(38, 43)"},
    {6, "(48, 53)"},
    {7, "(60, )"}
};

// 自定义删除器
struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

class AgeGenderDetector {
private:
    // TensorRT组件
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context;

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
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity != Severity::kINFO)
                std::cout << msg << std::endl;
        }
    } logger;

public:
    // 年龄性别识别结果结构体
    struct DetectionResult {
        std::vector<float> gender_probs;  // 性别概率 [female, male]
        std::vector<float> age_probs;     // 年龄组概率
        std::vector<int> predicted_ages;  // 预测的年龄组索引
        std::vector<std::string> age_group_names;  // 年龄组名称
    };

    AgeGenderDetector(const std::string& onnxModelPath) {
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

        // 创建CUDA流
        cudaStreamCreate(&stream);

        // 分配GPU内存
        size_t inputSize = inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
        size_t outputSize1 = outputDims1.d[1] * outputDims1.d[2] * sizeof(float);
        size_t outputSize2 = outputDims2.d[1] * outputDims2.d[2] * sizeof(float);

        cudaMalloc(&deviceInputBuffer, inputSize);
        cudaMalloc(&deviceOutputBuffer1, outputSize1);
        cudaMalloc(&deviceOutputBuffer2, outputSize2);
    }

    // 从人脸检测网络的bbox中裁剪和处理图像
    DetectionResult detect(const cv::Mat& image, const std::vector<float>& bbox_regressions) {
        // 假设bbox_regressions的格式为 [x, y, width, height]
        // 从bbox中裁剪人脸区域
        cv::Rect face_roi(
            bbox_regressions[0], 
            bbox_regressions[1], 
            bbox_regressions[2], 
            bbox_regressions[3]
        );

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
        resizedImage.convertTo(resizedImage, CV_32F, 1.0/255.0);

        // 将图像拷贝到GPU
        cudaMemcpyAsync(deviceInputBuffer, resizedImage.ptr<float>(), 
                        240 * 240 * 3 * sizeof(float), 
                        cudaMemcpyHostToDevice, stream);

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
                        gender_probs.size() * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(age_probs.data(), deviceOutputBuffer2, 
                        age_probs.size() * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);

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

    ~AgeGenderDetector() {
        // 资源清理
        cudaFree(deviceInputBuffer);
        cudaFree(deviceOutputBuffer1);
        cudaFree(deviceOutputBuffer2);
        cudaStreamDestroy(stream);
    }
};

// class FaceDetector {
//     // ... 之前的FaceDetector代码保持不变 ...

//     // 添加一个方法，同时进行人脸检测和年龄性别识别
//     std::vector<std::pair<std::vector<float>, AgeGenderDetector::DetectionResult>> 
//     detectWithAgeGender(const cv::Mat& image, AgeGenderDetector& ageGenderDetector) {
//         // 执行人脸检测
//         auto detection_results = detect(image);

//         std::vector<std::pair<std::vector<float>, AgeGenderDetector::DetectionResult>> combined_results;

//         // 对每个检测到的人脸进行年龄性别识别
//         for (const auto& bbox : detection_results[0]) {
//             // 假设bbox_regressions每4个元素代表一个人脸的边界框
//             std::vector<float> face_bbox = {
//                 bbox, bbox+1, bbox+2, bbox+3
//             };

//             // 进行年龄性别识别
//             auto age_gender_result = ageGenderDetector.detect(image, face_bbox);

//             combined_results.emplace_back(face_bbox, age_gender_result);
//         }

//         return combined_results;
//     }
// };

// int main() {
//     try {
//         // 初始化人脸检测器
//         FaceDetector faceDetector("face_detection.onnx");
        
//         // 初始化年龄性别识别器
//         AgeGenderDetector ageGenderDetector("age_gender_detection.onnx");

//         // 加载测试图像
//         cv::Mat testImage = cv::imread("test_image.jpg");

//         // 同时进行人脸检测和年龄性别识别
//         auto results = faceDetector.detectWithAgeGender(testImage, ageGenderDetector);

//         // 打印结果
//         for (const auto& result : results) {
//             const auto& bbox = result.first;
//             const auto& age_gender = result.second;

//             std::cout << "人脸位置: [x:" << bbox[0] 
//                       << ", y:" << bbox[1] 
//                       << ", width:" << bbox[2] 
//                       << ", height:" << bbox[3] << "]" << std::endl;

//             std::cout << "性别概率 [女性, 男性]: ["
//                       << age_gender.gender_probs[0] << ", " 
//                       << age_gender.gender_probs[1] << "]" << std::endl;

//             std::cout << "年龄组: " << age_gender.age_group_names[0] 
//                       << " (概率: " << age_gender.age_probs[age_gender.predicted_ages[0]] << ")" << std::endl;
//         }
//     }
//     catch (const std::exception& e) {
//         std::cerr << "错误: " << e.what() << std::endl;
//         return -1;
//     }

//     return 0;
// }