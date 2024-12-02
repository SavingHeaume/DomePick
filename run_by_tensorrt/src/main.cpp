#include "age_gender.h"
#include "face_detector.h"

int main() {
  try {


    // 初始化年龄性别识别器
    AgeGenderDetector ageGenderDetector("D:\\Projects\\Python\\DomePick\\weights\\age_gender.onnx");

    // 初始化人脸检测器
    FaceDetector faceDetector("D:\\Projects\\Python\\DomePick\\weights\\retinaface.onnx");

    // 加载测试图像
    cv::Mat testImage = cv::imread("D:\\Downloads\\00004.jpg");

    // 同时进行人脸检测和年龄性别识别
    auto results =
        faceDetector.detectWithAgeGender(testImage, ageGenderDetector);

    // 打印结果
    for (const auto& result : results) {
      const auto& bbox = result.first;
      const auto& age_gender = result.second;

      std::cout << "人脸位置: [x:" << bbox[0] << ", y:" << bbox[1]
                << ", width:" << bbox[2] << ", height:" << bbox[3] << "]"
                << std::endl;

      std::cout << "性别概率 [女性, 男性]: [" << age_gender.gender_probs[0]
                << ", " << age_gender.gender_probs[1] << "]" << std::endl;

      std::cout << "年龄组: " << age_gender.age_group_names[0] << " (概率: "
                << age_gender.age_probs[age_gender.predicted_ages[0]] << ")"
                << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}