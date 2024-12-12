#include "run.h"
#include <string>
#include "utils.h"
#include <algorithm>
#include "decode_face.h"
#include <json/json.h>
#include "json_converter.h"
#include "httplib.h"

Run::Run() {
  age_gender_detector_ = new AgeGenderDetector("D:\\Projects\\Python\\DomePick\\weights\\age_gender.onnx");
  face_detector_ = new FaceDetector("D:\\Projects\\Python\\DomePick\\weights\\retinaface.onnx");
}

void Run::Get() {
  //std::string path = "D:\\Downloads\\00004.jpg";
  std::string path = "D:\\Pictures\\600x800.jpg";
  std::vector<std::vector<float>> final_boxes = RunFaceDetector(path);

  auto final_result = RunAgeGender(final_boxes, path);

  JsonConverter json_converter(final_result);
  http_send(json_converter);
}

std::vector<AgeGenderDetector::DetectionResult> Run::RunAgeGender(std::vector<std::vector<float>> final_boxes, std::string path) {
  cv::Mat testImage = cv::imread(path);
  std::vector<AgeGenderDetector::DetectionResult> final_result;
  
  for (const auto& box : final_boxes) {
    if (box.size() < 4) continue; 

    // 获取边界框的归一化坐标
    float x_min = box[0];
    float y_min = box[1];
    float x_max = box[2];
    float y_max = box[3];

    // 转换为像素坐标
    int width = testImage.cols;
    int height = testImage.rows;
    int x1 = std::max(0, static_cast<int>(x_min * width));
    int y1 = std::max(0, static_cast<int>(y_min * height));
    int x2 = std::min(width, static_cast<int>(x_max * width));
    int y2 = std::min(height, static_cast<int>(y_max * height));

    // 定义 ROI 区域并裁剪图像
    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    if (roi.width > 0 && roi.height > 0) { // 确保 ROI 有效
      cv::Mat face_img = testImage(roi).clone();

      final_result.emplace_back(age_gender_detector_->detect(face_img));
    }
  }
  return final_result;
}

std::vector<std::vector<float>> Run::RunFaceDetector(std::string path) {
  cv::Mat testImage = cv::imread(path, cv::IMREAD_COLOR);

  auto detection_results = face_detector_->detect(testImage);
  DecodeFace decode_face(detection_results[0], detection_results[1]);
  return decode_face.GetFinalBoxes();
}

void Run::http_send(JsonConverter json_converter) {

  const std::string json_string = json_converter.ToJsonString();

  // 3. 创建HTTP客户端
  httplib::Client cli("http://127.0.0.1:5000/stream");

  // 4. 设置HTTP头
  httplib::Headers headers;
  headers.emplace("Content-Type", "application/json");

  // 5. 发送POST请求
  auto res = cli.Post("/api/endpoint", headers, json_string, "application/json");

  // 6. 处理响应
  if (res && res->status == 200) {
    std::cout << "Response: " << res->body << std::endl;
  }
  else {
    std::cout << "Failed to send request." << std::endl;
  }
}