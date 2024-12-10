#include "run.h"
#include <string>
#include "utils.h"
#include <algorithm>
#include "decode_face.h"

Run::Run() {
  //age_gender_detector = new AgeGenderDetector("D:\\Projects\\Python\\DomePick\\weights\\age_gender.onnx");
  face_detector_ = new FaceDetector("D:\\Projects\\Python\\DomePick\\weights\\retinaface.onnx");
}

void Run::Get() {
  //std::string path = "D:\\Downloads\\00004.jpg";
  std::string path = "D:\\Pictures\\600x800.jpg";
  std::vector<std::vector<float>> final_boxes = RunFaceDetector(path);
  // x_min, y_min, x_max, y_max ¹éÒ»»¯×ø±ê(0~1)

  //std::cout << final_boxes[0][0] << " " << final_boxes[0][1] << " " << final_boxes[0][2] << " " << final_boxes[0][3] << "\n";
}

void Run::RunAgeGender(std::vector<std::vector<float>> final_boxes, std::string path) {

}

std::vector<std::vector<float>> Run::RunFaceDetector(std::string path) {
  cv::Mat testImage = cv::imread(path, cv::IMREAD_COLOR);

  auto detection_results = face_detector_->detect(testImage);
  DecodeFace decode_face(detection_results[0], detection_results[1]);
  return decode_face.GetFinalBoxes();
}