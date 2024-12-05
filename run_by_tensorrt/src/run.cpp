#include "run.h"
#include <string>

Run::Run() {
  //age_gender_detector = new AgeGenderDetector("D:\\Projects\\Python\\DomePick\\weights\\age_gender.onnx");
  face_detector_ = new FaceDetector("D:\\Projects\\Python\\DomePick\\weights\\retinaface.onnx");
}

void Run::get() {
  //std::string path = "D:\\Downloads\\00004.jpg";
  std::string path = "D:\\Pictures\\600x800.jpg";
  run_face_detector(path);
}

void Run::run_face_detector(std::string path) {
  cv::Mat testImage = cv::imread(path, cv::IMREAD_COLOR);
  auto detection_results = face_detector_->detect(testImage);
  std::vector<float> classifications = detection_results[1];

  for (int i = 1; i < classifications.size(); i += 2) {
    if (classifications[i] > 0.5) {
      std::cout << classifications[i - 1] << " " << classifications[i] << "\n";
    }
  }
}
void Run::run_age_gender() {
   
}