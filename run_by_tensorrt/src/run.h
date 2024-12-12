#pragma once
#include "face_detector.h"
#include "age_gender.h"

class Run {
private:
  FaceDetector* face_detector_ = nullptr;
  AgeGenderDetector* age_gender_detector_ = nullptr;

public:
  Run();
  void Get();
  std::vector<std::vector<float>> RunFaceDetector(std::string path);
  std::vector<AgeGenderDetector::DetectionResult> RunAgeGender(std::vector<std::vector<float>> final_boxes, std::string path);
};