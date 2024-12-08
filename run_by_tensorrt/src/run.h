#pragma once
#include "face_detector.h"
#include "age_gender.h"

class Run {
private:
  FaceDetector* face_detector_ = nullptr;
  AgeGenderDetector* age_gender_detector = nullptr;

public:
  Run();
  void Get();
  std::vector<std::vector<float>> RunFaceDetector(std::string path);
  void RunAgeGender(std::vector<std::vector<float>> final_boxes, std::string path);
};