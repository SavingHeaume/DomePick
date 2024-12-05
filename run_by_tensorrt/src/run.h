#pragma once
#include "face_detector.h"
#include "age_gender.h"

class Run {
private:
  FaceDetector* face_detector_ = nullptr;
  AgeGenderDetector* age_gender_detector = nullptr;

public:
  Run();
  void get();
  void run_face_detector(std::string path);
  void run_age_gender();
};