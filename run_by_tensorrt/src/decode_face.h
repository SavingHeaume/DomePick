#pragma once

#include "utils.h"
#include <vector>
#include <opencv2/opencv.hpp>

class DecodeFace {

private:
  PriorBoxConfig *cfg_;
  std::vector<float> bbox_regressions_;
  std::vector<float> classifications_;


public:
  DecodeFace(std::vector<float> bbox_regressions, std::vector<float> classifications);
  std::vector<std::vector<float>> GetFinalBoxes();
  std::vector<std::vector<float>> GeneratePriors(const cv::Size& image_size);
  std::vector<std::vector<float>> DecodeBboxes(std::vector<std::vector<float>>& priors, const std::vector<float>&variances);
  std::vector<int> Nms(const std::vector<std::vector<float>>& dets, float nms_threshold);
};

void keep_top_k(std::vector<std::vector<float>>& boxes, std::vector<float>& scores, int top_k);
void filter_by_confidence(std::vector<std::vector<float>>& boxes, std::vector<float>& scores, float confidence_threshold);
