#include "decode_face.h"
#include <vector>
#include <numeric>
#include "utils.h"

DecodeFace::DecodeFace(std::vector<float> bbox_regressions, std::vector<float> classifications) 
  : bbox_regressions_(std::move(bbox_regressions)), classifications_(std::move(classifications)) {
  cfg_ = new PriorBoxConfig(
    { {16, 32}, {64, 128}, {256, 512} },
    { 8, 16, 32 },
    { 0.1f, 0.2f },
    false);
}

std::vector<std::vector<float>> DecodeFace::GetFinalBoxes() {
  std::vector<std::vector<float>> priors = GeneratePriors(cv::Size(640, 640));

  if (bbox_regressions_.size() / 4 != classifications_.size() / 2 ||
    bbox_regressions_.size() / 4 != priors.size()) {
    throw std::runtime_error("Mismatch between bbox_regressions, classifications, and priors sizes!");
  }

  // Decode bounding boxes
  std::vector<std::vector<float>> boxes = DecodeBboxes(priors, cfg_->variance);

  // Extract face probabilities
  std::vector<float> face_scores;
  for (size_t i = 0; i < classifications_.size(); i += 2) {
    face_scores.push_back(classifications_[i + 1]); // Assuming second value is face probability
  }

  filter_by_confidence(boxes, face_scores, 0.8);
  keep_top_k(boxes, face_scores, 5000);

  // Append scores to boxes
  for (size_t i = 0; i < boxes.size(); ++i) {
    boxes[i].push_back(face_scores[i]);
  }

  // Perform NMS
  std::vector<int> keep = Nms(boxes, 0.4);
  std::vector<std::vector<float>> final_boxes;
  for (int idx : keep) {
    final_boxes.push_back(boxes[idx]);
  }

  // Limit final output to top 750 boxes
  keep_top_k(final_boxes, face_scores, 750);

  return final_boxes;
}



std::vector<std::vector<float>> DecodeFace::GeneratePriors(const cv::Size& image_size) {
  std::vector<std::vector<float>> priors;
  for (size_t k = 0; k < cfg_->steps.size(); ++k) {
    int feature_map_h = std::ceil(static_cast<float>(image_size.height) / cfg_->steps[k]);
    int feature_map_w = std::ceil(static_cast<float>(image_size.width) / cfg_->steps[k]);

    for (int i = 0; i < feature_map_h; ++i) {
      for (int j = 0; j < feature_map_w; ++j) {
        for (const auto& min_size : cfg_->min_sizes[k]) {
          float cx = (j + 0.5f) * cfg_->steps[k] / image_size.width;
          float cy = (i + 0.5f) * cfg_->steps[k] / image_size.height;
          float s_kx = min_size / static_cast<float>(image_size.width);
          float s_ky = min_size / static_cast<float>(image_size.height);

          priors.push_back({ cx, cy, s_kx, s_ky });
        }
      }
    }
  }

  if (cfg_->clip) {
    for (auto& prior : priors) {
      for (float& val : prior) {
        val = std::max(0.0f, std::min(1.0f, val));
      }
    }
  }

  return priors;
}

std::vector<std::vector<float>> DecodeFace::DecodeBboxes(std::vector<std::vector<float>>& priors, const std::vector<float>& variances) {
  std::vector<std::vector<float>> boxes;
  for (size_t i = 0; i < priors.size(); ++i) {
    float prior_cx = priors[i][0];
    float prior_cy = priors[i][1];
    float prior_w = priors[i][2];
    float prior_h = priors[i][3];

    float pred_cx = variances[0] * bbox_regressions_[i * 4] * prior_w + prior_cx;
    float pred_cy = variances[0] * bbox_regressions_[i * 4 + 1] * prior_h + prior_cy;
    float pred_w = std::exp(variances[1] * bbox_regressions_[i * 4 + 2]) * prior_w;
    float pred_h = std::exp(variances[1] * bbox_regressions_[i * 4 + 3]) * prior_h;

    float x_min = pred_cx - pred_w / 2.0f;
    float y_min = pred_cy - pred_h / 2.0f;
    float x_max = pred_cx + pred_w / 2.0f;
    float y_max = pred_cy + pred_h / 2.0f;

    // Clip to image boundaries
    x_min = std::max(0.0f, std::min(1.0f, x_min));
    y_min = std::max(0.0f, std::min(1.0f, y_min));
    x_max = std::max(0.0f, std::min(1.0f, x_max));
    y_max = std::max(0.0f, std::min(1.0f, y_max));

    boxes.push_back({ x_min, y_min, x_max, y_max });
  }
  return boxes;
}

void filter_by_confidence(std::vector<std::vector<float>>& boxes, std::vector<float>& scores, float confidence_threshold) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] <= confidence_threshold) {
      scores.erase(scores.begin() + i);
      boxes.erase(boxes.begin() + i);
      --i; // Adjust index after removal
    }
  }
}

std::vector<int> DecodeFace::Nms(const std::vector<std::vector<float>>& dets, float nms_threshold) {
  std::vector<int> order(dets.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&dets](int a, int b) { return dets[a][4] > dets[b][4]; });

  std::vector<int> keep;
  while (!order.empty()) {
    int idx = order[0];
    keep.push_back(idx);
    order.erase(order.begin());

    std::vector<int> new_order;
    for (size_t i = 0; i < order.size(); ++i) {
      int other_idx = order[i];
      float xx1 = std::max(dets[idx][0], dets[other_idx][0]);
      float yy1 = std::max(dets[idx][1], dets[other_idx][1]);
      float xx2 = std::min(dets[idx][2], dets[other_idx][2]);
      float yy2 = std::min(dets[idx][3], dets[other_idx][3]);

      float w = std::max(0.0f, xx2 - xx1);
      float h = std::max(0.0f, yy2 - yy1);
      float inter = w * h;
      float area_a = (dets[idx][2] - dets[idx][0]) * (dets[idx][3] - dets[idx][1]);
      float area_b = (dets[other_idx][2] - dets[other_idx][0]) * (dets[other_idx][3] - dets[other_idx][1]);
      float ovr = inter / (area_a + area_b - inter);

      if (ovr <= nms_threshold) {
        new_order.push_back(other_idx);
      }
    }
    order = new_order;
  }

  return keep;
}

void keep_top_k(std::vector<std::vector<float>>& boxes, std::vector<float>& scores, int top_k) {
  if (top_k >= 0 && static_cast<size_t>(top_k) < scores.size()) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::nth_element(indices.begin(), indices.begin() + top_k, indices.end(), [&](int a, int b) {
      return scores[a] > scores[b];
      });
    indices.resize(top_k);

    std::vector<std::vector<float>> top_boxes;
    std::vector<float> top_scores;
    for (int idx : indices) {
      top_boxes.push_back(boxes[idx]);
      top_scores.push_back(scores[idx]);
    }
    boxes = top_boxes;
    scores = top_scores;
  }
}