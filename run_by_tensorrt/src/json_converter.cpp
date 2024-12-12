#include "json_converter.h"

Json::Value JsonConverter::ToJson() const {
  Json::Value json_obj;
  Json::Value json_array(Json::arrayValue);

  for (const auto& result : results_) {
    json_array.append(DetectionResultToJson(result));
  }

  json_obj["count"] = static_cast<int>(results_.size());
  json_obj["results"] = json_array;

  return json_obj;
}

// 获取 JSON 字符串
std::string JsonConverter::ToJsonString() const {
  Json::StreamWriterBuilder writer;
  Json::Value json_obj = ToJson();
  return Json::writeString(writer, json_obj);
}

// 打印 JSON 字符串
void JsonConverter::PrintJson() const {
  std::cout << ToJsonString() << std::endl;
}


  // 将单个 DetectionResult 转换为 JSON
Json::Value JsonConverter::DetectionResultToJson(const AgeGenderDetector::DetectionResult& result) {
  Json::Value json_obj;
  json_obj["gender"] = result.gender;
  json_obj["age"] = result.age;
  return json_obj;
}