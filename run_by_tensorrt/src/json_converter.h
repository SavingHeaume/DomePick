#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <json/json.h>
#include "age_gender.h"


class JsonConverter {
public:
    // 构造函数
    explicit JsonConverter(const std::vector<AgeGenderDetector::DetectionResult>& results)
        : results_(results) {}

    // 转换为 JSON 对象
    Json::Value ToJson() const;

    // 获取 JSON 字符串
    std::string ToJsonString() const;

    // 打印 JSON 字符串
    void PrintJson() const;

private:
    std::vector<AgeGenderDetector::DetectionResult> results_;

    // 将单个 DetectionResult 转换为 JSON
    static Json::Value DetectionResultToJson(const AgeGenderDetector::DetectionResult& result);
};