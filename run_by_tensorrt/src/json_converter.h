#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <json/json.h>
#include "age_gender.h"


class JsonConverter {
public:
    // ���캯��
    explicit JsonConverter(const std::vector<AgeGenderDetector::DetectionResult>& results)
        : results_(results) {}

    // ת��Ϊ JSON ����
    Json::Value ToJson() const;

    // ��ȡ JSON �ַ���
    std::string ToJsonString() const;

    // ��ӡ JSON �ַ���
    void PrintJson() const;

private:
    std::vector<AgeGenderDetector::DetectionResult> results_;

    // ������ DetectionResult ת��Ϊ JSON
    static Json::Value DetectionResultToJson(const AgeGenderDetector::DetectionResult& result);
};