#pragma once
#include "screw/types.hpp"
namespace screw {
cv::Mat segment(const cv::Mat& image, const Config& config);
std::vector<std::vector<cv::Point>> candidates(const cv::Mat& mask, const Config& config);
}
