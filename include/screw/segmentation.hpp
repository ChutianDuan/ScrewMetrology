#pragma once

#include "screw/types.hpp"
#include "screw/diagnostics.hpp"

namespace screw {
cv::Mat segment(const cv::Mat& image, const Config& config);
std::vector<std::vector<cv::Point>> candidates(const cv::Mat& mask, const Config& config);
// 可选旁路记录；原接口继续保留，nullptr 不收集数据。
cv::Mat segment(const cv::Mat& image, const Config& config, SegmentationTrace* trace);
std::vector<std::vector<cv::Point>> candidates(const cv::Mat& mask, const Config& config,
                                              CandidateTrace* trace);
} // namespace screw
