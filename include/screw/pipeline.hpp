#pragma once

#include "screw/types.hpp"
#include "screw/diagnostics.hpp"

namespace screw {
// 固定候选的对照实验入口：不重新分割或筛选，仍按质心排序编号。
// contours 与原始 gray 使用同一坐标系；尺寸、质量、尺度规则复用正式流程。
std::vector<Object> measureContours(const std::vector<std::vector<cv::Point>>& contours,
                                   const cv::Mat& gray, const Config& config,
                                   const std::optional<Scale>& scale,
                                   const std::optional<Recipe>& recipe, bool single_spec);
// mask 与原始 gray 必须同尺寸、同坐标系；读图和分割由调用方负责。
// 返回完整目标结果，按质心 (y,x) 排序并从 1 编号；异常交给入口处理。
std::vector<Object> measureObjects(const cv::Mat& mask, const cv::Mat& gray,
                                   const Config& config, const std::optional<Scale>& scale,
                                   const std::optional<Recipe>& recipe, bool single_spec);
// 可选旁路记录；原接口继续保留，nullptr 不收集数据。
std::vector<Object> measureObjects(const cv::Mat& mask, const cv::Mat& gray,
                                   const Config& config, const std::optional<Scale>& scale,
                                   const std::optional<Recipe>& recipe, bool single_spec,
                                   PipelineTrace* trace);
} // namespace screw
