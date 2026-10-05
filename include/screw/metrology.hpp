#pragma once

#include "screw/types.hpp"
#include "screw/diagnostics.hpp"

namespace screw {
// gray 是定位所用原图对应的 CV_8U 灰度图，与轮廓使用同一图像坐标系。
// 原位填入像素/细化尺寸和质量原因，不用旋转后的掩膜作为灰度采样图。
void measureObject(Object& object, const cv::Mat& gray, const Config& config);
// 从粗边缘沿单位 normal 采样；成功返回有符号像素偏移，失败返回 nullopt。
std::optional<double> refineEdge(const cv::Mat& gray, cv::Point2d point,
                                cv::Vec2d normal, const Config& config);
// 可选旁路记录；原接口继续保留，nullptr 不收集数据。
void measureObject(Object& object, const cv::Mat& gray, const Config& config,
                   MeasurementTrace* trace);
std::optional<double> refineEdge(const cv::Mat& gray, cv::Point2d point,
                                cv::Vec2d normal, const Config& config, EdgeTrace* trace);
} // namespace screw
