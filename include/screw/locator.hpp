#pragma once
#include "screw/types.hpp"
namespace screw {
// 输入外轮廓与整张图尺寸，求图像质心、三种主轴方向及几何质量；不在这里测长度。
Object locate(const std::vector<cv::Point>& contour, cv::Size size, const Config& config);
// 图像坐标 (x,y) <-> 工件坐标 (s,t)，两边都是像素单位，Object 提供原点与单位主轴。
cv::Point2d toLocal(cv::Point2d point, const Object& object);
cv::Point2d toImage(cv::Point2d point, const Object& object);
}
