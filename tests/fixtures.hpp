#pragma once

#include <opencv2/imgproc.hpp>
#include <cmath>

inline cv::Mat screwImage(double angle = 0, double length = 200, double head_width = 40,
                          double shaft_width = 20, double contrast = 180, double sigma = 0,
                          cv::Point2d center = {320, 240}) {
    constexpr int factor = 4;
    cv::Mat high_resolution(480 * factor, 640 * factor, CV_8UC3, cv::Scalar::all(225));
    const double theta = angle * CV_PI / 180;
    const double head_end = -length / 2 + 0.15 * length;
    const std::vector<cv::Point2d> local{
        {-length / 2, -head_width / 2}, {head_end, -head_width / 2},
        {head_end, -shaft_width / 2}, {length / 2, -shaft_width / 2},
        {length / 2, shaft_width / 2}, {head_end, shaft_width / 2},
        {head_end, head_width / 2}, {-length / 2, head_width / 2}};
    std::vector<cv::Point> polygon;
    for (const auto& point : local) {
        polygon.emplace_back(
            cvRound(factor * (center.x + point.x * std::cos(theta) - point.y * std::sin(theta))),
            cvRound(factor * (center.y + point.x * std::sin(theta) + point.y * std::cos(theta))));
    }
    cv::fillPoly(high_resolution, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar::all(225 - contrast));
    cv::Mat image;
    cv::resize(high_resolution, image, {640, 480}, 0, 0, cv::INTER_AREA);
    if (sigma > 0) {
        cv::GaussianBlur(image, image, {0, 0}, sigma);
    }
    return image;
}
