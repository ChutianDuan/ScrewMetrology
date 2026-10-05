#include "screw/locator.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>

namespace screw {
namespace {
constexpr double kTouchingSolidityThreshold = 0.55;

// 图像 y 向下，正角度在图中顺时针；返回无向主轴角度。
double directionAngle(cv::Vec2d direction) {
    return normalizeAngle(std::atan2(direction[1], direction[0]) * 180 / CV_PI);
}

void estimateAxes(Object& object, const std::vector<cv::Point>& contour,
                  const cv::Rect& bounds, const Config& config) {
    cv::Mat filled = cv::Mat::zeros(bounds.size(), CV_8U);
    std::vector<cv::Point> shifted;
    for (auto point : contour) {
        shifted.push_back(point - bounds.tl());
    }
    // PCA 使用填充区域全部像素，避免仅由边界采样密度决定方向；内孔也按填充处理。
    cv::drawContours(filled, std::vector<std::vector<cv::Point>>{shifted}, 0, 255, cv::FILLED);
    std::vector<cv::Point> points;
    cv::findNonZero(filled, points);
    cv::Mat samples(static_cast<int>(points.size()), 2, CV_64F);
    for (int i = 0; i < samples.rows; ++i) {
        samples.at<double>(i, 0) = points[i].x;
        samples.at<double>(i, 1) = points[i].y;
    }
    cv::PCA pca(samples, cv::Mat(), cv::PCA::DATA_AS_ROW);
    // ROI 平移不改变协方差方向；分母下限只避免除零，不保证退化目标可靠。
    object.eigen_ratio = pca.eigenvalues.at<double>(0)
                         / std::max(1e-9, pca.eigenvalues.at<double>(1));
    object.pca_angle = directionAngle({pca.eigenvectors.at<double>(0, 0),
                                      pca.eigenvectors.at<double>(0, 1)});
    cv::Vec4f line;
    cv::fitLine(contour, line, cv::DIST_L2, 0, 0.01, 0.01);
    object.fit_angle = directionAngle({line[0], line[1]});
    // RotatedRect.angle 对应 width 轴，height 更长时补 90°。
    object.rect_angle = normalizeAngle(object.box.angle
        + (object.box.size.height > object.box.size.width ? 90 : 0));
    if (config.axis == "pca") {
        object.angle = object.pca_angle;
    } else if (config.axis == "fitline") {
        object.angle = object.fit_angle;
    } else {
        object.angle = object.rect_angle;
    }
    double radians = object.angle * CV_PI / 180;
    object.axis = {std::cos(radians), std::sin(radians)};
}

void checkGeometry(Object& object, const cv::Rect& bounds, cv::Size size, const Config& config) {
    // 可输出可见部分尺寸，但碰边时禁止可靠公差判定。
    if (bounds.x <= 0 || bounds.y <= 0 || bounds.br().x >= size.width || bounds.br().y >= size.height) {
        object.reliable = false;
        object.issues.push_back("truncated");
    }
    if (object.eigen_ratio < config.axis_ratio) {
        object.reliable = false;
        object.issues.push_back("axis_ambiguous");
    }
    // 严重凹陷只是疑似粘连的启发式，不是通用实例分离算法。
    if (object.solidity < kTouchingSolidityThreshold) {
        object.reliable = false;
        object.issues.push_back("possible_touching");
    }
}
} // namespace

cv::Point2d toLocal(cv::Point2d point, const Object& object) {
    // 先减原点，再点乘 e_x=(vx,vy)、e_y=(-vy,vx)，即 R^T(P-O)。
    point -= object.center;
    return {point.x * object.axis[0] + point.y * object.axis[1],
            -point.x * object.axis[1] + point.y * object.axis[0]};
}

cv::Point2d toImage(cv::Point2d point, const Object& object) {
    // 逆变换 O+s*e_x+t*e_y，测量线和分界映射回原图。
    return {object.center.x + point.x * object.axis[0] - point.y * object.axis[1],
            object.center.y + point.x * object.axis[1] + point.y * object.axis[0]};
}

Object locate(const std::vector<cv::Point>& contour, cv::Size size, const Config& config) {
    if (contour.size() < 5) {
        throw std::runtime_error("Contour too small");
    }
    Object object;
    object.contour = contour;
    object.box = cv::minAreaRect(contour);
    cv::Moments moments = cv::moments(contour);
    if (moments.m00 <= 0) {
        throw std::runtime_error("Degenerate contour");
    }
    // 面积质心不一定等于旋转框中心。
    object.center = {moments.m10 / moments.m00, moments.m01 / moments.m00};
    cv::Rect bounds = cv::boundingRect(contour);
    estimateAxes(object, contour, bounds, config);
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    object.solidity = cv::contourArea(contour) / cv::contourArea(hull);
    checkGeometry(object, bounds, size, config);
    return object;
}
} // namespace screw
