#include "screw/locator.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>

namespace screw {
namespace {
// 图像 y 向下，atan2 得到的正角度在图上顺时针；归一化为无向主轴角度。
double direction(cv::Vec2d v) { return normalizeAngle(std::atan2(v[1], v[0]) * 180 / CV_PI); }
}
cv::Point2d toLocal(cv::Point2d p, const Object& o) {
    // 先减去原点 O，再分别点乘 e_x=(vx,vy)、e_y=(-vy,vx)，即 R^T(P-O)。
    p -= o.center;
    return {p.x * o.axis[0] + p.y * o.axis[1], -p.x * o.axis[1] + p.y * o.axis[0]};
}
cv::Point2d toImage(cv::Point2d p, const Object& o) {
    // 逆变换 O+s*e_x+t*e_y，把局部 ROI、分界和测量线映射回原图。
    return {o.center.x + p.x * o.axis[0] - p.y * o.axis[1],
            o.center.y + p.x * o.axis[1] + p.y * o.axis[0]};
}
Object locate(const std::vector<cv::Point>& contour, cv::Size size, const Config& c) {
    if (contour.size() < 5) throw std::runtime_error("Contour too small");
    Object o; o.contour = contour; o.box = cv::minAreaRect(contour);
    // 轮廓矩求面积质心：x=m10/m00，y=m01/m00；它不一定等于旋转框中心。
    cv::Moments moments = cv::moments(contour);
    if (moments.m00 <= 0) throw std::runtime_error("Degenerate contour");
    o.center = {moments.m10 / moments.m00, moments.m01 / moments.m00};
    // 将外轮廓填充为实心区域，PCA 使用全部前景像素，避免只由边界采样密度决定方向。
    // 在包围 ROI 内计算即可：整体平移不改变协方差及特征向量，内孔在此按填充处理。
    cv::Rect bounds = cv::boundingRect(contour);
    cv::Mat filled = cv::Mat::zeros(bounds.size(), CV_8U);
    std::vector<cv::Point> shifted;
    for (auto p : contour) shifted.push_back(p - bounds.tl());
    cv::drawContours(filled, std::vector<std::vector<cv::Point>>{shifted}, 0, 255, cv::FILLED);
    std::vector<cv::Point> points; cv::findNonZero(filled, points);
    cv::Mat samples(static_cast<int>(points.size()), 2, CV_64F);
    for (int i = 0; i < samples.rows; ++i) {
        samples.at<double>(i, 0) = points[i].x; samples.at<double>(i, 1) = points[i].y;
    }
    cv::PCA pca(samples, cv::Mat(), cv::PCA::DATA_AS_ROW);
    // 每行是一个二维点；最大特征值的特征向量为主轴，特征值比衡量长轴是否明显。
    // 分母下限只防止除零，不意味着退化形状能提供稳定方向。
    o.eigen_ratio = pca.eigenvalues.at<double>(0) / std::max(1e-9, pca.eigenvalues.at<double>(1));
    o.pca_angle = direction({pca.eigenvectors.at<double>(0, 0), pca.eigenvectors.at<double>(0, 1)});
    cv::Vec4f line; cv::fitLine(contour, line, cv::DIST_L2, 0, 0.01, 0.01);
    // fitLine 返回 (vx,vy,x0,y0)，前两项是方向；这里只保留方向用于方法对比。
    o.fit_angle = direction({line[0], line[1]});
    // RotatedRect.angle 对应 width 轴；height 更长时加 90°，统一取长边方向。
    o.rect_angle = normalizeAngle(o.box.angle + (o.box.size.height > o.box.size.width ? 90 : 0));
    o.angle = c.axis == "pca" ? o.pca_angle : c.axis == "fitline" ? o.fit_angle : o.rect_angle;
    double radians = o.angle * CV_PI / 180;
    o.axis = {std::cos(radians), std::sin(radians)};
    std::vector<cv::Point> hull; cv::convexHull(contour, hull);
    o.solidity = cv::contourArea(contour) / cv::contourArea(hull);
    // 接触图像边缘意味着可能只看到了工件的一部分：可输出可见尺寸，但禁止可靠判定。
    if (bounds.x <= 0 || bounds.y <= 0 || bounds.br().x >= size.width || bounds.br().y >= size.height) {
        o.reliable = false; o.issues.push_back("truncated");
    }
    if (o.eigen_ratio < c.axis_ratio) {
        // 接近圆形的目标没有明确长轴；不能仅因 atan2 得到了角度就相信该方向。
        o.reliable = false; o.issues.push_back("axis_ambiguous");
    }
    // 严重凹陷是疑似粘连的形状启发式，不是通用实例分离算法，仍可能误报或漏报。
    if (o.solidity < 0.55) {
        o.reliable = false; o.issues.push_back("possible_touching");
    }
    return o;
}
}
