#include "screw/segmentation.hpp"
#include "screw/config.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace screw {
// 分割输出 CV_8U 掩膜：255 为候选工件，0 为背景。这里只选像素，还没有定位。
cv::Mat segment(const cv::Mat& image, const Config& c) {
    validateConfig(c);
    if (image.empty() || image.type() != CV_8UC3) throw std::runtime_error("Expected nonempty 8-bit BGR image");
    cv::Mat filtered, gray, mask;
    // 平滑用于稳定分割；后续亚像素边缘使用另一条原始灰度图路径，避免过度模糊。
    cv::GaussianBlur(image, filtered, cv::Size(c.blur, c.blur), 0);
    cv::cvtColor(filtered, gray, cv::COLOR_BGR2GRAY);
    int polarity = c.invert ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
    if (c.segmentation == "otsu") {
        // Otsu 从整张图的灰度分布求一个阈值；invert=true 时暗部变成白色前景。
        cv::threshold(gray, mask, 0, 255, polarity | cv::THRESH_OTSU);
    } else if (c.segmentation == "adaptive") {
        // 局部高斯加权均值减 adaptive_c 形成阈值，窗口大小由 adaptive_block 决定。
        cv::adaptiveThreshold(gray, mask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, polarity, c.adaptive_block, c.adaptive_c);
    } else {
        cv::Mat lab; cv::cvtColor(filtered, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> channels; cv::split(lab, channels);
        // 木纹偏黄，部分金属的 b 通道较低；另用 L 保留暗色金属，两条件取并集。
        // 这里的通道值是 OpenCV 的 8 位 Lab 编码；阈值仅针对当前背景和光照。
        mask = (channels[2] <= c.lab_b_max) | (channels[0] <= c.lab_dark_l);
        if (!c.invert) cv::bitwise_not(mask, mask);
    }
    // 开运算去小白点，闭运算补小裂缝；核过大也会削掉细杆或连接邻近目标。
    if (c.morph_open > 1) cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {c.morph_open, c.morph_open}));
    if (c.morph_close > 1) cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {c.morph_close, c.morph_close}));
    return mask;
}
std::vector<std::vector<cv::Point>> candidates(const cv::Mat& mask, const Config& c) {
    cv::Mat labels, stats, centers;
    // labels 存每个像素所属区域的编号；编号 0 是背景，8 邻域也接受对角相连。
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centers, 8);
    cv::Mat kept = cv::Mat::zeros(mask.size(), CV_8U);
    std::vector<unsigned char> accepted(count, 0);
    // 先按实际前景像素数筛掉过小碎片和过大区域，再重建掩膜，减少轮廓处理量。
    for (int i = 1; i < count; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= c.min_area && area <= c.max_area) accepted[i] = 255;
    }
    for (int y = 0; y < labels.rows; ++y) {
        const int* src = labels.ptr<int>(y); auto* dst = kept.ptr<unsigned char>(y);
        for (int x = 0; x < labels.cols; ++x) dst[x] = accepted[src[x]];
    }
    std::vector<std::vector<cv::Point>> contours, result;
    // 只取外轮廓，所以螺母内孔不参与尺寸定义；保留全部轮廓点用于投影与拟合。
    cv::findContours(kept, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for (auto& contour : contours) {
        // contourArea 是几何面积，和前面的白色像素数不同，带孔目标尤其明显。
        double area = cv::contourArea(contour);
        if (contour.size() < 5 || area < c.min_area || area > c.max_area) continue;
        std::vector<cv::Point> hull; cv::convexHull(contour, hull);
        double hull_area = cv::contourArea(hull);
        // solidity=面积/凸包面积：严重破碎或凹陷的区域通常占比较低。
        // 此处仅筛候选，后面的定位与宽度剖面还会检查是否适合可靠测量。
        if (hull_area > 0 && area / hull_area >= c.min_solidity) result.push_back(std::move(contour));
    }
    return result;
}
}
