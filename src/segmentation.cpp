#include "screw/segmentation.hpp"
#include "screw/config.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>

namespace screw {
// 输出 CV_8U：255 为候选工件，0 为背景；此处还没有定位。
cv::Mat segment(const cv::Mat& image, const Config& config) {
    return segment(image, config, nullptr);
}

cv::Mat segment(const cv::Mat& image, const Config& config, SegmentationTrace* trace) {
    validateConfig(config);
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::runtime_error("Expected nonempty 8-bit BGR image");
    }
    if (trace) {
        *trace = {};
    }
    cv::Mat filtered;
    cv::Mat gray;
    cv::Mat mask;
    // 平滑只稳定分割；后续细化使用原始灰度，避免过度模糊亮度边缘。
    cv::GaussianBlur(image, filtered, cv::Size(config.blur, config.blur), 0);
    cv::cvtColor(filtered, gray, cv::COLOR_BGR2GRAY);
    int polarity = config.invert ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
    if (config.segmentation == "otsu") {
        cv::threshold(gray, mask, 0, 255, polarity | cv::THRESH_OTSU);
    } else if (config.segmentation == "adaptive") {
        // 阈值为局部高斯加权均值减 adaptive_c。
        cv::adaptiveThreshold(gray, mask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              polarity, config.adaptive_block, config.adaptive_c);
    } else {
        cv::Mat lab;
        cv::cvtColor(filtered, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> channels;
        cv::split(lab, channels);
        // 木纹偏黄，部分金属 b 较低；L 条件补暗金属。阈值作用于 8 位 Lab 编码。
        const cv::Mat color_condition = channels[2] <= config.lab_b_max;
        const cv::Mat dark_condition = channels[0] <= config.lab_dark_l;
        mask = color_condition | dark_condition;
        if (trace) {
            for (int i = 0; i < 3; ++i) {
                trace->lab_channels[i] = channels[i].clone();
            }
            trace->color_condition = color_condition.clone();
            trace->dark_condition = dark_condition.clone();
        }
        if (!config.invert) {
            cv::bitwise_not(mask, mask);
        }
    }
    if (trace) {
        trace->threshold_mask = mask.clone();
    }
    // 开运算去小白点，闭运算补裂缝；大核会削掉细杆或连接邻近工件。
    if (config.morph_open > 1) {
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, {config.morph_open, config.morph_open}));
    }
    if (trace) {
        trace->opened_mask = mask.clone();
    }
    if (config.morph_close > 1) {
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, {config.morph_close, config.morph_close}));
    }
    if (trace) {
        trace->final_mask = mask.clone();
    }
    return mask;
}

std::vector<std::vector<cv::Point>> candidates(const cv::Mat& mask, const Config& config) {
    return candidates(mask, config, nullptr);
}

std::vector<std::vector<cv::Point>> candidates(const cv::Mat& mask, const Config& config,
                                              CandidateTrace* trace) {
    if (trace) {
        *trace = {};
    }
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centers;
    // 标签 0 是背景；8 邻域也接受对角相连。
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centers, 8);
    cv::Mat kept = cv::Mat::zeros(mask.size(), CV_8U);
    std::vector<unsigned char> accepted(count, 0);
    for (int i = 1; i < count; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= config.min_area && area <= config.max_area) {
            accepted[i] = 255;
        }
        if (trace) {
            const cv::Rect bounds(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                                  stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
            std::string reason = "accepted";
            if (area < config.min_area) {
                reason = "pixel_area_below_min";
            } else if (area > config.max_area) {
                reason = "pixel_area_above_max";
            }
            trace->components.push_back({i, area, bounds, accepted[i] != 0, reason});
        }
    }
    // 先按前景像素数筛选，再重建掩膜，减少轮廓处理量。
    for (int y = 0; y < labels.rows; ++y) {
        const int* source = labels.ptr<int>(y);
        auto* destination = kept.ptr<unsigned char>(y);
        for (int x = 0; x < labels.cols; ++x) {
            destination[x] = accepted[source[x]];
        }
    }
    if (trace) {
        trace->kept_mask = kept.clone();
    }
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> result;
    // 只取外轮廓，保留全部边界点；内孔不参与外形尺寸定义。
    cv::findContours(kept, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for (auto& contour : contours) {
        double area = cv::contourArea(contour);
        ContourTrace* observation = nullptr;
        if (trace) {
            trace->contours.push_back({contour, area, std::nullopt, std::nullopt, false, ""});
            observation = &trace->contours.back();
        }
        // 几何面积与前景像素数不同，带孔目标尤其明显。
        if (contour.size() < 5 || area < config.min_area || area > config.max_area) {
            if (observation) {
                if (contour.size() < 5) {
                    observation->reason = "too_few_contour_points";
                } else if (area < config.min_area) {
                    observation->reason = "contour_area_below_min";
                } else {
                    observation->reason = "contour_area_above_max";
                }
            }
            continue;
        }
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hull_area = cv::contourArea(hull);
        if (observation) {
            observation->hull_area = hull_area;
            if (hull_area > 0) {
                observation->solidity = area / hull_area;
            }
            observation->reason = hull_area > 0 ? "solidity_below_min" : "degenerate_hull";
        }
        if (hull_area > 0 && area / hull_area >= config.min_solidity) {
            if (observation) {
                observation->accepted = true;
                observation->reason = "accepted";
            }
            result.push_back(std::move(contour));
        }
    }
    return result;
}
} // namespace screw
