#pragma once

#include "screw/types.hpp"
#include <array>

namespace screw {

// 学习用旁路数据，不加入 Object 或正式 JSON；只有传入观察记录时才保存。
struct SegmentationTrace {
    std::array<cv::Mat, 3> lab_channels;
    cv::Mat color_condition;
    cv::Mat dark_condition;
    cv::Mat threshold_mask;
    cv::Mat opened_mask;
    cv::Mat final_mask;
};

struct ComponentTrace {
    int label;
    int pixel_area;
    cv::Rect bounds;
    bool accepted;
    std::string reason;
};

struct ContourTrace {
    std::vector<cv::Point> contour;
    double area;
    std::optional<double> hull_area;
    std::optional<double> solidity;
    bool accepted = false;
    std::string reason;
};

struct CandidateTrace {
    cv::Mat kept_mask;
    std::vector<ComponentTrace> components;
    std::vector<ContourTrace> contours;
};

struct ProfileSlice {
    double s;
    double low;
    double high;
    double width;
};

struct EdgeTrace {
    cv::Point2d point;
    cv::Vec2d normal{1, 0};
    std::string role;
    int search_radius = 0;
    std::vector<double> intensities; // 从 -radius-1 开始；出界时保留已采样部分。
    std::vector<double> gradients;  // 从 -radius 开始。
    int peak = -1;
    std::optional<int> competing_peak;
    std::optional<double> delta;
    std::optional<double> offset;
    std::string reason;
};

struct MeasurementTrace {
    int object_id = 0;
    cv::Point2d center;
    cv::Mat local_mask;
    double local_s_origin = 0;
    double local_t_origin = 0;
    std::vector<ProfileSlice> slices;
    std::vector<double> profile;
    std::optional<double> shaft_baseline;
    std::optional<double> narrow_baseline;
    std::optional<double> negative_end_width;
    std::optional<double> positive_end_width;
    bool head_attempted = false;
    bool negative_end_wide = false;
    bool positive_end_wide = false;
    int head_end = 0;
    std::optional<int> boundary_index;
    std::optional<double> boundary_s;
    int attempted_slices = 0;
    int refined_slices = 0;
    std::vector<EdgeTrace> edges;
};

struct PipelineTrace {
    CandidateTrace candidates;
    std::vector<MeasurementTrace> objects; // 与返回的 Object 同序、同编号。
};

// 只读取旁路记录绘图和保存；不重新分割、定位或测量。
void saveDiagnostics(const cv::Mat& image, const Config& config,
                     const SegmentationTrace& segmentation, const PipelineTrace& pipeline,
                     const std::vector<Object>& objects, const fs::path& output);

} // namespace screw
