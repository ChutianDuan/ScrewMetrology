#pragma once
#include <opencv2/core.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace screw {
namespace fs = std::filesystem;
struct Config {
    // 算法参数：核尺寸、面积和搜索范围以像素为尺度；Lab 阈值作用于 8 位通道。
    std::string segmentation = "lab", axis = "pca";
    bool invert = true, subpixel = true;
    int blur = 3, adaptive_block = 51, morph_open = 3, morph_close = 11;
    double adaptive_c = 8, lab_b_max = 145, lab_dark_l = 90;
    double min_area = 1000, max_area = 200000, min_solidity = 0.25;
    double axis_ratio = 1.3, head_ratio = 1.25, gradient_min = 5;
    int edge_search = 4;
};
struct Dimensions {
    // optional 区分“尺寸不可测”和数值 0；导出时不可测字段写为 JSON null。
    // width 是整体横向投影范围，shaft_width 是接受杆段的截面宽度中位数。
    std::optional<double> length, width, head_length, head_width, shaft_width;
};
struct Object {
    // 图像内按质心位置重新编号，id 不具有跨图像追踪含义。
    int id = 0;
    // center 为外轮廓面积质心，也是局部坐标原点；box.center 为旋转框中心。
    cv::Point2d center;
    cv::RotatedRect box;
    cv::Vec2d axis{1, 0};
    // axis 是单位主轴 e_x；所有这里记录的角度均为度，主轴按 180° 周期表示。
    double angle = 0, rect_angle = 0, fit_angle = 0, pca_angle = 0;
    double eigen_ratio = 0, solidity = 0;
    // 能得到数值不等于数值可靠；reliable=false 时禁止给出可靠公差判定。
    bool reliable = true;
    std::vector<cv::Point> contour;
    std::vector<std::string> issues;
    // pixel/refined 分别保留像素/亚像素结果；mm 使用逐项优选值乘适用尺度。
    Dimensions pixel, refined, mm;
    std::string subpixel_status = "disabled", status = "NOT_EVALUATED";
    std::vector<std::string> reasons;
    cv::Point2d length_start, length_end;
    // 头杆分界在局部主轴上的 s 坐标，可为负；不是图像 x 坐标或数组下标。
    std::optional<double> head_boundary;
};
struct Scale {
    // 同分辨率只是必要条件；固定拍摄设置、同平面与近似正视仍须使用者保证。
    std::string reference;
    cv::Size image_size;
    cv::Point2d p1, p2;
    double known_mm = 0, mm_per_px = 0;
};
struct Recipe {
    std::string name, unit;
    // measurement -> [下限, 上限]，包含边界；同一配方内使用统一的 px 或 mm。
    std::map<std::string, std::pair<double, double>> limits;
};
struct ImageResult {
    std::string image, error;
    cv::Size size;
    Config config;
    std::optional<Scale> scale;
    std::string recipe, mode = "mixed";
    std::optional<Recipe> recipe_config;
    std::vector<Object> objects;
};
double normalizeAngle(double degrees);
double angleError(double a, double b);
std::optional<double> dimension(const Dimensions& dims, const std::string& name);
std::vector<std::string> dimensionNames();
Dimensions preferredDimensions(const Object& object);
} // namespace screw
