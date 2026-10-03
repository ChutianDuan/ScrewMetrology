#include "screw/config.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace screw {
namespace {
template<class T> void read(const cv::FileStorage& file, const char* key, T& value) {
    if (!file[key].empty()) file[key] >> value;
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
}
std::vector<std::string> dimensionNames() {
    return {"length", "width", "head_length", "head_width", "shaft_width"};
}
Dimensions preferredDimensions(const Object& o) {
    // 逐项优选，不能整组替换：整体细化成功时，头长仍可能只有像素值，头宽也可单独失败。
    auto pick=[](std::optional<double> refined, std::optional<double> pixel) {return refined ? refined : pixel;};
    return {pick(o.refined.length,o.pixel.length),pick(o.refined.width,o.pixel.width),
            pick(o.refined.head_length,o.pixel.head_length),pick(o.refined.head_width,o.pixel.head_width),
            pick(o.refined.shaft_width,o.pixel.shaft_width)};
}
std::optional<double> dimension(const Dimensions& d, const std::string& name) {
    if (name == "length") return d.length;
    if (name == "width") return d.width;
    if (name == "head_length") return d.head_length;
    if (name == "head_width") return d.head_width;
    if (name == "shaft_width") return d.shaft_width;
    throw std::runtime_error("Unknown dimension: " + name);
}
double normalizeAngle(double a) {
    // 主轴无头尾方向，θ 和 θ+180° 等价；fmod 对负数保留符号，需要补回正区间。
    a = std::fmod(a, 180.0);
    return a < 0 ? a + 180 : a;
}
double angleError(double a, double b) {
    // 比较无向轴要取绕回后的短角，例如 179° 与 1° 的误差为 2°。
    double delta = std::abs(normalizeAngle(a) - normalizeAngle(b));
    return std::min(delta, 180 - delta);
}
void validateConfig(const Config& c) {
    require(c.segmentation == "otsu" || c.segmentation == "adaptive" || c.segmentation == "lab", "Invalid segmentation");
    require(c.axis == "pca" || c.axis == "fitline" || c.axis == "rect", "Invalid axis method");
    auto odd = [](int n) { return n >= 1 && n <= 501 && n % 2 == 1; };
    require(odd(c.blur) && odd(c.morph_open) && odd(c.morph_close), "Kernel sizes must be odd, 1..501");
    require(odd(c.adaptive_block) && c.adaptive_block >= 3, "adaptive_block must be odd and >=3");
    require(std::isfinite(c.min_area) && std::isfinite(c.max_area) && c.min_area > 0 && c.max_area >= c.min_area, "Invalid area limits");
    require(std::isfinite(c.min_solidity) && c.min_solidity > 0 && c.min_solidity <= 1, "Invalid min_solidity");
    require(std::isfinite(c.axis_ratio) && c.axis_ratio > 1 && std::isfinite(c.head_ratio) && c.head_ratio > 1, "Axis/head ratios must exceed 1");
    require(std::isfinite(c.adaptive_c) && std::isfinite(c.lab_b_max) && c.lab_b_max >= 0 && c.lab_b_max <= 255 && std::isfinite(c.lab_dark_l) && c.lab_dark_l >= 0 && c.lab_dark_l <= 255, "Invalid threshold");
    require(c.edge_search >= 2 && c.edge_search <= 20 && std::isfinite(c.gradient_min) && c.gradient_min > 0, "Invalid subpixel parameters");
}
Config loadConfig(const fs::path& path) {
    Config c;
    if (!path.empty()) {
        cv::FileStorage f(path.string(), cv::FileStorage::READ);
        require(f.isOpened(), "Cannot open config: " + path.string());
        const std::vector<std::string> keys = {"segmentation", "axis", "invert", "subpixel", "blur", "adaptive_block", "adaptive_c", "morph_open", "morph_close", "lab_b_max", "lab_dark_l", "min_area", "max_area", "min_solidity", "axis_ratio", "head_ratio", "gradient_min", "edge_search"};
        for (auto node : f.root()) {
            // 拒绝拼错的参数名，避免配置看似生效而程序实际仍使用默认值。
            require(std::find(keys.begin(), keys.end(), node.name()) != keys.end(), "Unknown config key: " + node.name());
        }
        read(f, "segmentation", c.segmentation); read(f, "axis", c.axis);
        int invert = c.invert, subpixel = c.subpixel;
        read(f, "invert", invert); read(f, "subpixel", subpixel);
        require((invert == 0 || invert == 1) && (subpixel == 0 || subpixel == 1), "Boolean config values must be 0/1");
        c.invert = invert; c.subpixel = subpixel;
        read(f, "blur", c.blur); read(f, "adaptive_block", c.adaptive_block);
        read(f, "adaptive_c", c.adaptive_c); read(f, "morph_open", c.morph_open);
        read(f, "morph_close", c.morph_close); read(f, "lab_b_max", c.lab_b_max);
        read(f, "lab_dark_l", c.lab_dark_l); read(f, "min_area", c.min_area);
        read(f, "max_area", c.max_area); read(f, "min_solidity", c.min_solidity);
        read(f, "axis_ratio", c.axis_ratio); read(f, "head_ratio", c.head_ratio);
        read(f, "gradient_min", c.gradient_min); read(f, "edge_search", c.edge_search);
    }
    validateConfig(c);
    return c;
}
Recipe loadRecipe(const fs::path& path) {
    cv::FileStorage f(path.string(), cv::FileStorage::READ);
    require(f.isOpened(), "Cannot open recipe: " + path.string());
    Recipe r; f["name"] >> r.name; f["unit"] >> r.unit;
    require(!r.name.empty() && (r.unit == "px" || r.unit == "mm"), "Recipe needs name and unit (px/mm)");
    auto limits = f["limits"];
    require(limits.isMap() && !limits.empty(), "Recipe limits must be a nonempty map");
    for (auto node : limits) {
        // 复用尺寸名查找来验证支持项；每项必须是包含边界的 [min,max]，单位来自配方。
        dimension(Dimensions{}, node.name());
        require(node.isSeq() && node.size() == 2, "Each limit must be [min, max]");
        double low = static_cast<double>(node[0]), high = static_cast<double>(node[1]);
        require(std::isfinite(low) && std::isfinite(high) && low >= 0 && high >= low, "Invalid tolerance range");
        r.limits[node.name()] = {low, high};
    }
    return r;
}
} // namespace screw
