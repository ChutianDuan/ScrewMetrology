#include "screw/config.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace screw {
namespace {
template<class T>
void read(const cv::FileStorage& file, const char* key, T& value) {
    if (!file[key].empty()) {
        file[key] >> value;
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
} // namespace

std::vector<std::string> dimensionNames() {
    return {"length", "width", "head_length", "head_width", "shaft_width"};
}

Dimensions preferredDimensions(const Object& object) {
    // 逐项优选：整体细化成功时，头长仍可能只有像素值，头宽也可单独失败。
    auto pick = [](std::optional<double> refined, std::optional<double> pixel) {
        return refined ? refined : pixel;
    };
    return {pick(object.refined.length, object.pixel.length),
            pick(object.refined.width, object.pixel.width),
            pick(object.refined.head_length, object.pixel.head_length),
            pick(object.refined.head_width, object.pixel.head_width),
            pick(object.refined.shaft_width, object.pixel.shaft_width)};
}

std::optional<double> dimension(const Dimensions& dimensions, const std::string& name) {
    if (name == "length") {
        return dimensions.length;
    }
    if (name == "width") {
        return dimensions.width;
    }
    if (name == "head_length") {
        return dimensions.head_length;
    }
    if (name == "head_width") {
        return dimensions.head_width;
    }
    if (name == "shaft_width") {
        return dimensions.shaft_width;
    }
    throw std::runtime_error("Unknown dimension: " + name);
}

double normalizeAngle(double degrees) {
    // 主轴无头尾方向，θ 与 θ+180° 等价；fmod 对负数保留符号。
    degrees = std::fmod(degrees, 180.0);
    return degrees < 0 ? degrees + 180 : degrees;
}

double angleError(double first, double second) {
    double delta = std::abs(normalizeAngle(first) - normalizeAngle(second));
    return std::min(delta, 180 - delta);
}

void validateConfig(const Config& config) {
    require(config.segmentation == "otsu" || config.segmentation == "adaptive"
            || config.segmentation == "lab", "Invalid segmentation");
    require(config.axis == "pca" || config.axis == "fitline" || config.axis == "rect",
            "Invalid axis method");
    auto odd = [](int size) {
        return size >= 1 && size <= 501 && size % 2 == 1;
    };
    require(odd(config.blur) && odd(config.morph_open) && odd(config.morph_close),
            "Kernel sizes must be odd, 1..501");
    require(odd(config.adaptive_block) && config.adaptive_block >= 3,
            "adaptive_block must be odd and >=3");
    require(std::isfinite(config.min_area) && std::isfinite(config.max_area)
            && config.min_area > 0 && config.max_area >= config.min_area, "Invalid area limits");
    require(std::isfinite(config.min_solidity) && config.min_solidity > 0
            && config.min_solidity <= 1, "Invalid min_solidity");
    require(std::isfinite(config.axis_ratio) && config.axis_ratio > 1
            && std::isfinite(config.head_ratio) && config.head_ratio > 1,
            "Axis/head ratios must exceed 1");
    require(std::isfinite(config.adaptive_c) && std::isfinite(config.lab_b_max)
            && config.lab_b_max >= 0 && config.lab_b_max <= 255
            && std::isfinite(config.lab_dark_l) && config.lab_dark_l >= 0
            && config.lab_dark_l <= 255, "Invalid threshold");
    require(config.edge_search >= 2 && config.edge_search <= 20
            && std::isfinite(config.gradient_min) && config.gradient_min > 0,
            "Invalid subpixel parameters");
}

Config loadConfig(const fs::path& path) {
    Config config;
    if (!path.empty()) {
        cv::FileStorage file(path.string(), cv::FileStorage::READ);
        require(file.isOpened(), "Cannot open config: " + path.string());
        const std::vector<std::string> keys = {
            "segmentation", "axis", "invert", "subpixel", "blur", "adaptive_block",
            "adaptive_c", "morph_open", "morph_close", "lab_b_max", "lab_dark_l",
            "min_area", "max_area", "min_solidity", "axis_ratio", "head_ratio",
            "gradient_min", "edge_search"};
        for (auto node : file.root()) {
            // 拒绝拼错的参数名，避免配置看似生效却使用默认值。
            require(std::find(keys.begin(), keys.end(), node.name()) != keys.end(),
                    "Unknown config key: " + node.name());
        }
        read(file, "segmentation", config.segmentation);
        read(file, "axis", config.axis);
        int invert = config.invert;
        int subpixel = config.subpixel;
        read(file, "invert", invert);
        read(file, "subpixel", subpixel);
        require((invert == 0 || invert == 1) && (subpixel == 0 || subpixel == 1),
                "Boolean config values must be 0/1");
        config.invert = invert;
        config.subpixel = subpixel;
        read(file, "blur", config.blur);
        read(file, "adaptive_block", config.adaptive_block);
        read(file, "adaptive_c", config.adaptive_c);
        read(file, "morph_open", config.morph_open);
        read(file, "morph_close", config.morph_close);
        read(file, "lab_b_max", config.lab_b_max);
        read(file, "lab_dark_l", config.lab_dark_l);
        read(file, "min_area", config.min_area);
        read(file, "max_area", config.max_area);
        read(file, "min_solidity", config.min_solidity);
        read(file, "axis_ratio", config.axis_ratio);
        read(file, "head_ratio", config.head_ratio);
        read(file, "gradient_min", config.gradient_min);
        read(file, "edge_search", config.edge_search);
    }
    validateConfig(config);
    return config;
}

Recipe loadRecipe(const fs::path& path) {
    cv::FileStorage file(path.string(), cv::FileStorage::READ);
    require(file.isOpened(), "Cannot open recipe: " + path.string());
    Recipe recipe;
    file["name"] >> recipe.name;
    file["unit"] >> recipe.unit;
    require(!recipe.name.empty() && (recipe.unit == "px" || recipe.unit == "mm"),
            "Recipe needs name and unit (px/mm)");
    auto limits = file["limits"];
    require(limits.isMap() && !limits.empty(), "Recipe limits must be a nonempty map");
    for (auto node : limits) {
        dimension(Dimensions{}, node.name());
        require(node.isSeq() && node.size() == 2, "Each limit must be [min, max]");
        double low = static_cast<double>(node[0]);
        double high = static_cast<double>(node[1]);
        require(std::isfinite(low) && std::isfinite(high) && low >= 0 && high >= low,
                "Invalid tolerance range");
        recipe.limits[node.name()] = {low, high};
    }
    return recipe;
}
} // namespace screw
