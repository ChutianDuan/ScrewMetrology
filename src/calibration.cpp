#include "screw/calibration.hpp"
#include "screw/io.hpp"
#include <cmath>
#include <stdexcept>

namespace screw {
namespace {
Dimensions scaledDimensions(const Dimensions& dimensions, double scale) {
    // 缺失尺寸不能在换算时变成 0。
    auto multiply = [scale](std::optional<double> value) -> std::optional<double> {
        if (value) {
            return *value * scale;
        }
        return std::nullopt;
    };
    return {multiply(dimensions.length), multiply(dimensions.width),
            multiply(dimensions.head_length), multiply(dimensions.head_width),
            multiply(dimensions.shaft_width)};
}
} // namespace

Scale makeScale(const std::string& reference, cv::Size size, cv::Point2d p1,
                cv::Point2d p2, double mm) {
    auto inside = [&](cv::Point2d point) {
        return std::isfinite(point.x) && std::isfinite(point.y)
               && point.x >= 0 && point.y >= 0
               && point.x <= size.width - 1 && point.y <= size.height - 1;
    };
    double distance = cv::norm(p1 - p2);
    // 短参考距离会放大人工选点误差；至少相距 1 px。
    if (reference.empty() || size.width <= 0 || size.height <= 0
        || !inside(p1) || !inside(p2) || !std::isfinite(mm) || mm <= 0 || distance < 1) {
        throw std::runtime_error(
            "Invalid scale: points must be inside image and at least 1 px apart; mm must be positive");
    }
    return {reference, size, p1, p2, mm, mm / distance};
}

void saveScale(const Scale& scale, const fs::path& path) {
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
    cv::FileStorage file(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!file.isOpened()) {
        throw std::runtime_error("Cannot write scale: " + path.string());
    }
    file << "reference" << scale.reference
         << "width" << scale.image_size.width << "height" << scale.image_size.height
         << "p1" << scale.p1 << "p2" << scale.p2 << "known_mm" << scale.known_mm
         << "mm_per_px" << scale.mm_per_px
         << "assumption" << "fixed_setup_same_plane_near_frontal";
}

Scale loadScale(const fs::path& path) {
    cv::FileStorage file(path.string(), cv::FileStorage::READ);
    if (!file.isOpened()) {
        throw std::runtime_error("Cannot read scale: " + path.string());
    }
    Scale scale;
    file["reference"] >> scale.reference;
    file["width"] >> scale.image_size.width;
    file["height"] >> scale.image_size.height;
    file["p1"] >> scale.p1;
    file["p2"] >> scale.p2;
    file["known_mm"] >> scale.known_mm;
    file["mm_per_px"] >> scale.mm_per_px;
    Scale checked = makeScale(scale.reference, scale.image_size, scale.p1, scale.p2, scale.known_mm);
    // 重算尺度，拒绝与参考信息不一致的 mm_per_px。
    if (!std::isfinite(scale.mm_per_px)
        || std::abs(checked.mm_per_px - scale.mm_per_px) > checked.mm_per_px * 1e-6) {
        throw std::runtime_error("Inconsistent scale file");
    }
    return scale;
}

void applyScaleAndRecipe(Object& object, cv::Size size, const std::optional<Scale>& scale,
                         const std::optional<Recipe>& recipe, bool single_spec) {
    object.mm = {};
    object.status = "NOT_EVALUATED";
    object.reasons.clear();
    const Dimensions used = preferredDimensions(object);
    if (scale) {
        // 分辨率一致也不能自动证明焦距、距离和工件平面一致。
        if (size != scale->image_size) {
            object.reasons.push_back("scale_resolution_mismatch");
        } else {
            object.mm = scaledDimensions(used, scale->mm_per_px);
        }
    }
    // 可以保留展示用毫米值，但判定前提必须全部满足；single 不是自动规格识别。
    if (!single_spec) {
        object.reasons.push_back("mixed_display_mode");
        return;
    }
    if (!recipe) {
        object.reasons.push_back("no_recipe");
        return;
    }
    if (!object.reliable) {
        object.reasons.push_back("unreliable_geometry");
        return;
    }
    if (recipe->unit == "mm" && (!scale || size != scale->image_size)) {
        object.reasons.push_back("no_applicable_scale");
        return;
    }
    const Dimensions& dimensions = recipe->unit == "mm" ? object.mm : used;
    bool missing = false;
    bool out_of_range = false;
    for (const auto& [name, range] : recipe->limits) {
        auto value = dimension(dimensions, name);
        if (!value) {
            missing = true;
            object.reasons.push_back(name + ":unavailable");
        } else if (*value < range.first || *value > range.second) {
            out_of_range = true;
            object.reasons.push_back(name + ":out_of_range");
        }
    }
    // 上下限包含边界；必要项缺失时即使另一项越界也不作完整判定。
    if (!missing) {
        object.status = out_of_range ? "NG" : "OK";
        if (!out_of_range) {
            object.reasons.push_back("all_configured_limits_passed");
        }
    }
}
} // namespace screw
