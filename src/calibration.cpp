#include "screw/calibration.hpp"
#include "screw/io.hpp"
#include <cmath>
#include <stdexcept>

namespace screw {
// 两参考点的像素距离对应已知毫米距离：s=known_mm/distance，单位 mm/px。
// 单一尺度不处理镜头畸变或透视变化，仅适合同设置、同平面、近似正视场景。
Scale makeScale(const std::string& reference, cv::Size size, cv::Point2d p1, cv::Point2d p2, double mm) {
    auto inside = [&](cv::Point2d p) { return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0 && p.y >= 0 && p.x <= size.width - 1 && p.y <= size.height - 1; };
    double distance = cv::norm(p1 - p2);
    // 拒绝出界、重合或过近参考点；很短的参考距离容易放大人工选点误差。
    if (reference.empty() || size.width <= 0 || size.height <= 0 || !inside(p1) || !inside(p2) || !std::isfinite(mm) || mm <= 0 || distance < 1)
        throw std::runtime_error("Invalid scale: points must be inside image and at least 1 px apart; mm must be positive");
    return {reference, size, p1, p2, mm, mm / distance};
}
void saveScale(const Scale& s, const fs::path& path) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    cv::FileStorage f(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!f.isOpened()) throw std::runtime_error("Cannot write scale: " + path.string());
    f << "reference" << s.reference << "width" << s.image_size.width << "height" << s.image_size.height
      << "p1" << s.p1 << "p2" << s.p2 << "known_mm" << s.known_mm << "mm_per_px" << s.mm_per_px
      << "assumption" << "fixed_setup_same_plane_near_frontal";
}
Scale loadScale(const fs::path& path) {
    cv::FileStorage f(path.string(), cv::FileStorage::READ);
    if (!f.isOpened()) throw std::runtime_error("Cannot read scale: " + path.string());
    Scale s;
    f["reference"] >> s.reference; f["width"] >> s.image_size.width; f["height"] >> s.image_size.height;
    f["p1"] >> s.p1; f["p2"] >> s.p2; f["known_mm"] >> s.known_mm;
    f["mm_per_px"] >> s.mm_per_px;
    Scale checked = makeScale(s.reference, s.image_size, s.p1, s.p2, s.known_mm);
    // 由原始参考信息重算尺度，避免文件中的 mm_per_px 与参考距离不一致。
    if (!std::isfinite(s.mm_per_px) || std::abs(checked.mm_per_px - s.mm_per_px) > checked.mm_per_px * 1e-6)
        throw std::runtime_error("Inconsistent scale file");
    return s;
}
namespace {
Dimensions scaled(const Dimensions& d, double scale) {
    // 不可测尺寸仍保留 nullopt，换算毫米不能把“缺失”变成数值 0。
    auto mul = [scale](std::optional<double> v) -> std::optional<double> { return v ? std::optional<double>(*v * scale) : std::nullopt; };
    return {mul(d.length), mul(d.width), mul(d.head_length), mul(d.head_width), mul(d.shaft_width)};
}
}
void applyScaleAndRecipe(Object& o, cv::Size size, const std::optional<Scale>& scale,
                         const std::optional<Recipe>& recipe, bool single_spec) {
    o.mm = {}; o.status = "NOT_EVALUATED"; o.reasons.clear();
    const Dimensions used = preferredDimensions(o);
    if (scale) {
        // 分辨率不一致时尺度不适用；一致也不能自动证明焦距、距离和工件平面一致。
        if (size != scale->image_size) o.reasons.push_back("scale_resolution_mismatch");
        else o.mm = scaled(used, scale->mm_per_px);
    }
    // 先检查判定前提。single_spec 是使用者确认单规格，不是算法自动识别型号。
    // 前面可保留展示用毫米值，但质量不可靠时仍不得据此给出 OK/NG。
    if (!single_spec) { o.reasons.push_back("mixed_display_mode"); return; }
    if (!recipe) { o.reasons.push_back("no_recipe"); return; }
    if (!o.reliable) { o.reasons.push_back("unreliable_geometry"); return; }
    if (recipe->unit == "mm" && (!scale || size != scale->image_size)) {
        o.reasons.push_back("no_applicable_scale"); return;
    }
    const Dimensions& d = recipe->unit == "mm" ? o.mm : used;
    bool missing = false, out_of_range = false;
    // 范围包含上下边界；记录每项缺失或越界原因，便于区分“不可测”和“明确超差”。
    for (const auto& [name, range] : recipe->limits) {
        auto value = dimension(d, name);
        if (!value) { missing = true; o.reasons.push_back(name + ":unavailable"); }
        else if (*value < range.first || *value > range.second) {
            out_of_range = true; o.reasons.push_back(name + ":out_of_range");
        }
    }
    // 任一必要尺寸缺失时保持 NOT_EVALUATED，即使其他项已越界也不作完整判定。
    if (!missing) {
        o.status = out_of_range ? "NG" : "OK";
        if (!out_of_range) o.reasons.push_back("all_configured_limits_passed");
    }
}
}
