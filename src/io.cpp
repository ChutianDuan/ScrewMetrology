#include "screw/io.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace screw {
std::string quote(const std::string& value) {
    // 路径、配方名和错误信息也可能含引号/换行；转义后才能作为合法 JSON 字符串输出。
    std::ostringstream out; out << '"';
    for (unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch) << std::dec;
            else out << ch;
        }
    }
    out << '"'; return out.str();
}
std::string jsonNumber(std::optional<double> value) {
    // JSON 不接受 NaN/Inf；不可测或非有限值统一写 null，保留“无结果”的含义。
    if (!value || !std::isfinite(*value)) return "null";
    std::ostringstream out; out << std::setprecision(12) << *value; return out.str();
}
void writeStrings(std::ostream& out, const std::vector<std::string>& values) {
    out << '[';
    for (size_t i=0; i<values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
    out << ']';
}
std::ofstream openOutput(const fs::path& path) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream file(path);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    if (!file) throw std::runtime_error("Cannot write: " + path.string());
    return file;
}
namespace {
void dims(std::ostream& out, const Dimensions& d) {
    out << '{'; bool first = true;
    for (const auto& name : dimensionNames()) {
        if (!first) out << ',';
        first = false; out << quote(name) << ':' << jsonNumber(dimension(d,name));
    }
    out << '}';
}
std::optional<double> optionalNumber(cv::FileNode node) {
    // 读回时只接受有限数值，不能把 null 或非数值节点转换成虚假的 0 尺寸。
    if (node.isReal() || node.isInt()) {
        double v = static_cast<double>(node);
        if (std::isfinite(v)) return v;
    }
    return std::nullopt;
}
Dimensions readDims(cv::FileNode node) {
    return {optionalNumber(node["length"]),optionalNumber(node["width"]),optionalNumber(node["head_length"]),optionalNumber(node["head_width"]),optionalNumber(node["shaft_width"])};
}
std::vector<fs::path> paths(const fs::path& input, bool images) {
    if (!fs::exists(input)) throw std::runtime_error("Input does not exist: " + input.string());
    if (fs::is_regular_file(input)) return {input};
    if (!fs::is_directory(input)) throw std::runtime_error("Expected file or directory");
    std::vector<fs::path> result;
    for (const auto& entry : fs::directory_iterator(input)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return std::tolower(c);});
        if (images ? (ext==".png" || ext==".jpg" || ext==".jpeg" || ext==".bmp" || ext==".tif" || ext==".tiff") : ext==".json")
            result.push_back(entry.path());
    }
    std::sort(result.begin(),result.end());
    if (result.empty()) throw std::runtime_error("No input files: " + input.string());
    return result;
}
}
std::vector<fs::path> imagePaths(const fs::path& input) { return paths(input,true); }
std::vector<fs::path> resultPaths(const fs::path& input) { return paths(input,false); }
void saveResult(const ImageResult& r, const fs::path& path) {
    // 保存参数、尺度和配方的快照，便于追溯结果；原始/细化/优选尺寸分别保留供比较。
    auto out = openOutput(path);
    out << "{\n\"schema\":\"screw_measurement_v1\",\n\"image\":" << quote(r.image)
        << ",\n\"image_size\":[" << r.size.width << ',' << r.size.height << "],\n\"error\":" << (r.error.empty() ? "null" : quote(r.error))
        << ",\n\"mode\":" << quote(r.mode) << ",\n\"recipe\":";
    if (r.recipe_config) {
        out << "{\"name\":" << quote(r.recipe_config->name) << ",\"unit\":" << quote(r.recipe_config->unit) << ",\"limits\":{";
        bool first=true;
        for (const auto& [name,range] : r.recipe_config->limits) {
            if (!first) out << ',';
            first=false;out << quote(name) << ":[" << jsonNumber(range.first) << ',' << jsonNumber(range.second) << ']';
        }
        out << "}}";
    } else out << "null";
    out << ",\n\"config\":{\"segmentation\":" << quote(r.config.segmentation) << ",\"axis\":" << quote(r.config.axis)
        << ",\"invert\":" << r.config.invert << ",\"subpixel\":" << r.config.subpixel
        << ",\"blur\":" << r.config.blur << ",\"morph_open\":" << r.config.morph_open << ",\"morph_close\":" << r.config.morph_close
        << ",\"adaptive_block\":" << r.config.adaptive_block << ",\"adaptive_c\":" << r.config.adaptive_c
        << ",\"lab_b_max\":" << r.config.lab_b_max << ",\"lab_dark_l\":" << r.config.lab_dark_l
        << ",\"min_area\":" << r.config.min_area << ",\"max_area\":" << r.config.max_area
        << ",\"min_solidity\":" << r.config.min_solidity << ",\"axis_ratio\":" << r.config.axis_ratio
        << ",\"head_ratio\":" << r.config.head_ratio << ",\"gradient_min\":" << r.config.gradient_min
        << ",\"edge_search\":" << r.config.edge_search << "},\n\"scale\":";
    if (r.scale) out << "{\"reference\":" << quote(r.scale->reference) << ",\"mm_per_px\":" << jsonNumber(r.scale->mm_per_px) << ",\"image_size\":[" << r.scale->image_size.width << ',' << r.scale->image_size.height << "],\"assumption\":\"fixed_setup_same_plane_near_frontal\"}";
    else out << "null";
    out << ",\n\"objects\":[\n";
    for (size_t i=0; i<r.objects.size(); ++i) {
        if (i) out << ",\n";
        const auto& o = r.objects[i];
        out << "{\"id\":" << o.id << ",\"center\":[" << jsonNumber(o.center.x) << ',' << jsonNumber(o.center.y)
            << "],\"angle_deg\":" << jsonNumber(o.angle)
            << ",\"axis\":[" << jsonNumber(o.axis[0]) << ',' << jsonNumber(o.axis[1])
            << "],\"comparison_angles_deg\":{\"pca\":" << jsonNumber(o.pca_angle) << ",\"fitline\":" << jsonNumber(o.fit_angle) << ",\"min_area_rect\":" << jsonNumber(o.rect_angle)
            << "},\"rotated_box\":{\"center\":[" << jsonNumber(o.box.center.x) << ',' << jsonNumber(o.box.center.y)
            << "],\"size\":[" << jsonNumber(o.box.size.width) << ',' << jsonNumber(o.box.size.height)
            << "],\"angle_deg\":" << jsonNumber(o.box.angle) << "},\"reliable\":" << (o.reliable ? "true" : "false")
            << ",\"eigen_ratio\":" << jsonNumber(o.eigen_ratio) << ",\"solidity\":" << jsonNumber(o.solidity)
            << ",\"quality_issues\":"; writeStrings(out,o.issues);
        out << ",\"pixel\":"; dims(out,o.pixel);
        out << ",\"subpixel\":"; dims(out,o.refined);
        out << ",\"preferred_px\":"; dims(out,preferredDimensions(o));
        out << ",\"mm\":"; dims(out,o.mm);
        out << ",\"subpixel_status\":" << quote(o.subpixel_status) << ",\"status\":" << quote(o.status)
            << ",\"reasons\":"; writeStrings(out,o.reasons); out << '}';
    }
    out << "\n]}\n";
}
ImageResult loadResult(const fs::path& path) {
    // 评价只恢复已导出的定位、尺寸和质量字段，不重跑分割，也不向检测算法输入标注。
    cv::FileStorage f(path.string(),cv::FileStorage::READ);
    if (!f.isOpened()) throw std::runtime_error("Cannot read result: " + path.string());
    if (static_cast<std::string>(f["schema"]) != "screw_measurement_v1") throw std::runtime_error("Not a measurement JSON: " + path.string());
    ImageResult r; f["image"] >> r.image;
    auto size = f["image_size"]; r.size = {static_cast<int>(size[0]),static_cast<int>(size[1])};
    if (f["error"].isString()) f["error"] >> r.error;
    for (auto node : f["objects"]) {
        Object o; o.id = static_cast<int>(node["id"]);
        auto center = node["center"]; o.center = {static_cast<double>(center[0]),static_cast<double>(center[1])};
        o.angle = static_cast<double>(node["angle_deg"]);
        auto box = node["rotated_box"], bc = box["center"], bs = box["size"];
        o.box = cv::RotatedRect(cv::Point2f(static_cast<float>(bc[0]),static_cast<float>(bc[1])),cv::Size2f(static_cast<float>(bs[0]),static_cast<float>(bs[1])),static_cast<float>(box["angle_deg"]));
        o.reliable = static_cast<int>(node["reliable"]) != 0;
        o.pixel = readDims(node["pixel"]); o.refined = readDims(node["subpixel"]); o.mm = readDims(node["mm"]);
        node["status"] >> o.status;
        r.objects.push_back(std::move(o));
    }
    return r;
}
}
