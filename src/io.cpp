#include "screw/io.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace screw {
std::string quote(const std::string& value) {
    // 路径、配方名和错误也可能含引号/换行，必须转义成合法 JSON 字符串。
    std::ostringstream output;
    output << '"';
    for (unsigned char ch : value) {
        switch (ch) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (ch < 32) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(ch) << std::dec;
            } else {
                output << ch;
            }
        }
    }
    output << '"';
    return output.str();
}

std::string jsonNumber(std::optional<double> value) {
    // JSON 不接受 NaN/Inf，缺失或非有限值统一写 null。
    if (!value || !std::isfinite(*value)) {
        return "null";
    }
    std::ostringstream output;
    output << std::setprecision(12) << *value;
    return output.str();
}

void writeStrings(std::ostream& output, const std::vector<std::string>& values) {
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) {
            output << ',';
        }
        output << quote(values[i]);
    }
    output << ']';
}

std::ofstream openOutput(const fs::path& path) {
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
    std::ofstream file(path);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    if (!file) {
        throw std::runtime_error("Cannot write: " + path.string());
    }
    return file;
}

namespace {
void writeDimensions(std::ostream& output, const Dimensions& dimensions) {
    output << '{';
    bool first = true;
    for (const auto& name : dimensionNames()) {
        if (!first) {
            output << ',';
        }
        first = false;
        output << quote(name) << ':' << jsonNumber(dimension(dimensions, name));
    }
    output << '}';
}

void writeRecipe(std::ostream& output, const std::optional<Recipe>& recipe) {
    if (!recipe) {
        output << "null";
        return;
    }
    output << "{\"name\":" << quote(recipe->name) << ",\"unit\":" << quote(recipe->unit)
           << ",\"limits\":{";
    bool first = true;
    for (const auto& [name, range] : recipe->limits) {
        if (!first) {
            output << ',';
        }
        first = false;
        output << quote(name) << ":[" << jsonNumber(range.first) << ',' << jsonNumber(range.second) << ']';
    }
    output << "}}";
}

void writeConfig(std::ostream& output, const Config& config) {
    output << "{\"segmentation\":" << quote(config.segmentation) << ",\"axis\":" << quote(config.axis)
           << ",\"invert\":" << config.invert << ",\"subpixel\":" << config.subpixel
           << ",\"blur\":" << config.blur << ",\"morph_open\":" << config.morph_open
           << ",\"morph_close\":" << config.morph_close
           << ",\"adaptive_block\":" << config.adaptive_block << ",\"adaptive_c\":" << config.adaptive_c
           << ",\"lab_b_max\":" << config.lab_b_max << ",\"lab_dark_l\":" << config.lab_dark_l
           << ",\"min_area\":" << config.min_area << ",\"max_area\":" << config.max_area
           << ",\"min_solidity\":" << config.min_solidity << ",\"axis_ratio\":" << config.axis_ratio
           << ",\"head_ratio\":" << config.head_ratio << ",\"gradient_min\":" << config.gradient_min
           << ",\"edge_search\":" << config.edge_search << '}';
}

void writeScale(std::ostream& output, const std::optional<Scale>& scale) {
    if (!scale) {
        output << "null";
        return;
    }
    output << "{\"reference\":" << quote(scale->reference)
           << ",\"mm_per_px\":" << jsonNumber(scale->mm_per_px)
           << ",\"image_size\":[" << scale->image_size.width << ',' << scale->image_size.height
           << "],\"assumption\":\"fixed_setup_same_plane_near_frontal\"}";
}

void writeObject(std::ostream& output, const Object& object) {
    output << "{\"id\":" << object.id << ",\"center\":[" << jsonNumber(object.center.x)
           << ',' << jsonNumber(object.center.y) << "],\"angle_deg\":" << jsonNumber(object.angle)
           << ",\"axis\":[" << jsonNumber(object.axis[0]) << ',' << jsonNumber(object.axis[1])
           << "],\"comparison_angles_deg\":{\"pca\":" << jsonNumber(object.pca_angle)
           << ",\"fitline\":" << jsonNumber(object.fit_angle)
           << ",\"min_area_rect\":" << jsonNumber(object.rect_angle)
           << "},\"rotated_box\":{\"center\":[" << jsonNumber(object.box.center.x)
           << ',' << jsonNumber(object.box.center.y) << "],\"size\":["
           << jsonNumber(object.box.size.width) << ',' << jsonNumber(object.box.size.height)
           << "],\"angle_deg\":" << jsonNumber(object.box.angle)
           << "},\"reliable\":" << (object.reliable ? "true" : "false")
           << ",\"eigen_ratio\":" << jsonNumber(object.eigen_ratio)
           << ",\"solidity\":" << jsonNumber(object.solidity) << ",\"quality_issues\":";
    writeStrings(output, object.issues);
    output << ",\"pixel\":";
    writeDimensions(output, object.pixel);
    output << ",\"subpixel\":";
    writeDimensions(output, object.refined);
    output << ",\"preferred_px\":";
    writeDimensions(output, preferredDimensions(object));
    output << ",\"mm\":";
    writeDimensions(output, object.mm);
    output << ",\"subpixel_status\":" << quote(object.subpixel_status)
           << ",\"status\":" << quote(object.status) << ",\"reasons\":";
    writeStrings(output, object.reasons);
    output << '}';
}

std::optional<double> optionalNumber(cv::FileNode node) {
    // 不把 null 或非数值节点转换成虚假的 0 尺寸。
    if (node.isReal() || node.isInt()) {
        double value = static_cast<double>(node);
        if (std::isfinite(value)) {
            return value;
        }
    }
    return std::nullopt;
}

Dimensions readDimensions(cv::FileNode node) {
    return {optionalNumber(node["length"]), optionalNumber(node["width"]),
            optionalNumber(node["head_length"]), optionalNumber(node["head_width"]),
            optionalNumber(node["shaft_width"])};
}

Object readObject(cv::FileNode node) {
    Object object;
    object.id = static_cast<int>(node["id"]);
    auto center = node["center"];
    object.center = {static_cast<double>(center[0]), static_cast<double>(center[1])};
    object.angle = static_cast<double>(node["angle_deg"]);
    auto box = node["rotated_box"];
    auto box_center = box["center"];
    auto box_size = box["size"];
    object.box = cv::RotatedRect(
        cv::Point2f(static_cast<float>(box_center[0]), static_cast<float>(box_center[1])),
        cv::Size2f(static_cast<float>(box_size[0]), static_cast<float>(box_size[1])),
        static_cast<float>(box["angle_deg"]));
    object.reliable = static_cast<int>(node["reliable"]) != 0;
    object.pixel = readDimensions(node["pixel"]);
    object.refined = readDimensions(node["subpixel"]);
    object.mm = readDimensions(node["mm"]);
    node["status"] >> object.status;
    return object;
}

std::vector<fs::path> collectPaths(const fs::path& input, bool images) {
    if (!fs::exists(input)) {
        throw std::runtime_error("Input does not exist: " + input.string());
    }
    if (fs::is_regular_file(input)) {
        return {input};
    }
    if (!fs::is_directory(input)) {
        throw std::runtime_error("Expected file or directory");
    }
    std::vector<fs::path> result;
    for (const auto& entry : fs::directory_iterator(input)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
            return std::tolower(ch);
        });
        bool accepted;
        if (images) {
            accepted = extension == ".png" || extension == ".jpg" || extension == ".jpeg"
                       || extension == ".bmp" || extension == ".tif" || extension == ".tiff";
        } else {
            accepted = extension == ".json";
        }
        if (accepted) {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end());
    if (result.empty()) {
        throw std::runtime_error("No input files: " + input.string());
    }
    return result;
}
} // namespace

std::vector<fs::path> imagePaths(const fs::path& input) {
    return collectPaths(input, true);
}

std::vector<fs::path> resultPaths(const fs::path& input) {
    return collectPaths(input, false);
}

void saveResult(const ImageResult& result, const fs::path& path) {
    // 参数与尺度快照可追溯；像素、细化、优选尺寸分别保留供比较。
    auto output = openOutput(path);
    output << "{\n\"schema\":\"screw_measurement_v1\",\n\"image\":" << quote(result.image)
           << ",\n\"image_size\":[" << result.size.width << ',' << result.size.height
           << "],\n\"error\":" << (result.error.empty() ? "null" : quote(result.error))
           << ",\n\"mode\":" << quote(result.mode) << ",\n\"recipe\":";
    writeRecipe(output, result.recipe_config);
    output << ",\n\"config\":";
    writeConfig(output, result.config);
    output << ",\n\"scale\":";
    writeScale(output, result.scale);
    output << ",\n\"objects\":[\n";
    for (std::size_t i = 0; i < result.objects.size(); ++i) {
        if (i) {
            output << ",\n";
        }
        writeObject(output, result.objects[i]);
    }
    output << "\n]}\n";
}

ImageResult loadResult(const fs::path& path) {
    // 只恢复评价所需字段，不重跑分割，也不向检测算法输入标注。
    cv::FileStorage file(path.string(), cv::FileStorage::READ);
    if (!file.isOpened()) {
        throw std::runtime_error("Cannot read result: " + path.string());
    }
    if (static_cast<std::string>(file["schema"]) != "screw_measurement_v1") {
        throw std::runtime_error("Not a measurement JSON: " + path.string());
    }
    ImageResult result;
    file["image"] >> result.image;
    auto size = file["image_size"];
    result.size = {static_cast<int>(size[0]), static_cast<int>(size[1])};
    if (file["error"].isString()) {
        file["error"] >> result.error;
    }
    for (auto node : file["objects"]) {
        result.objects.push_back(readObject(node));
    }
    return result;
}
} // namespace screw
