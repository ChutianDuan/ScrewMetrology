#include "screw/evaluation.hpp"
#include "screw/io.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <cmath>

namespace screw {
namespace {
constexpr double kMatchingIoU = 0.5;
constexpr double kDirectionalBoxRatio = 1.3;

struct Stats {
    // 有符号误差与重复测量值使用同一容器，但统计含义不同。
    std::vector<double> values;

    void add(double value) {
        values.push_back(value);
    }

    void writeRepeatability(std::ostream& output) const {
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double squared = 0;
        for (double value : values) {
            squared += (value - mean) * (value - mean);
        }
        output << "{\"count\":" << values.size() << ",\"mean\":" << jsonNumber(mean)
               << ",\"min\":" << jsonNumber(*std::min_element(values.begin(), values.end()))
               << ",\"max\":" << jsonNumber(*std::max_element(values.begin(), values.end()))
               << ",\"std\":" << jsonNumber(values.size() > 1
                   ? std::optional<double>(std::sqrt(squared / (values.size() - 1))) : std::nullopt) << '}';
    }

    void write(std::ostream& output) const {
        output << "{\"count\":" << values.size();
        if (values.empty()) {
            output << ",\"mean\":null,\"mae\":null,\"max_abs\":null,\"std\":null}";
            return;
        }
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double absolute = 0;
        double maximum = 0;
        double squared = 0;
        for (double value : values) {
            absolute += std::abs(value);
            maximum = std::max(maximum, std::abs(value));
            squared += (value - mean) * (value - mean);
        }
        output << ",\"mean\":" << jsonNumber(mean)
               << ",\"mae\":" << jsonNumber(absolute / values.size())
               << ",\"max_abs\":" << jsonNumber(maximum)
               << ",\"std\":" << jsonNumber(values.size() > 1
                   ? std::optional<double>(std::sqrt(squared / (values.size() - 1))) : std::nullopt) << '}';
    }
};

struct TruthBox {
    int id = 0;
    int category = 0;
    cv::RotatedRect box;
};

struct Annotations {
    std::map<int, std::string> images;
    std::map<int, cv::Size> sizes;
    std::map<int, std::vector<TruthBox>> boxes;
    std::set<std::string> names;
};

struct Match {
    double iou;
    std::size_t prediction;
    std::size_t actual;
};

struct BoxMatches {
    std::vector<Match> pairs;
    std::vector<bool> used_predictions;
    std::vector<bool> used_truth;
};

struct DetectionStats {
    int tp = 0;
    int fp = 0;
    int fn = 0;
    int evaluated = 0;
    int failed = 0;
    int unreliable = 0;
    Stats centers;
    Stats origins;
    Stats angles;
    Stats ious;
    std::map<int, std::pair<int, int>> categories;
    std::vector<std::string> missing_images;
    std::vector<std::string> failed_images;
};

struct TruthMeasurement {
    std::string image;
    int target_id;
    std::string part;
    std::string measurement;
    std::string unit;
    std::optional<double> actual;
    std::string unique_key;
};

struct DimensionSample {
    std::optional<double> value;
    std::optional<double> pixel;
    std::optional<double> subpixel;
    std::string reason;
};

struct DimensionStats {
    int rows = 0;
    int excluded = 0;
    std::map<std::string, Stats> errors;
    std::map<std::string, Stats> pixel_errors;
    std::map<std::string, Stats> subpixel_errors;
    std::map<std::string, Stats> repeats;
};

std::map<std::string, ImageResult> loadResults(const fs::path& input) {
    std::map<std::string, ImageResult> results;
    for (auto path : resultPaths(input)) {
        cv::FileStorage file(path.string(), cv::FileStorage::READ);
        if (!file.isOpened()) {
            throw std::runtime_error("Cannot read JSON: " + path.string());
        }
        // 结果目录可能混有批次汇总或报告，只读取单图测量 schema。
        if (static_cast<std::string>(file["schema"]) != "screw_measurement_v1") {
            continue;
        }
        file.release();
        auto result = loadResult(path);
        std::string name = fs::path(result.image).filename().string();
        if (!results.emplace(name, std::move(result)).second) {
            throw std::runtime_error("Duplicate result image: " + name);
        }
    }
    if (results.empty()) {
        throw std::runtime_error("No measurement results");
    }
    return results;
}

std::vector<std::string> csvRow(const std::string& line) {
    std::vector<std::string> row;
    std::string field;
    bool quoted = false;
    bool closed = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];
        if (quoted) {
            if (ch == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field += '"';
                    ++i;
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                field += ch;
            }
        } else if (ch == ',') {
            row.push_back(field);
            field.clear();
            closed = false;
        } else if (ch == '"' && field.empty() && !closed) {
            quoted = true;
        } else if (ch == '\r' && i + 1 == line.size()) {
            // 接受 CRLF，但不把行尾回车放进最后一个字段。
        } else if (closed || ch == '"') {
            throw std::runtime_error("Malformed CSV quoting");
        } else {
            field += ch;
        }
    }
    if (quoted) {
        throw std::runtime_error("Unterminated CSV quote");
    }
    row.push_back(field);
    return row;
}

Annotations readAnnotations(const fs::path& path) {
    cv::FileStorage file(path.string(), cv::FileStorage::READ);
    if (!file.isOpened() || !file["images"].isSeq() || !file["annotations"].isSeq()) {
        throw std::runtime_error("Invalid MVTec annotation file");
    }
    Annotations annotations;
    for (auto node : file["images"]) {
        int id = static_cast<int>(node["id"]);
        std::string name = static_cast<std::string>(node["file_name"]);
        if (name.empty() || !annotations.names.insert(name).second
            || !annotations.images.emplace(id, name).second) {
            throw std::runtime_error("Duplicate/invalid annotation image");
        }
        annotations.sizes[id] = {static_cast<int>(node["width"]), static_cast<int>(node["height"])};
    }
    for (auto node : file["annotations"]) {
        auto box = node["bbox"];
        if (!box.isSeq() || box.size() != 5) {
            throw std::runtime_error("MVTec bbox requires [row,col,width,height,phi]");
        }
        int image_id = static_cast<int>(node["image_id"]);
        if (!annotations.images.count(image_id)) {
            throw std::runtime_error("Annotation references unknown image");
        }
        double row = static_cast<double>(box[0]);
        double col = static_cast<double>(box[1]);
        double width = static_cast<double>(box[2]);
        double height = static_cast<double>(box[3]);
        double phi = static_cast<double>(box[4]);
        if (!std::isfinite(row) || !std::isfinite(col) || !std::isfinite(width)
            || !std::isfinite(height) || !std::isfinite(phi) || width <= 0 || height <= 0) {
            throw std::runtime_error("Invalid ground truth box");
        }
        // 原包为 row/col 中心，原点偏移 +0.5；逆时针弧度转图像坐标需取负。
        annotations.boxes[image_id].push_back({
            static_cast<int>(node["id"]), static_cast<int>(node["category_id"]),
            cv::RotatedRect({static_cast<float>(col - 0.5), static_cast<float>(row - 0.5)},
                           {static_cast<float>(width), static_cast<float>(height)},
                           static_cast<float>(-phi * 180 / CV_PI))});
    }
    return annotations;
}

BoxMatches matchBoxes(const std::vector<Object>& predictions, const std::vector<TruthBox>& truth) {
    std::vector<Match> candidates;
    for (std::size_t p = 0; p < predictions.size(); ++p) {
        for (std::size_t g = 0; g < truth.size(); ++g) {
            double iou = rotatedIoU(predictions[p].box, truth[g].box);
            if (iou >= kMatchingIoU) {
                candidates.push_back({iou, p, g});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Match& first, const Match& second) {
        if (first.iou != second.iou) {
            return first.iou > second.iou;
        }
        return std::tie(first.prediction, first.actual) < std::tie(second.prediction, second.actual);
    });
    BoxMatches matches;
    matches.used_predictions.resize(predictions.size(), false);
    matches.used_truth.resize(truth.size(), false);
    // 固定为 IoU 降序贪心一对一匹配，不是最大权重二分图匹配。
    for (const auto& candidate : candidates) {
        if (matches.used_predictions[candidate.prediction] || matches.used_truth[candidate.actual]) {
            continue;
        }
        matches.used_predictions[candidate.prediction] = true;
        matches.used_truth[candidate.actual] = true;
        matches.pairs.push_back(candidate);
    }
    return matches;
}

void writeImageEvaluation(std::ostream& output, const std::string& name, const ImageResult& result,
                          const std::vector<TruthBox>& truth, const BoxMatches& matches,
                          DetectionStats& stats) {
    output << "{\"image\":" << quote(name) << ",\"matches\":[";
    int matched = 0;
    for (const auto& match : matches.pairs) {
        if (matched++) {
            output << ',';
        }
        const auto& prediction = result.objects[match.prediction];
        const auto& actual = truth[match.actual];
        // 标注是框中心真值：与轮廓质心的偏移分开统计。
        double center_error = cv::norm(prediction.box.center - actual.box.center);
        double origin_distance = cv::norm(prediction.center - cv::Point2d(actual.box.center));
        stats.centers.add(center_error);
        stats.origins.add(origin_distance);
        stats.ious.add(match.iou);
        double ratio = std::max(actual.box.size.width, actual.box.size.height)
                       / std::min(actual.box.size.width, actual.box.size.height);
        std::optional<double> angle;
        if (ratio >= kDirectionalBoxRatio) {
            angle = angleError(prediction.angle, actual.box.angle
                + (actual.box.size.height > actual.box.size.width ? 90 : 0));
            stats.angles.add(*angle);
        }
        ++stats.categories[actual.category].first;
        output << "{\"target_id\":" << prediction.id << ",\"annotation_id\":" << actual.id
               << ",\"iou\":" << jsonNumber(match.iou)
               << ",\"center_error_px\":" << jsonNumber(center_error)
               << ",\"centroid_to_box_center_distance_px\":" << jsonNumber(origin_distance)
               << ",\"angle_error_deg\":" << jsonNumber(angle) << '}';
    }
    stats.tp += matched;
    stats.fp += static_cast<int>(result.objects.size()) - matched;
    stats.fn += static_cast<int>(truth.size()) - matched;
    output << "],\"false_positive_ids\":[";
    bool first = true;
    for (std::size_t p = 0; p < result.objects.size(); ++p) {
        if (!matches.used_predictions[p]) {
            if (!first) {
                output << ',';
            }
            first = false;
            output << result.objects[p].id;
        }
    }
    output << "],\"missed_annotation_ids\":[";
    first = true;
    for (std::size_t g = 0; g < truth.size(); ++g) {
        ++stats.categories[truth[g].category].second;
        if (!matches.used_truth[g]) {
            if (!first) {
                output << ',';
            }
            first = false;
            output << truth[g].id;
        }
    }
    output << "],\"error\":" << (result.error.empty() ? "null" : quote(result.error)) << "}\n";
}

void drawBox(cv::Mat& image, const cv::RotatedRect& box, const cv::Scalar& color,
             const std::string& label) {
    cv::Point2f corners[4];
    box.points(corners);
    for (int i = 0; i < 4; ++i) {
        cv::line(image, corners[i], corners[(i + 1) % 4], color, 3, cv::LINE_AA);
    }
    cv::putText(image, label, box.center, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
}

void saveFailureOverlay(const ImageResult& result, const std::string& name,
                        const std::vector<TruthBox>& truth, const BoxMatches& matches,
                        const fs::path& output) {
    if (matches.pairs.size() >= truth.size() && matches.pairs.size() >= result.objects.size()) {
        return;
    }
    cv::Mat image = cv::imread(result.image);
    if (image.empty()) {
        return;
    }
    for (std::size_t g = 0; g < truth.size(); ++g) {
        if (!matches.used_truth[g]) {
            drawBox(image, truth[g].box, {0, 0, 255}, "miss:" + std::to_string(truth[g].id));
        }
    }
    for (std::size_t p = 0; p < result.objects.size(); ++p) {
        if (!matches.used_predictions[p]) {
            drawBox(image, result.objects[p].box, {0, 255, 255},
                    "fp:" + std::to_string(result.objects[p].id));
        }
    }
    if (image.cols > 960) {
        cv::resize(image, image, {960, static_cast<int>(image.rows * 960.0 / image.cols)});
    }
    fs::create_directories(output / "failures");
    if (!cv::imwrite((output / "failures" / (fs::path(name).stem().string() + ".jpg")).string(),
                     image, {cv::IMWRITE_JPEG_QUALITY, 80})) {
        throw std::runtime_error("Cannot write failure overlay");
    }
}

void saveDetectionReport(const fs::path& path, const fs::path& annotation_file,
                         std::size_t image_count, const DetectionStats& stats) {
    auto output = openOutput(path);
    output << "{\"schema\":\"screw_detection_evaluation_v1\",\"annotation_file\":"
           << quote(fs::absolute(annotation_file).string())
           << ",\"iou_threshold\":0.5,\"matching\":\"greedy_descending_iou_one_to_one\""
           << ",\"annotation_images\":" << image_count << ",\"evaluated_images\":" << stats.evaluated
           << ",\"failed_images\":" << stats.failed << ",\"missing_result_images\":";
    writeStrings(output, stats.missing_images);
    output << ",\"failed_image_names\":";
    writeStrings(output, stats.failed_images);
    output << ",\"tp\":" << stats.tp << ",\"fp\":" << stats.fp << ",\"fn\":" << stats.fn
           << ",\"recall\":" << jsonNumber(stats.tp + stats.fn
               ? std::optional<double>(static_cast<double>(stats.tp) / (stats.tp + stats.fn)) : std::nullopt)
           << ",\"precision\":" << jsonNumber(stats.tp + stats.fp
               ? std::optional<double>(static_cast<double>(stats.tp) / (stats.tp + stats.fp)) : std::nullopt)
           << ",\"unreliable_predictions\":" << stats.unreliable << ",\"center_error_px\":";
    stats.centers.write(output);
    output << ",\"centroid_to_box_center_distance_px\":";
    stats.origins.write(output);
    output << ",\"angle_error_deg\":";
    stats.angles.write(output);
    output << ",\"matched_iou\":";
    stats.ious.write(output);
    output << ",\"categories\":[";
    bool first = true;
    for (auto [id, counts] : stats.categories) {
        if (!first) {
            output << ',';
        }
        first = false;
        output << "{\"id\":" << id << ",\"matched\":" << counts.first << ",\"total\":" << counts.second
               << ",\"recall\":" << jsonNumber(static_cast<double>(counts.first) / counts.second) << '}';
    }
    output << "]}\n";
}

TruthMeasurement parseTruthMeasurement(const std::vector<std::string>& row, int row_number) {
    if (row.size() != 6) {
        throw std::runtime_error("CSV field count at data row " + std::to_string(row_number));
    }
    TruthMeasurement measurement;
    measurement.image = fs::path(row[0]).filename().string();
    measurement.part = row[2];
    measurement.measurement = row[3];
    measurement.unit = row[5];
    dimension(Dimensions{}, measurement.measurement);
    if (measurement.part.empty() || (measurement.unit != "mm" && measurement.unit != "px")) {
        throw std::runtime_error("Invalid part_id or unit at CSV row " + std::to_string(row_number));
    }
    std::size_t used = 0;
    measurement.target_id = std::stoi(row[1], &used);
    if (used != row[1].size() || measurement.target_id <= 0) {
        throw std::runtime_error("Invalid target_id");
    }
    // truth 可空，只参与重复性，不用于误差统计。
    if (!row[4].empty()) {
        used = 0;
        double value = std::stod(row[4], &used);
        if (used != row[4].size() || !std::isfinite(value) || value <= 0) {
            throw std::runtime_error("Invalid truth");
        }
        measurement.actual = value;
    }
    measurement.unique_key = measurement.image + "|" + row[1] + "|"
                             + measurement.measurement + "|" + measurement.unit;
    return measurement;
}

DimensionSample findDimensionSample(const std::map<std::string, ImageResult>& results,
                                    const TruthMeasurement& measurement) {
    DimensionSample sample;
    auto found = results.find(measurement.image);
    if (found == results.end()) {
        sample.reason = "missing_result";
        return sample;
    }
    const Object* object = nullptr;
    for (const auto& candidate : found->second.objects) {
        if (candidate.id == measurement.target_id) {
            object = &candidate;
            break;
        }
    }
    if (!object) {
        sample.reason = "missing_target";
        return sample;
    }
    if (!object->reliable) {
        sample.reason = "unreliable_geometry";
        return sample;
    }
    sample.pixel = dimension(object->pixel, measurement.measurement);
    sample.subpixel = dimension(object->refined, measurement.measurement);
    if (measurement.unit == "mm") {
        sample.value = dimension(object->mm, measurement.measurement);
    } else {
        sample.value = sample.subpixel ? sample.subpixel : sample.pixel;
    }
    if (!sample.value) {
        sample.reason = "unavailable_dimension";
    }
    return sample;
}

void accumulateDimension(DimensionStats& stats, const TruthMeasurement& measurement,
                         const DimensionSample& sample) {
    if (!sample.reason.empty()) {
        ++stats.excluded;
        return;
    }
    std::string key = measurement.measurement + "|" + measurement.unit;
    stats.repeats[measurement.part + "|" + key].add(*sample.value);
    if (measurement.actual) {
        stats.errors[key].add(*sample.value - *measurement.actual);
        if (measurement.unit == "px" && sample.pixel) {
            stats.pixel_errors[key].add(*sample.pixel - *measurement.actual);
        }
        if (measurement.unit == "px" && sample.subpixel) {
            stats.subpixel_errors[key].add(*sample.subpixel - *measurement.actual);
        }
    }
}

void writeDimensionSample(std::ostream& output, const TruthMeasurement& measurement,
                          const DimensionSample& sample) {
    output << "{\"image\":" << quote(measurement.image) << ",\"target_id\":" << measurement.target_id
           << ",\"part_id\":" << quote(measurement.part)
           << ",\"measurement\":" << quote(measurement.measurement)
           << ",\"unit\":" << quote(measurement.unit) << ",\"truth\":" << jsonNumber(measurement.actual)
           << ",\"measured\":" << jsonNumber(sample.value)
           << ",\"error\":" << jsonNumber(sample.value && measurement.actual
               ? std::optional<double>(*sample.value - *measurement.actual) : std::nullopt)
           << ",\"excluded_reason\":" << (sample.reason.empty() ? "null" : quote(sample.reason)) << "}\n";
}

void writeStatsMap(std::ostream& output, const std::map<std::string, Stats>& values,
                   bool repeatability = false) {
    bool first = true;
    for (const auto& [key, stats] : values) {
        if (!first) {
            output << ',';
        }
        first = false;
        output << quote(key) << ':';
        if (repeatability) {
            stats.writeRepeatability(output);
        } else {
            stats.write(output);
        }
    }
}

void saveDimensionReport(const fs::path& path, const DimensionStats& stats) {
    auto output = openOutput(path);
    output << "{\"schema\":\"screw_dimension_evaluation_v1\",\"rows\":" << stats.rows
           << ",\"excluded\":" << stats.excluded << ",\"errors\":{";
    writeStatsMap(output, stats.errors);
    output << "},\"pixel_errors\":{";
    writeStatsMap(output, stats.pixel_errors);
    output << "},\"subpixel_errors\":{";
    writeStatsMap(output, stats.subpixel_errors);
    output << "},\"repeatability\":{";
    writeStatsMap(output, stats.repeats, true);
    output << "}}\n";
}
} // namespace

double rotatedIoU(const cv::RotatedRect& first, const cv::RotatedRect& second) {
    if (first.size.area() <= 0 || second.size.area() <= 0) {
        return 0;
    }
    std::vector<cv::Point2f> overlap;
    int status = cv::rotatedRectangleIntersection(first, second, overlap);
    double intersection = 0;
    if (status == cv::INTERSECT_FULL) {
        intersection = std::min(first.size.area(), second.size.area());
    } else if (overlap.size() >= 3) {
        intersection = std::abs(cv::contourArea(overlap));
    }
    double total = first.size.area() + second.size.area() - intersection;
    return total > 0 ? std::clamp(intersection / total, 0.0, 1.0) : 0;
}

void evaluateBoxes(const fs::path& input, const fs::path& annotation_file, const fs::path& output) {
    auto results = loadResults(input);
    auto annotations = readAnnotations(annotation_file);
    for (const auto& entry : results) {
        if (!annotations.names.count(entry.first)) {
            throw std::runtime_error("Result image not in annotations: " + entry.first);
        }
    }
    fs::create_directories(output);
    auto details = openOutput(output / "images.jsonl");
    DetectionStats stats;
    for (const auto& [image_id, name] : annotations.images) {
        auto found = results.find(name);
        if (found == results.end()) {
            stats.missing_images.push_back(name);
            continue;
        }
        ++stats.evaluated;
        const auto& result = found->second;
        const auto& truth = annotations.boxes[image_id];
        if (!result.error.empty()) {
            ++stats.failed;
            stats.failed_images.push_back(name);
        } else if (result.size != annotations.sizes[image_id]) {
            throw std::runtime_error("Result/annotation resolution mismatch: " + name);
        }
        for (const auto& object : result.objects) {
            if (!object.reliable) {
                ++stats.unreliable;
            }
        }
        auto matches = matchBoxes(result.objects, truth);
        writeImageEvaluation(details, name, result, truth, matches, stats);
        saveFailureOverlay(result, name, truth, matches, output);
    }
    saveDetectionReport(output / "report.json", annotation_file, annotations.images.size(), stats);
}

void evaluateDimensions(const fs::path& input, const fs::path& truth_file, const fs::path& output) {
    auto results = loadResults(input);
    std::ifstream truth(truth_file);
    if (!truth) {
        throw std::runtime_error("Cannot read CSV: " + truth_file.string());
    }
    std::string line;
    if (!std::getline(truth, line)) {
        throw std::runtime_error("Empty CSV");
    }
    auto header = csvRow(line);
    if (header != std::vector<std::string>{"image", "target_id", "part_id", "measurement", "truth", "unit"}) {
        throw std::runtime_error("CSV header must be image,target_id,part_id,measurement,truth,unit");
    }
    fs::create_directories(output);
    auto details = openOutput(output / "dimensions.jsonl");
    DimensionStats stats;
    std::set<std::string> seen;
    while (std::getline(truth, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }
        ++stats.rows;
        auto measurement = parseTruthMeasurement(csvRow(line), stats.rows);
        if (!seen.insert(measurement.unique_key).second) {
            throw std::runtime_error("Duplicate CSV measurement");
        }
        auto sample = findDimensionSample(results, measurement);
        accumulateDimension(stats, measurement, sample);
        writeDimensionSample(details, measurement, sample);
    }
    if (!stats.rows) {
        throw std::runtime_error("No CSV data rows");
    }
    saveDimensionReport(output / "report.json", stats);
}
} // namespace screw
