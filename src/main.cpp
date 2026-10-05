#include "screw/calibration.hpp"
#include "screw/config.hpp"
#include "screw/evaluation.hpp"
#include "screw/io.hpp"
#include "screw/pipeline.hpp"
#include "screw/segmentation.hpp"
#include "screw/visualization.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace {
using namespace screw;
using Options = std::map<std::string, std::string>;

struct MeasurementOptions {
    fs::path input;
    fs::path output;
    Config config;
    std::string mode;
    std::optional<Scale> scale;
    std::optional<Recipe> recipe;
};

struct BatchStats {
    int failures = 0;
    int objects = 0;
    int refined = 0;
    int ok = 0;
    int ng = 0;
    int not_evaluated = 0;

    void add(const ImageResult& result) {
        objects += static_cast<int>(result.objects.size());
        for (const auto& object : result.objects) {
            if (object.subpixel_status == "refined") {
                ++refined;
            }
            if (object.status == "OK") {
                ++ok;
            } else if (object.status == "NG") {
                ++ng;
            } else {
                ++not_evaluated;
            }
        }
    }
};

void printHelp() {
    std::cout << R"(ScrewMetrology - C++ / OpenCV screw metrology

measure --input IMAGE_OR_DIRECTORY [--config YAML] [--output DIRECTORY]
        [--segmentation lab|otsu|adaptive] [--axis pca|fitline|rect]
        [--scale YAML] [--recipe YAML] [--mode mixed|single] [--no-subpixel]
calibrate-scale --input REFERENCE_IMAGE --p1 X,Y --p2 X,Y --mm DISTANCE --output YAML
evaluate --results JSON_OR_DIRECTORY (--annotations MVTEC_JSON | --truth CSV)
         [--output DIRECTORY]

Defaults: mixed display mode, pixel units, config/measurement.yaml if present.
Scale applies only to fixed setup, same plane, near-frontal images at matching resolution.
Exit codes: 0 success (including no objects); 1 invalid arguments or processing failure.
)";
}

Options parseOptions(int argc, char** argv, int start, const std::set<std::string>& allowed) {
    Options options;
    for (int i = start; i < argc; ++i) {
        std::string key = argv[i];
        if (!allowed.count(key)) {
            throw std::runtime_error("Unknown option: " + key);
        }
        if (options.count(key)) {
            throw std::runtime_error("Duplicate option: " + key);
        }
        if (key == "--no-subpixel") {
            options[key] = "1";
        } else {
            if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
                throw std::runtime_error("Missing value: " + key);
            }
            options[key] = argv[++i];
        }
    }
    return options;
}

std::string optionValue(const Options& options, const std::string& key,
                        const std::string& fallback = "") {
    auto found = options.find(key);
    return found == options.end() ? fallback : found->second;
}

std::string requiredOption(const Options& options, const std::string& key) {
    auto value = optionValue(options, key);
    if (value.empty()) {
        throw std::runtime_error("Required option: " + key);
    }
    return value;
}

double parseNumber(const std::string& text) {
    std::size_t used = 0;
    double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
        throw std::runtime_error("Invalid number: " + text);
    }
    return value;
}

cv::Point2d parsePoint(const std::string& text) {
    auto comma = text.find(',');
    if (comma == std::string::npos || text.find(',', comma + 1) != std::string::npos) {
        throw std::runtime_error("Point must be X,Y");
    }
    return {parseNumber(text.substr(0, comma)), parseNumber(text.substr(comma + 1))};
}

MeasurementOptions loadMeasurementOptions(const Options& options) {
    MeasurementOptions measurement;
    measurement.input = requiredOption(options, "--input");
    measurement.output = optionValue(options, "--output", "result/measurement");
    fs::path config_path = optionValue(options, "--config",
                                      fs::exists("config/measurement.yaml") ? "config/measurement.yaml" : "");
    measurement.config = loadConfig(config_path);
    // YAML 提供场景参数，命令行只覆盖显式选项，覆盖后再验证。
    if (options.count("--segmentation")) {
        measurement.config.segmentation = optionValue(options, "--segmentation");
    }
    if (options.count("--axis")) {
        measurement.config.axis = optionValue(options, "--axis");
    }
    if (options.count("--no-subpixel")) {
        measurement.config.subpixel = false;
    }
    validateConfig(measurement.config);
    measurement.mode = optionValue(options, "--mode", "mixed");
    if (measurement.mode != "mixed" && measurement.mode != "single") {
        throw std::runtime_error("Mode must be mixed or single");
    }
    if (options.count("--scale")) {
        measurement.scale = loadScale(optionValue(options, "--scale"));
    }
    if (options.count("--recipe")) {
        measurement.recipe = loadRecipe(optionValue(options, "--recipe"));
    }
    return measurement;
}

void checkOutputNames(const std::vector<fs::path>& paths) {
    std::set<std::string> stems;
    // a.png 与 a.jpg 会覆盖相同产物，批次开始前拒绝冲突。
    for (const auto& path : paths) {
        if (!stems.insert(path.stem().string()).second) {
            throw std::runtime_error("Output name collision: " + path.stem().string());
        }
    }
}

ImageResult prepareResult(const fs::path& path, const MeasurementOptions& measurement) {
    ImageResult result;
    result.image = fs::absolute(path).string();
    result.config = measurement.config;
    result.scale = measurement.scale;
    result.mode = measurement.mode;
    if (measurement.recipe) {
        result.recipe = measurement.recipe->name;
        result.recipe_config = measurement.recipe;
    }
    return result;
}

void processImage(const fs::path& path, const MeasurementOptions& measurement, ImageResult& result) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Cannot decode image");
    }
    result.size = image.size();
    // 分割使用平滑图；亚像素采样保留原图的灰度边缘。
    cv::Mat mask = segment(image, measurement.config);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    result.objects = measureObjects(mask, gray, measurement.config, measurement.scale,
                                    measurement.recipe, measurement.mode == "single");
    fs::path prefix = measurement.output / path.stem();
    if (!cv::imwrite(prefix.string() + ".mask.png", mask)
        || !cv::imwrite(prefix.string() + ".overlay.png", visualize(image, result))) {
        throw std::runtime_error("Cannot write image output");
    }
}

void saveBatchSummary(const fs::path& output, std::size_t image_count,
                      const BatchStats& stats, double seconds) {
    auto summary = openOutput(output / "batch_summary.json");
    summary << "{\"schema\":\"screw_batch_v1\",\"images\":" << image_count
            << ",\"failed\":" << stats.failures << ",\"objects\":" << stats.objects
            << ",\"subpixel_refined\":" << stats.refined << ",\"ok\":" << stats.ok
            << ",\"ng\":" << stats.ng << ",\"not_evaluated\":" << stats.not_evaluated
            << ",\"seconds\":" << jsonNumber(seconds) << "}\n";
}

int runMeasurement(const Options& options) {
    auto measurement = loadMeasurementOptions(options);
    auto paths = imagePaths(measurement.input);
    checkOutputNames(paths);
    fs::create_directories(measurement.output);
    BatchStats stats;
    auto start = std::chrono::steady_clock::now();
    for (const auto& path : paths) {
        auto result = prepareResult(path, measurement);
        try {
            processImage(path, measurement, result);
        } catch (const std::exception& error) {
            // 坏图或处理失败时清空部分结果，保留错误，继续下一张。
            result.error = error.what();
            result.objects.clear();
            ++stats.failures;
            std::cerr << path.filename().string() << ": " << result.error << '\n';
        }
        // JSON 写入仍在逐图处理的 catch 外；保持原有输出失败边界。
        saveResult(result, measurement.output / (path.stem().string() + ".json"));
        stats.add(result);
        std::cout << path.filename().string() << ": " << result.objects.size() << " objects"
                  << (result.error.empty() ? "" : " (failed)") << '\n';
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    saveBatchSummary(measurement.output, paths.size(), stats, seconds);
    std::cout << "Processed " << paths.size() << " images, " << stats.objects << " objects, "
              << stats.failures << " failures in " << seconds << " s\n";
    // 处理成功与尺寸合格是两个概念：无目标可返回 0，任何处理失败返回 1。
    return stats.failures ? 1 : 0;
}

int runScaleCalibration(const Options& options) {
    std::string input = requiredOption(options, "--input");
    cv::Mat image = cv::imread(input);
    if (image.empty()) {
        throw std::runtime_error("Cannot decode reference image");
    }
    Scale scale = makeScale(fs::absolute(input).string(), image.size(),
                            parsePoint(requiredOption(options, "--p1")),
                            parsePoint(requiredOption(options, "--p2")),
                            parseNumber(requiredOption(options, "--mm")));
    saveScale(scale, requiredOption(options, "--output"));
    std::cout << "Scale: " << scale.mm_per_px << " mm/px\n";
    return 0;
}

int runEvaluation(const Options& options) {
    if (options.count("--annotations") == options.count("--truth")) {
        throw std::runtime_error("Specify exactly one of --annotations or --truth");
    }
    fs::path input = requiredOption(options, "--results");
    fs::path output = optionValue(options, "--output", "result/evaluation");
    if (options.count("--annotations")) {
        evaluateBoxes(input, requiredOption(options, "--annotations"), output);
    } else {
        evaluateDimensions(input, requiredOption(options, "--truth"), output);
    }
    std::cout << "Evaluation saved: " << output / "report.json" << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
            printHelp();
            return 0;
        }
        // 保持单线程，减少时间比较中的调度变化，不改变算法参数。
        cv::setNumThreads(1);
        std::string command = argv[1];
        if (command == "measure") {
            return runMeasurement(parseOptions(argc, argv, 2,
                {"--input", "--config", "--output", "--segmentation", "--axis",
                 "--scale", "--recipe", "--mode", "--no-subpixel"}));
        }
        if (command == "calibrate-scale") {
            return runScaleCalibration(parseOptions(argc, argv, 2,
                {"--input", "--p1", "--p2", "--mm", "--output"}));
        }
        if (command == "evaluate") {
            return runEvaluation(parseOptions(argc, argv, 2,
                {"--results", "--annotations", "--truth", "--output"}));
        }
        throw std::runtime_error("Unknown command: " + command);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
