#include "screw/config.hpp"
#include "screw/io.hpp"
#include "screw/pipeline.hpp"
#include "screw/segmentation.hpp"
#include "screw/visualization.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
using namespace screw;
const std::vector<std::string> kMethods{"pca", "fitline", "rect"};

std::vector<fs::path> readInputs(const fs::path& input, const fs::path& list) {
    std::vector<fs::path> paths;
    if (list.empty()) {
        paths = imagePaths(input);
    } else {
        std::ifstream file(list);
        if (!file) {
            throw std::runtime_error("Cannot read image list: " + list.string());
        }
        std::string name;
        while (std::getline(file, name)) {
            if (!name.empty() && name.back() == '\r') {
                name.pop_back();
            }
            if (name.empty()) {
                continue;
            }
            if (fs::path(name).filename().string() != name || !fs::is_regular_file(input / name)) {
                throw std::runtime_error("Invalid/missing listed image: " + name);
            }
            paths.push_back(input / name);
        }
    }
    if (paths.empty()) {
        throw std::runtime_error("No comparison images");
    }
    std::set<std::string> stems;
    for (const auto& path : paths) {
        if (!stems.insert(path.stem().string()).second) {
            throw std::runtime_error("Duplicate output stem: " + path.stem().string());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

void saveImage(const fs::path& path, const cv::Mat& image) {
    fs::create_directories(path.parent_path());
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write: " + path.string());
    }
}

void saveContours(const fs::path& path, const std::vector<std::vector<cv::Point>>& contours) {
    auto file = openOutput(path);
    file << "{\"coordinate_system\":\"original_image_xy\",\"contours\":[";
    for (std::size_t i = 0; i < contours.size(); ++i) {
        if (i) {
            file << ',';
        }
        file << '[';
        for (std::size_t j = 0; j < contours[i].size(); ++j) {
            if (j) {
                file << ',';
            }
            file << '[' << contours[i][j].x << ',' << contours[i][j].y << ']';
        }
        file << ']';
    }
    file << "]}\n";
}

bool compareImage(const fs::path& path, const Config& base_config,
                  const fs::path& output, std::ostream& manifest) {
    cv::Mat image, gray, mask;
    std::vector<std::vector<cv::Point>> contours;
    std::string shared_error;
    try {
        image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Cannot decode image: " + path.string());
        }
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        // 每图只分割和筛选一次；三个方法共享同一份只读轮廓和原始灰度。
        mask = segment(image, base_config);
        contours = candidates(mask, base_config);
    } catch (const std::exception& error) {
        shared_error = error.what();
    }
    const auto stem = path.stem().string();
    if (shared_error.empty()) {
        saveImage(output / "shared" / (stem + ".mask.png"), mask);
        saveContours(output / "shared" / (stem + ".contours.json"), contours);
    }
    bool failed = !shared_error.empty();
    std::vector<cv::Mat> panels;
    manifest << "{\"image\":" << quote(path.filename().string())
             << ",\"candidate_count\":" << contours.size()
             << ",\"mask_nonzero\":" << (mask.empty() ? 0 : cv::countNonZero(mask))
             << ",\"shared_error\":" << (shared_error.empty() ? "null" : quote(shared_error))
             << ",\"method_errors\":{";
    for (std::size_t i = 0; i < kMethods.size(); ++i) {
        ImageResult result;
        result.image = fs::absolute(path).string();
        result.size = image.size();
        result.config = base_config;
        result.config.axis = kMethods[i];
        result.error = shared_error;
        if (shared_error.empty()) {
            try {
                result.objects = measureContours(contours, gray, result.config,
                                                  std::nullopt, std::nullopt, false);
            } catch (const std::exception& error) {
                result.error = error.what();
            }
        }
        failed = failed || !result.error.empty();
        const auto folder = output / "methods" / kMethods[i];
        saveResult(result, folder / (stem + ".json"));
        if (!image.empty()) {
            const auto overlay = visualize(image, result);
            saveImage(folder / (stem + ".png"), overlay);
            cv::Mat panel;
            const double factor = std::min(1.0, 640.0 / image.cols);
            cv::resize(overlay, panel, {}, factor, factor, cv::INTER_AREA);
            cv::copyMakeBorder(panel, panel, 30, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all(255));
            cv::putText(panel, kMethods[i], {10, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                        cv::Scalar::all(0), 1, cv::LINE_AA);
            panels.push_back(panel);
        }
        if (i) {
            manifest << ',';
        }
        manifest << quote(kMethods[i]) << ':'
                 << (result.error.empty() ? "null" : quote(result.error));
    }
    manifest << "}}\n";
    if (panels.size() == kMethods.size()) {
        cv::Mat combined;
        cv::hconcat(panels, combined);
        saveImage(output / "panels" / (stem + ".png"), combined);
    }
    return failed;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "Usage: compare_axes IMAGE_OR_DIRECTORY CONFIG OUTPUT_DIRECTORY [IMAGE_LIST]\n";
            return 1;
        }
        cv::setNumThreads(1);
        const auto config = screw::loadConfig(argv[2]);
        const auto inputs = readInputs(argv[1], argc == 5 ? argv[4] : "");
        const screw::fs::path output = argv[3];
        // 独立实验目录避免旧文件混入统计，也不覆盖历史报告。
        if (screw::fs::exists(output) && !screw::fs::is_empty(output)) {
            throw std::runtime_error("Comparison output must be empty: " + output.string());
        }
        auto metadata = screw::openOutput(output / "metadata.json");
        metadata << "{\"schema\":\"screw_axis_comparison_v1\",\"opencv_version\":"
                 << screw::quote(CV_VERSION) << ",\"methods\":[\"pca\",\"fitline\",\"rect\"]"
                 << ",\"policy\":\"one_mask_one_candidate_set_per_image\",\"image_count\":"
                 << inputs.size() << "}\n";
        auto manifest = screw::openOutput(output / "manifest.jsonl");
        int failures = 0;
        for (const auto& path : inputs) {
            failures += compareImage(path, config, output, manifest);
        }
        std::cout << "Compared " << inputs.size() << " images; " << failures << " failed images\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
