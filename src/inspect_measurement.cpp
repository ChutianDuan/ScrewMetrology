#include "screw/config.hpp"
#include "screw/diagnostics.hpp"
#include "screw/io.hpp"
#include "screw/pipeline.hpp"
#include "screw/segmentation.hpp"
#include "screw/visualization.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdexcept>

// 无窗口学习入口：一个输入、一份配置、一个独立输出目录。
int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: inspect_measurement IMAGE CONFIG_YAML OUTPUT_DIRECTORY\n";
            return 1;
        }
        cv::setNumThreads(1);
        const screw::fs::path input = argv[1];
        const screw::fs::path output = argv[3];
        const auto config = screw::loadConfig(argv[2]);
        const auto image = cv::imread(input.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Cannot decode image: " + input.string());
        }
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        screw::SegmentationTrace segmentation;
        const auto mask = screw::segment(image, config, &segmentation);
        screw::PipelineTrace pipeline;
        screw::ImageResult result;
        result.image = screw::fs::absolute(input).string();
        result.size = image.size();
        result.config = config;
        result.objects = screw::measureObjects(mask, gray, config, std::nullopt, std::nullopt, false, &pipeline);
        screw::saveDiagnostics(image, config, segmentation, pipeline, result.objects, output);
        screw::saveResult(result, output / "measurement.json");
        if (!cv::imwrite((output / "mask.png").string(), mask)
            || !cv::imwrite((output / "overlay.png").string(), screw::visualize(image, result))) {
            throw std::runtime_error("Cannot write measurement images");
        }
        std::cout << "Saved " << result.objects.size() << " object panels to " << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
