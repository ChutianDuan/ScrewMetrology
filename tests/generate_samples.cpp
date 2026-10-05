#include "fixtures.hpp"
#include "screw/io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

void saveTruthSample(const screw::fs::path& root, std::ostream& truth, const cv::Mat& image,
                     const std::string& name, const std::string& part,
                     double length, double width, double shaft_width) {
    if (!cv::imwrite((root / "images" / name).string(), image)) {
        throw std::runtime_error("Cannot write sample");
    }
    const std::vector<std::pair<std::string, double>> dimensions{
        {"length", length}, {"width", width}, {"head_length", 0.15 * length},
        {"head_width", width}, {"shaft_width", shaft_width}};
    for (const auto& dimension : dimensions) {
        truth << name << ",1," << part << ',' << dimension.first << ',' << dimension.second << ",px\n";
    }
}

void generateTruthSamples(const screw::fs::path& root, std::ostream& truth) {
    int index = 0;
    for (double angle : std::vector<double>{0, 17, 45, 80, 110, 159}) {
        const std::string name = "rotation_" + std::to_string(index++) + ".png";
        saveTruthSample(root, truth, screwImage(angle), name, "rotation_demo", 200, 40, 20);
    }
    saveTruthSample(root, truth, screwImage(25, 160, 34, 16), "small.png", "small", 160, 34, 16);
    saveTruthSample(root, truth, screwImage(65, 240, 48, 24, 180, 1.2), "blur.png", "blur", 240, 48, 24);
    saveTruthSample(root, truth, screwImage(35, 200, 40, 20, 20, 0.8), "low_contrast.png", "low_contrast", 200, 40, 20);

    std::mt19937 generator(20261003);
    std::normal_distribution<double> noise(0, 1.5);
    for (int i = 0; i < 30; ++i) {
        cv::Mat image = screwImage(17, 200, 40, 20, 180, 0.8);
        // 每像素只抽一次噪声，三个通道共用；循环顺序决定固定种子下的样本。
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                const double sample = noise(generator);
                for (int channel = 0; channel < 3; ++channel) {
                    image.at<cv::Vec3b>(y, x)[channel] = cv::saturate_cast<unsigned char>(
                        image.at<cv::Vec3b>(y, x)[channel] + sample);
                }
            }
        }
        std::ostringstream name;
        name << "repeat_" << std::setw(2) << std::setfill('0') << i << ".png";
        saveTruthSample(root, truth, image, name.str(), "simulated_repeat", 200, 40, 20);
    }
}

void generateEdgeCases(const screw::fs::path& root) {
    cv::imwrite((root / "edge_cases" / "truncated.png").string(), screwImage(0, 200, 40, 20, 180, 0, {50, 240}));
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar::all(225));
    cv::imwrite((root / "edge_cases" / "blank.png").string(), blank);
    cv::Mat nut = blank.clone();
    cv::circle(nut, {320, 240}, 30, {40, 40, 40}, -1);
    cv::circle(nut, {320, 240}, 12, {225, 225, 225}, -1);
    cv::imwrite((root / "edge_cases" / "nut.png").string(), nut);
    cv::Mat touching = screwImage(0);
    cv::rectangle(touching, {340, 245, 140, 30}, {45, 45, 45}, -1);
    cv::imwrite((root / "edge_cases" / "touching.png").string(), touching);
    auto bad = screw::openOutput(root / "edge_cases" / "bad.png");
    bad << "not an image";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const screw::fs::path root = argc > 1 ? argv[1] : "data/synthetic";
        screw::fs::create_directories(root / "images");
        screw::fs::create_directories(root / "edge_cases");
        auto truth = screw::openOutput(root / "truth.csv");
        truth << "image,target_id,part_id,measurement,truth,unit\n";
        generateTruthSamples(root, truth);
        generateEdgeCases(root);
        std::cout << "Generated 39 truth samples, 30 simulated repeats, and 5 edge cases in " << root << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
