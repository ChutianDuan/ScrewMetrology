#include "fixtures.hpp"
#include "screw/io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>

// 机制演示，不计入正常合成尺寸真值；保留图像及说明以便复盘。
int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Usage: generate_axis_cases OUTPUT_DIRECTORY");
        }
        const screw::fs::path output = argv[1];
        screw::fs::create_directories(output);
        cv::Mat round(480, 640, CV_8UC3, cv::Scalar::all(225));
        cv::circle(round, {320, 240}, 40, cv::Scalar::all(45), cv::FILLED);
        cv::Mat touching = screwImage(0);
        cv::rectangle(touching, {340, 245, 140, 30}, cv::Scalar::all(45), cv::FILLED);
        cv::Mat occluded = screwImage(17);
        cv::rectangle(occluded, {295, 180, 18, 120}, cv::Scalar::all(225), cv::FILLED);
        const std::vector<std::pair<std::string, cv::Mat>> cases{
            {"near_round", round}, {"huge_head", screwImage(17, 200, 280, 20)},
            {"touching", touching}, {"occluded", occluded}};
        for (const auto& item : cases) {
            if (!cv::imwrite((output / (item.first + ".png")).string(), item.second)) {
                throw std::runtime_error("Cannot write case: " + item.first);
            }
        }
        auto notes = screw::openOutput(output / "cases.txt");
        notes << "near_round: no identifiable long-axis truth\n"
              << "huge_head: construction shaft direction 17 deg; head width 280 px\n"
              << "touching: rectangle attached to a 0 deg screw; no merged-object size truth\n"
              << "occluded: background strip across a 17 deg screw; visible fragments have no complete size truth\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
