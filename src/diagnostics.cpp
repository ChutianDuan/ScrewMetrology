#include "screw/diagnostics.hpp"
#include "screw/io.hpp"
#include "screw/locator.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace screw {
namespace {
constexpr int kTileWidth = 560;
constexpr int kTileHeight = 340;
const cv::Scalar kRawColor{145, 145, 145};
const cv::Scalar kProfileColor{0, 220, 255};
const cv::Scalar kBaselineColor{80, 220, 80};
const cv::Scalar kThresholdColor{0, 160, 255};
const cv::Scalar kBoundaryColor{230, 80, 230};

void text(cv::Mat& image, const std::string& value, cv::Point point, double size = 0.42,
          cv::Scalar color = {225, 225, 225}) {
    cv::putText(image, value, point, cv::FONT_HERSHEY_SIMPLEX, size, color, 1, cv::LINE_AA);
}

std::string number(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

cv::Mat tile(const std::string& title) {
    cv::Mat output(kTileHeight, kTileWidth, CV_8UC3, cv::Scalar(24, 24, 24));
    text(output, title, {12, 24}, 0.5);
    return output;
}

cv::Mat imageTile(const cv::Mat& image, const std::string& title, bool binary = false) {
    cv::Mat output = tile(title);
    if (image.empty()) {
        text(output, "Not recorded for this segmentation method", {20, 170});
        return output;
    }
    cv::Mat color;
    if (image.channels() == 1) {
        cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
    } else {
        color = image;
    }
    const double scale = std::min((kTileWidth - 20.0) / image.cols,
                                  (kTileHeight - 50.0) / image.rows);
    cv::Mat resized;
    cv::resize(color, resized, {std::max(1, cvRound(image.cols * scale)),
                                std::max(1, cvRound(image.rows * scale))}, 0, 0,
               binary ? cv::INTER_NEAREST : cv::INTER_AREA);
    resized.copyTo(output(cv::Rect((kTileWidth - resized.cols) / 2,
                                  40 + (kTileHeight - 50 - resized.rows) / 2,
                                  resized.cols, resized.rows)));
    return output;
}

void place(cv::Mat& panel, const cv::Mat& image, int row, int column) {
    image.copyTo(panel(cv::Rect(column * kTileWidth, row * kTileHeight, kTileWidth, kTileHeight)));
}

struct Plot {
    cv::Mat image;
    cv::Rect area{58, 64, 460, 220};
    double x_min;
    double x_max;
    double y_max;

    Plot(const std::string& title, double minimum, double maximum, double vertical_maximum,
         const std::string& x_label, const std::string& y_label)
        : image(tile(title)), x_min(minimum), x_max(std::max(maximum, minimum + 1)),
          y_max(std::max(vertical_maximum, 1.0)) {
        cv::rectangle(image, area, {80, 80, 80}, 1);
        text(image, y_label, {12, 48}, 0.35);
        for (int i = 0; i <= 2; ++i) {
            double x = x_min + (x_max - x_min) * i / 2;
            double y = y_max * i / 2;
            text(image, number(x), {point(x, 0).x - 15, area.br().y + 16}, 0.32);
            text(image, number(y), {3, point(x_min, y).y + 4}, 0.3);
        }
        text(image, x_label, {180, 330}, 0.38);
    }

    cv::Point point(double x, double y) const {
        return {cvRound(area.x + (x - x_min) / (x_max - x_min) * (area.width - 1)),
                cvRound(area.y + (1 - y / y_max) * (area.height - 1))};
    }

    void horizontal(double y, const cv::Scalar& color) {
        cv::line(image, point(x_min, y), point(x_max, y), color, 1, cv::LINE_AA);
    }

    void dashedHorizontal(double y, const cv::Scalar& color) {
        const int row = point(x_min, y).y;
        for (int x = area.x; x < area.br().x; x += 12) {
            cv::line(image, {x, row}, {std::min(x + 6, area.br().x - 1), row}, color, 1);
        }
    }

    void vertical(double x, const cv::Scalar& color) {
        cv::line(image, point(x, 0), point(x, y_max), color, 1, cv::LINE_AA);
    }

    void series(const std::vector<double>& x, const std::vector<double>& y, const cv::Scalar& color) {
        for (std::size_t i = 1; i < y.size(); ++i) {
            cv::line(image, point(x[i - 1], y[i - 1]), point(x[i], y[i]), color, 1, cv::LINE_AA);
        }
    }
};

cv::Mat profilePlot(const MeasurementTrace& trace, const Config& config) {
    if (trace.slices.empty()) {
        return tile("WIDTH PROFILE | no valid slices");
    }
    std::vector<double> positions;
    std::vector<double> widths;
    for (const auto& slice : trace.slices) {
        positions.push_back(slice.s);
        widths.push_back(slice.width);
    }
    double maximum = *std::max_element(widths.begin(), widths.end());
    if (trace.head_attempted && trace.shaft_baseline) {
        maximum = std::max(maximum, config.head_ratio * *trace.shaft_baseline);
    }
    if (trace.head_attempted && trace.narrow_baseline) {
        maximum = std::max(maximum, config.head_ratio * *trace.narrow_baseline);
    }
    Plot plot("WIDTH PROFILE | raw / median / shaft / threshold", positions.front(), positions.back(),
               maximum * 1.15, "local s (px)", "width (px)");
    plot.series(positions, widths, kRawColor);
    plot.series(positions, trace.profile, kProfileColor);
    if (trace.shaft_baseline) {
        plot.horizontal(*trace.shaft_baseline, kBaselineColor);
        if (trace.head_attempted) {
            plot.horizontal(config.head_ratio * *trace.shaft_baseline, kThresholdColor);
        }
    }
    if (trace.head_attempted && trace.narrow_baseline) {
        plot.dashedHorizontal(config.head_ratio * *trace.narrow_baseline, {255, 200, 60});
    }
    if (trace.boundary_s) {
        plot.vertical(*trace.boundary_s, kBoundaryColor);
    }
    text(plot.image, "Raw/median/shaft/head/touch: gray/yellow/green/orange/cyan", {58, 310}, 0.3);
    return plot.image;
}

const EdgeTrace* selectEdge(const MeasurementTrace& trace, bool accepted) {
    const EdgeTrace* fallback = nullptr;
    for (const auto& edge : trace.edges) {
        if (edge.offset.has_value() != accepted) {
            continue;
        }
        if (!fallback) {
            fallback = &edge;
        }
        if (accepted || edge.reason == "competing_peak") {
            return &edge;
        }
        if (!edge.gradients.empty() && (!fallback || fallback->gradients.empty())) {
            fallback = &edge;
        }
    }
    return fallback;
}

cv::Mat edgePlot(const EdgeTrace* edge, const Config& config, bool accepted) {
    if (!edge) {
        return tile(accepted ? "EDGE | no accepted sample" : "EDGE | no rejected sample");
    }
    double maximum = config.gradient_min;
    if (!edge->gradients.empty()) {
        maximum = std::max(maximum, *std::max_element(edge->gradients.begin(), edge->gradients.end()));
    }
    Plot plot("EDGE | " + edge->role + " | " + edge->reason,
               -edge->search_radius - 1, edge->search_radius + 1, maximum * 1.2,
               "offset from coarse edge (px)", "Gradient: left axis; intensity: right axis");
    std::vector<double> intensity_positions;
    for (std::size_t i = 0; i < edge->intensities.size(); ++i) {
        intensity_positions.push_back(static_cast<int>(i) - edge->search_radius - 1);
    }
    std::vector<double> gradient_positions;
    for (std::size_t i = 0; i < edge->gradients.size(); ++i) {
        gradient_positions.push_back(static_cast<int>(i) - edge->search_radius);
    }
    std::vector<double> scaled_intensities;
    for (double value : edge->intensities) {
        scaled_intensities.push_back(value / 255 * plot.y_max);
    }
    plot.series(intensity_positions, scaled_intensities, {255, 180, 60});
    for (int i = 0; i <= 2; ++i) {
        text(plot.image, std::to_string(cvRound(255.0 * i / 2)),
             {plot.area.br().x + 3, plot.point(plot.x_max, plot.y_max * i / 2).y + 4}, 0.3, {255, 180, 60});
    }
    plot.series(gradient_positions, edge->gradients, kProfileColor);
    plot.horizontal(config.gradient_min, kThresholdColor);
    plot.vertical(0, kRawColor);
    if (edge->offset) {
        plot.vertical(*edge->offset, kBaselineColor);
    }
    if (edge->peak >= 0) {
        // 空心圆是峰两侧邻点；只有通过峰检查才进入拟合，实心红点是最大峰。
        for (int i = std::max(0, edge->peak - 1);
             i <= std::min(static_cast<int>(edge->gradients.size()) - 1, edge->peak + 1); ++i) {
            cv::circle(plot.image, plot.point(gradient_positions[i], edge->gradients[i]),
                       4, {0, 0, 255}, i == edge->peak ? cv::FILLED : 1, cv::LINE_AA);
        }
    }
    if (edge->offset && edge->delta && edge->peak > 0
        && edge->peak + 1 < static_cast<int>(edge->gradients.size())) {
        const double left = edge->gradients[edge->peak - 1];
        const double mid = edge->gradients[edge->peak];
        const double right = edge->gradients[edge->peak + 1];
        std::vector<double> positions;
        std::vector<double> parabola;
        for (int i = 0; i <= 20; ++i) {
            const double delta = -1 + i / 10.0;
            positions.push_back(gradient_positions[edge->peak] + delta);
            parabola.push_back(mid + 0.5 * (right - left) * delta
                              + 0.5 * (left - 2 * mid + right) * delta * delta);
        }
        plot.series(positions, parabola, kBoundaryColor);
    }
    if (edge->competing_peak) {
        int i = *edge->competing_peak;
        cv::circle(plot.image, plot.point(gradient_positions[i], edge->gradients[i]),
                   6, kBoundaryColor, 2, cv::LINE_AA);
    }
    text(plot.image, "Cyan: I (right)  Yellow: gradient (left)  Purple: fit", {58, 310}, 0.32);
    return plot.image;
}

cv::Mat localMaskTile(const MeasurementTrace& trace) {
    if (trace.local_mask.empty()) {
        return tile("LOCAL MASK | not recorded");
    }
    cv::Mat color;
    cv::cvtColor(trace.local_mask, color, cv::COLOR_GRAY2BGR);
    if (trace.boundary_s) {
        const int x = cvRound(*trace.boundary_s - trace.local_s_origin);
        cv::line(color, {x, 0}, {x, color.rows - 1}, kBoundaryColor, 1);
    }
    auto output = imageTile(color, "LOCAL MASK | horizontal s; magenta = boundary", true);
    text(output, "s origin = " + number(trace.local_s_origin)
         + "  t origin = " + number(trace.local_t_origin), {12, 332}, 0.34);
    return output;
}

cv::Mat objectTile(const cv::Mat& image, const Object& object, const MeasurementTrace& trace) {
    cv::Mat overlay = image.clone();
    cv::drawContours(overlay, std::vector<std::vector<cv::Point>>{object.contour}, 0, kBaselineColor, 2);
    cv::circle(overlay, object.center, 3, {0, 0, 255}, cv::FILLED);
    cv::line(overlay, object.length_start, object.length_end, kProfileColor, 2, cv::LINE_AA);
    cv::arrowedLine(overlay, object.center, toImage({40, 0}, object), {255, 120, 50}, 2, cv::LINE_AA);
    if (object.head_boundary && object.pixel.width) {
        cv::line(overlay, toImage({*object.head_boundary, -*object.pixel.width / 2}, object),
                 toImage({*object.head_boundary, *object.pixel.width / 2}, object), kBoundaryColor, 2);
    }
    for (bool accepted : {true, false}) {
        const auto* edge = selectEdge(trace, accepted);
        if (edge) {
            const auto color = accepted ? kBaselineColor : cv::Scalar(0, 0, 255);
            cv::Point2d delta(edge->normal[0] * edge->search_radius, edge->normal[1] * edge->search_radius);
            cv::line(overlay, edge->point - delta, edge->point + delta, color, 2, cv::LINE_AA);
            cv::circle(overlay, edge->point, 4, color, 1, cv::LINE_AA);
        }
    }
    cv::Rect bounds = cv::boundingRect(object.contour);
    bounds = cv::Rect(bounds.x - 20, bounds.y - 20, bounds.width + 40, bounds.height + 40)
             & cv::Rect(0, 0, image.cols, image.rows);
    return imageTile(overlay(bounds), "OBJECT #" + std::to_string(object.id) + " | actual sampling locations");
}

cv::Mat summaryTile(const Object& object, const MeasurementTrace& trace, const Config& config) {
    cv::Mat output = tile("DECISIONS | orange = threshold; magenta = boundary");
    std::vector<std::string> lines{
        "axis = " + number(object.angle) + " deg (undirected)",
        "shaft baseline = " + jsonNumber(trace.shaft_baseline) + " px",
        "head threshold = " + jsonNumber(trace.head_attempted && trace.shaft_baseline
            ? std::optional<double>(*trace.shaft_baseline * config.head_ratio) : std::nullopt) + " px",
        "narrow / touching threshold = " + jsonNumber(trace.narrow_baseline) + " / "
            + jsonNumber(trace.head_attempted && trace.narrow_baseline
                ? std::optional<double>(*trace.narrow_baseline * config.head_ratio) : std::nullopt),
        "end medians: " + jsonNumber(trace.negative_end_width) + " / " + jsonNumber(trace.positive_end_width),
        "head eligible = " + std::string(trace.head_attempted ? "yes" : "no")
            + "; head end = " + std::to_string(trace.head_end),
        "boundary index / s = " + (trace.boundary_index ? std::to_string(*trace.boundary_index) : "null")
            + " / " + jsonNumber(trace.boundary_s),
        "refined slices = " + std::to_string(trace.refined_slices) + " / " + std::to_string(trace.attempted_slices),
        "subpixel = " + object.subpixel_status + "; reliable = " + (object.reliable ? "yes" : "no")};
    int y = 55;
    for (const auto& line : lines) {
        text(output, line, {12, y}, 0.39);
        y += 22;
    }
    for (const auto& issue : object.issues) {
        text(output, issue, {12, y}, 0.36, kThresholdColor);
        y += 19;
        if (y > 325) {
            break; // 完整原因保留在 observations.md 和 measurement.json。
        }
    }
    return output;
}

cv::Mat candidateTile(const cv::Mat& image, const CandidateTrace& trace, bool components) {
    cv::Mat overlay = image.clone();
    if (components) {
        for (const auto& component : trace.components) {
            cv::rectangle(overlay, component.bounds,
                          component.accepted ? kBaselineColor : cv::Scalar(0, 0, 255), 1);
        }
    } else {
        for (const auto& contour : trace.contours) {
            cv::drawContours(overlay, std::vector<std::vector<cv::Point>>{contour.contour}, 0,
                             contour.accepted ? kBaselineColor : cv::Scalar(0, 0, 255), 2);
        }
    }
    return imageTile(overlay, components ? "COMPONENT AREA | green keep / red reject"
                                        : "EXTERNAL CONTOURS | green keep / red reject");
}

cv::Mat segmentationPanel(const cv::Mat& image, const Config& config, const SegmentationTrace& trace,
                           const CandidateTrace& candidates) {
    cv::Mat output(3 * kTileHeight, 4 * kTileWidth, CV_8UC3);
    place(output, imageTile(image, "SOURCE BGR | " + config.segmentation), 0, 0);
    place(output, imageTile(trace.lab_channels[0], "Lab L (8-bit)"), 0, 1);
    place(output, imageTile(trace.lab_channels[1], "Lab a (8-bit)"), 0, 2);
    place(output, imageTile(trace.lab_channels[2], "Lab b (8-bit)"), 0, 3);
    place(output, imageTile(trace.color_condition, "b <= " + number(config.lab_b_max), true), 1, 0);
    place(output, imageTile(trace.dark_condition, "L <= " + number(config.lab_dark_l), true), 1, 1);
    place(output, imageTile(trace.threshold_mask, "THRESHOLD | before morphology", true), 1, 2);
    place(output, imageTile(trace.opened_mask, "AFTER OPEN | kernel = " + std::to_string(config.morph_open), true), 1, 3);
    place(output, imageTile(trace.final_mask, "AFTER CLOSE | final mask", true), 2, 0);
    place(output, imageTile(candidates.kept_mask, "AFTER COMPONENT AREA FILTER", true), 2, 1);
    place(output, candidateTile(image, candidates, true), 2, 2);
    place(output, candidateTile(image, candidates, false), 2, 3);
    return output;
}

void saveImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write diagnostics: " + path.string());
    }
}

void saveCandidates(const CandidateTrace& trace, const fs::path& path) {
    auto output = openOutput(path);
    output << "stage,index,pixel_area,contour_area,hull_area,solidity,accepted,reason\n";
    for (const auto& component : trace.components) {
        output << "component," << component.label << ',' << component.pixel_area
               << ",,,," << component.accepted << ',' << component.reason << '\n';
    }
    for (std::size_t i = 0; i < trace.contours.size(); ++i) {
        const auto& contour = trace.contours[i];
        output << "contour," << i + 1 << ",," << jsonNumber(contour.area) << ','
               << jsonNumber(contour.hull_area) << ',' << jsonNumber(contour.solidity) << ','
               << contour.accepted << ',' << contour.reason << '\n';
    }
}

void saveProfile(const MeasurementTrace& trace, const Config& config, const fs::path& path) {
    auto output = openOutput(path);
    output << "index,s_px,low_px,high_px,raw_width_px,median_width_px,shaft_baseline_px,head_threshold_px,narrow_baseline_px,touching_threshold_px,boundary\n";
    for (std::size_t i = 0; i < trace.slices.size(); ++i) {
        const auto& slice = trace.slices[i];
        const std::optional<double> smooth = i < trace.profile.size()
            ? std::optional<double>(trace.profile[i]) : std::nullopt;
        output << i << ',' << jsonNumber(slice.s) << ',' << jsonNumber(slice.low) << ','
               << jsonNumber(slice.high) << ',' << jsonNumber(slice.width) << ',' << jsonNumber(smooth)
               << ',' << jsonNumber(trace.shaft_baseline) << ','
               << jsonNumber(trace.head_attempted && trace.shaft_baseline ? std::optional<double>(*trace.shaft_baseline * config.head_ratio)
                                                  : std::nullopt)
               << ',' << jsonNumber(trace.narrow_baseline)
               << ',' << jsonNumber(trace.head_attempted && trace.narrow_baseline
                    ? std::optional<double>(*trace.narrow_baseline * config.head_ratio) : std::nullopt)
               << ',' << (trace.boundary_index && static_cast<int>(i) == *trace.boundary_index) << '\n';
    }
}

void saveEdges(const MeasurementTrace& trace, const fs::path& path) {
    auto output = openOutput(path);
    output << "edge,role,reason,x,y,nx,ny,sample_offset_px,intensity,gradient,is_peak,is_peak_neighbor,is_competing,refined_offset_px\n";
    for (std::size_t index = 0; index < trace.edges.size(); ++index) {
        const auto& edge = trace.edges[index];
        // 出界且第一个点就失败的记录也保留，不能因为没有曲线而消失。
        for (std::size_t i = 0; i < std::max<std::size_t>(1, edge.intensities.size()); ++i) {
            int gradient_index = static_cast<int>(i) - 1;
            bool has_gradient = gradient_index >= 0 && gradient_index < static_cast<int>(edge.gradients.size());
            output << index << ',' << edge.role << ',' << edge.reason << ','
                   << jsonNumber(edge.point.x) << ',' << jsonNumber(edge.point.y) << ','
                   << jsonNumber(edge.normal[0]) << ',' << jsonNumber(edge.normal[1]) << ','
                   << static_cast<int>(i) - edge.search_radius - 1 << ','
                   << jsonNumber(i < edge.intensities.size() ? std::optional<double>(edge.intensities[i]) : std::nullopt)
                   << ',' << jsonNumber(has_gradient ? std::optional<double>(edge.gradients[gradient_index]) : std::nullopt)
                   << ',' << (has_gradient && gradient_index == edge.peak)
                   << ',' << (has_gradient && edge.peak >= 0 && std::abs(gradient_index - edge.peak) <= 1)
                   << ',' << (has_gradient && edge.competing_peak && gradient_index == *edge.competing_peak)
                   << ',' << jsonNumber(edge.offset) << '\n';
        }
    }
}
} // namespace

void saveDiagnostics(const cv::Mat& image, const Config& config,
                     const SegmentationTrace& segmentation, const PipelineTrace& pipeline,
                     const std::vector<Object>& objects, const fs::path& output) {
    if (image.empty() || image.type() != CV_8UC3 || pipeline.objects.size() != objects.size()) {
        throw std::runtime_error("Diagnostics require BGR image and aligned object traces");
    }
    fs::create_directories(output);
    saveImage(output / "segmentation.png", segmentationPanel(image, config, segmentation, pipeline.candidates));
    saveCandidates(pipeline.candidates, output / "candidates.csv");
    auto notes = openOutput(output / "observations.md");
    notes << "# 测量观察记录\n\n本记录读取同一次计算的旁路数据，没有重新计算尺寸。\n\n"
          << "分割：`" << config.segmentation << "`，主轴：`" << config.axis << "`，目标数：" << objects.size()
          << "。阈值 L/b：" << config.lab_dark_l << '/' << config.lab_b_max
          << "，head_ratio：" << config.head_ratio << "。\n\n"
          << "[分割与筛选面板](segmentation.png)，[候选记录](candidates.csv)。红色候选已被过滤；绿色被保留。\n\n"
          << "对象面板：灰色原始宽度、黄色中位数剖面、绿色杆宽、橙色头部阈值、紫色分界、青色虚线粘连检查阈值。"
          << "梯度图横轴为相对粗边缘的采样偏移；局部宽度图横轴为 s 坐标，单位均为 px。\n\n";
    for (std::size_t i = 0; i < objects.size(); ++i) {
        const auto& object = objects[i];
        const auto& trace = pipeline.objects[i];
        if (trace.object_id != object.id || trace.center != object.center) {
            throw std::runtime_error("Diagnostics object/trace order mismatch");
        }
        const std::string stem = "object_" + std::to_string(object.id);
        cv::Mat panel(3 * kTileHeight, 2 * kTileWidth, CV_8UC3);
        place(panel, objectTile(image, object, trace), 0, 0);
        place(panel, localMaskTile(trace), 0, 1);
        place(panel, profilePlot(trace, config), 1, 0);
        place(panel, summaryTile(object, trace, config), 1, 1);
        place(panel, edgePlot(selectEdge(trace, true), config, true), 2, 0);
        place(panel, edgePlot(selectEdge(trace, false), config, false), 2, 1);
        saveImage(output / (stem + ".png"), panel);
        saveProfile(trace, config, output / (stem + ".profile.csv"));
        saveEdges(trace, output / (stem + ".edges.csv"));
        notes << "## 目标 " << object.id << "\n\n[过程面板](" << stem << ".png)，[宽度剖面]("
              << stem << ".profile.csv)，[全部采样记录](" << stem << ".edges.csv)。\n\n"
              << "质心：(" << jsonNumber(object.center.x) << ", " << jsonNumber(object.center.y)
              << ")；无向主轴：" << jsonNumber(object.angle) << "°。\n\n"
              << "杆宽基线：" << jsonNumber(trace.shaft_baseline)
              << "；低四分位基线：" << jsonNumber(trace.narrow_baseline)
              << "；两端中位数：" << jsonNumber(trace.negative_end_width) << " / " << jsonNumber(trace.positive_end_width)
              << "；头部端：" << trace.head_end << "；分界索引／s："
              << (trace.boundary_index ? std::to_string(*trace.boundary_index) : "null")
              << " / " << jsonNumber(trace.boundary_s) << "。\n\n"
              << "截面细化成功／尝试：" << trace.refined_slices << '/' << trace.attempted_slices
              << "；整体细化状态：`" << object.subpixel_status << "`；几何可靠："
              << (object.reliable ? "是" : "否") << "。\n\n质量原因：";
        for (const auto& issue : object.issues) {
            notes << '`' << issue << "` ";
        }
        notes << "\n\n";
    }
}
} // namespace screw
