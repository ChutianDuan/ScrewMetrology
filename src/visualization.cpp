#include "screw/visualization.hpp"
#include "screw/locator.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace screw {
namespace {

cv::Scalar objectColor(const Object& object) {
    // 橙色先提示几何不可靠，红色表示 NG；默认展示模式也可画绿色框，不能据颜色判 OK。
    if (!object.reliable) {
        return {0, 160, 255};
    }
    if (object.status == "NG") {
        return {0, 0, 255};
    }
    return {0, 210, 0};
}

void drawGeometry(cv::Mat& output, const Object& object, const cv::Scalar& color) {
    cv::Point2f corners[4];
    object.box.points(corners);
    for (int i = 0; i < 4; ++i) {
        cv::line(output, corners[i], corners[(i + 1) % 4], color, 2, cv::LINE_AA);
    }
    cv::drawContours(output, std::vector<std::vector<cv::Point>>{object.contour}, 0, {255, 160, 0}, 1);
    cv::circle(output, object.center, 3, {0, 0, 255}, cv::FILLED);
    cv::arrowedLine(output, object.center, toImage({40, 0}, object), {255, 0, 0}, 2, cv::LINE_AA);
    cv::arrowedLine(output, object.center, toImage({0, 25}, object), {255, 0, 255}, 2, cv::LINE_AA);
    cv::line(output, object.length_start, object.length_end, {0, 255, 255}, 1, cv::LINE_AA);
    if (object.head_boundary && object.pixel.width) {
        cv::line(output, toImage({*object.head_boundary, -*object.pixel.width / 2}, object),
                 toImage({*object.head_boundary, *object.pixel.width / 2}, object), {255, 0, 255}, 1);
    }
}

std::vector<std::string> labelLines(const Object& object) {
    const Dimensions dimensions = preferredDimensions(object);
    std::ostringstream first;
    std::ostringstream second;
    first << '#' << object.id << "  " << std::fixed << std::setprecision(1)
          << dimensions.length.value_or(0) << " x " << dimensions.width.value_or(0) << " px";
    const std::string status = object.status == "NOT_EVALUATED" ? "N/E" : object.status;
    second << std::fixed << std::setprecision(1) << object.angle << " deg | " << status;
    if (!object.reliable) {
        second << " | CHECK";
    }
    std::vector<std::string> lines{first.str(), second.str()};
    if (object.mm.length && object.mm.width) {
        std::ostringstream millimeters;
        millimeters << std::fixed << std::setprecision(2) << *object.mm.length
                    << " x " << *object.mm.width << " mm";
        lines.push_back(millimeters.str());
    }
    return lines;
}

cv::Rect chooseLabelPosition(const Object& object, const ImageResult& result, cv::Size image_size,
                             cv::Size label_size, int line_height, const std::vector<cv::Rect>& labels) {
    const cv::Rect bounds = object.box.boundingRect();
    cv::Rect chosen(0, 0, label_size.width, label_size.height);
    double best = std::numeric_limits<double>::max();
    // 在目标附近枚举标签位置，优先避免标签互相遮挡、其次避免压住工件，并限制在图内。
    // cost 是展示位置的代价，不是测量误差或置信度；引线关联移动后的标签与目标。
    for (int dy = -3; dy <= 3; ++dy) {
        for (int side = 0; side < 4; ++side) {
            int x = bounds.x;
            if (side == 1) {
                x = bounds.br().x + 8;
            } else if (side == 2) {
                x = bounds.x - label_size.width - 8;
            }
            int y = static_cast<int>(object.center.y) - label_size.height / 2;
            if (side == 0) {
                y = bounds.y - label_size.height - 8;
            } else if (side == 3) {
                y = bounds.br().y + 8;
            }
            y += dy * line_height;
            const cv::Rect tag(std::clamp(x, 0, image_size.width - label_size.width),
                               std::clamp(y, 0, image_size.height - label_size.height),
                               label_size.width, label_size.height);
            double cost = cv::norm(cv::Point2d(tag.x + tag.width / 2.0, tag.y + tag.height / 2.0) - object.center);
            for (const auto& other : labels) {
                cost += (tag & other).area() * 50;
            }
            for (const auto& target : result.objects) {
                cost += (tag & target.box.boundingRect()).area() * 2;
            }
            if (cost < best) {
                best = cost;
                chosen = tag;
            }
        }
    }
    return chosen;
}

void drawLabel(cv::Mat& output, const Object& object, const std::vector<std::string>& lines,
               const cv::Rect& bounds, const cv::Scalar& color, double font, int line_height) {
    const cv::Point connector(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    cv::line(output, connector, object.center, {70, 70, 70}, 1, cv::LINE_AA);
    cv::rectangle(output, bounds, {25, 25, 25}, cv::FILLED);
    cv::rectangle(output, bounds, color, 1);
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::putText(output, lines[i], {bounds.x + 6, bounds.y + static_cast<int>(i + 1) * line_height - 4},
                    cv::FONT_HERSHEY_SIMPLEX, font, {255, 255, 255}, 1, cv::LINE_AA);
    }
}

} // namespace

cv::Mat visualize(const cv::Mat& image, const ImageResult& result) {
    cv::Mat output = image.clone();
    const double font = std::clamp(image.cols / 3200.0, 0.4, 0.65);
    const int line_height = static_cast<int>(24 * font) + 9;
    std::vector<cv::Rect> labels;
    for (const auto& object : result.objects) {
        const cv::Scalar color = objectColor(object);
        drawGeometry(output, object, color);
        const auto lines = labelLines(object);
        int label_width = 0;
        int baseline = 0;
        for (const auto& line : lines) {
            label_width = std::max(label_width,
                cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, font, 1, &baseline).width + 12);
        }
        label_width = std::min(label_width, image.cols);
        const int label_height = std::min(static_cast<int>(lines.size()) * line_height + 4, image.rows);
        const cv::Rect chosen = chooseLabelPosition(object, result, image.size(),
                                                    {label_width, label_height}, line_height, labels);
        labels.push_back(chosen);
        drawLabel(output, object, lines, chosen, color, font, line_height);
    }
    return output;
}

} // namespace screw
