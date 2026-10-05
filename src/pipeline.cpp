#include "screw/pipeline.hpp"
#include "screw/calibration.hpp"
#include "screw/locator.hpp"
#include "screw/metrology.hpp"
#include "screw/segmentation.hpp"
#include <algorithm>
#include <tuple>
#include <utility>

namespace screw {
namespace {
std::vector<Object> measureContoursImpl(const std::vector<std::vector<cv::Point>>& contours,
                                       const cv::Mat& gray, const Config& config,
                                       const std::optional<Scale>& scale,
                                       const std::optional<Recipe>& recipe, bool single_spec,
                                       PipelineTrace* trace) {
    std::vector<Object> objects;
    for (const auto& contour : contours) {
        Object object = locate(contour, gray.size(), config);
        if (trace) {
            MeasurementTrace measurement;
            measureObject(object, gray, config, &measurement);
            trace->objects.push_back(std::move(measurement));
        } else {
            measureObject(object, gray, config);
        }
        applyScaleAndRecipe(object, gray.size(), scale, recipe, single_spec);
        objects.push_back(std::move(object));
    }
    // 本图编号不是跨帧追踪 ID；保持两个入口原有的排序方式。
    std::sort(objects.begin(), objects.end(), [](const Object& first, const Object& second) {
        return std::tie(first.center.y, first.center.x) < std::tie(second.center.y, second.center.x);
    });
    if (trace) {
        // 初始顺序和比较规则都与 objects 相同，等质心时也保持相同的排序行为。
        std::sort(trace->objects.begin(), trace->objects.end(), [](const MeasurementTrace& first,
                                                                 const MeasurementTrace& second) {
            return std::tie(first.center.y, first.center.x) < std::tie(second.center.y, second.center.x);
        });
    }
    for (std::size_t i = 0; i < objects.size(); ++i) {
        objects[i].id = static_cast<int>(i) + 1;
        if (trace) {
            trace->objects[i].object_id = objects[i].id;
        }
    }
    return objects;
}
} // namespace

std::vector<Object> measureContours(const std::vector<std::vector<cv::Point>>& contours,
                                   const cv::Mat& gray, const Config& config,
                                   const std::optional<Scale>& scale,
                                   const std::optional<Recipe>& recipe, bool single_spec) {
    return measureContoursImpl(contours, gray, config, scale, recipe, single_spec, nullptr);
}

std::vector<Object> measureObjects(const cv::Mat& mask, const cv::Mat& gray,
                                   const Config& config, const std::optional<Scale>& scale,
                                   const std::optional<Recipe>& recipe, bool single_spec) {
    return measureObjects(mask, gray, config, scale, recipe, single_spec, nullptr);
}

std::vector<Object> measureObjects(const cv::Mat& mask, const cv::Mat& gray,
                                   const Config& config, const std::optional<Scale>& scale,
                                   const std::optional<Recipe>& recipe, bool single_spec,
                                   PipelineTrace* trace) {
    if (trace) {
        *trace = {};
    }
    const auto contours = candidates(mask, config, trace ? &trace->candidates : nullptr);
    return measureContoursImpl(contours, gray, config, scale, recipe, single_spec, trace);
}
} // namespace screw
