#include "screw/metrology.hpp"
#include "screw/locator.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace screw {
namespace {
constexpr int kLocalPadding = 2;
constexpr std::size_t kMinimumProfileSlices = 10;
constexpr std::size_t kSmoothingRadius = 2;
constexpr int kBoundaryMinimumRun = 3;
constexpr double kMinimumRefinedFraction = 0.6;
constexpr double kCompetingPeakRatio = 0.8;
constexpr double kMinimumShaftBaseline = 2;
constexpr double kMinimumElongation = 1.5;
constexpr double kInteriorWidthRatio = 1.7;
constexpr int kDistantPeakSeparation = 2;
constexpr double kExtremumCoordinateTolerance = 1e-5;
constexpr double kMinimumNegativeCurvature = -1e-6;
constexpr double kMaximumPeakOffset = 0.75;

// 投影范围来自原始轮廓，不能用旋转栅格的宽高替代。
struct Projection {
    double s_min = std::numeric_limits<double>::max();
    double s_max = -std::numeric_limits<double>::max();
    double t_min = std::numeric_limits<double>::max();
    double t_max = -std::numeric_limits<double>::max();
    cv::Point2d length_start;
    cv::Point2d length_end;
};

// 一个局部截面：s 是主轴位置，low/high 是横向边界，单位均为像素。
using Slice = ProfileSlice;

struct HeadShaftAnalysis {
    int head_end = 0;       // -1/+1：局部主轴负端/正端，不是原图左右。
    int boundary = -1;     // 截面索引；保存到 Object 时再转换成局部 s 坐标。
};

struct HeadEdges {
    cv::Point2d low;
    cv::Point2d high;
};

double median(std::vector<double> values) {
    // 中位数不易受孤立毛刺影响；nth_element 无须将整个序列排序。
    if (values.empty()) {
        return 0;
    }
    auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    double result = *middle;
    if (values.size() % 2 == 0) {
        result = (result + *std::max_element(values.begin(), middle)) / 2;
    }
    return result;
}

std::optional<double> sampleGray(const cv::Mat& gray, cv::Point2d point) {
    // 双线性采样需要四邻域；邻域出界时不能继续拟合。
    if (point.x < 0 || point.y < 0 || point.x >= gray.cols - 1 || point.y >= gray.rows - 1) {
        return std::nullopt;
    }
    int x = static_cast<int>(std::floor(point.x));
    int y = static_cast<int>(std::floor(point.y));
    double dx = point.x - x;
    double dy = point.y - y;
    return (1 - dy) * ((1 - dx) * gray.at<unsigned char>(y, x)
                      + dx * gray.at<unsigned char>(y, x + 1))
         + dy * ((1 - dx) * gray.at<unsigned char>(y + 1, x)
                 + dx * gray.at<unsigned char>(y + 1, x + 1));
}

Projection projectContour(Object& object) {
    Projection projection;
    for (auto point : object.contour) {
        auto local = toLocal(point, object);
        if (local.x < projection.s_min) {
            projection.s_min = local.x;
            projection.length_start = local;
        }
        if (local.x > projection.s_max) {
            projection.s_max = local.x;
            projection.length_end = local;
        }
        projection.t_min = std::min(projection.t_min, local.y);
        projection.t_max = std::max(projection.t_max, local.y);
    }
    // 边界像素中心的投影范围不统一加 1；连续灰度边缘距离留给细化。
    object.pixel.length = projection.s_max - projection.s_min;
    object.pixel.width = projection.t_max - projection.t_min;
    object.length_start = toImage(projection.length_start, object);
    object.length_end = toImage(projection.length_end, object);
    return projection;
}

std::vector<Slice> buildWidthProfile(const Object& object, const Projection& projection,
                                    MeasurementTrace* trace) {
    int width = static_cast<int>(std::ceil(projection.s_max - projection.s_min))
                + 2 * kLocalPadding + 1;
    int height = static_cast<int>(std::ceil(projection.t_max - projection.t_min))
                 + 2 * kLocalPadding + 1;
    cv::Rect bounds = cv::boundingRect(object.contour);
    cv::Mat component = cv::Mat::zeros(bounds.size(), CV_8U);
    std::vector<cv::Point> shifted;
    for (auto point : object.contour) {
        shifted.push_back(point - bounds.tl());
    }
    // 只填充当前外轮廓；已经合并的粘连工件不会在这里自动拆开。
    cv::drawContours(component, std::vector<std::vector<cv::Point>>{shifted}, 0, 255, cv::FILLED);
    cv::Point2d local_origin = toLocal(bounds.tl(), object);
    // 栅格坐标 u=s-s_min+pad、v=t-t_min+pad；留白防止边界被 ROI 裁切。
    cv::Matx23d transform(
        object.axis[0], object.axis[1], local_origin.x - projection.s_min + kLocalPadding,
        -object.axis[1], object.axis[0], local_origin.y - projection.t_min + kLocalPadding);
    cv::Mat local_mask;
    cv::warpAffine(component, local_mask, transform, {width, height},
                   cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

    if (trace) {
        trace->local_mask = local_mask.clone();
        trace->local_s_origin = projection.s_min - kLocalPadding;
        trace->local_t_origin = projection.t_min - kLocalPadding;
    }
    std::vector<Slice> slices;
    for (int x = 0; x < local_mask.cols; ++x) {
        int low = local_mask.rows;
        int high = -1;
        for (int y = 0; y < local_mask.rows; ++y) {
            if (local_mask.at<unsigned char>(y, x)) {
                low = std::min(low, y);
                high = y;
            }
        }
        // 宽度是边界坐标差，会跨过内孔，不是白色像素数。
        if (high > low) {
            slices.push_back({x + projection.s_min - kLocalPadding,
                              low + projection.t_min - kLocalPadding,
                              high + projection.t_min - kLocalPadding,
                              static_cast<double>(high - low)});
        }
    }
    return slices;
}

std::vector<double> smoothProfile(const std::vector<Slice>& slices) {
    std::vector<double> profile;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        std::vector<double> window;
        std::size_t begin = i > kSmoothingRadius ? i - kSmoothingRadius : 0;
        std::size_t end = std::min(slices.size(), i + kSmoothingRadius + 1);
        for (std::size_t j = begin; j < end; ++j) {
            window.push_back(slices[j].width);
        }
        profile.push_back(median(window));
    }
    return profile;
}

HeadShaftAnalysis analyzeHeadShaft(Object& object, const std::vector<Slice>& slices,
                                  const Config& config, MeasurementTrace* trace) {
    auto profile = smoothProfile(slices);
    std::vector<double> middle;
    for (std::size_t i = slices.size() * 3 / 10; i < slices.size() * 7 / 10; ++i) {
        middle.push_back(profile[i]);
    }
    double shaft_baseline = median(middle);

    // 第二个粘连工件可能撑宽中段：另取较低四分位检查两端异常。
    std::vector<double> baseline;
    for (std::size_t i = slices.size() / 5; i < slices.size() * 4 / 5; ++i) {
        baseline.push_back(profile[i]);
    }
    std::sort(baseline.begin(), baseline.end());
    double narrow_baseline = baseline[baseline.size() / 4];
    if (trace) {
        trace->profile = profile;
        trace->shaft_baseline = shaft_baseline;
        trace->narrow_baseline = narrow_baseline;
    }
    HeadShaftAnalysis analysis;
    if (!(shaft_baseline > kMinimumShaftBaseline
          && *object.pixel.length > kMinimumElongation * *object.pixel.width
          && object.eigen_ratio >= config.axis_ratio)) {
        return analysis;
    }

    if (trace) {
        trace->head_attempted = true;
    }
    std::vector<double> first;
    std::vector<double> last;
    for (std::size_t i = slices.size() / 20; i < slices.size() / 5; ++i) {
        first.push_back(profile[i]);
        last.push_back(profile[slices.size() - 1 - i]);
    }
    const double negative_end_width = median(first);
    const double positive_end_width = median(last);
    bool at_left = negative_end_width > config.head_ratio * shaft_baseline;
    bool at_right = positive_end_width > config.head_ratio * shaft_baseline;
    if (trace) {
        trace->negative_end_width = negative_end_width;
        trace->positive_end_width = positive_end_width;
        trace->negative_end_wide = at_left;
        trace->positive_end_wide = at_right;
    }
    if (negative_end_width > config.head_ratio * narrow_baseline
        && positive_end_width > config.head_ratio * narrow_baseline) {
        object.reliable = false;
        object.issues.push_back("possible_touching_two_ends");
    }
    if (at_left != at_right) {
        analysis.head_end = at_left ? -1 : 1;
        int run = 0;
        for (int step = 0; step < static_cast<int>(slices.size()) / 2; ++step) {
            int i = at_left ? step : static_cast<int>(slices.size()) - 1 - step;
            run = profile[i] <= shaft_baseline * config.head_ratio ? run + 1 : 0;
            // 连续三个截面回到杆宽范围，取这段的第一个点作为分界。
            if (run >= kBoundaryMinimumRun && step > static_cast<int>(slices.size()) / 20) {
                analysis.boundary = at_left ? i - (kBoundaryMinimumRun - 1)
                                            : i + (kBoundaryMinimumRun - 1);
                break;
            }
        }
    }
    int interior_wide = 0;
    for (std::size_t i = slices.size() * 3 / 10; i < slices.size() * 7 / 10; ++i) {
        if (profile[i] > shaft_baseline * kInteriorWidthRatio) {
            ++interior_wide;
        }
    }
    if ((at_left && at_right) || interior_wide > static_cast<int>(slices.size()) / 10) {
        object.reliable = false;
        object.issues.push_back("possible_touching_profile");
    }
    return analysis;
}

bool isHeadSlice(std::size_t index, const HeadShaftAnalysis& analysis) {
    if (analysis.head_end < 0) {
        return static_cast<int>(index) < analysis.boundary;
    }
    return static_cast<int>(index) > analysis.boundary;
}

bool isShaftSlice(std::size_t index, std::size_t count, const HeadShaftAnalysis& analysis) {
    return !isHeadSlice(index, analysis) && index >= count / 5 && index < count * 4 / 5;
}

HeadEdges measureHeadShaft(Object& object, const Projection& projection,
                          const std::vector<Slice>& slices, const HeadShaftAnalysis& analysis) {
    HeadEdges edges;
    if (analysis.boundary < 0) {
        object.issues.push_back("head_shaft_unavailable");
        return edges;
    }
    double boundary_s = slices[analysis.boundary].s;
    object.head_boundary = boundary_s;
    object.pixel.head_length = analysis.head_end < 0
        ? boundary_s - projection.s_min : projection.s_max - boundary_s;
    std::vector<double> shaft_widths;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        if (isShaftSlice(i, slices.size(), analysis)) {
            shaft_widths.push_back(slices[i].width);
        }
    }
    double head_min = std::numeric_limits<double>::max();
    double head_max = -head_min;
    for (auto point : object.contour) {
        auto local = toLocal(point, object);
        bool in_head = analysis.head_end < 0 ? local.x <= boundary_s : local.x >= boundary_s;
        if (in_head) {
            if (local.y < head_min) {
                head_min = local.y;
                edges.low = local;
            }
            if (local.y > head_max) {
                head_max = local.y;
                edges.high = local;
            }
        }
    }
    if (head_max > head_min) {
        object.pixel.head_width = head_max - head_min;
    }
    object.pixel.shaft_width = median(shaft_widths);
    return edges;
}

void fallbackSubpixel(Object& object, const std::string& issue) {
    object.refined = {};
    object.subpixel_status = "fallback";
    object.issues.push_back(issue);
}

// 记录实际调用的采样与拒绝原因，绘图阶段不会再调用 refineEdge。
std::optional<double> refineObservedEdge(const cv::Mat& gray, cv::Point2d point, cv::Vec2d normal,
                                         const Config& config, MeasurementTrace* trace,
                                         const char* role) {
    if (!trace) {
        return refineEdge(gray, point, normal, config);
    }
    EdgeTrace edge;
    auto offset = refineEdge(gray, point, normal, config, &edge);
    edge.role = role;
    trace->edges.push_back(std::move(edge));
    return offset;
}

void refineDimensions(Object& object, const cv::Mat& gray, const Config& config,
                      const Projection& projection, const std::vector<Slice>& slices,
                      const HeadShaftAnalysis& analysis, const HeadEdges& head_edges,
                      MeasurementTrace* trace) {
    cv::Vec2d transverse{-object.axis[1], object.axis[0]};
    std::vector<double> refined_widths;
    std::vector<double> shaft_widths;
    int attempted = 0;
    int refined_count = 0;
    for (std::size_t i = 1; i + 1 < slices.size(); ++i) {
        ++attempted;
        auto low = refineObservedEdge(gray, toImage({slices[i].s, slices[i].low}, object),
                                      transverse, config, trace, "slice_low");
        auto high = refineObservedEdge(gray, toImage({slices[i].s, slices[i].high}, object),
                                       transverse, config, trace, "slice_high");
        if (!low || !high) {
            continue;
        }
        // 两侧都沿正横轴采样，因此偏移相减，不能直接相加。
        double width = slices[i].width + *high - *low;
        if (width <= 0) {
            continue;
        }
        ++refined_count;
        refined_widths.push_back(width);
        if (analysis.boundary >= 0 && isShaftSlice(i, slices.size(), analysis)) {
            shaft_widths.push_back(width);
        }
    }
    if (trace) {
        trace->attempted_slices = attempted;
        trace->refined_slices = refined_count;
    }
    auto low_end = refineObservedEdge(gray, object.length_start, object.axis, config, trace, "length_start");
    auto high_end = refineObservedEdge(gray, object.length_end, object.axis, config, trace, "length_end");
    if (!low_end || !high_end || refined_widths.empty()
        || refined_count < kMinimumRefinedFraction * attempted) {
        // 此时原实现未清空已有 refined；保持这个提前返回的语义。
        object.subpixel_status = "fallback";
        object.issues.push_back("subpixel_low_quality");
        return;
    }
    object.refined.length = *object.pixel.length + *high_end - *low_end;

    // 整体宽度从原始轮廓 t 极值出发，不替换成旋转掩膜某列的最大宽度。
    cv::Point2d bottom;
    cv::Point2d top;
    for (auto point : object.contour) {
        auto local = toLocal(point, object);
        if (std::abs(local.y - projection.t_min) < kExtremumCoordinateTolerance) {
            bottom = local;
        }
        if (std::abs(local.y - projection.t_max) < kExtremumCoordinateTolerance) {
            top = local;
        }
    }
    auto low_width = refineObservedEdge(gray, toImage(bottom, object), transverse, config, trace, "width_low");
    auto high_width = refineObservedEdge(gray, toImage(top, object), transverse, config, trace, "width_high");
    if (!low_width || !high_width) {
        fallbackSubpixel(object, "subpixel_width_unavailable");
        return;
    }
    object.refined.width = *object.pixel.width + *high_width - *low_width;
    if (*object.refined.length <= 0 || *object.refined.width <= 0) {
        fallbackSubpixel(object, "subpixel_invalid_geometry");
        return;
    }
    if (object.pixel.head_width) {
        auto low_head = refineObservedEdge(gray, toImage(head_edges.low, object), transverse, config, trace, "head_low");
        auto high_head = refineObservedEdge(gray, toImage(head_edges.high, object), transverse, config, trace, "head_high");
        if (low_head && high_head && *object.pixel.head_width + *high_head - *low_head > 0) {
            object.refined.head_width = *object.pixel.head_width + *high_head - *low_head;
        } else {
            object.issues.push_back("subpixel_head_width_unavailable");
        }
    }
    if (!shaft_widths.empty()) {
        object.refined.shaft_width = median(shaft_widths);
    }
    // 头杆分界是宽度阈值，不是轴向灰度边缘；refined.head_length 保持空值。
    object.subpixel_status = "refined";
}
} // namespace

std::optional<double> refineEdge(const cv::Mat& gray, cv::Point2d point,
                                cv::Vec2d normal, const Config& config) {
    return refineEdge(gray, point, normal, config, nullptr);
}

std::optional<double> refineEdge(const cv::Mat& gray, cv::Point2d point,
                                cv::Vec2d normal, const Config& config, EdgeTrace* trace) {
    if (trace) {
        *trace = {};
        trace->point = point;
        trace->normal = normal;
        trace->search_radius = config.edge_search;
    }
    auto reject = [trace](const char* reason) -> std::optional<double> {
        if (trace) {
            trace->reason = reason;
        }
        return std::nullopt;
    };
    std::vector<double> intensities;
    std::vector<double> gradients;
    int radius = config.edge_search;
    // 搜索 [-radius,+radius]，两侧各多采一点用于中心差分。
    for (int i = -radius - 1; i <= radius + 1; ++i) {
        auto value = sampleGray(gray, {point.x + i * normal[0], point.y + i * normal[1]});
        if (!value) {
            return reject("out_of_image");
        }
        intensities.push_back(*value);
        if (trace) {
            trace->intensities.push_back(*value);
        }
    }
    for (int i = 1; i < static_cast<int>(intensities.size()) - 1; ++i) {
        gradients.push_back(std::abs(intensities[i + 1] - intensities[i - 1]) / 2);
    }
    int peak = static_cast<int>(std::max_element(gradients.begin(), gradients.end()) - gradients.begin());
    if (trace) {
        trace->gradients = gradients;
        trace->peak = peak;
    }
    if (peak == 0 || peak == static_cast<int>(gradients.size()) - 1
        || gradients[peak] < config.gradient_min) {
        return reject(peak == 0 || peak == static_cast<int>(gradients.size()) - 1
                      ? "peak_at_search_boundary" : "weak_gradient");
    }
    for (int i = 1; i < static_cast<int>(gradients.size()) - 1; ++i) {
        if (std::abs(i - peak) > kDistantPeakSeparation && gradients[i] >= kCompetingPeakRatio * gradients[peak]
            && gradients[i] >= gradients[i - 1] && gradients[i] >= gradients[i + 1]) {
            if (trace) {
                trace->competing_peak = i;
            }
            return reject("competing_peak");
        }
    }
    double left = gradients[peak - 1];
    double mid = gradients[peak];
    double right = gradients[peak + 1];
    double curvature = left - 2 * mid + right;
    if (curvature >= kMinimumNegativeCurvature) {
        return reject("flat_peak");
    }
    double delta = 0.5 * (left - right) / curvature;
    if (trace) {
        trace->delta = delta;
    }
    if (std::abs(delta) > kMaximumPeakOffset) {
        return reject("unstable_fit");
    }
    // gradients[0] 对应 -radius；返回相对粗边缘的有符号像素偏移。
    const double offset = peak - radius + delta;
    if (trace) {
        trace->offset = offset;
        trace->reason = "accepted";
    }
    return offset;
}

void measureObject(Object& object, const cv::Mat& gray, const Config& config) {
    measureObject(object, gray, config, nullptr);
}

void measureObject(Object& object, const cv::Mat& gray, const Config& config, MeasurementTrace* trace) {
    if (trace) {
        *trace = {};
        trace->center = object.center;
    }
    Projection projection = projectContour(object);
    auto slices = buildWidthProfile(object, projection, trace);
    if (trace) {
        trace->slices = slices;
    }
    if (slices.size() < kMinimumProfileSlices) {
        if (config.subpixel) {
            object.subpixel_status = "fallback";
            object.issues.push_back("profile_too_short");
        }
        return;
    }
    HeadShaftAnalysis analysis = analyzeHeadShaft(object, slices, config, trace);
    HeadEdges head_edges = measureHeadShaft(object, projection, slices, analysis);
    if (trace) {
        trace->head_end = analysis.head_end;
        if (analysis.boundary >= 0) {
            trace->boundary_index = analysis.boundary;
            trace->boundary_s = slices[analysis.boundary].s;
        }
    }
    if (config.subpixel) {
        refineDimensions(object, gray, config, projection, slices, analysis, head_edges, trace);
    }
}
} // namespace screw
