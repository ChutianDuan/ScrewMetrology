#include "screw/metrology.hpp"
#include "screw/locator.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace screw {
namespace {
double median(std::vector<double> values) {
    // 中位数比均值更不易受孤立毛刺影响；nth_element 只保证中间元素位置，无须全排序。
    if (values.empty()) return 0;
    auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    double result = *middle;
    if (values.size() % 2 == 0) result = (result + *std::max_element(values.begin(), middle)) / 2;
    return result;
}
std::optional<double> sample(const cv::Mat& gray, cv::Point2d p) {
    // 用四邻域灰度按距离加权，允许在非整数坐标取样；邻域出界时不能继续拟合。
    if (p.x < 0 || p.y < 0 || p.x >= gray.cols - 1 || p.y >= gray.rows - 1) return std::nullopt;
    int x = static_cast<int>(std::floor(p.x)), y = static_cast<int>(std::floor(p.y));
    double dx = p.x - x, dy = p.y - y;
    // 先沿 x 对上下两行插值，再沿 y 插值，就是双线性插值。
    return (1-dy) * ((1-dx)*gray.at<unsigned char>(y,x) + dx*gray.at<unsigned char>(y,x+1))
         + dy * ((1-dx)*gray.at<unsigned char>(y+1,x) + dx*gray.at<unsigned char>(y+1,x+1));
}
// 一个局部截面：s 是主轴位置，low/high 是横向 t 边界，width=high-low，均为像素尺度。
struct Slice { double s, low, high, width; std::optional<double> refined; };
}
// p 为粗边缘图像坐标，n 为单位采样方向；返回沿 n 的有符号像素偏移，不是绝对坐标。
// 两个边缘都沿同一单位轴采样，细化距离因此为原距离 + 末端偏移 - 起点偏移。
std::optional<double> refineEdge(const cv::Mat& gray, cv::Point2d p, cv::Vec2d n, const Config& c) {
    std::vector<double> intensities, gradients;
    int radius = c.edge_search;
    // 搜索区间是 [-radius,+radius]，两侧各多取一点供中心差分使用。
    for (int i = -radius-1; i <= radius+1; ++i) {
        auto value = sample(gray, {p.x + i*n[0], p.y + i*n[1]});
        if (!value) return std::nullopt;
        intensities.push_back(*value);
    }
    // |I(k+1)-I(k-1)|/2 不区分亮到暗或暗到亮：两种边缘都用梯度幅值找峰。
    for (int i = 1; i < static_cast<int>(intensities.size())-1; ++i)
        gradients.push_back(std::abs(intensities[i+1] - intensities[i-1]) / 2);
    int peak = static_cast<int>(std::max_element(gradients.begin(), gradients.end()) - gradients.begin());
    // 峰在搜索端点时缺邻点，弱峰则不足以定位；两种情况都返回不可细化。
    if (peak == 0 || peak == static_cast<int>(gradients.size())-1 || gradients[peak] < c.gradient_min) return std::nullopt;
    // 相距超过 2 个采样点的近等强峰可能来自反光、螺纹或另一边缘，不能盲选最大峰。
    for (int i = 1; i < static_cast<int>(gradients.size())-1; ++i) {
        if (std::abs(i-peak) > 2 && gradients[i] >= 0.8*gradients[peak] && gradients[i] >= gradients[i-1] && gradients[i] >= gradients[i+1]) return std::nullopt;
    }
    double left = gradients[peak-1], mid = gradients[peak], right = gradients[peak+1];
    // 三点抛物线顶点偏移：δ=0.5*(left-right)/(left-2*mid+right)。
    // 曲率须为负；近乎平坦或过大偏移说明拟合不稳定，保留像素级结果更合适。
    double curvature = left - 2*mid + right;
    if (curvature >= -1e-6) return std::nullopt;
    double delta = 0.5 * (left - right) / curvature;
    if (std::abs(delta) > 0.75) return std::nullopt;
    // gradients[0] 对应沿 n 的 -radius 位置，将数组下标转回相对 p 的偏移。
    return peak - radius + delta;
}
void measureObject(Object& o, const cv::Mat& gray, const Config& c) {
    // 第 1 步：投影外轮廓到工件两轴。范围之差与原点平移无关，避免倾斜时轴对齐框变大。
    double smin = std::numeric_limits<double>::max(), smax = -smin, tmin = smin, tmax = -smin;
    cv::Point2d left, right;
    for (auto p : o.contour) {
        auto q = toLocal(p, o);
        if (q.x < smin) { smin = q.x; left = q; }
        if (q.x > smax) { smax = q.x; right = q; }
        tmin = std::min(tmin, q.y); tmax = std::max(tmax, q.y);
    }
    // 这里测的是边界像素中心的投影范围，不统一加 1；灰度边缘距离留给后续细化。
    o.pixel.length = smax - smin; o.pixel.width = tmax - tmin;
    o.length_start = toImage(left, o); o.length_end = toImage(right, o);
    const int pad = 2;
    int width = static_cast<int>(std::ceil(smax - smin)) + 2*pad + 1;
    int height = static_cast<int>(std::ceil(tmax - tmin)) + 2*pad + 1;
    cv::Rect bounds = cv::boundingRect(o.contour);
    // 第 2 步：只填充当前候选外轮廓，再旋转到局部 ROI，避免带入其他候选区域。
    // 已被分割合并进同一轮廓的粘连工件无法靠此步骤分离，须由质量检查提示。
    cv::Mat component = cv::Mat::zeros(bounds.size(), CV_8U);
    std::vector<cv::Point> shifted;
    for (auto p : o.contour) shifted.push_back(p - bounds.tl());
    cv::drawContours(component, std::vector<std::vector<cv::Point>>{shifted}, 0, 255, cv::FILLED);
    cv::Point2d local_origin = toLocal(bounds.tl(), o);
    // component 的坐标从 bounds.tl() 开始；先旋转到 (s,t)，再减去投影下限并加留白。
    // 即目标局部栅格坐标 u=s-smin+pad、v=t-tmin+pad，pad 防止轮廓被 ROI 裁切。
    cv::Matx23d transform(o.axis[0], o.axis[1], local_origin.x-smin+pad,
                         -o.axis[1], o.axis[0], local_origin.y-tmin+pad);
    cv::Mat local;
    // 二值掩膜采用最近邻插值，避免生成中间灰度；原始灰度采样不在这张旋转掩膜上做。
    cv::warpAffine(component, local, transform, {width,height}, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
    std::vector<Slice> slices;
    // 每列沿横向找最小/最大前景坐标，得到 width(s)；范围会跨过内孔，不是在统计白点数。
    for (int x = 0; x < local.cols; ++x) {
        int low = local.rows, high = -1;
        for (int y = 0; y < local.rows; ++y) if (local.at<unsigned char>(y,x)) {
            low = std::min(low,y); high = y;
        }
        if (high > low) slices.push_back({x+smin-pad, low+tmin-pad, high+tmin-pad, static_cast<double>(high-low), std::nullopt});
    }
    if (slices.size() < 10) {
        if (c.subpixel) {o.subpixel_status = "fallback";o.issues.push_back("profile_too_short");}
        return;
    }
    std::vector<double> profile;
    // 第 3 步：用最多 5 个邻近截面的中位数抑制局部螺纹波动，再估计中间段的杆宽基线。
    for (size_t i = 0; i < slices.size(); ++i) {
        std::vector<double> window;
        size_t begin = i > 2 ? i-2 : 0, end = std::min(slices.size(), i+3);
        for (size_t j = begin; j < end; ++j) window.push_back(slices[j].width);
        profile.push_back(median(window));
    }
    std::vector<double> middle;
    for (size_t i = slices.size()*3/10; i < slices.size()*7/10; ++i) middle.push_back(profile[i]);
    double shaft = median(middle);
    // 第二个粘连工件可能占据中段多数位置，使中位数基线变大而掩盖异常。
    // 另取中间截面的较低四分位作粘连检查基线；最终杆宽仍用接受杆段的中位数。
    std::vector<double> baseline;
    for (size_t i=slices.size()/5;i<slices.size()*4/5;++i) baseline.push_back(profile[i]);
    std::sort(baseline.begin(),baseline.end());
    double narrow_baseline=baseline[baseline.size()/4];
    int head_end = 0, boundary = -1;
    // 只对有明显长轴的细长目标尝试头杆分离；螺母等形状可以只有整体长宽。
    if (shaft > 2 && *o.pixel.length > 1.5 * *o.pixel.width && o.eigen_ratio >= c.axis_ratio) {
        std::vector<double> first, last;
        // 比较两端靠内 5%～20% 的截面，减轻尖端、边界栅格化和端面过渡的影响。
        for (size_t i = slices.size()/20; i < slices.size()/5; ++i) {
            first.push_back(profile[i]); last.push_back(profile[slices.size()-1-i]);
        }
        bool at_left = median(first) > c.head_ratio*shaft, at_right = median(last) > c.head_ratio*shaft;
        if (median(first)>c.head_ratio*narrow_baseline && median(last)>c.head_ratio*narrow_baseline) {
            o.reliable=false;o.issues.push_back("possible_touching_two_ends");
        }
        if (at_left != at_right) {
            // 只接受恰好一端明显扩大；-1/+1 表示局部主轴负端/正端，不是固定图像左右。
            head_end = at_left ? -1 : 1;
            int run = 0;
            for (int step = 0; step < static_cast<int>(slices.size())/2; ++step) {
                int i = at_left ? step : static_cast<int>(slices.size())-1-step;
                run = profile[i] <= shaft * c.head_ratio ? run+1 : 0;
                // 连续 3 个截面回到杆宽范围后才确定分界，避免单个凹口被误作头部结束。
                if (run >= 3 && step > static_cast<int>(slices.size())/20) {
                    boundary = at_left ? i-2 : i+2; break;
                }
            }
        }
        // 两端同时扩大或中段持续异常变宽时记录疑似粘连，但不尝试恢复被遮挡的尺寸。
        int interior_wide = 0;
        for (size_t i = slices.size()*3/10; i < slices.size()*7/10; ++i)
            if (profile[i] > shaft*1.7) ++interior_wide;
        if ((at_left && at_right) || interior_wide > static_cast<int>(slices.size())/10) {
            o.reliable = false; o.issues.push_back("possible_touching_profile");
        }
    }
    cv::Point2d head_low, head_high;
    // 第 4 步：按分界分组测量。头长是主轴范围；头宽是头部原始轮廓的横向投影范围。
    if (boundary >= 0) {
        double s = slices[boundary].s;
        o.head_boundary = s;
        o.pixel.head_length = head_end < 0 ? s-smin : smax-s;
        std::vector<double> shaft_values;
        // 杆宽排除已识别头部及两端过渡区，取有效中段截面宽度的中位数。
        for (size_t i = 0; i < slices.size(); ++i) {
            bool in_head = head_end < 0 ? static_cast<int>(i) < boundary : static_cast<int>(i) > boundary;
            if (!in_head && i >= slices.size()/5 && i < slices.size()*4/5) shaft_values.push_back(slices[i].width);
        }
        double head_min=std::numeric_limits<double>::max(), head_max=-head_min;
        for (auto point:o.contour) {
            auto q=toLocal(point,o);
            if (head_end<0 ? q.x<=s : q.x>=s) {
                if (q.y<head_min) {head_min=q.y;head_low=q;}
                if (q.y>head_max) {head_max=q.y;head_high=q;}
            }
        }
        if (head_max>head_min) o.pixel.head_width=head_max-head_min;
        o.pixel.shaft_width = median(shaft_values);
    } else o.issues.push_back("head_shaft_unavailable");
    if (!c.subpixel) return;
    // 第 5 步：沿原始灰度图的 e_y 细化截面两侧。此时仍使用像素剖面给出的粗位置。
    cv::Vec2d transverse{-o.axis[1], o.axis[0]};
    std::vector<double> refined_widths, shaft_widths;
    int attempted = 0, refined = 0;
    for (size_t i = 1; i+1 < slices.size(); ++i) {
        ++attempted;
        auto low = refineEdge(gray, toImage({slices[i].s,slices[i].low},o), transverse,c);
        auto high = refineEdge(gray, toImage({slices[i].s,slices[i].high},o), transverse,c);
        if (!low || !high) continue;
        // 两侧都沿正 e_y 求偏移，因此是 high 的偏移减 low 的偏移，不能直接相加。
        double w = slices[i].width + *high - *low;
        if (w <= 0) continue;
        ++refined; slices[i].refined = w; refined_widths.push_back(w);
        if (boundary >= 0) {
            bool in_head = head_end < 0 ? static_cast<int>(i) < boundary : static_cast<int>(i) > boundary;
            if (!in_head && i >= slices.size()/5 && i < slices.size()*4/5) shaft_widths.push_back(w);
        }
    }
    auto low_end = refineEdge(gray, o.length_start, o.axis, c);
    auto high_end = refineEdge(gray, o.length_end, o.axis, c);
    // 总长两端沿 e_x 细化；整体长宽要求两端有效且至少 60% 被尝试截面细化成功。
    // 达不到条件就保留像素值并明确降级，不能用零值代替失败或强行输出精确尺寸。
    if (!low_end || !high_end || refined_widths.empty() || refined < 0.6*attempted) {
        o.subpixel_status = "fallback"; o.issues.push_back("subpixel_low_quality"); return;
    }
    o.refined.length = *o.pixel.length + *high_end - *low_end;
    // 整体宽度细化仍从原始轮廓的 t 极值出发，保持“投影范围”的定义。
    // 不改成旋转掩膜某列的最大宽度：两者定义不同，还会受 ROI 插值量化影响。
    cv::Point2d bottom, top;
    for (auto p : o.contour) {
        auto q = toLocal(p,o);
        if (std::abs(q.y-tmin) < 1e-5) bottom = q;
        if (std::abs(q.y-tmax) < 1e-5) top = q;
    }
    auto low_width = refineEdge(gray,toImage(bottom,o),transverse,c);
    auto high_width = refineEdge(gray,toImage(top,o),transverse,c);
    if (!low_width || !high_width) {
        o.refined = {}; o.subpixel_status = "fallback"; o.issues.push_back("subpixel_width_unavailable"); return;
    }
    o.refined.width = *o.pixel.width + *high_width - *low_width;
    if (*o.refined.length <= 0 || *o.refined.width <= 0) {
        o.refined = {}; o.subpixel_status = "fallback"; o.issues.push_back("subpixel_invalid_geometry"); return;
    }
    if (o.pixel.head_width) {
        // 头宽独立细化；失败时只让该项回退，逐项优选逻辑仍可使用其他有效细化尺寸。
        auto low_head=refineEdge(gray,toImage(head_low,o),transverse,c);
        auto high_head=refineEdge(gray,toImage(head_high,o),transverse,c);
        if (low_head && high_head && *o.pixel.head_width+*high_head-*low_head>0)
            o.refined.head_width=*o.pixel.head_width+*high_head-*low_head;
        else o.issues.push_back("subpixel_head_width_unavailable");
    }
    if (!shaft_widths.empty()) {
        o.refined.shaft_width = median(shaft_widths);
        // 头杆分界由宽度阈值确定，不是沿主轴的灰度边缘；refined.head_length 保持空值。
    }
    o.subpixel_status = "refined";
}
}
