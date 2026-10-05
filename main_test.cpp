#include "screw/config.hpp"
#include "screw/diagnostics.hpp"
#include "screw/locator.hpp"
#include "screw/pipeline.hpp"
#include "screw/segmentation.hpp"
#include "screw/visualization.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <array>
#include <sstream>

// 三种分割共用同一条测量流程，避免比较时遗漏某种方法的 PCA 或亚像素测量。
// 调试时在调用处按 F11 进入，依次观察 contour、object 和 result。
static screw::ImageResult measureMask(const cv::Mat& image, const cv::Mat& gray,
                                      const cv::Mat& mask, const screw::Config& config,
                                      const std::string& image_name,
                                      screw::PipelineTrace* trace = nullptr) {
    screw::ImageResult result;
    result.image = image_name;
    result.size = image.size();
    result.config = config;
    // MVTec 没有本拍摄设置的毫米尺度和单规格配方，默认只展示像素尺寸。
    result.mode = "mixed";

    // 与正式入口共用候选处理与编号；跨分割方法的编号仍不保证对应同一工件。
    result.objects = screw::measureObjects(mask, gray, config, result.scale,
                                           result.recipe_config, result.mode == "single", trace);
    std::cout << config.segmentation << " / " << config.axis << "：检测到 "
              << result.objects.size() << " 个候选目标\n";
    return result;
}

// 总览图只画编号和方向，完整尺寸标签留在放大视图，避免缩小时文字遮挡工件。
static cv::Mat overviewTile(const cv::Mat& image, const screw::ImageResult& result,
                            const cv::Scalar& axis_color, cv::Size size) {
    cv::Mat tile;
    cv::resize(image, tile, size, 0, 0, cv::INTER_AREA);
    const double scale = static_cast<double>(size.width) / image.cols;
    auto point = [scale](cv::Point2d p) {
        return cv::Point(cvRound(p.x * scale), cvRound(p.y * scale));
    };
    for (const auto& object : result.objects) {
        const cv::Scalar color = object.reliable ? cv::Scalar(70, 220, 90)
                                                 : cv::Scalar(0, 160, 255);
        cv::Point2f corners[4];
        object.box.points(corners);
        for (int i = 0; i < 4; ++i) {
            cv::line(tile, point(corners[i]), point(corners[(i + 1) % 4]), color, 1, cv::LINE_AA);
        }
        const cv::Point center = point(object.center);
        // 主轴是无向轴，以穿过质心的线段表示，避免箭头暗示螺钉头尾方向。
        const double half_length = std::clamp(object.pixel.length.value_or(40) * scale * 0.45, 8.0, 80.0);
        const cv::Point delta(cvRound(object.axis[0] * half_length), cvRound(object.axis[1] * half_length));
        cv::line(tile, center - delta, center + delta, axis_color, 2, cv::LINE_AA);
        cv::circle(tile, center, 2, {0, 0, 255}, cv::FILLED);
        const std::string label = std::to_string(object.id);
        const cv::Point label_position(std::clamp(center.x + 5, 0, size.width - 30),
                                       std::clamp(center.y - 5, 15, size.height - 5));
        cv::putText(tile, label, label_position, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 3, cv::LINE_AA);
        cv::putText(tile, label, label_position, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 255, 255}, 1, cv::LINE_AA);
    }
    return tile;
}

// 标题放在独立区域，避免覆盖图像；OpenCV 内置字体使用英文标签。
static cv::Mat titledImage(const cv::Mat& image, const std::string& title,
                           const std::string& subtitle, const cv::Scalar& accent) {
    cv::Mat output(image.rows + 58, image.cols, CV_8UC3, cv::Scalar(28, 28, 28));
    image.copyTo(output(cv::Rect(0, 58, image.cols, image.rows)));
    cv::line(output, {0, 0}, {image.cols - 1, 0}, accent, 3);
    cv::putText(output, title, {12, 23}, cv::FONT_HERSHEY_SIMPLEX, 0.55, accent, 1, cv::LINE_AA);
    cv::putText(output, subtitle, {12, 45}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {220, 220, 220}, 1, cv::LINE_AA);
    return output;
}

static cv::Mat maskTile(const cv::Mat& mask, const std::string& segmentation, cv::Size tile_size) {
    cv::Mat small_mask;
    cv::Mat colored_mask;
    // 二值图用最近邻缩放，保持 0/255，不引入灰色伪边缘。
    cv::resize(mask, small_mask, tile_size, 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(small_mask, colored_mask, cv::COLOR_GRAY2BGR);
    const double foreground = 100.0 * cv::countNonZero(mask) / mask.total();
    std::ostringstream stats;
    stats << "White = foreground | " << cvRound(foreground) << "% pixels";
    return titledImage(colored_mask, segmentation + " | MASK", stats.str(), {210, 210, 210});
}

struct MeasurementView {
    cv::Mat overview;
    cv::Mat detail;
};

static MeasurementView measurementView(const cv::Mat& image, const screw::ImageResult& result,
                                       const std::string& title, const cv::Scalar& axis_color,
                                       cv::Size tile_size) {
    const auto check_count = std::count_if(result.objects.begin(), result.objects.end(),
        [](const screw::Object& object) { return !object.reliable; });
    std::ostringstream stats;
    stats << result.objects.size() << " objects | " << check_count << " CHECK | unit: px";
    const cv::Mat compact = overviewTile(image, result, axis_color, tile_size);
    MeasurementView view;
    view.overview = titledImage(compact, title, stats.str(), axis_color);
    view.detail = titledImage(screw::visualize(image, result), title,
                             stats.str() + " | N/E = not evaluated", axis_color);
    return view;
}

static screw::Config experimentConfig() {
    // 手动调参区：显式列出全部参数，初始值与项目默认配置一致。
    // 这里直接修改即可；仅创建配置不会自动执行分割或测量。
    screw::Config config = {};

    // 1. 分割与去噪
    config.segmentation = "lab";  // 基础配置；下方对比流程会分别设置三种分割方法
    config.invert = true;         // Otsu/自适应：暗目标为前景；Lab：保留下面的选中条件
    config.blur = 3;              // 高斯核边长，奇数 1..501；1 表示不平滑
    config.lab_b_max = 145;       // Lab：b <= 此值 或 L <= 下值成为前景，范围 0..255
    config.lab_dark_l = 90;       // Lab：补充暗目标；增大会选中更多暗背景
    config.adaptive_block = 51;   // 仅自适应阈值使用，局部窗口边长，奇数 3..501
    config.adaptive_c = 8;        // 仅自适应阈值使用：阈值 = 局部加权均值 - C

    // 2. 形态学与候选过滤（尺寸随图像分辨率变化，需要观察掩膜调节）
    config.morph_open = 3;        // 开运算核边长：去小白点；奇数 1..501，1 跳过
    config.morph_close = 11;      // 闭运算核边长：补裂缝；过大会连接邻近目标，1 跳过
    config.min_area = 1000;       // 候选面积下限，px²；减小可保留小目标，也会保留噪声
    config.max_area = 200000;     // 候选面积上限，px²；必须 >= min_area
    config.min_solidity = 0.25;   // 面积 / 凸包面积下限，范围 (0,1]；增大过滤更严格

    // 3. 主轴与头杆测量
    config.axis = "pca";          // 基础配置；下方对比流程会分别设置三种主轴方法
    config.axis_ratio = 1.3;      // PCA 特征值比下限，必须 >1；过低则标记主轴不明确
    config.head_ratio = 1.25;     // 端部宽度超过杆宽的此倍数时识别为头部，必须 >1

    // 4. 亚像素边缘（定位与像素测量之后才使用）
    config.subpixel = true;       // false：仅保留像素测量；true：尝试细化边缘
    config.edge_search = 4;       // 沿边缘法线搜索半径，px，整数 2..20
    config.gradient_min = 5;      // 灰度梯度峰下限，必须 >0；增大更严格，低对比可能降级

    return config;
}

// 构建固定的 3×4 总览并保存九张详情；窗口事件由显示函数处理。
static bool saveComparison(const cv::Mat& image, const cv::Mat& gray,
                           const screw::Config& config, const screw::fs::path& image_path,
                           const screw::fs::path& output, cv::Mat& comparison) {
    // 三行独立配置：在这里逐项覆盖参数，可对某一种阈值方法单独调参。
    screw::Config lab_config = config;
    lab_config.segmentation = "lab";
    screw::Config otsu_config = config;
    otsu_config.segmentation = "otsu";
    screw::Config adaptive_config = config;
    adaptive_config.segmentation = "adaptive";
    const std::array<screw::Config, 3> row_configs{lab_config, otsu_config, adaptive_config};
    const std::array<std::string, 3> axes{"pca", "fitline", "rect"};
    const std::array<cv::Scalar, 3> axis_colors{
        cv::Scalar(255, 220, 60), cv::Scalar(220, 120, 255), cv::Scalar(30, 230, 255)};
    std::array<cv::Mat, 3> masks;
    std::array<std::array<screw::ImageResult, 3>, 3> results;
    std::array<std::array<cv::Mat, 3>, 3> details;

    const int tile_width = 448;
    const int gap = 8;
    const int top = 64;
    const int footer = 50;
    // 保持输入图像纵横比，同一源像素在所有格子中的显示位置相同。
    const cv::Size tile_size(tile_width, std::max(1, cvRound(image.rows * tile_width / double(image.cols))));
    const int cell_height = tile_size.height + 58;
    comparison = cv::Mat(top + 3 * cell_height + 4 * gap + footer,
                       4 * tile_width + 5 * gap, CV_8UC3, cv::Scalar(18, 18, 18));
    cv::putText(comparison, "SEGMENTATION x AXIS | same image, same scale", {16, 28},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, {245, 245, 245}, 1, cv::LINE_AA);
    cv::putText(comparison, "Rows: Lab / Otsu / Adaptive    Columns: Mask / PCA / fitLine / Rect", {16, 51},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {180, 180, 180}, 1, cv::LINE_AA);

    screw::fs::create_directories(output);
    for (std::size_t row = 0; row < row_configs.size(); ++row) {
        const auto& row_config = row_configs[row];
        screw::validateConfig(row_config);
        screw::SegmentationTrace segmentation;
        masks[row] = screw::segment(image, row_config, row == 0 ? &segmentation : nullptr);
        const cv::Mat mask_cell = maskTile(masks[row], row_config.segmentation, tile_size);
        mask_cell.copyTo(comparison(cv::Rect(gap, top + gap + row * (cell_height + gap),
                                             tile_width, cell_height)));

        for (std::size_t col = 0; col < axes.size(); ++col) {
            // 同一行复用同一张掩膜；只改变 axis，然后重新定位和测量。
            // 不能只替换显示角度，否则沿主轴计算的尺寸仍是旧方法的结果。
            screw::Config method_config = row_config;
            method_config.axis = axes[col];
            auto& result = results[row][col];
            screw::PipelineTrace pipeline;
            const bool observe = row == 0 && col == 0;
            result = measureMask(image, gray, masks[row], method_config, image_path.filename().string(),
                                 observe ? &pipeline : nullptr);
            if (observe) {
                // Lab/PCA 的中间数据来自当前这次计算；其余八个组合保持原有展示。
                try {
                    screw::saveDiagnostics(image, method_config, segmentation, pipeline, result.objects,
                                           output / "process");
                } catch (const std::exception& error) {
                    std::cerr << "保存过程图失败：" << error.what() << '\n';
                    return false;
                }
            }
            const std::string title = row_config.segmentation + " / " + axes[col];
            const auto view = measurementView(image, result, title, axis_colors[col], tile_size);
            view.overview.copyTo(comparison(cv::Rect(gap + (col + 1) * (tile_width + gap),
                top + gap + row * (cell_height + gap), tile_width, cell_height)));
            details[row][col] = view.detail;
            if (!cv::imwrite((output / (row_config.segmentation + "_" + axes[col] + ".png")).string(),
                             details[row][col])) {
                std::cerr << "保存详情图失败\n";
                return false;
            }
        }
    }
    const int footer_y = comparison.rows - footer + 18;
    cv::putText(comparison, "Green box: geometry usable | Orange box: CHECK | Colored line: selected axis (undirected)",
                {16, footer_y}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1, cv::LINE_AA);
    cv::putText(comparison, "Fixed 3 x 4 panel | Full-size images: result/main_test/ | Q/Esc = quit", {16, footer_y + 21},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1, cv::LINE_AA);
    if (!cv::imwrite((output / "comparison.png").string(), comparison)) {
        std::cerr << "保存对比面板失败\n";
        return false;
    }
    std::cout << "对比面板与九张详情图已保存至 " << output << std::endl;
    return true;
}

static void showComparison(const cv::Mat& comparison) {
    // 固定显示一个 3×4 面板，全部组合同时可见，无需切换方法。
    cv::namedWindow("Comparison", cv::WINDOW_NORMAL);
    cv::resizeWindow("Comparison", 1280, cvRound(comparison.rows * 1280.0 / comparison.cols));
    cv::imshow("Comparison", comparison);
    while (true) {
        // 定期处理窗口事件，也允许点击窗口关闭按钮退出。
        int key = cv::waitKey(50);
        if (cv::getWindowProperty("Comparison", cv::WND_PROP_VISIBLE) < 1) {
            break;
        }
        if (key < 0) {
            continue;
        }
        key &= 0xff;
        if (key >= 'A' && key <= 'Z') {
            key += 'a' - 'A';
        }
        if (key == 27 || key == 'q') {
            break;
        }
    }
    cv::destroyAllWindows();
}

int main() {
    // 固定学习入口：图像、参数和 3×4 面板布局直接在本文件中调整，无命令行参数。
    // 想先学纯几何，可换成 data/synthetic/images/rotation_1.png（干净背景，单目标）。
    // CMake 提供项目根目录；从 build/ 执行 ./main_test 也能找到数据。
    const screw::fs::path project_dir = SCREW_TEST_PROJECT_DIR;
    const screw::fs::path image_path = project_dir / "data/mvtec_screws/images/screws_153.png";
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    // 读图失败时及时退出，避免后面的显示或测量产生难以理解的错误。
    if (image.empty()) {
        std::cerr << "无法读取测试图像：" << image_path << "，请检查图像路径。\n";
        return 1;
    }

    const screw::Config config = experimentConfig();

    // 调参时尽早检查非法取值，避免到后面的算法阶段才报错。
    try {
        screw::validateConfig(config);
    } catch (const std::exception& error) {
        std::cerr << "配置错误：" << error.what() << '\n';
        return 1;
    }

    // 三种方法共用原始灰度图，供亚像素边缘采样；不从二值掩膜生成灰度。
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat comparison;
    const screw::fs::path output = project_dir / "result/main_test";
    if (!saveComparison(image, gray, config, image_path, output, comparison)) {
        return 1;
    }
    showComparison(comparison);

    return 0;
}
