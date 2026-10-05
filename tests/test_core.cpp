#include "fixtures.hpp"
#include "screw/calibration.hpp"
#include "screw/config.hpp"
#include "screw/diagnostics.hpp"
#include "screw/evaluation.hpp"
#include "screw/io.hpp"
#include "screw/locator.hpp"
#include "screw/metrology.hpp"
#include "screw/pipeline.hpp"
#include "screw/segmentation.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace {
using namespace screw;

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<class Function>
void rejects(Function function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, message);
}

Object detect(const cv::Mat& image, const Config& config) {
    const auto contours = candidates(segment(image, config), config);
    check(contours.size() == 1, "Expected one component");
    Object object = locate(contours[0], image.size(), config);
    object.id = 1;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    measureObject(object, gray, config);
    return object;
}

void checkRotationAndHeadShaft(const Config& config) {
    check(angleError(179, 1) == 2 && normalizeAngle(-10) == 170, "Angle conventions");
    for (double angle : std::vector<double>{0, 17, 45, 80, 110, 159}) {
        const auto object = detect(screwImage(angle), config);
        check(object.reliable, "Clean screw reliable");
        check(angleError(object.angle, angle) < 0.8, "Rotation recovery");
        check(std::abs(*object.pixel.length - 200) < 3 && std::abs(*object.pixel.width - 40) < 3,
              "Projection dimensions");
        check(object.pixel.head_width && object.pixel.shaft_width
              && std::abs(*object.pixel.shaft_width - 20) < 2, "Head and shaft recovery");
        check(object.pixel.head_length && std::abs(*object.pixel.head_length - 30) < 2,
              "Head length recovery");
        check(cv::norm(toImage(toLocal({100.5, 80.3}, object), object) - cv::Point2d(100.5, 80.3)) < 1e-8,
              "Local coordinate inverse");
        std::cout << "rotation " << angle << " -> " << object.angle
                  << ", subpixel=" << object.subpixel_status << '\n';
    }
}

void checkSubpixel(const Config& config) {
    cv::Mat step(40, 220, CV_8U);
    for (int y = 0; y < step.rows; ++y) {
        for (int x = 0; x < step.cols; ++x) {
            step.at<unsigned char>(y, x) = cv::saturate_cast<unsigned char>(
                30 + 180 * 0.5 * (1 + std::erf((x - 154.37) / 1.2)));
        }
    }
    const auto edge = refineEdge(step, {154, 20}, {1, 0}, config);
    check(edge && std::abs(154 + *edge - 154.37) < 0.15, "Subpixel analytic edge accuracy");
    const cv::Mat flat(40, 220, CV_8U, cv::Scalar(100));
    check(!refineEdge(flat, {154, 20}, {1, 0}, config), "Flat edge rejected");
    check(!refineEdge(step, {0, 20}, {1, 0}, config), "Out-of-image profile rejected");
    cv::Mat double_edge(40, 220, CV_8U, cv::Scalar(20));
    double_edge.colRange(152, 157).setTo(220);
    check(!refineEdge(double_edge, {154, 20}, {1, 0}, config), "Multiple edges rejected");
    const auto object = detect(screwImage(), config);
    check(object.subpixel_status == "refined" && object.refined.length, "Clean subpixel path");
}

Scale checkScale(const fs::path& root) {
    const Scale scale = makeScale("reference.png", {640, 480}, {100, 100}, {300, 100}, 20);
    check(std::abs(scale.mm_per_px - 0.1) < 1e-9, "Scale calculation");
    saveScale(scale, root / "scale.yaml");
    check(std::abs(loadScale(root / "scale.yaml").mm_per_px - 0.1) < 1e-9, "Scale roundtrip");
    rejects([] { makeScale("ref", {640, 480}, {100, 100}, {100, 100}, 20); },
            "Coincident reference rejected");
    rejects([] { makeScale("ref", {640, 480}, {-1, 100}, {100, 100}, 20); },
            "Outside reference rejected");
    rejects([] { makeScale("ref", {640, 480}, {100, 100}, {300, 100}, -20); },
            "Negative scale rejected");
    return scale;
}

void checkTolerance(Object& object, const Scale& scale) {
    const Recipe recipe{"test", "mm", {{"length", {19, 21}}, {"head_length", {2, 4}}}};
    applyScaleAndRecipe(object, {640, 480}, scale, recipe, true);
    check(object.status == "OK" && object.mm.head_length, "Mixed pixel/subpixel fields supported");
    applyScaleAndRecipe(object, {640, 480}, std::nullopt, recipe, true);
    check(object.status == "NOT_EVALUATED", "Missing mm scale");
    applyScaleAndRecipe(object, {800, 600}, scale, recipe, true);
    check(object.status == "NOT_EVALUATED" && !object.mm.length, "Scale resolution mismatch");
    applyScaleAndRecipe(object, {640, 480}, scale, recipe, false);
    check(object.status == "NOT_EVALUATED", "Mixed-mode tolerance disabled");
    const double measured = *preferredDimensions(object).length;
    Recipe bound{"bound", "px", {{"length", {measured, measured}}}};
    applyScaleAndRecipe(object, {640, 480}, std::nullopt, bound, true);
    check(object.status == "OK", "Inclusive bounds");
    bound.limits["length"] = {0, measured - 1};
    applyScaleAndRecipe(object, {640, 480}, std::nullopt, bound, true);
    check(object.status == "NG", "Out-of-tolerance result");
}

Object checkQualityAndArtifacts(const fs::path& root, const Config& config, const Scale& scale) {
    auto truncated = detect(screwImage(0, 200, 40, 20, 180, 0, {50, 240}), config);
    check(!truncated.reliable, "Truncation detected");
    const Recipe recipe{"test", "mm", {{"length", {19, 21}}, {"head_length", {2, 4}}}};
    applyScaleAndRecipe(truncated, {640, 480}, scale, recipe, true);
    check(truncated.status == "NOT_EVALUATED", "Truncated target cannot pass");
    cv::Mat touching = screwImage();
    cv::rectangle(touching, {340, 245, 140, 30}, {45, 45, 45}, -1);
    check(!detect(touching, config).reliable, "Touching profile is flagged");

    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar::all(225));
    // CLI 的 CTest 继续复用这些产物，包含读图失败与配置错误的输入。
    cv::imwrite((root / "blank.png").string(), blank);
    cv::imwrite((root / "clean.png").string(), screwImage());
    {
        auto output = openOutput(root / "bad.png");
        output << "invalid image data";
    }
    check(candidates(segment(blank, config), config).empty(), "No-object image");
    cv::circle(blank, {320, 240}, 30, {40, 40, 40}, -1);
    auto nut = detect(blank, config);
    check(!nut.reliable && !nut.pixel.shaft_width, "Circular target direction ambiguous");
    const Recipe shaft{"shaft", "px", {{"shaft_width", {10, 30}}}};
    nut.reliable = true;
    applyScaleAndRecipe(nut, {640, 480}, std::nullopt, shaft, true);
    check(nut.status == "NOT_EVALUATED", "Unavailable dimension");
    return nut;
}

void checkConfiguration(const fs::path& root, const Config& config) {
    Config invalid = config;
    invalid.blur = 2;
    rejects([&] { validateConfig(invalid); }, "Invalid kernel");
    {
        auto output = openOutput(root / "bad_config.yaml");
        output << "%YAML:1.0\nmin_arrea: 100.\n";
    }
    rejects([&] { loadConfig(root / "bad_config.yaml"); }, "Config typo rejected");
    {
        auto output = openOutput(root / "bad_recipe.yaml");
        output << "%YAML:1.0\nname: bad\nunit: px\nlimits:\n  length: [20., 10.]\n";
    }
    {
        auto output = openOutput(root / "good_recipe.yaml");
        output << "%YAML:1.0\nname: sample\nunit: mm\nlimits:\n  length: [19., 21.]\n";
    }
    rejects([&] { loadRecipe(root / "bad_recipe.yaml"); }, "Invalid recipe bounds");
}

void checkResultIO(const fs::path& root, const Object& object, const Object& nut) {
    ImageResult result;
    result.image = "synthetic.png";
    result.size = {640, 480};
    result.objects = {object, nut};
    result.objects[1].id = 2;
    saveResult(result, root / "result.json");
    const auto loaded = loadResult(root / "result.json");
    check(loaded.objects.size() == 2 && !loaded.objects[1].pixel.shaft_width, "JSON null roundtrip");
    check(jsonNumber(std::numeric_limits<double>::quiet_NaN()) == "null", "Finite JSON numbers");
}

void checkEvaluation(const fs::path& root, Object object) {
    const cv::RotatedRect box({30, 40}, {100, 20}, 37);
    check(std::abs(rotatedIoU(box, box) - 1) < 1e-6, "Identical box IoU");
    check(rotatedIoU(box, cv::RotatedRect({300, 400}, {100, 20}, 37)) == 0, "Disjoint box IoU");
    // 验证数据集 row/col、半像素原点，以及逆时针弧度到图像角度的转换。
    ImageResult box_result;
    box_result.image = "fixture.png";
    box_result.size = {640, 480};
    Object prediction;
    prediction.id = 1;
    prediction.box = box;
    prediction.center = box.center;
    prediction.angle = 37;
    box_result.objects = {prediction};
    saveResult(box_result, root / "boxes" / "fixture.json");
    {
        auto output = openOutput(root / "annotations.json");
        output << "{\"images\":[{\"id\":1,\"file_name\":\"fixture.png\",\"width\":640,\"height\":480}],"
               << "\"annotations\":[{\"id\":1,\"image_id\":1,\"category_id\":1,\"bbox\":[40.5,30.5,100,20,"
               << -37 * CV_PI / 180 << "]}]}";
    }
    evaluateBoxes(root / "boxes", root / "annotations.json", root / "box_report");
    cv::FileStorage report((root / "box_report" / "report.json").string(), cv::FileStorage::READ);
    check(static_cast<int>(report["tp"]) == 1
          && static_cast<double>(report["angle_error_deg"]["mae"]) < 0.01, "Dataset conversion evaluation");

    object.reliable = true;
    object.pixel.length = 200;
    object.refined = {};
    ImageResult dimension_result;
    dimension_result.image = "synthetic.png";
    dimension_result.size = {640, 480};
    dimension_result.objects = {object};
    saveResult(dimension_result, root / "dimensions" / "sample.json");
    {
        auto output = openOutput(root / "truth.csv");
        output << "image,target_id,part_id,measurement,truth,unit\nsynthetic.png,1,part,length,199,px\n";
    }
    evaluateDimensions(root / "dimensions", root / "truth.csv", root / "dimension_report");
    cv::FileStorage dimension_report((root / "dimension_report" / "report.json").string(), cv::FileStorage::READ);
    check(static_cast<double>(dimension_report["errors"]["length|px"]["mean"]) == 1, "Signed dimensional error");
}

void checkSharedPipeline(const Config& config) {
    const Recipe recipe{"pipeline", "px", {{"length", {190, 210}}}};
    cv::Mat gray(480, 640, CV_8U, cv::Scalar(225));
    cv::Mat empty_mask = cv::Mat::zeros(gray.size(), CV_8U);
    check(measureObjects(empty_mask, gray, config, std::nullopt, recipe, true).empty(),
          "Pipeline accepts no targets");

    cv::Mat image = screwImage(0, 200, 40, 20, 180, 0, {450, 120});
    cv::min(image, screwImage(0, 200, 40, 20, 180, 0, {180, 120}), image);
    cv::min(image, screwImage(0, 200, 40, 20, 180, 0, {180, 340}), image);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const auto objects = measureObjects(segment(image, config), gray, config, std::nullopt, recipe, true);
    check(objects.size() == 3, "Pipeline retains separated targets");
    const std::vector<cv::Point2d> expected_centers{{180, 120}, {450, 120}, {180, 340}};
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& object = objects[i];
        // 螺钉质心受宽头影响偏离生成中心，用包含范围检验位置与排序。
        check(object.id == static_cast<int>(i + 1)
              && cv::norm(object.center - expected_centers[i]) < 20, "Pipeline orders by centroid y then x");
        check(object.reliable && object.status == "OK"
              && object.reasons == std::vector<std::string>{"all_configured_limits_passed"}
              && object.subpixel_status == "refined" && object.refined.length, "Pipeline preserves quality and tolerance");
    }
    image = screwImage(0, 200, 40, 20, 180, 0, {50, 240});
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const auto truncated = measureObjects(segment(image, config), gray, config, std::nullopt, recipe, true);
    check(truncated.size() == 1 && !truncated[0].reliable
          && truncated[0].status == "NOT_EVALUATED"
          && std::find(truncated[0].issues.begin(), truncated[0].issues.end(), "truncated") != truncated[0].issues.end(),
          "Pipeline keeps invalid geometry unevaluated");
}

std::string fileText(const fs::path& path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void checkFixedCandidates(const fs::path& root, const Config& base_config) {
    cv::Mat image = screwImage(17, 200, 40, 20, 180, 0, {180, 120});
    cv::min(image, screwImage(65, 200, 40, 20, 180, 0, {450, 340}), image);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const auto mask = segment(image, base_config);
    const auto contours = candidates(mask, base_config);
    const auto snapshot = contours;
    const Recipe recipe{"fixed-candidates", "px", {{"length", {190, 210}}}};
    std::vector<Object> reference;
    for (const auto& method : std::vector<std::string>{"pca", "fitline", "rect"}) {
        Config config = base_config;
        config.axis = method;
        ImageResult original;
        original.image = "fixed.png";
        original.config = config;
        original.size = image.size();
        original.objects = measureObjects(mask, gray, config, std::nullopt, recipe, true);
        ImageResult fixed = original;
        fixed.objects = measureContours(contours, gray, config, std::nullopt, recipe, true);
        saveResult(original, root / "fixed_original.json");
        saveResult(fixed, root / "fixed_contours.json");
        check(fileText(root / "fixed_original.json") == fileText(root / "fixed_contours.json"),
              "Fixed contours preserve every serialized field and recipe decision");
        check(contours == snapshot, "Measurement does not mutate shared candidates");
        check(fixed.objects.size() == 2, "Shared candidates retain both targets");
        if (reference.empty()) {
            reference = fixed.objects;
        }
        for (std::size_t i = 0; i < reference.size(); ++i) {
            const auto& object = fixed.objects[i];
            double expected = object.pca_angle;
            if (method == "fitline") {
                expected = object.fit_angle;
            } else if (method == "rect") {
                expected = object.rect_angle;
            }
            check(object.id == reference[i].id && object.center == reference[i].center
                  && object.box.center == reference[i].box.center
                  && object.box.size == reference[i].box.size && object.box.angle == reference[i].box.angle
                  && object.contour == reference[i].contour && object.angle == expected,
                  "Axis comparisons retain identity and select the requested method");
            // 用完全相同的轮廓坐标核对逐对象调用，避免另生成平移图的浮点取整差异。
            auto manual = locate(object.contour, gray.size(), config);
            measureObject(manual, gray, config);
            check(object.pixel.length == manual.pixel.length && object.pixel.width == manual.pixel.width,
                  "Fixed candidates actually remeasure dimensions along the selected axis");
        }
    }
    check(measureContours({}, gray, base_config, std::nullopt, std::nullopt, false).empty(),
          "Fixed candidates accept empty input");
    image = screwImage(0, 200, 40, 20, 180, 0, {50, 240});
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const auto invalid = measureContours(candidates(segment(image, base_config), base_config),
                                         gray, base_config, std::nullopt, recipe, true);
    check(invalid.size() == 1 && !invalid[0].reliable && invalid[0].status == "NOT_EVALUATED",
          "Fixed candidates preserve geometry rejection");
}

void checkObservedScene(const fs::path& root, const cv::Mat& scene, const cv::Mat& gray,
                        const Config& config, const Scale& scale, const Recipe& recipe,
                        bool& saved_panel) {
    SegmentationTrace segmentation;
    const auto mask = segment(scene, config, &segmentation);
    check(cv::norm(mask, segment(scene, config), cv::NORM_INF) == 0,
          "Observation preserves segmentation");
    check(cv::norm(mask, segmentation.final_mask, cv::NORM_INF) == 0,
          "Trace captures final mask");
    if (config.segmentation == "lab") {
        cv::Mat conditions = segmentation.color_condition | segmentation.dark_condition;
        if (!config.invert) {
            cv::bitwise_not(conditions, conditions);
        }
        check(cv::norm(conditions, segmentation.threshold_mask, cv::NORM_INF) == 0,
              "Lab trace captures conditions before morphology and inversion semantics");
    }
    PipelineTrace pipeline;
    ImageResult observed;
    observed.image = "diagnostic_fixture.png";
    observed.size = scene.size();
    observed.config = config;
    observed.objects = measureObjects(mask, gray, config, scale, recipe, true, &pipeline);
    ImageResult plain = observed;
    plain.objects = measureObjects(mask, gray, config, scale, recipe, true);
    saveResult(plain, root / "observation_off.json");
    saveResult(observed, root / "observation_on.json");
    check(fileText(root / "observation_off.json") == fileText(root / "observation_on.json"),
          "Observation preserves serialized dimensions, IDs, states and reasons");
    check(pipeline.objects.size() == observed.objects.size(), "Trace count matches objects");
    for (std::size_t i = 0; i < observed.objects.size(); ++i) {
        const auto& trace = pipeline.objects[i];
        const auto& object = observed.objects[i];
        check(trace.object_id == object.id && trace.center == object.center
              && trace.boundary_s == object.head_boundary, "Sorted trace follows its object");
        check(plain.objects[i].length_start == object.length_start
              && plain.objects[i].length_end == object.length_end
              && plain.objects[i].head_boundary == object.head_boundary,
              "Observation preserves non-JSON geometry");
        if (!config.subpixel) {
            check(trace.edges.empty(), "Disabled subpixel does not sample for display");
        }
        for (const auto& slice : trace.slices) {
            int x = cvRound(slice.s - trace.local_s_origin);
            int low = cvRound(slice.low - trace.local_t_origin);
            int high = cvRound(slice.high - trace.local_t_origin);
            check(trace.local_mask.at<unsigned char>(low, x) == 255
                  && trace.local_mask.at<unsigned char>(high, x) == 255
                  && slice.width == high - low, "Profile trace coordinates match local mask");
        }
        if (trace.boundary_index) {
            check(trace.slices[*trace.boundary_index].s == *trace.boundary_s,
                  "Boundary index maps to recorded local s");
        }
    }
    if (config.segmentation == "otsu" && config.invert && config.subpixel && !saved_panel) {
        saveDiagnostics(scene, config, segmentation, pipeline, observed.objects, root / "diagnostics");
        saveResult(observed, root / "observation_after_render.json");
        check(fileText(root / "observation_on.json") == fileText(root / "observation_after_render.json"),
              "Rendering leaves measurement unchanged");
        const auto panel = cv::imread((root / "diagnostics" / "object_1.png").string());
        check(!panel.empty() && panel.cols == 1120 && panel.rows == 1020,
              "Object process panel saved");
        saved_panel = true;
    }
    if (observed.objects.empty() && config.segmentation == "otsu" && config.invert && config.subpixel) {
        saveDiagnostics(scene, config, segmentation, pipeline, observed.objects, root / "empty_diagnostics");
        check(fs::exists(root / "empty_diagnostics" / "segmentation.png"),
              "No-object diagnostics still save segmentation");
    }
}

void checkEdgeTrace(const Config& base_config) {
    cv::Mat edge_image(40, 220, CV_8U);
    for (int y = 0; y < edge_image.rows; ++y) {
        for (int x = 0; x < edge_image.cols; ++x) {
            edge_image.at<unsigned char>(y, x) = cv::saturate_cast<unsigned char>(
                30 + 180 * 0.5 * (1 + std::erf((x - 154.37) / 1.2)));
        }
    }
    EdgeTrace accepted;
    const auto offset = refineEdge(edge_image, {154, 20}, {1, 0}, base_config, &accepted);
    check(offset && accepted.offset == offset && accepted.reason == "accepted"
          && accepted.intensities.size() == static_cast<std::size_t>(2 * base_config.edge_search + 3)
          && accepted.gradients.size() == static_cast<std::size_t>(2 * base_config.edge_search + 1),
          "Accepted edge records actual samples and fitted offset");
    EdgeTrace rejected;
    check(!refineEdge(edge_image, {0, 20}, {1, 0}, base_config, &rejected)
          && rejected.reason == "out_of_image", "Out-of-image reason recorded");
    edge_image.setTo(20);
    edge_image.colRange(154, edge_image.cols).setTo(22);
    check(!refineEdge(edge_image, {154, 20}, {1, 0}, base_config, &rejected)
          && rejected.reason == "weak_gradient" && !rejected.delta,
          "Weak internal peak rejected before fitting");
    edge_image.setTo(20);
    edge_image.colRange(152, 157).setTo(220);
    check(!refineEdge(edge_image, {154, 20}, {1, 0}, base_config, &rejected)
          && rejected.reason == "competing_peak" && rejected.competing_peak
          && !rejected.delta, "Competing peak identified before fitting");
}

void checkDiagnostics(const fs::path& root, const Config& base_config) {
    cv::Mat multi = screwImage(0, 200, 40, 20, 180, 0, {450, 120});
    cv::min(multi, screwImage(0, 200, 40, 20, 180, 0, {180, 120}), multi);
    cv::min(multi, screwImage(17, 200, 40, 20, 180, 0, {180, 340}), multi);
    cv::Mat touching = screwImage();
    cv::rectangle(touching, {340, 245, 140, 30}, {45, 45, 45}, cv::FILLED);
    const std::vector<cv::Mat> scenes{multi, touching,
        cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(225))};
    const Recipe recipe{"diagnostics", "px", {{"length", {190, 210}}}};
    const Scale scale = makeScale("reference.png", {640, 480}, {100, 100}, {300, 100}, 20);
    bool saved_panel = false;
    for (const auto& scene : scenes) {
        cv::Mat gray;
        cv::cvtColor(scene, gray, cv::COLOR_BGR2GRAY);
        for (const auto& method : std::vector<std::string>{"lab", "otsu", "adaptive"}) {
            for (bool invert : {true, false}) {
                for (bool subpixel : {true, false}) {
                    Config config = base_config;
                    config.segmentation = method;
                    config.invert = invert;
                    config.subpixel = subpixel;
                    checkObservedScene(root, scene, gray, config, scale, recipe, saved_panel);
                }
            }
        }
    }
    checkEdgeTrace(base_config);
    PipelineTrace short_pipeline;
    cv::Mat short_gray(40, 40, CV_8U, cv::Scalar(225));
    cv::Mat short_mask = cv::Mat::zeros(short_gray.size(), CV_8U);
    cv::rectangle(short_mask, {15, 15, 6, 6}, 255, cv::FILLED);
    Config small_config = base_config;
    small_config.min_area = 1;
    const auto small = measureObjects(short_mask, short_gray, small_config, std::nullopt,
                                      std::nullopt, false, &short_pipeline);
    check(small.size() == 1 && short_pipeline.objects[0].profile.empty()
          && short_pipeline.objects[0].edges.empty() && small[0].subpixel_status == "fallback",
          "Short profile trace preserves early return");
}

} // namespace

int main(int argc, char** argv) {
    try {
        cv::setNumThreads(1);
        const screw::fs::path root = argc > 1 ? argv[1] : "test_artifacts";
        screw::fs::create_directories(root);
        screw::Config config;
        config.segmentation = "otsu";
        config.blur = 1;
        config.morph_open = 1;
        config.morph_close = 1;

        checkRotationAndHeadShaft(config);
        checkSubpixel(config);
        const auto scale = checkScale(root);
        auto object = detect(screwImage(), config);
        checkTolerance(object, scale);
        const auto nut = checkQualityAndArtifacts(root, config, scale);
        checkConfiguration(root, config);
        checkResultIO(root, object, nut);
        checkEvaluation(root, object);
        checkSharedPipeline(config);
        checkFixedCandidates(root, config);
        checkDiagnostics(root, config);
        std::cout << "All core checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
