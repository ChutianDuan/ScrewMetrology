#include "screw/calibration.hpp"
#include "screw/config.hpp"
#include "screw/evaluation.hpp"
#include "screw/io.hpp"
#include "screw/locator.hpp"
#include "screw/metrology.hpp"
#include "screw/segmentation.hpp"
#include "screw/visualization.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace {
using namespace screw;
using Options=std::map<std::string,std::string>;
void help() {
    std::cout << R"(ScrewMetrology - C++ / OpenCV screw metrology

measure --input IMAGE_OR_DIRECTORY [--config YAML] [--output DIRECTORY]
        [--segmentation lab|otsu|adaptive] [--axis pca|fitline|rect]
        [--scale YAML] [--recipe YAML] [--mode mixed|single] [--no-subpixel]
calibrate-scale --input REFERENCE_IMAGE --p1 X,Y --p2 X,Y --mm DISTANCE --output YAML
evaluate --results JSON_OR_DIRECTORY (--annotations MVTEC_JSON | --truth CSV)
         [--output DIRECTORY]

Defaults: mixed display mode, pixel units, config/measurement.yaml if present.
Scale applies only to fixed setup, same plane, near-frontal images at matching resolution.
Exit codes: 0 success (including no objects); 1 invalid arguments or processing failure.
)";
}
Options options(int argc,char** argv,int start,const std::set<std::string>& allowed) {
    Options out;
    for (int i=start;i<argc;++i) {
        std::string key=argv[i];
        if (!allowed.count(key)) throw std::runtime_error("Unknown option: "+key);
        if (out.count(key)) throw std::runtime_error("Duplicate option: "+key);
        if (key=="--no-subpixel") out[key]="1";
        else {
            if (i+1>=argc || std::string(argv[i+1]).rfind("--",0)==0) throw std::runtime_error("Missing value: "+key);
            out[key]=argv[++i];
        }
    }
    return out;
}
std::string value(const Options& o,const std::string& key,const std::string& fallback="") {
    auto it=o.find(key);return it==o.end()?fallback:it->second;
}
std::string required(const Options& o,const std::string& key) {
    auto result=value(o,key);if (result.empty()) throw std::runtime_error("Required option: "+key);return result;
}
double number(const std::string& text) {
    size_t used=0;double result=std::stod(text,&used);
    if (used!=text.size() || !std::isfinite(result)) throw std::runtime_error("Invalid number: "+text);
    return result;
}
cv::Point2d point(const std::string& text) {
    auto comma=text.find(',');
    if (comma==std::string::npos || text.find(',',comma+1)!=std::string::npos) throw std::runtime_error("Point must be X,Y");
    return {number(text.substr(0,comma)),number(text.substr(comma+1))};
}
int measure(const Options& o) {
    fs::path input=required(o,"--input"),output=value(o,"--output","result/measurement");
    fs::path config_path=value(o,"--config",fs::exists("config/measurement.yaml")?"config/measurement.yaml":"");
    Config config=loadConfig(config_path);
    // YAML 提供场景参数，命令行只覆盖显式指定的选项；覆盖后再验证合法性。
    if (o.count("--segmentation")) config.segmentation=value(o,"--segmentation");
    if (o.count("--axis")) config.axis=value(o,"--axis");
    if (o.count("--no-subpixel")) config.subpixel=false;
    validateConfig(config);
    std::string mode=value(o,"--mode","mixed");
    if (mode!="mixed" && mode!="single") throw std::runtime_error("Mode must be mixed or single");
    std::optional<Scale> scale;std::optional<Recipe> recipe;
    if (o.count("--scale")) scale=loadScale(value(o,"--scale"));
    if (o.count("--recipe")) recipe=loadRecipe(value(o,"--recipe"));
    auto paths=imagePaths(input);std::set<std::string> stems;
    // 图像扩展名不进入产物前缀；a.png 与 a.jpg 会覆盖同一 JSON，故批次开始前拒绝冲突。
    for (const auto& path:paths) if (!stems.insert(path.stem().string()).second) throw std::runtime_error("Output name collision: "+path.stem().string());
    fs::create_directories(output);
    int failures=0,objects=0,refined=0,ok=0,ng=0,not_evaluated=0;
    auto start=std::chrono::steady_clock::now();
    for (const auto& path:paths) {
        ImageResult r;r.image=fs::absolute(path).string();r.config=config;r.scale=scale;r.mode=mode;
        if (recipe) {r.recipe=recipe->name;r.recipe_config=recipe;}
        try {
            cv::Mat image=cv::imread(path.string(),cv::IMREAD_COLOR);
            if (image.empty()) throw std::runtime_error("Cannot decode image");
            r.size=image.size();
            // 主流程 1：去噪分割得到掩膜；另从原图生成灰度供亚像素采样，保留亮度边缘。
            cv::Mat mask=segment(image,config),gray;cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
            for (const auto& contour:candidates(mask,config)) {
                // 主流程 2：候选轮廓 -> 质心与主轴 -> 局部几何/亚像素 -> 尺度和公差。
                // 每个阶段保留质量原因；有尺寸值的目标也可能禁止可靠判定。
                Object object=locate(contour,image.size(),config);
                measureObject(object,gray,config);
                applyScaleAndRecipe(object,image.size(),scale,recipe,mode=="single");
                r.objects.push_back(std::move(object));
            }
            // 编号按每张图的质心位置生成，可用于本图 CSV 对应，不是跨帧追踪 ID。
            std::sort(r.objects.begin(),r.objects.end(),[](const Object& a,const Object& b){
                return std::tie(a.center.y,a.center.x)<std::tie(b.center.y,b.center.x);
            });
            for (size_t i=0;i<r.objects.size();++i) r.objects[i].id=static_cast<int>(i)+1;
            fs::path prefix=output/path.stem();
            if (!cv::imwrite(prefix.string()+".mask.png",mask) || !cv::imwrite(prefix.string()+".overlay.png",visualize(image,r))) throw std::runtime_error("Cannot write image output");
        } catch (const std::exception& e) {
            // 某张坏图或图像处理失败时清空部分结果，保留错误记录，并继续下一张图。
            r.error=e.what();r.objects.clear();++failures;std::cerr << path.filename().string() << ": " << r.error << '\n';
        }
        // 失败图也写 JSON，让全量评价能把处理失败计入漏检，而不是静默遗漏该图。
        saveResult(r,output/(path.stem().string()+".json"));
        objects+=static_cast<int>(r.objects.size());
        for (const auto& object:r.objects) {
            if (object.subpixel_status=="refined") ++refined;
            if (object.status=="OK") ++ok;
            else if (object.status=="NG") ++ng;
            else ++not_evaluated;
        }
        std::cout << path.filename().string() << ": " << r.objects.size() << " objects" << (r.error.empty()?"":" (failed)") << '\n';
    }
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    auto summary=openOutput(output/"batch_summary.json");
    // 批处理成功与尺寸合格是两个概念：无目标可返回 0，任何处理失败使最终返回 1。
    summary << "{\"schema\":\"screw_batch_v1\",\"images\":" << paths.size() << ",\"failed\":" << failures
        << ",\"objects\":" << objects << ",\"subpixel_refined\":" << refined << ",\"ok\":" << ok << ",\"ng\":" << ng
        << ",\"not_evaluated\":" << not_evaluated << ",\"seconds\":" << jsonNumber(seconds) << "}\n";
    std::cout << "Processed " << paths.size() << " images, " << objects << " objects, " << failures << " failures in " << seconds << " s\n";
    return failures?1:0;
}
}
int main(int argc,char** argv) {
    try {
        if (argc<2 || std::string(argv[1])=="--help" || std::string(argv[1])=="help") {help();return 0;}
        // 使用单线程减少线程开销与运行时间比较中的调度变化，不改变算法参数。
        cv::setNumThreads(1);
        std::string command=argv[1];
        if (command=="measure") return measure(options(argc,argv,2,{"--input","--config","--output","--segmentation","--axis","--scale","--recipe","--mode","--no-subpixel"}));
        if (command=="calibrate-scale") {
            auto o=options(argc,argv,2,{"--input","--p1","--p2","--mm","--output"});
            std::string input=required(o,"--input");cv::Mat image=cv::imread(input);
            if (image.empty()) throw std::runtime_error("Cannot decode reference image");
            Scale s=makeScale(fs::absolute(input).string(),image.size(),point(required(o,"--p1")),point(required(o,"--p2")),number(required(o,"--mm")));
            saveScale(s,required(o,"--output"));std::cout << "Scale: " << s.mm_per_px << " mm/px\n";return 0;
        }
        if (command=="evaluate") {
            auto o=options(argc,argv,2,{"--results","--annotations","--truth","--output"});
            if (o.count("--annotations")==o.count("--truth")) throw std::runtime_error("Specify exactly one of --annotations or --truth");
            fs::path input=required(o,"--results"),output=value(o,"--output","result/evaluation");
            if (o.count("--annotations")) evaluateBoxes(input,required(o,"--annotations"),output);
            else evaluateDimensions(input,required(o,"--truth"),output);
            std::cout << "Evaluation saved: " << output/"report.json" << '\n';return 0;
        }
        throw std::runtime_error("Unknown command: "+command);
    } catch (const std::exception& e) {std::cerr << "Error: " << e.what() << '\n';return 1;}
}
