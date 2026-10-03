#include "screw/evaluation.hpp"
#include "screw/io.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace screw {
namespace {
struct Stats {
    // values 可装误差或重复测量值，但统计含义不同：误差看 MAE，重复性看测量值波动。
    std::vector<double> values;
    void add(double x) { values.push_back(x); }
    void writeRepeatability(std::ostream& out) const {
        // 同一实物同一测量项按 N-1 求样本标准差；只有一次测量时不能估计重复性。
        double mean=std::accumulate(values.begin(),values.end(),0.0)/values.size(), squared=0;
        for (double v:values) squared+=(v-mean)*(v-mean);
        out << "{\"count\":" << values.size() << ",\"mean\":" << jsonNumber(mean)
            << ",\"min\":" << jsonNumber(*std::min_element(values.begin(),values.end()))
            << ",\"max\":" << jsonNumber(*std::max_element(values.begin(),values.end()))
            << ",\"std\":" << jsonNumber(values.size()>1 ? std::optional<double>(std::sqrt(squared/(values.size()-1))) : std::nullopt) << '}';
    }
    void write(std::ostream& out) const {
        // 尺寸评价传入“测量-真值”，mean 保留偏差方向，mae 则对误差取绝对值。
        out << "{\"count\":" << values.size();
        if (values.empty()) { out << ",\"mean\":null,\"mae\":null,\"max_abs\":null,\"std\":null}"; return; }
        double mean=std::accumulate(values.begin(),values.end(),0.0)/values.size(), absolute=0,maximum=0,squared=0;
        for (double v : values) { absolute+=std::abs(v); maximum=std::max(maximum,std::abs(v)); squared+=(v-mean)*(v-mean); }
        out << ",\"mean\":" << jsonNumber(mean) << ",\"mae\":" << jsonNumber(absolute/values.size())
            << ",\"max_abs\":" << jsonNumber(maximum) << ",\"std\":"
            << jsonNumber(values.size()>1 ? std::optional<double>(std::sqrt(squared/(values.size()-1))) : std::nullopt) << '}';
    }
};
struct TruthBox { int id=0, category=0; cv::RotatedRect box; };
std::map<std::string,ImageResult> loadResults(const fs::path& input) {
    std::map<std::string,ImageResult> results;
    for (auto path : resultPaths(input)) {
        cv::FileStorage f(path.string(),cv::FileStorage::READ);
        if (!f.isOpened()) throw std::runtime_error("Cannot read JSON: " + path.string());
        // 结果目录还可能有批次汇总或评价报告；只读取单图测量 schema。
        if (static_cast<std::string>(f["schema"]) != "screw_measurement_v1") continue;
        f.release();
        auto r=loadResult(path); std::string name=fs::path(r.image).filename().string();
        if (!results.emplace(name,std::move(r)).second) throw std::runtime_error("Duplicate result image: " + name);
    }
    if (results.empty()) throw std::runtime_error("No measurement results");
    return results;
}
std::vector<std::string> csvRow(const std::string& line) {
    std::vector<std::string> row; std::string field; bool quoted=false, closed=false;
    for (size_t i=0;i<line.size();++i) {
        char c=line[i];
        if (quoted) {
            if (c=='"') {
                if (i+1<line.size() && line[i+1]=='"') { field+='"'; ++i; }
                else {quoted=false;closed=true;}
            } else field+=c;
        } else if (c==',') { row.push_back(field);field.clear();closed=false; }
        else if (c=='"' && field.empty() && !closed) quoted=true;
        else if (c=='\r' && i+1==line.size()) {}
        else if (closed || c=='"') throw std::runtime_error("Malformed CSV quoting");
        else field+=c;
    }
    if (quoted) throw std::runtime_error("Unterminated CSV quote");
    row.push_back(field); return row;
}
void drawBox(cv::Mat& image, const cv::RotatedRect& box, cv::Scalar color, const std::string& label) {
    cv::Point2f p[4]; box.points(p);
    for (int i=0;i<4;++i) cv::line(image,p[i],p[(i+1)%4],color,3,cv::LINE_AA);
    cv::putText(image,label,box.center,cv::FONT_HERSHEY_SIMPLEX,0.7,color,2);
}
}
double rotatedIoU(const cv::RotatedRect& a, const cv::RotatedRect& b) {
    // IoU=交集面积/(面积 A+面积 B-交集面积)，必须对旋转框求交，不能用轴对齐框代替。
    if (a.size.area()<=0 || b.size.area()<=0) return 0;
    std::vector<cv::Point2f> overlap;
    int status=cv::rotatedRectangleIntersection(a,b,overlap);
    double intersection=0;
    // 完全包含时交集就是小框；部分重叠时返回交集多边形，再计算其面积。
    if (status==cv::INTERSECT_FULL) intersection=std::min(a.size.area(),b.size.area());
    else if (overlap.size()>=3) intersection=std::abs(cv::contourArea(overlap));
    double total=a.size.area()+b.size.area()-intersection;
    return total>0 ? std::clamp(intersection/total,0.0,1.0) : 0;
}
void evaluateBoxes(const fs::path& input, const fs::path& annotation_file, const fs::path& output) {
    auto results=loadResults(input);
    cv::FileStorage f(annotation_file.string(),cv::FileStorage::READ);
    if (!f.isOpened() || !f["images"].isSeq() || !f["annotations"].isSeq()) throw std::runtime_error("Invalid MVTec annotation file");
    std::map<int,std::string> images;
    std::map<int,cv::Size> sizes;
    std::map<int,std::vector<TruthBox>> truth;
    std::set<std::string> names;
    for (auto node:f["images"]) {
        int id=static_cast<int>(node["id"]); std::string name=static_cast<std::string>(node["file_name"]);
        if (name.empty() || !names.insert(name).second || !images.emplace(id,name).second) throw std::runtime_error("Duplicate/invalid annotation image");
        sizes[id]={static_cast<int>(node["width"]),static_cast<int>(node["height"])};
    }
    for (auto node:f["annotations"]) {
        auto b=node["bbox"];
        if (!b.isSeq() || b.size()!=5) throw std::runtime_error("MVTec bbox requires [row,col,width,height,phi]");
        int image_id=static_cast<int>(node["image_id"]);
        if (!images.count(image_id)) throw std::runtime_error("Annotation references unknown image");
        double row=static_cast<double>(b[0]), col=static_cast<double>(b[1]),w=static_cast<double>(b[2]),h=static_cast<double>(b[3]),phi=static_cast<double>(b[4]);
        if (!std::isfinite(row) || !std::isfinite(col) || !std::isfinite(w) || !std::isfinite(h) || !std::isfinite(phi) || w<=0 || h<=0) throw std::runtime_error("Invalid ground truth box");
        // 原包定义 bbox=[row,col,width,height,phi]，不是通常 COCO 的左上角 x/y/w/h。
        // row/col 的原点比 OpenCV 像素中心偏移 +0.5，故中心换成 (col-0.5,row-0.5)。
        // phi 为数学逆时针弧度，而图像 y 向下，故 OpenCV 框角度需取负并换成度。
        truth[image_id].push_back({static_cast<int>(node["id"]),static_cast<int>(node["category_id"]),
            cv::RotatedRect({static_cast<float>(col-0.5),static_cast<float>(row-0.5)},
                            {static_cast<float>(w),static_cast<float>(h)},static_cast<float>(-phi*180/CV_PI))});
    }
    for (const auto& entry:results) if (!names.count(entry.first)) throw std::runtime_error("Result image not in annotations: "+entry.first);
    fs::create_directories(output);
    auto details=openOutput(output/"images.jsonl");
    int tp=0,fp=0,fn=0,evaluated=0,failed=0,unreliable=0;
    Stats centers, origins, angles, ious;
    std::map<int,std::pair<int,int>> categories;
    std::vector<std::string> missing,failed_images;
    for (const auto& [image_id,name]:images) {
        auto it=results.find(name);
        // 子集评价明确列出未提供结果的图；若提供了失败图 JSON，它仍进入本次评价。
        if (it==results.end()) { missing.push_back(name); continue; }
        ++evaluated; const auto& r=it->second; const auto& gt=truth[image_id];
        if (!r.error.empty()) {++failed;failed_images.push_back(name);}
        else if (r.size!=sizes[image_id]) throw std::runtime_error("Result/annotation resolution mismatch: "+name);
        for (const auto& o:r.objects) if (!o.reliable) ++unreliable;
        struct Match {double iou;size_t pred,actual;};
        std::vector<Match> options;
        // 枚举 IoU>=0.5 的匹配，再按 IoU 降序贪心配对；每个预测和真值最多使用一次。
        // 这是固定的评价口径，不是类别识别，也不是最大权重二分图匹配。
        for (size_t p=0;p<r.objects.size();++p) for (size_t g=0;g<gt.size();++g) {
            double iou=rotatedIoU(r.objects[p].box,gt[g].box);
            if (iou>=0.5) options.push_back({iou,p,g});
        }
        std::sort(options.begin(),options.end(),[](const Match& a,const Match& b){
            if (a.iou!=b.iou) return a.iou>b.iou;
            return std::tie(a.pred,a.actual)<std::tie(b.pred,b.actual);
        });
        std::vector<bool> used_pred(r.objects.size(),false),used_gt(gt.size(),false);
        int matched=0;
        details << "{\"image\":" << quote(name) << ",\"matches\":[";
        for (const auto& m:options) {
            if (used_pred[m.pred] || used_gt[m.actual]) continue;
            used_pred[m.pred]=true;used_gt[m.actual]=true;
            if (matched++) details << ',';
            const auto& p=r.objects[m.pred]; const auto& g=gt[m.actual];
            // 标注提供旋转框中心，不提供面积质心：框中心误差与质心偏移必须分别记录。
            double center_error=cv::norm(p.box.center-g.box.center);
            double origin_distance=cv::norm(p.center-cv::Point2d(g.box.center));
            centers.add(center_error);origins.add(origin_distance);ious.add(m.iou);
            double ratio=std::max(g.box.size.width,g.box.size.height)/std::min(g.box.size.width,g.box.size.height);
            std::optional<double> angle;
            // 近圆真值框方向不稳定，不纳入角误差；其他框统一取长边方向并按 180° 比较。
            if (ratio>=1.3) { angle=angleError(p.angle,g.box.angle+(g.box.size.height>g.box.size.width?90:0)); angles.add(*angle); }
            ++categories[g.category].first;
            details << "{\"target_id\":" << p.id << ",\"annotation_id\":" << g.id << ",\"iou\":" << jsonNumber(m.iou)
                << ",\"center_error_px\":" << jsonNumber(center_error) << ",\"centroid_to_box_center_distance_px\":" << jsonNumber(origin_distance)
                << ",\"angle_error_deg\":" << jsonNumber(angle) << '}';
        }
        // TP=匹配数，FP=未匹配预测，FN=未匹配真值；定位检出统计包括质量不可靠的预测。
        tp+=matched;fp+=static_cast<int>(r.objects.size())-matched;fn+=static_cast<int>(gt.size())-matched;
        details << "],\"false_positive_ids\":[";bool first=true;
        for (size_t p=0;p<r.objects.size();++p) if (!used_pred[p]) {if (!first) details << ',';first=false;details << r.objects[p].id;}
        details << "],\"missed_annotation_ids\":[";first=true;
        for (size_t g=0;g<gt.size();++g) {
            ++categories[gt[g].category].second;
            if (!used_gt[g]) {if (!first) details << ',';first=false;details << gt[g].id;}
        }
        details << "],\"error\":" << (r.error.empty()?"null":quote(r.error)) << "}\n";
        if (matched<static_cast<int>(gt.size()) || matched<static_cast<int>(r.objects.size())) {
            cv::Mat image=cv::imread(r.image);
            if (!image.empty()) {
                for (size_t g=0;g<gt.size();++g) if (!used_gt[g]) drawBox(image,gt[g].box,{0,0,255},"miss:"+std::to_string(gt[g].id));
                for (size_t p=0;p<r.objects.size();++p) if (!used_pred[p]) drawBox(image,r.objects[p].box,{0,255,255},"fp:"+std::to_string(r.objects[p].id));
                if (image.cols>960) cv::resize(image,image,{960,static_cast<int>(image.rows*960.0/image.cols)});
                fs::create_directories(output/"failures");
                if (!cv::imwrite((output/"failures"/(fs::path(name).stem().string()+".jpg")).string(),image,{cv::IMWRITE_JPEG_QUALITY,80})) throw std::runtime_error("Cannot write failure overlay");
            }
        }
    }
    auto out=openOutput(output/"report.json");
    out << "{\"schema\":\"screw_detection_evaluation_v1\",\"annotation_file\":" << quote(fs::absolute(annotation_file).string())
        << ",\"iou_threshold\":0.5,\"matching\":\"greedy_descending_iou_one_to_one\",\"annotation_images\":" << images.size()
        << ",\"evaluated_images\":" << evaluated << ",\"failed_images\":" << failed << ",\"missing_result_images\":";
    writeStrings(out,missing);out << ",\"failed_image_names\":";writeStrings(out,failed_images);
    out << ",\"tp\":" << tp << ",\"fp\":" << fp << ",\"fn\":" << fn
        << ",\"recall\":" << jsonNumber(tp+fn?std::optional<double>(static_cast<double>(tp)/(tp+fn)):std::nullopt)
        << ",\"precision\":" << jsonNumber(tp+fp?std::optional<double>(static_cast<double>(tp)/(tp+fp)):std::nullopt)
        << ",\"unreliable_predictions\":" << unreliable << ",\"center_error_px\":";centers.write(out);
    out << ",\"centroid_to_box_center_distance_px\":";origins.write(out);
    out << ",\"angle_error_deg\":";angles.write(out);out << ",\"matched_iou\":";ious.write(out);
    out << ",\"categories\":[";bool first=true;
    for (auto [id,counts]:categories) {if (!first) out << ',';first=false;out << "{\"id\":" << id << ",\"matched\":" << counts.first << ",\"total\":" << counts.second << ",\"recall\":" << jsonNumber(static_cast<double>(counts.first)/counts.second) << '}';}
    out << "]}\n";
}
void evaluateDimensions(const fs::path& input, const fs::path& truth_file, const fs::path& output) {
    auto results=loadResults(input);std::ifstream truth(truth_file);
    if (!truth) throw std::runtime_error("Cannot read CSV: "+truth_file.string());
    std::string line;if (!std::getline(truth,line)) throw std::runtime_error("Empty CSV");
    auto header=csvRow(line);
    if (header!=std::vector<std::string>{"image","target_id","part_id","measurement","truth","unit"}) throw std::runtime_error("CSV header must be image,target_id,part_id,measurement,truth,unit");
    fs::create_directories(output);auto details=openOutput(output/"dimensions.jsonl");
    std::map<std::string,Stats> errors,pixel_errors,subpixel_errors,repeats;
    // CSV 用 image+target_id 找本图目标，用 part_id 把不同图中的同一实物关联起来。
    int rows=0,excluded=0;std::set<std::string> seen;
    while (std::getline(truth,line)) {
        if (line.empty() || line=="\r") continue;
        ++rows;auto row=csvRow(line);
        if (row.size()!=6) throw std::runtime_error("CSV field count at data row "+std::to_string(rows));
        std::string image=fs::path(row[0]).filename().string(),measurement=row[3],unit=row[5];
        dimension(Dimensions{},measurement);
        if (row[2].empty() || (unit!="mm" && unit!="px")) throw std::runtime_error("Invalid part_id or unit at CSV row "+std::to_string(rows));
        size_t used=0;int id=std::stoi(row[1],&used);
        if (used!=row[1].size() || id<=0) throw std::runtime_error("Invalid target_id");
        std::optional<double> actual;
        // truth 可空：没有真值时仍能统计重复性，但不能计算测量误差。
        if (!row[4].empty()) {used=0;double v=std::stod(row[4],&used);if (used!=row[4].size() || !std::isfinite(v) || v<=0) throw std::runtime_error("Invalid truth");actual=v;}
        std::string unique=image+"|"+row[1]+"|"+measurement+"|"+unit;
        if (!seen.insert(unique).second) throw std::runtime_error("Duplicate CSV measurement");
        auto image_it=results.find(image);const Object* object=nullptr;std::string reason;
        if (image_it==results.end()) reason="missing_result";
        else {
            for (const auto& o:image_it->second.objects) if (o.id==id) {object=&o;break;}
            if (!object) reason="missing_target";
            else if (!object->reliable) reason="unreliable_geometry";
        }
        std::optional<double> value,pixel,subpixel;
        if (reason.empty()) {
            pixel=dimension(object->pixel,measurement);subpixel=dimension(object->refined,measurement);
            value=unit=="mm"?dimension(object->mm,measurement):(subpixel?subpixel:pixel);
            if (!value) reason="unavailable_dimension";
        }
        std::string key=measurement+"|"+unit;
        // 不可靠或不可测的尺寸单独记录排除原因，不用零值填充后混入精度统计。
        if (!reason.empty()) ++excluded;
        else {
            // 按实物、测量项、单位分组；不同规格或不同单位不能混成一个重复性样本。
            repeats[row[2]+"|"+key].add(*value);
            if (actual) {
                errors[key].add(*value-*actual);
                // 像素真值同时比较原始轮廓尺寸与有效细化尺寸；细化失败不冒充成功样本。
                if (unit=="px" && pixel) pixel_errors[key].add(*pixel-*actual);
                if (unit=="px" && subpixel) subpixel_errors[key].add(*subpixel-*actual);
            }
        }
        details << "{\"image\":" << quote(image) << ",\"target_id\":" << id << ",\"part_id\":" << quote(row[2])
            << ",\"measurement\":" << quote(measurement) << ",\"unit\":" << quote(unit) << ",\"truth\":" << jsonNumber(actual)
            << ",\"measured\":" << jsonNumber(value) << ",\"error\":" << jsonNumber(value && actual?std::optional<double>(*value-*actual):std::nullopt)
            << ",\"excluded_reason\":" << (reason.empty()?"null":quote(reason)) << "}\n";
    }
    if (!rows) throw std::runtime_error("No CSV data rows");
    auto out=openOutput(output/"report.json");
    out << "{\"schema\":\"screw_dimension_evaluation_v1\",\"rows\":" << rows << ",\"excluded\":" << excluded << ",\"errors\":{";
    bool first=true;for (auto& [key,stats]:errors) {if (!first) out << ',';first=false;out << quote(key) << ':';stats.write(out);}
    out << "},\"pixel_errors\":{";first=true;for (auto& [key,stats]:pixel_errors) {if (!first) out << ',';first=false;out << quote(key) << ':';stats.write(out);}
    out << "},\"subpixel_errors\":{";first=true;for (auto& [key,stats]:subpixel_errors) {if (!first) out << ',';first=false;out << quote(key) << ':';stats.write(out);}
    out << "},\"repeatability\":{";first=true;for (auto& [key,stats]:repeats) {if (!first) out << ',';first=false;out << quote(key) << ':';stats.writeRepeatability(out);}
    out << "}}\n";
}
}
