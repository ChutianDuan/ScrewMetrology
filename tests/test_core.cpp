#include "fixtures.hpp"
#include "screw/calibration.hpp"
#include "screw/config.hpp"
#include "screw/evaluation.hpp"
#include "screw/io.hpp"
#include "screw/locator.hpp"
#include "screw/metrology.hpp"
#include "screw/segmentation.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition,const std::string& message) {if (!condition) throw std::runtime_error(message);}
template<class F> void rejects(F function,const std::string& message) {
    bool threw=false;try {function();} catch (const std::exception&) {threw=true;} check(threw,message);
}
screw::Object detect(const cv::Mat& image,const screw::Config& config) {
    auto contours=screw::candidates(screw::segment(image,config),config);
    check(contours.size()==1,"Expected one component");
    auto o=screw::locate(contours[0],image.size(),config);o.id=1;
    cv::Mat gray;cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);screw::measureObject(o,gray,config);return o;
}
}
int main(int argc,char** argv) {
    using namespace screw;
    try {
        cv::setNumThreads(1);fs::path root=argc>1?argv[1]:"test_artifacts";fs::create_directories(root);
        Config c;c.segmentation="otsu";c.blur=1;c.morph_open=1;c.morph_close=1;
        check(angleError(179,1)==2 && normalizeAngle(-10)==170,"Angle conventions");
        cv::RotatedRect box({30,40},{100,20},37);
        check(std::abs(rotatedIoU(box,box)-1)<1e-6,"Identical box IoU");
        check(rotatedIoU(box,cv::RotatedRect({300,400},{100,20},37))==0,"Disjoint box IoU");
        for (double angle:std::vector<double>{0,17,45,80,110,159}) {
            auto o=detect(screwImage(angle),c);
            check(o.reliable,"Clean screw reliable");
            check(angleError(o.angle,angle)<0.8,"Rotation recovery");
            check(std::abs(*o.pixel.length-200)<3 && std::abs(*o.pixel.width-40)<3,"Projection dimensions");
            check(o.pixel.head_width && o.pixel.shaft_width && std::abs(*o.pixel.shaft_width-20)<2,"Head and shaft recovery");
            check(o.pixel.head_length && std::abs(*o.pixel.head_length-30)<2,"Head length recovery");
            check(cv::norm(toImage(toLocal({100.5,80.3},o),o)-cv::Point2d(100.5,80.3))<1e-8,"Local coordinate inverse");
            std::cout << "rotation " << angle << " -> " << o.angle << ", subpixel=" << o.subpixel_status << '\n';
        }
        cv::Mat step(40,220,CV_8U);
        for (int y=0;y<step.rows;++y) for (int x=0;x<step.cols;++x)
            step.at<unsigned char>(y,x)=cv::saturate_cast<unsigned char>(30+180*0.5*(1+std::erf((x-154.37)/1.2)));
        auto edge=refineEdge(step,{154,20},{1,0},c);
        check(edge && std::abs(154+*edge-154.37)<0.15,"Subpixel analytic edge accuracy");
        cv::Mat flat(40,220,CV_8U,cv::Scalar(100));
        check(!refineEdge(flat,{154,20},{1,0},c),"Flat edge rejected");
        check(!refineEdge(step,{0,20},{1,0},c),"Out-of-image profile rejected");
        cv::Mat double_edge(40,220,CV_8U,cv::Scalar(20));double_edge.colRange(152,157).setTo(220);
        check(!refineEdge(double_edge,{154,20},{1,0},c),"Multiple edges rejected");
        auto o=detect(screwImage(),c);
        check(o.subpixel_status=="refined" && o.refined.length,"Clean subpixel path");
        Scale s=makeScale("reference.png",{640,480},{100,100},{300,100},20);
        check(std::abs(s.mm_per_px-0.1)<1e-9,"Scale calculation");saveScale(s,root/"scale.yaml");
        check(std::abs(loadScale(root/"scale.yaml").mm_per_px-0.1)<1e-9,"Scale roundtrip");
        rejects([&]{makeScale("ref",{640,480},{100,100},{100,100},20);},"Coincident reference rejected");
        rejects([&]{makeScale("ref",{640,480},{-1,100},{100,100},20);},"Outside reference rejected");
        rejects([&]{makeScale("ref",{640,480},{100,100},{300,100},-20);},"Negative scale rejected");
        Recipe recipe{"test","mm",{{"length",{19,21}},{"head_length",{2,4}}}};
        applyScaleAndRecipe(o,{640,480},s,recipe,true);check(o.status=="OK" && o.mm.head_length,"Mixed pixel/subpixel fields supported");
        applyScaleAndRecipe(o,{640,480},std::nullopt,recipe,true);check(o.status=="NOT_EVALUATED","Missing mm scale");
        applyScaleAndRecipe(o,{800,600},s,recipe,true);check(o.status=="NOT_EVALUATED" && !o.mm.length,"Scale resolution mismatch");
        applyScaleAndRecipe(o,{640,480},s,recipe,false);check(o.status=="NOT_EVALUATED","Mixed-mode tolerance disabled");
        double measured=*preferredDimensions(o).length;Recipe bound{"bound","px",{{"length",{measured,measured}}}};
        applyScaleAndRecipe(o,{640,480},std::nullopt,bound,true);check(o.status=="OK","Inclusive bounds");
        bound.limits["length"]={0,measured-1};applyScaleAndRecipe(o,{640,480},std::nullopt,bound,true);check(o.status=="NG","Out-of-tolerance result");
        auto truncated=detect(screwImage(0,200,40,20,180,0,{50,240}),c);
        check(!truncated.reliable,"Truncation detected");applyScaleAndRecipe(truncated,{640,480},s,recipe,true);
        check(truncated.status=="NOT_EVALUATED","Truncated target cannot pass");
        cv::Mat touching=screwImage();cv::rectangle(touching,{340,245,140,30},{45,45,45},-1);
        check(!detect(touching,c).reliable,"Touching profile is flagged");
        cv::Mat blank(480,640,CV_8UC3,cv::Scalar::all(225));
        cv::imwrite((root/"blank.png").string(),blank);
        cv::imwrite((root/"clean.png").string(),screwImage());
        {auto out=openOutput(root/"bad.png");out << "invalid image data";}
        check(candidates(segment(blank,c),c).empty(),"No-object image");
        cv::circle(blank,{320,240},30,{40,40,40},-1);auto nut=detect(blank,c);
        check(!nut.reliable && !nut.pixel.shaft_width,"Circular target direction ambiguous");
        Recipe shaft{"shaft","px",{{"shaft_width",{10,30}}}};nut.reliable=true;
        applyScaleAndRecipe(nut,{640,480},std::nullopt,shaft,true);check(nut.status=="NOT_EVALUATED","Unavailable dimension");
        Config invalid=c;invalid.blur=2;rejects([&]{validateConfig(invalid);},"Invalid kernel");
        {auto out=openOutput(root/"bad_config.yaml");out << "%YAML:1.0\nmin_arrea: 100.\n";}
        rejects([&]{loadConfig(root/"bad_config.yaml");},"Config typo rejected");
        {auto out=openOutput(root/"bad_recipe.yaml");out << "%YAML:1.0\nname: bad\nunit: px\nlimits:\n  length: [20., 10.]\n";}
        {auto out=openOutput(root/"good_recipe.yaml");out << "%YAML:1.0\nname: sample\nunit: mm\nlimits:\n  length: [19., 21.]\n";}
        rejects([&]{loadRecipe(root/"bad_recipe.yaml");},"Invalid recipe bounds");
        ImageResult r;r.image="synthetic.png";r.size={640,480};r.objects={o,nut};r.objects[1].id=2;
        saveResult(r,root/"result.json");auto loaded=loadResult(root/"result.json");
        check(loaded.objects.size()==2 && !loaded.objects[1].pixel.shaft_width,"JSON null roundtrip");
        check(jsonNumber(std::numeric_limits<double>::quiet_NaN())=="null","Finite JSON numbers");
        // Exercise dataset row/col, half-pixel origin, and CCW-radian conversion.
        ImageResult gt_result;r.objects.clear();gt_result.image="fixture.png";gt_result.size={640,480};
        Object prediction;prediction.id=1;prediction.box=box;prediction.center=box.center;prediction.angle=37;gt_result.objects={prediction};
        saveResult(gt_result,root/"boxes"/"fixture.json");
        {auto out=openOutput(root/"annotations.json");out << "{\"images\":[{\"id\":1,\"file_name\":\"fixture.png\",\"width\":640,\"height\":480}],\"annotations\":[{\"id\":1,\"image_id\":1,\"category_id\":1,\"bbox\":[40.5,30.5,100,20," << -37*CV_PI/180 << "]}]}";}
        evaluateBoxes(root/"boxes",root/"annotations.json",root/"box_report");
        cv::FileStorage report((root/"box_report"/"report.json").string(),cv::FileStorage::READ);
        check(static_cast<int>(report["tp"])==1 && static_cast<double>(report["angle_error_deg"]["mae"])<0.01,"Dataset conversion evaluation");
        o.reliable=true;o.pixel.length=200;o.refined={};r.objects={o};saveResult(r,root/"dimensions"/"sample.json");
        {auto out=openOutput(root/"truth.csv");out << "image,target_id,part_id,measurement,truth,unit\nsynthetic.png,1,part,length,199,px\n";}
        evaluateDimensions(root/"dimensions",root/"truth.csv",root/"dimension_report");
        cv::FileStorage dr((root/"dimension_report"/"report.json").string(),cv::FileStorage::READ);
        check(static_cast<double>(dr["errors"]["length|px"]["mean"])==1,"Signed dimensional error");
        std::cout << "All core checks passed\n";return 0;
    } catch (const std::exception& e) {std::cerr << "FAIL: " << e.what() << '\n';return 1;}
}
