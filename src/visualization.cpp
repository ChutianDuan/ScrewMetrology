#include "screw/visualization.hpp"
#include "screw/locator.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace screw {
cv::Mat visualize(const cv::Mat& image, const ImageResult& result) {
    cv::Mat out=image.clone();
    double font=std::clamp(image.cols/3200.0,0.4,0.65);
    int line_height=static_cast<int>(24*font)+9;
    std::vector<cv::Rect> labels;
    for (const auto& o:result.objects) {
        // 橙色先提示几何不可靠，红色表示 NG；默认展示模式也可画绿色框，不能据颜色判 OK。
        cv::Scalar color=!o.reliable?cv::Scalar(0,160,255):o.status=="NG"?cv::Scalar(0,0,255):cv::Scalar(0,210,0);
        cv::Point2f corners[4];o.box.points(corners);
        for (int i=0;i<4;++i) cv::line(out,corners[i],corners[(i+1)%4],color,2,cv::LINE_AA);
        cv::drawContours(out,std::vector<std::vector<cv::Point>>{o.contour},0,{255,160,0},1);
        cv::circle(out,o.center,3,{0,0,255},cv::FILLED);
        cv::arrowedLine(out,o.center,toImage({40,0},o),{255,0,0},2,cv::LINE_AA);
        cv::arrowedLine(out,o.center,toImage({0,25},o),{255,0,255},2,cv::LINE_AA);
        cv::line(out,o.length_start,o.length_end,{0,255,255},1,cv::LINE_AA);
        if (o.head_boundary && o.pixel.width) cv::line(out,toImage({*o.head_boundary,-*o.pixel.width/2},o),toImage({*o.head_boundary,*o.pixel.width/2},o),{255,0,255},1);
        Dimensions d=preferredDimensions(o);
        std::ostringstream first,second;first << '#' << o.id << "  " << std::fixed << std::setprecision(1)
            << d.length.value_or(0) << " x " << d.width.value_or(0) << " px";
        second << std::fixed << std::setprecision(1) << o.angle << " deg | "
               << (o.status=="NOT_EVALUATED"?"N/E":o.status) << (!o.reliable?" | CHECK":"");
        std::vector<std::string> lines={first.str(),second.str()};
        if (o.mm.length && o.mm.width) {
            std::ostringstream mm;mm << std::fixed << std::setprecision(2) << *o.mm.length << " x " << *o.mm.width << " mm";
            lines.push_back(mm.str());
        }
        int label_width=0,baseline=0;
        for (const auto& line:lines) label_width=std::max(label_width,cv::getTextSize(line,cv::FONT_HERSHEY_SIMPLEX,font,1,&baseline).width+12);
        label_width=std::min(label_width,image.cols);
        int label_height=std::min(static_cast<int>(lines.size())*line_height+4,image.rows);
        cv::Rect bounds=o.box.boundingRect();
        cv::Rect chosen(0,0,label_width,label_height);double best=std::numeric_limits<double>::max();
        // 在目标附近枚举标签位置，优先避免标签互相遮挡、其次避免压住工件，并限制在图内。
        // cost 是展示位置的代价，不是测量误差或置信度；引线关联移动后的标签与目标。
        for (int dy=-3;dy<=3;++dy) for (int side=0;side<4;++side) {
            int x=side==0?bounds.x:side==1?bounds.br().x+8:side==2?bounds.x-label_width-8:bounds.x;
            int y=side==0?bounds.y-label_height-8:side==3?bounds.br().y+8:static_cast<int>(o.center.y)-label_height/2;
            y+=dy*line_height;
            cv::Rect tag(std::clamp(x,0,image.cols-label_width),std::clamp(y,0,image.rows-label_height),label_width,label_height);
            double cost=cv::norm(cv::Point2d(tag.x+tag.width/2.0,tag.y+tag.height/2.0)-o.center);
            for (const auto& other:labels) cost+=(tag&other).area()*50;
            for (const auto& target:result.objects) cost+=(tag&target.box.boundingRect()).area()*2;
            if (cost<best) {best=cost;chosen=tag;}
        }
        labels.push_back(chosen);
        cv::Point connector(chosen.x+chosen.width/2,chosen.y+chosen.height/2);
        cv::line(out,connector,o.center,{70,70,70},1,cv::LINE_AA);
        cv::rectangle(out,chosen,{25,25,25},cv::FILLED);
        cv::rectangle(out,chosen,color,1);
        for (size_t i=0;i<lines.size();++i) cv::putText(out,lines[i],{chosen.x+6,chosen.y+static_cast<int>(i+1)*line_height-4},cv::FONT_HERSHEY_SIMPLEX,font,{255,255,255},1,cv::LINE_AA);
    }
    return out;
}
}
