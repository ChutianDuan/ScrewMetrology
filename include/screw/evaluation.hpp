#pragma once

#include "screw/types.hpp"

namespace screw {
double rotatedIoU(const cv::RotatedRect& a, const cv::RotatedRect& b);
void evaluateBoxes(const fs::path& results, const fs::path& annotations, const fs::path& output);
void evaluateDimensions(const fs::path& results, const fs::path& truth, const fs::path& output);
} // namespace screw
