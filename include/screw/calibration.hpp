#pragma once

#include "screw/types.hpp"

namespace screw {
Scale makeScale(const std::string& reference, cv::Size size, cv::Point2d p1,
                cv::Point2d p2, double known_mm);
void saveScale(const Scale& scale, const fs::path& path);
Scale loadScale(const fs::path& path);
void applyScaleAndRecipe(Object& object, cv::Size size, const std::optional<Scale>& scale,
                         const std::optional<Recipe>& recipe, bool single_spec);
} // namespace screw
