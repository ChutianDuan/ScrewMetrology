#pragma once

#include "screw/types.hpp"
#include <ostream>
#include <fstream>

namespace screw {
std::string quote(const std::string& value);
std::string jsonNumber(std::optional<double> value);
void writeStrings(std::ostream& out, const std::vector<std::string>& values);
void saveResult(const ImageResult& result, const fs::path& path);
ImageResult loadResult(const fs::path& path);
std::vector<fs::path> imagePaths(const fs::path& input);
std::vector<fs::path> resultPaths(const fs::path& input);
std::ofstream openOutput(const fs::path& path);
} // namespace screw
