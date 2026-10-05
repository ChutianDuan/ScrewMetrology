#pragma once

#include "screw/types.hpp"

namespace screw {
Config loadConfig(const fs::path& path);
void validateConfig(const Config& config);
Recipe loadRecipe(const fs::path& path);
} // namespace screw
