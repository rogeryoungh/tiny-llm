#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tinyllm {

std::vector<uint32_t> utf8_to_codepoints(const std::string &input);

std::string replace_unicode_space(const std::string &input);

}
