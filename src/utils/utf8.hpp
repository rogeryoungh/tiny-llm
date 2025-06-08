#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tinyllm {

std::int32_t utf8_to_codepoint(const std::string_view input);

std::vector<uint32_t> utf8_to_codepoints(const std::string &input);

std::string replace_unicode_space(const std::string &input);

}
