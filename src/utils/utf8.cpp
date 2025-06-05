#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tinyllm {

std::vector<uint32_t> utf8_to_codepoints(const std::string &input) {
  std::vector<uint32_t> codepoints;
  for (std::size_t i = 0; i < input.size();) {
    uint32_t codepoint = 0;
    std::size_t bytes = 0;
    if ((input[i] & 0x80) == 0) {
      codepoint = input[i];
      bytes = 1;
    } else if ((input[i] & 0xE0) == 0xC0) {
      codepoint = input[i] & 0x1F;
      bytes = 2;
    } else if ((input[i] & 0xF0) == 0xE0) {
      codepoint = input[i] & 0x0F;
      bytes = 3;
    } else if ((input[i] & 0xF8) == 0xF0) {
      codepoint = input[i] & 0x07;
      bytes = 4;
    }
    for (std::size_t j = 1; j < bytes; ++j) {
      codepoint <<= 6;
      codepoint |= (input[i + j] & 0x3F);
    }
    codepoints.emplace_back(codepoint);
    i += bytes;
  }
  return codepoints;
}

std::string replace_unicode_space(const std::string &input) {
  std::string output;
  output.reserve(input.size());
  constexpr std::array<char, 3> unicode_space = {static_cast<char>(0xE2), static_cast<char>(0x96),
                                                 static_cast<char>(0x81)};
  std::size_t i = 0;
  while (i < input.size()) {
    if (i + 3 < input.size() && input[i] == unicode_space[0] && input[i + 1] == unicode_space[1] &&
        input[i + 2] == unicode_space[2]) {
      output += ' ';
      i += 3;
      continue;
    }
    output += input[i];
    i += 1;
  }
  return output;
}

} // namespace tinyllm
