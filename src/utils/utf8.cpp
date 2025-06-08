#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tinyllm {

std::pair<std::size_t, std::uint32_t> utf8_to_codepoint(const std::string_view input) {
  if (input.empty()) {
    return {0, 0};
  }
  std::uint32_t codepoint = 0;
  std::size_t bytes = 0;
  if ((input[0] & 0x80) == 0) {
    codepoint = input[0];
    bytes = 1;
  } else if ((input[0] & 0xE0) == 0xC0) {
    codepoint = input[0] & 0x1F;
    bytes = 2;
  } else if ((input[0] & 0xF0) == 0xE0) {
    codepoint = input[0] & 0x0F;
    bytes = 3;
  } else if ((input[0] & 0xF8) == 0xF0) {
    codepoint = input[0] & 0x07;
    bytes = 4;
  } else {
    return {0, 0};
  }

  if (bytes > input.size()) {
    return {0, 0};
  }

  for (std::size_t i = 1; i < bytes; ++i) {
    if ((input[i] & 0xC0) != 0x80)
      return {0, 0};
    codepoint <<= 6;
    codepoint |= (input[i] & 0x3F);
  }

  return {bytes, codepoint};
}

std::vector<uint32_t> utf8_to_codepoints(const std::string &input) {
  std::vector<uint32_t> codepoints;
  std::size_t i = 0;
  while (i < input.size()) {
    std::size_t start = i;
    auto [bytes, codepoint] = utf8_to_codepoint(input.substr(start));
    if (bytes == 0) {
      // Invalid UTF-8 sequence, skip it
      i += 1; // Skip one byte to avoid infinite loop
      continue;
    }
    codepoints.push_back(static_cast<uint32_t>(codepoint));
    i += bytes;
  }
  return codepoints;
}

std::string replace_unicode_space(const std::string &input) {
  std::string output;
  output.reserve(input.size());
  const std::uint32_t space = 0x2581; // ▁
  std::size_t i = 0;
  while (i < input.size()) {
    auto [bytes, codepoint] = utf8_to_codepoint(input.substr(i));
    if (bytes == 0) {
      output += input[i];
      i += 1;
      continue;
    }
    if (codepoint == space) {
      output += ' ';
    } else {
      for (std::size_t j = 0; j < bytes; ++j) {
        output += input[i + j];
      }
    }
    i += bytes;
  }
  return output;
}

} // namespace tinyllm
