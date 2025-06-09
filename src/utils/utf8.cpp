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

std::string gpt2_unicode_to_bytes(const std::string &input) {
  std::string output;
  output.reserve(input.size());
  std::size_t i = 0;
  while (i < input.size()) {
    std::uint8_t b = static_cast<std::uint8_t>(input[i]);
    if (0x20 <= b && b <= 0x7e) {
      output += input[i];
      i += 1;
    } else if (b == 0xc4) {
      std::uint8_t next = input[i + 1];
      output += next > 0xa0 ? next - 0x22 : next - 0x80;
      i += 2;
    } else if (b == 0xc5) {
      std::uint8_t next = input[i + 1];
      output += next > 0x82 ? next + 0x2a : next + 0x1e;
      i += 2;
    } else if (b == 0xc2) {
      output += input[i + 1];
      i += 2;
    } else if (b == 0xc3) {
      output += input[i + 1] + 0x40;
      i += 2;
    } else {
      output += input[i];
      i += 1;
    }
  }
  return output;
}

} // namespace tinyllm
