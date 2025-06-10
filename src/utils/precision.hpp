#pragma once

#include <cstdint>
#include <span>

namespace tinyllm {

inline float bf16_to_fp32(std::uint16_t bf16) {
  std::uint32_t fp32_bits = std::uint32_t(bf16) << 16;
  return std::bit_cast<float>(fp32_bits);
}

inline std::uint16_t fp32_to_bf16(float fp32) {
  std::uint32_t fp32_bits = std::bit_cast<std::uint32_t>(fp32);
  return std::uint16_t(fp32_bits >> 16);
}

template <typename T> inline float _cvt_to_fp32(T value) {
  if constexpr (sizeof(T) == sizeof(float)) {
    return value;
  } else {
    return bf16_to_fp32(value);
  }
}

void convert_bf16_to_fp32_inplace(std::span<std::byte> data);

void copy_fp32_to_bf16_n(const float *first, std::size_t n, std::uint16_t *result);

void copy_bf16_to_fp32_n(const std::uint16_t *first, std::size_t n, float *result);

} // namespace tinyllm
