#pragma once

#include "../core/float.hpp"
#include <span>

namespace tinyllm {

inline float bf16_to_fp32(bf16_t v) {
  // return float(std::bit_cast<bf16_t>(v));
  auto fp32_bits = std::uint32_t(v) << 16;
  return std::bit_cast<float>(fp32_bits);
}

inline bf16_t fp32_to_bf16(float v) {
  // return std::bit_cast<std::uint16_t>(bf16_t(v));
  auto fp32_bits = std::bit_cast<std::uint32_t>(v);
  return bf16_t(fp32_bits >> 16);
}

inline float fp16_to_fp32(fp16_t v) {
  return float(v);
  // Convert FP16 to FP32 using bit manipulation
  // auto fp32_bits = (std::uint32_t(v) << 13) | ((v & 0x7FFF) << 13);
  // return std::bit_cast<float>(fp32_bits);
}

inline fp16_t fp32_to_fp16(float v) {
  return fp16_t(v);
  // Convert FP32 to FP16 using bit manipulation
  // auto fp32_bits = std::bit_cast<std::uint32_t>(v);
  // return fp16_t(fp32_bits >> 13);
}

template <typename T> inline float _cvt_to_fp32(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, bf16_t>) {
    return bf16_to_fp32(value);
  } else {
    return fp16_to_fp32(value);
  }
}

void convert_bf16_to_fp32_inplace(std::span<std::byte> data);

void copy_fp32_to_bf16_n(const float *first, std::size_t n, bf16_t *result);

void copy_bf16_to_fp32_n(const bf16_t *first, std::size_t n, float *result);

void copy_fp32_to_fp16_n(const float *first, std::size_t n, fp16_t *result);

void copy_fp16_to_fp32_n(const fp16_t *first, std::size_t n, float *result);

} // namespace tinyllm
