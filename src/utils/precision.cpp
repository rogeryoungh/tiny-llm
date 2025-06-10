#include "precision.hpp"

namespace tinyllm {

void convert_bf16_to_fp32_inplace(std::span<std::byte> data) {
  auto *fp32_ptr = reinterpret_cast<float *>(data.data());
  auto *bf16_ptr = reinterpret_cast<const std::uint16_t *>(data.data() + data.size() / 2);
  std::size_t num_fp32 = data.size() / sizeof(std::uint32_t);
  for (std::size_t i = 0; i < num_fp32; ++i) {
    fp32_ptr[i] = bf16_to_fp32(bf16_ptr[i]);
  }
}

void copy_fp32_to_bf16_n(const float *first, std::size_t n, std::uint16_t *result) {
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = fp32_to_bf16(first[i]);
  }
}

void copy_bf16_to_fp32_n(const std::uint16_t *first, std::size_t n, float *result) {
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = bf16_to_fp32(first[i]);
  }
}

} // namespace tinyllm
