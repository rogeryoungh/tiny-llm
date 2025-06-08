#include "precision_convert.hpp"

namespace tinyllm {

void convert_bf16_to_fp32_inplace(std::span<std::byte> data) {
  std::float32_t *fp32_ptr = reinterpret_cast<std::float32_t *>(data.data());
  std::bfloat16_t *fp16_ptr = reinterpret_cast<std::bfloat16_t *>(data.data());
  std::size_t num_elements = data.size() / sizeof(std::float32_t);
  std::size_t i = num_elements - 1;
  do {

    auto f16 = fp16_ptr[i];
    auto f32 = std::float32_t(f16);
    fp32_ptr[i] = f32;
    i--;
  } while (i != 0);
}

} // namespace tinyllm
