#pragma once

#include <span>
#include <stdfloat>

namespace tinyllm {

void convert_bf16_to_fp32_inplace(std::span<std::byte> data);

} // namespace tinyllm