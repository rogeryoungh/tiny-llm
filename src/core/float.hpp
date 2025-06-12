#pragma once

#include <cstdint>
#include <stdfloat>

namespace tinyllm {

#ifdef __STDCPP_FLOAT16_T__
using fp16_t = std::float16_t;
#else
using fp16_t = _Float16;
#endif

// #ifdef __STDCPP_BFLOAT16_T__
// using bf16_t = std::bfloat16_t;
// #else
using bf16_t = std::uint16_t;
// #endif

} // namespace tinyllm
