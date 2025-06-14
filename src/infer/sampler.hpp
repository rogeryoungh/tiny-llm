#pragma once

#include "inference_ctx.hpp"
#include <random>

namespace tinyllm {

class Sampler {
public:
  Sampler(InferenceCtx &ctx, float temperature = 1.0f, float top_p = 1.0f, std::int32_t top_k = 0, int seed = 114514);

  std::int32_t sample_argmax();

  std::int32_t sample_top_k_top_p();

  std::int32_t sample();

  InferenceCtx &ctx;
  float temperature;
  float top_p;
  std::int32_t top_k;

protected:
  std::mt19937 gen;

  std::vector<std::pair<float, std::int32_t>> heap;
};

} // namespace tinyllm
