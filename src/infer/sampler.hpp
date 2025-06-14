#pragma once

#include <random>
#include <span>

namespace tinyllm {

class Sampler {
public:
  Sampler(float temperature = 1.0f, float top_p = 1.0f, std::int32_t top_k = 0, int seed = 114514);

  std::int32_t sample_argmax(const std::span<float> logits);

  std::int32_t sample_top_k_top_p(const std::span<float> logits);

  std::vector<std::pair<float, std::int32_t>> get_top_k(const std::span<float> logits);

  std::int32_t get_top_p( std::vector<std::pair<float, std::int32_t>> heap);

  float temperature;
  float top_p;
  std::int32_t top_k;

protected:
  std::mt19937 gen;
};

} // namespace tinyllm
