#include "sampler.hpp"

#include <algorithm>
#include <cmath>

namespace tinyllm {

Sampler::Sampler(float temperature, float top_p, std::int32_t top_k, int seed)
    : temperature(temperature), top_p(top_p), top_k(top_k), gen(seed) {}

std::int32_t Sampler::sample_argmax(const std::span<float> logits) {
  auto max_it = std::max_element(logits.begin(), logits.end());
  return static_cast<std::int32_t>(std::distance(logits.begin(), max_it));
}

std::int32_t Sampler::sample_top_k_top_p(const std::span<float> logits) {
  if (top_k <= 0) {
    return sample_argmax(logits);
  }
  auto heap = get_top_k(logits);
  return get_top_p(std::move(heap));
}

std::vector<std::pair<float, std::int32_t>> Sampler::get_top_k(const std::span<float> logits) {
  const size_t N = logits.size();

  std::vector<std::pair<float, std::int32_t>> heap;
  heap.reserve(top_k);
  auto cmp = [](const auto &a, const auto &b) { return a.first > b.first; };
  for (size_t i = 0; i < N; ++i) {
    float score = logits[i];
    if (heap.size() < top_k) {
      heap.emplace_back(score, int(i));
      if (heap.size() == top_k) {
        std::ranges::make_heap(heap, cmp);
      }
    } else if (score > heap.front().first) {
      std::ranges::pop_heap(heap, cmp);
      heap.back() = {score, int(i)};
      std::ranges::push_heap(heap, cmp);
    }
  }

  std::ranges::sort_heap(heap, cmp);

  return heap;
}

std::int32_t Sampler::get_top_p(std::vector<std::pair<float, std::int32_t>> heap) {
  double sum_exp = 0.0;
  for (auto &[p, _] : heap) {
    p = std::exp(p / temperature);
    sum_exp += p;
  }

  size_t pos = 0;
  double cum = 0.0;
  for (std::size_t i = 0; i < top_k; ++i) {
    cum += heap[i].first;
    if (cum >= top_p * sum_exp) {
      heap.resize(i + 1);
      break;
    }
  }

  std::uniform_real_distribution<float> uni(0.0f, float(cum));
  float u = uni(gen);
  float acc = 0.0f;
  for (const auto &[p, token] : heap) {
    acc += p;
    if (acc >= u) {
      return token;
    }
  }
  return heap.back().second;
}

} // namespace tinyllm
