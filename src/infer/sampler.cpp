#include "sampler.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

namespace tinyllm {

Sampler::Sampler(InferenceCtx &ctx, float temperature, float top_p, std::int32_t top_k, int seed)
    : ctx(ctx), temperature(temperature), top_p(top_p), top_k(top_k), gen(seed) {
  heap.reserve(top_k);
}

std::int32_t Sampler::sample_argmax() {
  const auto logits = ctx.get_logits();
  auto max_it = std::max_element(logits.begin(), logits.end());
  return static_cast<std::int32_t>(std::distance(logits.begin(), max_it));
}

std::int32_t Sampler::sample_top_k_top_p() {
  const auto &logits = ctx.get_logits();
  const size_t N = logits.size();
  heap.clear();
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

std::int32_t Sampler::sample() {
  if (top_k > 0 && top_p < 1.0f) {
    return sample_top_k_top_p();
  } else if (top_k > 0) {
    return sample_top_k_top_p();
  } else if (top_p < 1.0f) {
    assert(!"not implemented yet");
    return sample_argmax();
  } else {
    return sample_argmax();
  }
}

} // namespace tinyllm
