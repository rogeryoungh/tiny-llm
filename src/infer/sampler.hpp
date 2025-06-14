#pragma once

#include "inference_ctx.hpp"
#include <cmath>
#include <random>

namespace tinyllm {

class Sampler {
public:
  Sampler(InferenceCtx &ctx, float temperature = 1.0f, float top_p = 1.0f, std::int32_t top_k = 0, int seed = 114514)
      : ctx(ctx), temperature(temperature), top_p(top_p), top_k(top_k), gen(seed) {}

  std::int32_t sample_argmax() {
    const auto logits = ctx.get_logits();
    auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<std::int32_t>(std::distance(logits.begin(), max_it));
  }

  std::int32_t sample() {
    const auto &logits = ctx.get_logits();
    const size_t N = logits.size();

    auto probs = softmax(logits);

    std::vector<std::pair<double, std::int32_t>> candidates(N);
    for (std::int32_t i = 0; i < N; ++i) {
      candidates[i] = {probs[i], i};
    }
    std::sort(candidates.begin(), candidates.end(), std::greater<>());

    if (top_k > 0) {
      candidates.resize(std::min<size_t>(top_k, N));
    } else {
      return candidates[0].second;
    }
    if (top_p < 1.0f) {
      double cum = 0.0;
      std::size_t size = 0;
      while (size < candidates.size() && cum < top_p) {
        cum += candidates[size].first;
        ++size;
      }
      candidates.resize(size);
    }

    std::vector<double> final_probs;
    std::vector<std::int32_t> final_idx;
    final_probs.reserve(candidates.size());
    final_idx.reserve(candidates.size());
    double s = 0.0;
    for (auto &pr : candidates) {
      final_probs.push_back(pr.first);
      final_idx.push_back(pr.second);
      s += pr.first;
    }
    for (double &v : final_probs)
      v /= s;

    std::discrete_distribution<std::size_t> dist(final_probs.begin(), final_probs.end());
    size_t choice = dist(gen);
    return final_idx[choice];
  }

  InferenceCtx &ctx;
  float temperature;
  float top_p;
  std::int32_t top_k;

protected:
  std::mt19937 gen;

  std::vector<double> softmax(std::span<const float> logits) {
    std::vector<double> probs;
    double sum_exp = 0.0;
    probs.reserve(logits.size());
    for (const auto &logit : logits) {
      double exp_val = std::exp(logit / temperature);
      probs.push_back(exp_val);
      sum_exp += exp_val;
    }
    for (auto &prob : probs) {
      prob /= sum_exp;
    }
    return probs;
  }
};

} // namespace tinyllm
