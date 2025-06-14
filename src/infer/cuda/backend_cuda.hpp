#pragma once

#include "../../core/config.hpp"
#include "../../core/model_cuda.hpp"
#include "../../core/tensor.hpp"
#include "../inference_ctx.hpp"
#include "arena_alloc.hpp"

namespace tinyllm {

struct InferenceBackendCUDA : InferenceBackend {

  InferenceBackendCUDA(ModelCuda &model, std::size_t kv_size, DataType kv_dtype = DataType::F32);

  void forward(std::int32_t token, std::int32_t pos) override;

  void forward_prefill(std::int32_t token, std::int32_t pos) override;

  std::span<const float> get_logits() const override;

  std::size_t memory_usage() const override;

protected:
  const ModelCuda &model;
  Config &config;
  ArenaAlloc alloc;
  cuda::ArenaAlloc cuda_alloc;
  std::size_t kv_size;

  DataType kv_dtype;

  Tensor logits_cpu;

  float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *attn, *logits;
  std::vector<void *> k_cache, v_cache;

protected:
  void forward_block(std::size_t block_id, std::int32_t pos, std::int32_t kv_sink, std::int32_t kv_pos,
                     std::int32_t kv_len);
};

} // namespace tinyllm
