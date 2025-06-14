#pragma once

#include "../../core/config.hpp"
#include "../../core/model.hpp"
#include "../../core/tensor.hpp"
#include "../inference_ctx.hpp"
#include "arena_alloc.hpp"

namespace tinyllm {

namespace cuda {

struct Block {
  void *attn_q, *attn_k, *attn_v, *attn_o;
  void *attn_q_bias, *attn_k_bias, *attn_v_bias;
  void *attn_q_norm, *attn_k_norm;
  void *mlp_down, *mlp_gate, *mlp_up;
  void *input_norm, *post_norm;
};

struct ModelWeights {
  std::vector<Block> blocks;
  void *embed;
  void *norm;
  void *lm_head;
};

struct InferVars {
  float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *attn, *logits;
  std::vector<void *> k_cache, v_cache;
};

} // namespace cuda

struct InferenceBackendCUDA : InferenceBackend {

  InferenceBackendCUDA(Model &model, std::size_t kv_size, DataType kv_dtype = DataType::F32);

  void forward(std::int32_t token, std::int32_t pos) override;

  void forward_prefill(std::int32_t token, std::int32_t pos) override;

  std::uint32_t argmax() const override;

  std::size_t memory_usage() const override;

protected:
  const Model &model;
  Config &config;
  ArenaAlloc alloc;
  cuda::ArenaAlloc cuda_alloc;
  std::size_t kv_size;

  DataType kv_dtype;
  cuda::ModelWeights weights;

  Tensor logits_cpu;

  float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *attn, *logits;
  std::vector<void *> k_cache, v_cache;

protected:
  void forward_block(std::size_t block_id, std::int32_t pos, std::int32_t kv_sink,
                     std::int32_t kv_pos, std::int32_t kv_len);
};

} // namespace tinyllm
