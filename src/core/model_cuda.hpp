#pragma once

#include "../infer/cuda/arena_alloc.hpp"
#include "config.hpp"
#include "tensor.hpp"

namespace tinyllm {

struct Model;

struct ModelCuda {
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

  Config &config;
  cuda::ArenaAlloc cuda_alloc;
  DataType dtype;

  ModelWeights weight;

  ModelCuda(const Model &model);

  std::size_t memory_usage() const;
};

} // namespace tinyllm
