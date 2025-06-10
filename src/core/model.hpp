#pragma once

#include "config.hpp"
#include "tensor.hpp"

#include <vector>

namespace tinyllm {

struct Block {
  Tensor attn_q, attn_k, attn_v, attn_o;
  Tensor attn_q_bias, attn_k_bias, attn_v_bias;
  Tensor mlp_down, mlp_gate, mlp_up;
  Tensor input_norm, post_norm;
};

struct ModelWeights {
  std::vector<Block> blocks;
  Tensor embed;
  Tensor norm;
  Tensor lm_head;
};

struct Model {
  Config &config;
  TensorAlloc alloc;

  ModelWeights weight;

  Model(Config &cfg);

  ~Model();

  void load_weights();

  void _permute_qk(Tensor &q, std::size_t n_heads);
};

} // namespace tinyllm
