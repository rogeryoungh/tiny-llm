#pragma once

#include "config.hpp"
#include "tensor.hpp"

#include <vector>

namespace tinyllm {

struct Block {
  Tensor attn_q, attn_k, attn_v, attn_o;
  Tensor mlp_down, mlp_gate, mlp_up;
  Tensor input_norm, post_norm;
};

struct ModelWeights {
  std::vector<Block> blocks;
  Tensor embed;
  Tensor norm;
};

struct Model {
  Config &config;
  TensorAlloc alloc;

  ModelWeights weight;

  Model(Config &cfg);

  ~Model();

  void load_weights();
};

} // namespace tinyllm
