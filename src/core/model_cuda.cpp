#include "model_cuda.hpp"
#include "model.hpp"

namespace tinyllm {

ModelCuda::ModelCuda(const Model &model) : config(model.config), dtype(model.dtype) {
  weight.embed = cuda_alloc.upload(model.weight.embed.data);
  weight.norm = cuda_alloc.upload(model.weight.norm.data);
  weight.blocks.resize(config.num_hidden_layers);

  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    auto &block_cpu = model.weight.blocks[i];
    auto &block = weight.blocks[i];
    block.attn_q = cuda_alloc.upload(block_cpu.attn_q.data);
    block.attn_k = cuda_alloc.upload(block_cpu.attn_k.data);
    block.attn_v = cuda_alloc.upload(block_cpu.attn_v.data);
    block.attn_o = cuda_alloc.upload(block_cpu.attn_o.data);
    block.mlp_down = cuda_alloc.upload(block_cpu.mlp_down.data);
    block.mlp_gate = cuda_alloc.upload(block_cpu.mlp_gate.data);
    block.mlp_up = cuda_alloc.upload(block_cpu.mlp_up.data);
    block.input_norm = cuda_alloc.upload(block_cpu.input_norm.data);
    block.post_norm = cuda_alloc.upload(block_cpu.post_norm.data);

    if (config.model_type == "qwen2") {
      block.attn_k_bias = cuda_alloc.upload(block_cpu.attn_k_bias.data);
      block.attn_q_bias = cuda_alloc.upload(block_cpu.attn_q_bias.data);
      block.attn_v_bias = cuda_alloc.upload(block_cpu.attn_v_bias.data);
    } else {
      block.attn_k_bias = nullptr;
      block.attn_q_bias = nullptr;
      block.attn_v_bias = nullptr;
    }

    if (config.model_type == "qwen3") {
      block.attn_q_norm = cuda_alloc.upload(block_cpu.attn_q_norm.data);
      block.attn_k_norm = cuda_alloc.upload(block_cpu.attn_k_norm.data);
    } else {
      block.attn_q_norm = nullptr;
      block.attn_k_norm = nullptr;
    }
  }
  if (config.tie_word_embeddings) {
    weight.lm_head = weight.embed;
  } else {
    weight.lm_head = cuda_alloc.upload(model.weight.lm_head.data);
  }
}

std::size_t ModelCuda::memory_usage() const { return cuda_alloc.total_allocated; }

} // namespace tinyllm