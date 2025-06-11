#include "model.hpp"
#include "../utils/safetensors_reader.hpp"

#include <cassert>
#include <format>

namespace tinyllm {

Model::Model(Config &cfg, DataType mem_dtype) : config(cfg), dtype(mem_dtype) {}

void Model::load_weights() {
  SafeTensorsReader reader(config.model_path);

  auto _create_tensor = [&](const std::string &name) {
    const auto &meta = reader.get_tensor_meta(name);
    std::array<std::int32_t, 4> shape{1, 1, 1, 1};
    if (meta.shape.size() > 4 || meta.shape.size() < 1) {
      throw std::runtime_error("Invalid tensor shape for " + name);
    } else {
      std::copy(meta.shape.rbegin(), meta.shape.rend(), shape.begin());
    }
    auto tensor = alloc.alloc(dtype, shape);
    reader.load_tensor(name, tensor.data, dtype);
    return tensor;
  };

  weight.embed = _create_tensor("model.embed_tokens.weight");
  weight.norm = _create_tensor("model.norm.weight");

  weight.blocks.resize(config.num_hidden_layers);

  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    auto &block = weight.blocks[i];
    block.attn_q = _create_tensor(std::format("model.layers.{}.self_attn.q_proj.weight", i));
    block.attn_k = _create_tensor(std::format("model.layers.{}.self_attn.k_proj.weight", i));
    block.attn_v = _create_tensor(std::format("model.layers.{}.self_attn.v_proj.weight", i));
    block.attn_o = _create_tensor(std::format("model.layers.{}.self_attn.o_proj.weight", i));
    block.mlp_down = _create_tensor(std::format("model.layers.{}.mlp.down_proj.weight", i));
    block.mlp_gate = _create_tensor(std::format("model.layers.{}.mlp.gate_proj.weight", i));
    block.mlp_up = _create_tensor(std::format("model.layers.{}.mlp.up_proj.weight", i));
    block.input_norm = _create_tensor(std::format("model.layers.{}.input_layernorm.weight", i));
    block.post_norm = _create_tensor(std::format("model.layers.{}.post_attention_layernorm.weight", i));

    if (config.model_type == "qwen2") {
      block.attn_q_bias = _create_tensor(std::format("model.layers.{}.self_attn.q_proj.bias", i));
      block.attn_k_bias = _create_tensor(std::format("model.layers.{}.self_attn.k_proj.bias", i));
      block.attn_v_bias = _create_tensor(std::format("model.layers.{}.self_attn.v_proj.bias", i));
    }

    if (config.model_type == "qwen3") {
      block.attn_q_norm = _create_tensor(std::format("model.layers.{}.self_attn.q_norm.weight", i));
      block.attn_k_norm = _create_tensor(std::format("model.layers.{}.self_attn.k_norm.weight", i));
    }
  }

  if (config.tie_word_embeddings) {
    weight.lm_head = weight.embed;
  } else {
    weight.lm_head = _create_tensor("lm_head.weight");
  }
}

Model::~Model() {
  for (auto &block : weight.blocks) {
    alloc.dealloc(block.attn_q);
    alloc.dealloc(block.attn_k);
    alloc.dealloc(block.attn_v);
    alloc.dealloc(block.attn_o);
    alloc.dealloc(block.mlp_down);
    alloc.dealloc(block.mlp_gate);
    alloc.dealloc(block.mlp_up);
    alloc.dealloc(block.input_norm);
    alloc.dealloc(block.post_norm);

    alloc.dealloc(block.attn_q_bias);
    alloc.dealloc(block.attn_k_bias);
    alloc.dealloc(block.attn_v_bias);

    alloc.dealloc(block.attn_q_norm);
    alloc.dealloc(block.attn_k_norm);
  }
  alloc.dealloc(weight.embed);
  if (!config.tie_word_embeddings) {
    alloc.dealloc(weight.lm_head);
  }
  alloc.dealloc(weight.norm);
}

} // namespace tinyllm
