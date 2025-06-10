#include "model.hpp"
#include "../utils/safetensors_reader.hpp"

#include <cassert>
#include <cstring>
#include <format>

namespace tinyllm {

Model::Model(Config &cfg) : config(cfg) {}

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
    auto tensor = alloc.alloc_fp32(shape);
    reader.load_tensor(name, tensor.data, DataType::F32);
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

    _permute_qk(block.attn_q, config.num_attention_heads);
    _permute_qk(block.attn_k, config.num_key_value_heads);

    if (config.model_type == "qwen2") {
      block.attn_q_bias = _create_tensor(std::format("model.layers.{}.self_attn.q_proj.bias", i));
      block.attn_k_bias = _create_tensor(std::format("model.layers.{}.self_attn.k_proj.bias", i));
      block.attn_v_bias = _create_tensor(std::format("model.layers.{}.self_attn.v_proj.bias", i));

      _permute_qk(block.attn_q_bias, config.num_attention_heads);
      _permute_qk(block.attn_k_bias, config.num_key_value_heads);
    }
  }

  if (config.tie_word_embeddings) {
    weight.lm_head = weight.embed;
  } else {
    weight.lm_head = _create_tensor("lm_head.weight");
  }
}

void Model::_permute_qk(Tensor &q, std::size_t heads) {
  const size_t row = q.shape[0], col = q.shape[1];
  const size_t head_dim = config.hidden_size / config.num_attention_heads;
  const size_t half_dim = head_dim / 2;

  std::vector<float> tmp(row * col);
  std::copy_n(q.as<float>(), row * col, tmp.data());

  if (col > 1) {
    // 2D case: permute within each head
    for (size_t h = 0; h < heads; ++h) {
      float *head_data = q.as<float>() + h * head_dim * row;
      const float *src_data = tmp.data() + h * head_dim * row;

      for (size_t i = 0; i < head_dim; ++i) {
        size_t new_idx = (i % half_dim) * 2 + (i / half_dim);
        std::copy_n(src_data + i * row, row, head_data + new_idx * row);
      }
    }
  } else {
    // 1D case: permute rows
    for (size_t h = 0; h < heads; ++h) {
      for (size_t i = 0; i < head_dim; ++i) {
        size_t old_idx = h * head_dim + (i / 2) + (i % 2) * half_dim;
        q.as<float>()[h * head_dim + i] = tmp[old_idx];
      }
    }
  }
}

Model::~Model() {
  for (auto &block : weight.blocks) {
    alloc.dealloc(block.attn_q.data);
    alloc.dealloc(block.attn_k.data);
    alloc.dealloc(block.attn_v.data);
    alloc.dealloc(block.attn_o.data);
    alloc.dealloc(block.mlp_down.data);
    alloc.dealloc(block.mlp_gate.data);
    alloc.dealloc(block.mlp_up.data);
    alloc.dealloc(block.input_norm.data);
    alloc.dealloc(block.post_norm.data);

    if (!block.attn_q_bias.data.empty()) {
      alloc.dealloc(block.attn_q_bias.data);
      alloc.dealloc(block.attn_k_bias.data);
      alloc.dealloc(block.attn_v_bias.data);
    }
  }
  alloc.dealloc(weight.embed.data);
  alloc.dealloc(weight.norm.data);
  if (weight.lm_head.data.empty()) {
    alloc.dealloc(weight.lm_head.data);
  }
}

} // namespace tinyllm
