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
    const std::size_t hidden_dim = config.hidden_size;
    const std::size_t head_dim = config.hidden_size / config.num_attention_heads;

    _permute_qk(block.attn_q, config.num_attention_heads, head_dim, hidden_dim);

    _permute_qk(block.attn_k, config.num_key_value_heads, head_dim, hidden_dim);
  }

  if (config.tie_word_embeddings) {
    weight.lm_head = weight.embed;
  } else {
    weight.lm_head = _create_tensor("lm_head.weight");
  }
}

void Model::_permute_qk(Tensor &q, std::size_t heads, std::size_t head_dim, std::size_t hidden_dim) {
  const std::size_t block_elems = head_dim * hidden_dim;

  std::vector<float> wr(block_elems);

  const std::size_t half = head_dim / 2;

  for (std::size_t h = 0; h < heads; ++h) {
    auto *base = q.as<float>() + h * block_elems;

    std::copy_n(base, block_elems, wr.data());

    for (std::size_t i = 0; i < head_dim; ++i) {
      const std::size_t block = i / half;
      const std::size_t idx = i % half;
      const std::size_t new_row = idx * 2 + block;

      const float *src_row = wr.data() + i * hidden_dim;
      float *dst_row = base + new_row * hidden_dim;

      std::copy_n(src_row, hidden_dim, dst_row);
    }
  }
}

Model::~Model() {
  // Deallocate all tensors
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
  }
  alloc.dealloc(weight.embed.data);
  alloc.dealloc(weight.norm.data);
}

} // namespace tinyllm
