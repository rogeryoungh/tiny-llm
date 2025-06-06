#include "model.hpp"
#include "../utils/safetensors_reader.hpp"

#include <format>

namespace tinyllm {

Model::Model(Config &cfg) : config(cfg) {}

void Model::load_weights() {
  SafeTensorsReader reader(config.model_path);

  auto _create_tensor = [&](const std::string &name) {
    const auto &meta = reader.get_tensor_meta(name);
    Tensor tensor;
    if (meta.shape.size() > 4 || meta.shape.size() < 1) {
      throw std::runtime_error("Invalid tensor shape for " + name);
    } else {
      std::copy(meta.shape.begin(), meta.shape.end(), tensor.shape.begin());
    }
    tensor.dtype = string_to_dtype(meta.dtype);
    tensor.data = alloc.alloc(meta.data_offsets[1] - meta.data_offsets[0]);
    reader.load_tensor(name, tensor.data);
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
