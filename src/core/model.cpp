#include "model.hpp"
#include "../utils/precision.hpp"
#include "../utils/safetensors_reader.hpp"

#include <cassert>
#include <cstddef>
#include <format>

namespace tinyllm {

Model::Model(Config &cfg) : config(cfg) {}

void Model::load_weights() {
  SafeTensorsReader reader(config.model_path, alloc);

  auto get_tensor = [&](const std::string &name) {
    const auto &meta = reader.get_metadata(name);
    auto dtype = string_to_dtype(meta.dtype);
    return Tensor(meta.shape, dtype, meta.data);
  };

  weight.embed = get_tensor("model.embed_tokens.weight");
  weight.norm = get_tensor("model.norm.weight");

  dtype = weight.embed.dtype;

  weight.blocks.resize(config.num_hidden_layers);

  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    auto &block = weight.blocks[i];
    block.attn_q = get_tensor(std::format("model.layers.{}.self_attn.q_proj.weight", i));
    block.attn_k = get_tensor(std::format("model.layers.{}.self_attn.k_proj.weight", i));
    block.attn_v = get_tensor(std::format("model.layers.{}.self_attn.v_proj.weight", i));
    block.attn_o = get_tensor(std::format("model.layers.{}.self_attn.o_proj.weight", i));
    block.mlp_down = get_tensor(std::format("model.layers.{}.mlp.down_proj.weight", i));
    block.mlp_gate = get_tensor(std::format("model.layers.{}.mlp.gate_proj.weight", i));
    block.mlp_up = get_tensor(std::format("model.layers.{}.mlp.up_proj.weight", i));
    block.input_norm = get_tensor(std::format("model.layers.{}.input_layernorm.weight", i));
    block.post_norm = get_tensor(std::format("model.layers.{}.post_attention_layernorm.weight", i));

    if (config.model_type == "qwen2") {
      block.attn_q_bias = get_tensor(std::format("model.layers.{}.self_attn.q_proj.bias", i));
      block.attn_k_bias = get_tensor(std::format("model.layers.{}.self_attn.k_proj.bias", i));
      block.attn_v_bias = get_tensor(std::format("model.layers.{}.self_attn.v_proj.bias", i));
    }

    if (config.model_type == "qwen3") {
      block.attn_q_norm = get_tensor(std::format("model.layers.{}.self_attn.q_norm.weight", i));
      block.attn_k_norm = get_tensor(std::format("model.layers.{}.self_attn.k_norm.weight", i));
    }
  }

  if (config.tie_word_embeddings) {
    weight.lm_head = weight.embed;
  } else {
    weight.lm_head = get_tensor("lm_head.weight");
  }
}

void Model::to_dtype(DataType new_dtype) {
  ArenaAlloc other_alloc;
  bool in_old_alloc = false;
  bool in_new_alloc = false;

  auto convert = [&](Tensor &tensor) {
    if (tensor.data.empty()) {
      return;
    }
    if (tensor.dtype == new_dtype) {
      in_old_alloc = true;
      return;
    }
    auto tensor2 = Tensor::alloc(other_alloc, new_dtype, tensor.shape);
    if (tensor.dtype == DataType::F32 && new_dtype == DataType::BF16) {
      auto *data_ptr = reinterpret_cast<bf16_t *>(tensor2.data.data());
      copy_fp32_to_bf16_n(tensor.as<float>(), tensor.data.size() / sizeof(float), data_ptr);
      tensor.data = tensor2.data;
      tensor.dtype = DataType::BF16;
    } else if (tensor.dtype == DataType::BF16 && new_dtype == DataType::F32) {
      auto *data_ptr = reinterpret_cast<float *>(tensor.data.data());
      copy_bf16_to_fp32_n(tensor.as<bf16_t>(), tensor.data.size() / sizeof(bf16_t), data_ptr);
      tensor.data = tensor2.data;
      tensor.dtype = DataType::F32;
    } else {
      throw std::runtime_error("Unsupported dtype conversion");
    }
    in_new_alloc = true;
  };

  for (auto &block : weight.blocks) {
    convert(block.attn_q);
    convert(block.attn_k);
    convert(block.attn_v);
    convert(block.attn_o);

    convert(block.attn_q_bias);
    convert(block.attn_k_bias);
    convert(block.attn_v_bias);

    convert(block.attn_q_norm);
    convert(block.attn_k_norm);
    convert(block.mlp_down);
    convert(block.mlp_gate);
    convert(block.mlp_up);
    convert(block.input_norm);
    convert(block.post_norm);
  }

  convert(weight.embed);
  convert(weight.norm);
  if (config.tie_word_embeddings) {
    weight.lm_head = weight.embed;
  } else {
    convert(weight.lm_head);
  }

  if (in_old_alloc && in_new_alloc) {
    alloc.merge(other_alloc);
  } else if (in_old_alloc) {
    // do nothing
  } else if (in_new_alloc) {
    alloc.swap(other_alloc);
  }
  dtype = new_dtype;
}

std::size_t Model::memory_usage() const { return alloc.total_allocated; }

} // namespace tinyllm
