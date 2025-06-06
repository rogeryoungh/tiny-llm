#pragma once

#include <filesystem>

namespace tinyllm {

struct Config {
  std::filesystem::path model_path;
  std::string model_type;

  std::uint32_t bos_token_id;
  std::uint32_t eos_token_id;
  std::uint32_t vocab_size;

  std::string hidden_act;
  std::uint32_t hidden_size;
  std::uint32_t intermediate_size;
  std::uint32_t max_position_embeddings;
  std::uint32_t num_key_value_heads;
  std::uint32_t num_hidden_layers;
  std::uint32_t num_attention_heads;

  Config(const std::filesystem::path &path);
};

} // namespace tinyllm
