#pragma once

#include <filesystem>

namespace tinyllm {

struct Config {
  std::filesystem::path model_path;
  std::string model_type;

  std::int32_t bos_token_id;
  std::int32_t eos_token_id;
  std::int32_t vocab_size;

  std::string hidden_act;
  std::int32_t hidden_size;
  std::int32_t intermediate_size;
  std::int32_t max_position_embeddings;
  std::int32_t num_key_value_heads;
  std::int32_t num_hidden_layers;
  std::int32_t num_attention_heads;
  float rope_theta;
  float rms_norm_eps;

  bool tie_word_embeddings;

  Config(const std::filesystem::path &path);
};

} // namespace tinyllm
