#include "config.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace tinyllm {

Config::Config(const fs::path &path) : model_path(path) {
  // Load the model configuration from a JSON file
  const fs::path config_file = path / "config.json";
  std::ifstream file(config_file);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open config file: " + config_file.string());
  }

  nlohmann::json config_json;
  file >> config_json;

  model_type = config_json.at("model_type").get<std::string>();
  hidden_act = config_json.at("hidden_act").get<std::string>();
  hidden_size = config_json.value("hidden_size", -1);
  intermediate_size = config_json.value("intermediate_size", -1);
  num_key_value_heads = config_json.value("num_key_value_heads", -1);
  num_hidden_layers = config_json.value("num_hidden_layers", -1);
  num_attention_heads = config_json.value("num_attention_heads", -1);
  tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  rope_theta = config_json.value("rope_theta", 1000000.0f);
  rms_norm_eps = config_json.value("rms_norm_eps", 1e-6f);
  head_dim = config_json.value("head_dim", hidden_size / num_attention_heads);

  bos_token_id = config_json.value("bos_token_id", -1);
  eos_token_id = config_json.value("eos_token_id", -1);

  max_position_embeddings = config_json.value("max_position_embeddings", 16384);
  if (max_position_embeddings <= 0 || max_position_embeddings > 1e8) {
    max_position_embeddings = 1 << 24;
  }

  vocab_size = config_json.value("vocab_size", -1);

  const fs::path generation_config_file = path / "generation_config.json";

  if (!fs::exists(generation_config_file)) {
    do_sample = false;
    temperature = 1.0f;
    top_p = 0.9f;
    top_k = 50.0f;
  } else {
    std::ifstream gen_file(generation_config_file);
    if (!gen_file.is_open()) {
      throw std::runtime_error("Failed to open generation config file: " + generation_config_file.string());
    }
    nlohmann::json gen_json;
    gen_file >> gen_json;

    do_sample = gen_json.value("do_sample", false);
    temperature = gen_json.value("temperature", 1.0f);
    top_p = gen_json.value("top_p", 1.0f);
    top_k = gen_json.value("top_k", 0);
  }
}

} // namespace tinyllm
