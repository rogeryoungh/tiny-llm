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
  hidden_size = config_json.at("hidden_size").get<std::uint32_t>();
  intermediate_size = config_json.at("intermediate_size").get<std::uint32_t>();
  max_position_embeddings = config_json.at("max_position_embeddings").get<std::uint32_t>();
  num_key_value_heads = config_json.at("num_key_value_heads").get<std::uint32_t>();
  num_hidden_layers = config_json.at("num_hidden_layers").get<std::uint32_t>();
  num_attention_heads = config_json.at("num_attention_heads").get<std::uint32_t>();

  if (config_json.contains("bos_token_id")) {
    bos_token_id = config_json.at("bos_token_id").get<std::uint32_t>();
  } else {
    bos_token_id = static_cast<std::uint32_t>(-1);
  }

  if (config_json.contains("eos_token_id")) {
    eos_token_id = config_json.at("eos_token_id").get<std::uint32_t>();
  } else {
    eos_token_id = static_cast<std::uint32_t>(-1);
  }

  vocab_size = config_json.at("vocab_size").get<std::uint32_t>();
}

} // namespace tinyllm
