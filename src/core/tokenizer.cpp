#include "tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace tinyllm {

TokenizerTrieNode::TokenizerTrieNode(std::int32_t id) : token_id(id) {}

void TokenizerTrieNode::insert(const std::string_view word, std::int32_t token_id) {
  TokenizerTrieNode *p = this;
  for (const char c : word) {
    std::cout << "Inserting character: " << int(c) << std::endl;
    if (p->children.empty()) {
      p->children.resize(256); // Assuming ASCII characters
    }
    auto *q = &p->children[static_cast<std::uint8_t>(c)];
    p = q;
  }
  p->token_id = token_id;
}

Tokenizer::Tokenizer() = default;

void Tokenizer::load(const std::filesystem::path &path) {
  nlohmann::json tokenizer_config_json;
  std::ifstream file(path / "tokenizer_config.json");
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer config file");
  }
  file >> tokenizer_config_json;

  if (tokenizer_config_json["tokenizer_class"] != "LlamaTokenizer") {
    throw std::runtime_error("Unsupported tokenizer class: " +
                             tokenizer_config_json["tokenizer_class"].get<std::string>());
  }

  nlohmann::json tokenizer_json;
  std::ifstream tokenizer_file(path / "tokenizer.json");
  if (!tokenizer_file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer file");
  }
  tokenizer_file >> tokenizer_json;

  const auto &vocab_json = tokenizer_json["model"]["vocab"];

  // Load vocabulary and other settings from the JSON
  for (const auto &[key, value] : vocab_json.items()) {
    auto token_id = value.get<std::int32_t>();
    vocab.push_back(key);
    std::cout << "Inserting token: " << key << " with ID: " << token_id << std::endl;
    root.insert(key, token_id);
  }
}

std::vector<std::int32_t> Tokenizer::encode(const std::string &text) {
  // Tokenization logic goes here
  return {};
}

} // namespace tinyllm