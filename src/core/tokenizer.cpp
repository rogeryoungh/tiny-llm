#include "tokenizer.hpp"
#include "../utils/utf8.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace tinyllm {

TokenizerTrieNode::TokenizerTrieNode(std::int32_t id) : token_id(id) {}

void TokenizerTrieNode::insert(const std::string_view word, std::int32_t token_id) {
  TokenizerTrieNode *p = this;
  for (const char c : word) {
    if (p->children.empty()) {
      p->children.resize(256); // Assuming ASCII characters
    }
    auto *q = &p->children[static_cast<std::uint8_t>(c)];
    p = q;
  }
  p->token_id = token_id;
}

Tokenizer::Tokenizer(Config &cfg) : config(cfg) {
  nlohmann::json tokenizer_config_json;
  std::ifstream file(config.model_path / "tokenizer_config.json");
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer config file");
  }
  file >> tokenizer_config_json;

  if (tokenizer_config_json["tokenizer_class"] != "LlamaTokenizer") {
    throw std::runtime_error("Unsupported tokenizer class: " +
                             tokenizer_config_json["tokenizer_class"].get<std::string>());
  }

  if (tokenizer_config_json["add_bos_token"].get<bool>()) {
    bos_token_id = config.bos_token_id;
  }

  if (tokenizer_config_json["add_eos_token"].get<bool>()) {
    eos_token_id = config.eos_token_id;
  }
}

void Tokenizer::load_trie() {
  nlohmann::json tokenizer_json;
  std::ifstream tokenizer_file(config.model_path / "tokenizer.json");
  if (!tokenizer_file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer file");
  }
  tokenizer_file >> tokenizer_json;

  const auto &vocab_json = tokenizer_json["model"]["vocab"];
  vocab.resize(vocab_json.size() + 100);

  // Load vocabulary and other settings from the JSON
  for (const auto &[key, value] : vocab_json.items()) {
    auto token_id = value.get<std::int32_t>();
    const auto processed_key = replace_unicode_space(key);
    vocab[token_id] = processed_key;
    root.insert(processed_key, token_id);
  }
}

std::vector<std::int32_t> Tokenizer::encode(const std::string &text) {
  std::vector<std::int32_t> tokens;
  if (bos_token_id >= 0) {
    tokens.push_back(bos_token_id);
  }
  for (std::size_t i = 0; i < text.size(); ++i) {
    TokenizerTrieNode *p = &root, *valid_p = nullptr;
    std::size_t j = i, valid_j = i;
    while (j < text.size()) {
      const auto uc = static_cast<std::uint8_t>(text[j]);
      if (p->children.empty()) {
        break;
      } else {
        p = &p->children[uc];
        j += 1;
        if (p->token_id >= 0) {
          valid_p = p;
          valid_j = j;
        }
      }
    }
    if (valid_p) {
      tokens.push_back(valid_p->token_id);
      i = valid_j - 1;
    } else {
      tokens.push_back(-1);
    }
  }
  if (eos_token_id >= 0) {
    tokens.push_back(eos_token_id);
  }
  return tokens;
}

std::string Tokenizer::_debug_decode(const std::vector<std::int32_t> &tokens) {
  std::string decoded = "[";
  for (const auto token_id : tokens) {
    if (token_id < 0 || token_id >= static_cast<int>(vocab.size())) {
      decoded += "<UNK>"; // Unknown token
    } else {
      decoded += "`" + vocab[token_id] + "`";
    }
    decoded += ", ";
  }
  decoded.push_back(']');
  return decoded;
}

} // namespace tinyllm