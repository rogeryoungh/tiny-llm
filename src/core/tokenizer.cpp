#include "tokenizer.hpp"
#include "../utils/utf8.hpp"
#include <bit>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>

namespace tinyllm {

TokenizerTrieNode::TokenizerTrieNode(std::int32_t id) : token_id(id) {}

void TokenizerTrieNode::insert(const std::string_view word, std::int32_t token_id) {
  TokenizerTrieNode *p = this;
  for (const std::uint8_t c : word) {
    const std::uint32_t c0 = c % 64, c1 = c / 64;
    const std::uint64_t mask = 1ULL << c0;
    auto &cmask = p->mask64[c1];
    auto &cc = p->children[c1];

    if (cmask & mask) {
      int u = std::popcount(cmask & (mask - 1));
      p = &cc[u];

    } else {
      cmask |= mask;
      int u = std::popcount(cmask & (mask - 1));
      cc.insert(cc.begin() + u, TokenizerTrieNode{});
      p = &cc[u];
    }
  }
  p->token_id = token_id;
}

const TokenizerTrieNode *TokenizerTrieNode::get(std::uint8_t c) const {
  const std::uint32_t c0 = c % 64, c1 = c / 64;
  const std::uint64_t mask = 1ULL << c0;

  if (mask & mask64[c1]) {
    int u = std::popcount(mask64[c1] & (mask - 1));
    return &children[c1][u];
  } else {
    return nullptr;
  }
}

Tokenizer::Tokenizer(Config &cfg) : config(cfg) {
  nlohmann::json tokenizer_config_json;
  std::ifstream file(config.model_path / "tokenizer_config.json");
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer config file");
  }
  file >> tokenizer_config_json;

  if (tokenizer_config_json.value("add_bos_token", false)) {
    bos_token_id = config.bos_token_id;
  }

  if (tokenizer_config_json.value("add_eos_token", false)) {
    eos_token_id = config.eos_token_id;
  }
}

void Tokenizer::_add_token(const std::string &key, std::int32_t token_id) {
  std::string key_decoded;
  if (byte_fallback) {
    if (key.front() == '<' && key.back() == '>') {
      auto get_hex_value = [](char c) -> int {
        if (c >= '0' && c <= '9')
          return c - '0';
        if (c >= 'a' && c <= 'f')
          return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
          return c - 'A' + 10;
        return -1;
      };
      std::string s;
      for (std::size_t i = 3; i + 2 < key.size(); i += 2) {
        int c0 = get_hex_value(key[i]);
        int c1 = get_hex_value(key[i + 1]);
        if (c0 < 0 || c1 < 0) {
          key_decoded.clear();
          break;
        }
        key_decoded += (c0 << 4) | c1;
      }
    }
  } else {
    key_decoded = gpt2_unicode_to_bytes(key);
  }
  if (key_decoded.empty()) {
    key_decoded = replace_unicode_space(key);
  }
  // std::cout << "[DEBUG] Tokenizer: key = `" << key << "`, token_id = " << token_id << ", decoded = `" << key_decoded
  //           << "`" << std::endl;
  vocab[token_id] = key_decoded;
  root.insert(key_decoded, token_id);
}

void Tokenizer::load_trie() {
  nlohmann::json tokenizer_json;
  std::ifstream tokenizer_file(config.model_path / "tokenizer.json");
  if (!tokenizer_file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer file");
  }
  tokenizer_file >> tokenizer_json;

  const auto &vocab_json = tokenizer_json["model"]["vocab"];
  const auto &added_tokens = tokenizer_json["added_tokens"];
  vocab.resize(vocab_json.size() + added_tokens.size() + 100);

  byte_fallback = tokenizer_json["model"].value("byte_fallback", false);

  // Load vocabulary and other settings from the JSON
  for (const auto &[key, value] : vocab_json.items()) {
    auto token_id = value.get<std::int32_t>();
    _add_token(key, token_id);
  }
  for (const auto &data : added_tokens) {
    auto key = data["content"].get<std::string>();
    auto token_id = data["id"].get<std::int32_t>();
    if (vocab[token_id].empty()) {
      _add_token(key, token_id);
    }
  }
}

std::vector<std::int32_t> Tokenizer::encode_raw(const std::string &text) {
  std::vector<std::int32_t> tokens;
  std::size_t i = 0;
  while (i < text.size()) {
    const TokenizerTrieNode *p = &root, *valid_p = nullptr;
    std::size_t j = i, valid_j = i;
    while (j < text.size()) {
      const auto uc = static_cast<std::uint8_t>(text[j]);
      auto *nxt = p->get(uc);
      if (!nxt) {
        break;
      } else {
        p = nxt;
        j += 1;
        if (p->token_id >= 0) {
          valid_p = p;
          valid_j = j;
        }
      }
    }
    if (valid_p) {
      tokens.push_back(valid_p->token_id);
      i = valid_j;
    } else {
      tokens.push_back(-1);
      i += 1;
    }
  }
  return tokens;
}

std::vector<std::int32_t> Tokenizer::encode(const std::string &text) {
  std::vector<std::int32_t> tokens;
  if (bos_token_id >= 0) {
    tokens.push_back(bos_token_id);
  }
  tokens.append_range(encode_raw(text));
  if (eos_token_id >= 0) {
    tokens.push_back(eos_token_id);
  }
  return tokens;
}

std::vector<std::int32_t> Tokenizer::encode_padding(const std::string &text, std::size_t padding_size) {
  std::vector<std::int32_t> tokens_raw = encode_raw(text);
  std::vector<std::int32_t> space_tokens = encode_raw(" ");
  std::size_t eos_bos_size = (bos_token_id >= 0) + (eos_token_id >= 0);
  std::int64_t needed_padding = padding_size - static_cast<std::int64_t>(tokens_raw.size() + eos_bos_size);
  if (needed_padding <= 0) {
    return tokens_raw;
  }
  std::size_t repeat = needed_padding / space_tokens.size();
  std::vector<std::int32_t> tokens;
  tokens.reserve(padding_size);
  if (bos_token_id >= 0) {
    tokens.push_back(bos_token_id);
  }
  for (std::size_t i = 0; i < repeat; ++i) {
    tokens.append_range(space_tokens);
  }
  tokens.append_range(tokens_raw);
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

std::size_t Tokenizer::memory_usage() const {
  std::size_t size = sizeof(Tokenizer) + vocab.size() * sizeof(std::string);
  for (const auto &word : vocab) {
    size += word.capacity();
  }
  auto dfs_trie = [](auto &&self, const TokenizerTrieNode &node) -> std::size_t {
    std::size_t size = sizeof(TokenizerTrieNode);
    for (const auto &c : node.children) {
      for (const auto &c2 : c) {
        size += self(self, c2);
      }
    }
    return size;
  };
  size += dfs_trie(dfs_trie, root);
  return size;
}

} // namespace tinyllm