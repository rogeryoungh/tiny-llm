#include "tokenizer.hpp"
#include "../utils/utf8.hpp"
#include <bit>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>

namespace tinyllm {

void PopcountTrie::insert(const std::string_view word, std::int32_t token_id) {
  Node *p = &root;
  for (std::size_t i = 0; i < word.size(); ++i) {
    const std::uint8_t uc = std::uint8_t(word[i]);
    const std::uint32_t c0 = uc % 64, c1 = uc / 64;
    const std::uint64_t mask = 1ULL << c0;
    auto &cmask = p->mask64[c1];
    auto &cc = p->children[c1];
    int u = std::popcount(cmask & (mask - 1));
    if (cmask & mask) {
      p = &cc[u];
    } else {
      cmask |= mask;
      int u = std::popcount(cmask & (mask - 1));
      cc.insert(cc.begin() + u, Node{});
      p = &cc[u];
    }
  }
  p->token_id = token_id;
}

std::pair<std::int32_t, std::int32_t> PopcountTrie::longest_match(const std::string_view word) const {
  std::int32_t token_id = -1;
  std::int32_t length = 0;
  const Node *p = &root;
  for (std::size_t i = 0; i < word.size(); ++i) {
    const std::uint8_t uc = std::uint8_t(word[i]);
    const std::uint32_t c0 = uc % 64, c1 = uc / 64;
    const std::uint64_t mask = 1ULL << c0;
    const auto cmask = p->mask64[c1];
    std::uint64_t u = std::popcount(cmask & (mask - 1));
    if (cmask & mask) {
      p = &p->children[c1][u];
      if (p->token_id >= 0) {
        token_id = p->token_id;
        length = i + 1;
      }
    } else {
      break;
    }
  }
  return {token_id, length};
}

std::size_t PopcountTrie::memory_usage() const {
  std::size_t size = sizeof(PopcountTrie) - sizeof(root);
  auto dfs_trie = [&](auto &&self, const Node &node) -> std::size_t {
    std::size_t node_size = sizeof(node);
    for (const auto &child : node.children) {
      for (const auto &sub_node : child) {
        node_size += self(self, sub_node);
      }
      node_size += (child.capacity() - child.size()) * sizeof(Node);
    }
    return node_size;
  };
  size += dfs_trie(dfs_trie, root);
  return size;
}

void StaticDATrie::load_trie(const PopcountTrie &trie) { root_id = _insert(trie.root); }

std::int32_t StaticDATrie::_insert(const PopcountTrie::Node &node) {
  const std::uint64_t full_mask = node.mask64[0] | node.mask64[1] | node.mask64[2] | node.mask64[3];
  const std::int32_t id = node.token_id;

  if (full_mask == 0) {
    return -id - 2;
  }

  std::array<std::int32_t, 4> children{};

  for (std::size_t i = 0; i < 4; ++i) {
    std::vector<std::int32_t> child_ids;
    for (const auto &child : node.children[i]) {
      std::int32_t child_id = _insert(child);
      child_ids.push_back(child_id);
    }
    children[i] = next.size();
    next.insert_range(next.end(), child_ids);
  }
  meta.emplace_back(id, node.mask64, children);
  return std::int32_t(meta.size()) - 1;
}

std::size_t StaticDATrie::memory_usage() const {
  std::size_t size = sizeof(StaticDATrie);
  size += meta.size() * sizeof(Meta);
  size += next.size() * sizeof(std::int32_t);
  return size;
}

std::pair<std::int32_t, std::int32_t> StaticDATrie::longest_match(const std::string_view data) const {
  std::int32_t token_id = -1;
  std::int32_t length = 0;

  std::int32_t p = root_id;
  for (std::size_t i = 0; i < data.size(); ++i) {
    const std::uint8_t uc = std::uint8_t(data[i]);
    const std::uint32_t c0 = uc % 64, c1 = uc / 64;
    const std::uint64_t mask = 1ULL << c0;
    const auto cmask = meta[p].mask64[c1];
    if (cmask & mask) {
      std::uint64_t u = std::popcount(cmask & (mask - 1));
      p = next[meta[p].children[c1] + u];
      if (p < -1) {
        token_id = -p - 2;
        length = i + 1;
        break;
      }
      if (meta[p].token_id >= 0) {
        token_id = meta[p].token_id;
        length = i + 1;
      }
    } else {
      break;
    }
  }
  return {token_id, length};
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

std::string Tokenizer::_decoded_token_key(const std::string &key) const {
  if (byte_fallback) {
    char c0 = -1, c1 = -1;
    if (key.starts_with("<0x") && key.ends_with(">")) {
      auto get_hex_value = [](char c) -> int {
        if (c >= '0' && c <= '9') {
          return c - '0';
        } else {
          c &= 0xDF; // Convert to uppercase
          if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
          } else {
            return -1;
          }
        }
      };
      c0 = get_hex_value(key[3]);
      c1 = get_hex_value(key[4]);
    }
    if (c0 >= 0 && c1 >= 0) {
      return std::string(1, (c0 << 4) | c1);
    } else {
      return replace_unicode_space(key);
    }
  } else {
    return gpt2_unicode_to_bytes(key);
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
  const auto &added_tokens = tokenizer_json["added_tokens"];
  vocab.resize(vocab_json.size() + added_tokens.size() + 100);

  byte_fallback = tokenizer_json["model"].value("byte_fallback", false);

  PopcountTrie popcnt_trie;

  // Load vocabulary and other settings from the JSON
  for (const auto &[key, value] : vocab_json.items()) {
    auto token_id = value.get<std::int32_t>();
    const auto decoded = _decoded_token_key(key);
    popcnt_trie.insert(decoded, token_id);
    vocab[token_id] = decoded;
  }
  for (const auto &data : added_tokens) {
    auto key = data["content"].get<std::string>();
    auto token_id = data["id"].get<std::int32_t>();
    const auto decoded = _decoded_token_key(key);
    popcnt_trie.insert(decoded, token_id);
    vocab[token_id] = decoded;
  }
  trie.load_trie(popcnt_trie);
}

std::vector<std::int32_t> Tokenizer::encode_raw(const std::string &text) {
  std::vector<std::int32_t> tokens;
  std::size_t i = 0;
  while (i < text.size()) {
    const std::string_view sub = {text.begin() + i, text.end()};
    auto [token_id, length] = trie.longest_match(sub);
    if (token_id >= 0) {
      tokens.push_back(token_id);
      i += length;
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
  size += trie.memory_usage();
  return size;
}

} // namespace tinyllm
