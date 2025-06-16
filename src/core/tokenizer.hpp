#pragma once

#include "config.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace tinyllm {

struct TokenizerTrieNode {
  std::int32_t token_id;

  std::array<std::uint64_t, 4> mask64{};
  std::array<std::vector<TokenizerTrieNode>, 4> children{};

  TokenizerTrieNode(std::int32_t id = -1);

  void insert(const std::string_view word, std::int32_t token_id);

  const TokenizerTrieNode *get(std::uint8_t c) const;
};

struct Tokenizer {
  Config &config;
  std::vector<std::string> vocab;
  TokenizerTrieNode root;
  std::int32_t bos_token_id = -1;
  std::int32_t eos_token_id = -1;
  bool byte_fallback = false;

  Tokenizer(Config &cfg);

  void load_trie();

  std::vector<std::int32_t> encode(const std::string &text);

  std::vector<std::int32_t> encode_raw(const std::string &text);

  std::vector<std::int32_t> encode_padding(const std::string &text, std::size_t padding_size);

  std::size_t memory_usage() const;

  std::string _debug_decode(const std::vector<std::int32_t> &tokens);

protected:
  void _add_token(const std::string &key, std::int32_t token_id);
};

} // namespace tinyllm
