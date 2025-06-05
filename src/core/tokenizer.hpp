#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tinyllm {

struct TokenizerTrieNode {
  std::int32_t token_id;
  std::vector<TokenizerTrieNode> children;

  TokenizerTrieNode(std::int32_t id = -1);

  void insert(const std::string_view word, std::int32_t token_id);
};

struct Tokenizer {
  std::vector<std::string> vocab;
  TokenizerTrieNode root;
  std::int32_t bos_token_id = -1;
  std::int32_t eos_token_id = -1;

  Tokenizer();

  void load(const std::filesystem::path &path);

  std::vector<std::int32_t> encode(const std::string &text);

  std::string _debug_decode(const std::vector<std::int32_t> &tokens);
};

} // namespace tinyllm
