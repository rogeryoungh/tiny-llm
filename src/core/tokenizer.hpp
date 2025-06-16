#pragma once

#include "config.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace tinyllm {

struct PopcountTrie {
  struct Node {
    std::int32_t token_id = -1;
    std::array<std::uint64_t, 4> mask64{};
    std::array<std::vector<Node>, 4> children{};
  };

  Node root{};

  void insert(const std::string_view word, std::int32_t token_id);

  std::pair<std::int32_t, std::int32_t> longest_match(const std::string_view data) const;

  std::size_t memory_usage() const;
};

struct StaticDATrie {
  struct Meta {
    std::int32_t token_id = -1;
    std::array<std::uint64_t, 4> mask64{};
    std::array<std::int32_t, 4> children{};
  };
  std::vector<Meta> meta;
  std::vector<std::int32_t> next; // if < -1, is `-token_id - 2`
  std::int32_t root_id = -1;
  
  void load_trie(const PopcountTrie &trie);

  std::pair<std::int32_t, std::int32_t> longest_match(const std::string_view data) const;

  std::size_t memory_usage() const;

protected:
  std::int32_t _insert(const PopcountTrie::Node &node);

  std::string _encode_token(std::string s) const;
};

struct Tokenizer {
  Config &config;
  std::vector<std::string> vocab;
  StaticDATrie trie;
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
  std::string _decoded_token_key(const std::string &key) const;
};

} // namespace tinyllm
