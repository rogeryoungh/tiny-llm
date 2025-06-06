#include "core/config.hpp"
#include "core/model.hpp"
#include "core/tokenizer.hpp"
#include "utils/debug.hpp"

#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_llm_folder>" << std::endl;
    return 1;
  }

  const fs::path path = argv[1];

  std::cout << "Loading tensors from: " << path << std::endl;

  tinyllm::Config config(path);

  tinyllm::Tokenizer tokenizer(config);
  tokenizer.load_trie();

  tinyllm::Model model(config);
  model.load_weights();

  const std::string text = "Who is Linus?";
  std::cout << "Encoding text: " << text << std::endl;
  const auto tokens = tokenizer.encode(text);
  std::cout << "Encoded tokens: " << tokens << std::endl;
  std::cout << "Decoded text: " << tokenizer._debug_decode(tokens) << std::endl;

  std::cout << "Weight memory usage: " << (model.alloc.total_allocated >> 20) << " MB" << std::endl;

  return 0;
}
