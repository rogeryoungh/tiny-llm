#include "core/config.hpp"
#include "core/tokenizer.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include "utils/debug.hpp"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_llm_folder>" << std::endl;
    return 1;
  }

  const fs::path path = argv[1];

  std::cout << "Loading tensors from: " << path << std::endl;

  tinyllm::Config config;
  // config.load_tensors(path);

  tinyllm::Tokenizer tokenizer;
  tokenizer.load(path);

  const std::string text = "Who is Linus?";
  std::cout << "Encoding text: " << text << std::endl;
  const auto tokens = tokenizer.encode(text);
  std::cout << "Encoded tokens: " << tokens << std::endl;
  std::cout << "Decoded text: " << tokenizer._debug_decode(tokens) << std::endl;

  return 0;
}
