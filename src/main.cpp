#include "core/config.hpp"
#include "core/model.hpp"
#include "core/tokenizer.hpp"
#include "infer/inference_ctx.hpp"
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

  const std::string text = "用中文介绍一下雷军？";
  std::cout << "Encoding text: " << text << std::endl;
  const auto tokens = tokenizer.encode(text);
  std::cout << "Encoded tokens: " << tokens << std::endl;
  std::cout << "Decoded text: " << tokenizer._debug_decode(tokens) << std::endl;

  std::cout << "Weight memory usage: " << (model.alloc.total_allocated >> 20) << " MB" << std::endl;

  tinyllm::InferenceCtx ctx(config, 4096);

  std::cout << "Inference memory usage: " << (ctx.alloc.total_allocated >> 20) << " MB" << std::endl;

  // ctx.forward(model, 0, 0);
  for (std::int32_t i = 0; i < tokens.size(); ++i) {
    std::cout << "Forwarding token: " << tokens[i] << " at position: " << i << std::endl;
    ctx.forward(model, tokens[i], i);
    std::uint32_t token = ctx.argmax();
    std::string decoded_token = tokenizer.vocab[token];
  }

  std::string answer = "";
  for (std::int32_t i = 0; i < 4096; ++i) {
    std::uint32_t token = ctx.argmax();
    std::string decoded_token = tokenizer.vocab[token];
    std::cout << decoded_token << std::flush;
    answer += decoded_token;
    if (token == config.eos_token_id) {
      std::cout << "Reached EOS token, stopping." << std::endl;
      break;
    }
    ctx.forward(model, token, tokens.size() + i);
  }

  std::cout << "Generated answer: " << answer << std::endl;

  return 0;
}
