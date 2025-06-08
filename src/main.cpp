#include "core/config.hpp"
#include "core/model.hpp"
#include "core/tokenizer.hpp"
#include "infer/inference_ctx.hpp"
#include "utils/debug.hpp"
#include "utils/utf8.hpp"

#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_llm_folder>" << std::endl;
    return 1;
  }

  const fs::path path = argv[1];

  std::cout << "[DEBUG] Loading tensors from: " << path << std::endl;

  tinyllm::Config config(path);

  tinyllm::Tokenizer tokenizer(config);
  tokenizer.load_trie();
  std::cout << "[DEBUG] Tokenizer loaded." << std::endl;

  tinyllm::Model model(config);
  model.load_weights();
  std::cout << "[DEBUG] Model weights loaded." << std::endl;

  std::cout << ">>> " << std::flush;

  std::string text = "用中文介绍一下雷军？";
  std::cin >> text;
  const auto tokens = tokenizer.encode(text);
  std::cout << "[DEBUG] Encoded tokens: " << tokens << std::endl;
  std::cout << "[DEBUG] Decoded text: " << tokenizer._debug_decode(tokens) << std::endl;

  std::cout << "[DEBUG] Weight memory usage: " << (model.alloc.total_allocated >> 20) << " MB" << std::endl;

  tinyllm::InferenceCtx ctx(config, 4096);

  std::cout << "[DEBUG] Inference memory usage: " << (ctx.alloc.total_allocated >> 20) << " MB" << std::endl;

  // ctx.forward(model, 0, 0);
  std::cout << "[DEBUG] Forwarding prompt ..." << std::endl;

  for (std::int32_t i = 0; i < tokens.size(); ++i) {
    ctx.forward(model, tokens[i], i);
    std::uint32_t token = ctx.argmax();
    std::string decoded_token = tokenizer.vocab[token];
  }

  std::string answer_buffer;
  for (std::int32_t i = 0; i < 4096; ++i) {
    std::uint32_t token = ctx.argmax();
    if (token == config.eos_token_id) {
      std::cout << "[DEBUG] Reached EOS token, stopping." << std::endl;
      break;
    }
    std::string decoded_token = tokenizer.vocab[token];
    answer_buffer += decoded_token;
    if (tinyllm::utf8_to_codepoint(answer_buffer).first > 0) {
      std::cout << answer_buffer << std::flush;
      answer_buffer.clear();
    }
    ctx.forward(model, token, tokens.size() + i);
  }
  if (!answer_buffer.empty()) {
    std::cout << answer_buffer << std::flush;
  }

  return 0;
}
