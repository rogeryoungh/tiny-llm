#include "core/config.hpp"
#include "core/model.hpp"
#include "core/tokenizer.hpp"
#include "infer/inference_ctx.hpp"
#include "utils/debug.hpp"
#include "utils/stopwatch.hpp"
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

  tinyllm::Stopwatch tokenizer_load_timer;
  tinyllm::Tokenizer tokenizer(config);
  tokenizer.load_trie();
  std::cout << std::format("[DEBUG] Tokenizer loaded in {:3f} ms.", tokenizer_load_timer.elapsed_ms()) << std::endl;

  tinyllm::Stopwatch model_load_timer;
  tinyllm::Model model(config);
  model.load_weights();
  std::cout << std::format("[DEBUG] Model weights loaded in {:3f} ms.", model_load_timer.elapsed_ms()) << std::endl;

  tinyllm::InferenceCtx ctx(model, 4096, tinyllm::DeviceType::GPU, tinyllm::DataType::F16);

  std::cout << "[DEBUG] Weight memory usage: " << (model.memory_usage() >> 20) << " MB" << std::endl;
  std::cout << "[DEBUG] Inference memory usage: " << (ctx.memory_usage() >> 20) << " MB" << std::endl;
  std::cout << "[DEBUG] Tokenizer memory usage: " << (tokenizer.memory_usage() >> 20) << " MB" << std::endl;

  std::cout << ">>> " << std::flush;

  std::string text = "";
  // std::cin >> text;
  std::getline(std::cin, text);

  const auto tokens = tokenizer.encode(text);
  std::cout << "[DEBUG] Encoded tokens: " << tokens << std::endl;
  std::cout << "[DEBUG] Decoded text: " << tokenizer._debug_decode(tokens) << std::endl;

  // ctx.forward(model, 0, 0);
  std::cout << "[DEBUG] Forwarding prompt ..." << std::endl;

  tinyllm::Stopwatch prefill_timer;
  for (std::int32_t i = 0; i < tokens.size(); ++i) {
    if (i + 1 < tokens.size()) {
      ctx.forward_prefill(tokens[i], i);
    } else {
      ctx.forward(tokens[i], i);
    }
  }
  std::cout << std::format("[DEBUG] Prefill completed in {:3f} ms.", prefill_timer.elapsed_ms()) << std::endl;
  std::cout << std::format("[DEBUG] Prefill throughput: {:3f} tokens/s",
                           tokens.size() / prefill_timer.elapsed_seconds())
            << std::endl;
  tinyllm::Stopwatch answer_timer;

  std::string answer_buffer;
  std::size_t generate_tokens = 0;
  for (std::int32_t i = 0; i < 4096; ++i) {
    std::uint32_t token = ctx.argmax();
    generate_tokens += 1;
    if (token == config.eos_token_id) {
      break;
    }
    std::string decoded_token = tokenizer.vocab[token];
    answer_buffer += decoded_token;
    while (std::size_t length = tinyllm::utf8_to_codepoint(answer_buffer).first) {
      std::cout << std::string_view(answer_buffer.data(), length) << std::flush;
      answer_buffer.erase(0, length);
    }
    ctx.forward(token, tokens.size() + i);
  }
  if (!answer_buffer.empty()) {
    std::cout << answer_buffer << std::flush;
  }

  std::cout << std::endl;
  std::cout << std::format("[DEBUG] Generated {} tokens.", generate_tokens) << std::endl;
  std::cout << std::format("[DEBUG] Generate throughput: {:3f} tokens/s",
                           generate_tokens / answer_timer.elapsed_seconds())
            << std::endl;

  return 0;
}
