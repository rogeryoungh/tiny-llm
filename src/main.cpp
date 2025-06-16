#include "core/config.hpp"
#include "core/model.hpp"
#include "core/model_cuda.hpp"
#include "core/tokenizer.hpp"
#include "infer/inference_ctx.hpp"
#include "utils/stopwatch.hpp"
#include "utils/utf8.hpp"

#include <iostream>
#include <print>

#include <cxxopts.hpp>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  cxxopts::Options options("tinyllm", "A lightweight large language model inference engine.");
  // clang-format off
  options.add_options()
      ("h,help", "Show help")
      ("m,model", "Path to the LLM folder", cxxopts::value<std::string>())
      ("prompt", "Prompt text to use for inference", cxxopts::value<std::string>()->default_value(""))
      ("benchmark-prompt-size",  "The size of tokens in the benchmarking prompt, at least 32", cxxopts::value<std::size_t>()->default_value("0"))
      ("device", "Device to use for inference (cpu or cuda)", cxxopts::value<std::string>()->default_value("cuda"))
      ("kv-size", "Size of the key-value cache", cxxopts::value<std::size_t>()->default_value("4096"))
      ("max-tokens", "Maximum number of tokens to generate", cxxopts::value<std::size_t>()->default_value("4096"))
      ;
  // clang-format on

  auto result = options.parse(argc, argv);
  if (result.count("help") || argc < 2) {
    std::println("{}", options.help());
    return 0;
  }

  const fs::path model_path = result["model"].as<std::string>();
  auto prompt = result["prompt"].as<std::string>();
  auto device = result["device"].as<std::string>();
  auto kv_size = result["kv-size"].as<std::size_t>();
  auto max_tokens = result["max-tokens"].as<std::size_t>();
  auto benchmark_prompt_size = result["benchmark-prompt-size"].as<std::size_t>();

  std::println("[DEBUG] Model path: {}", model_path.string());
  std::println("[DEBUG] Using device: {}", device);

  tinyllm::Config config(model_path);
  if (config.do_sample) {
    std::println("[DEBUG] Sampling with temperature: {}, top-k: {}, top-p: {}", config.temperature, config.top_k,
                 config.top_p);
  } else {
    std::println("[DEBUG] Sampling disabled, using argmax.");
  }

  tinyllm::Stopwatch tokenizer_load_timer;
  tinyllm::Tokenizer tokenizer(config);
  tokenizer.load_trie();
  std::println("[DEBUG] Tokenizer loaded in {:3f} ms, memory usage {} MB.", tokenizer_load_timer.elapsed_ms(),
               tokenizer.memory_usage() >> 20);

  tinyllm::Stopwatch model_load_timer;

  std::unique_ptr<tinyllm::Model> model_ptr;
  std::unique_ptr<tinyllm::ModelCuda> model_cuda_ptr;
  std::unique_ptr<tinyllm::InferenceCtx> ctx_ptr;

  model_ptr = std::make_unique<tinyllm::Model>(config);
  model_ptr->load_weights();
  std::println("[DEBUG] Model loaded in {:3f} ms, memory usage {} MB.", model_load_timer.elapsed_ms(),
               model_ptr->memory_usage() >> 20);

  if (device == "cpu") {
    ctx_ptr = std::make_unique<tinyllm::InferenceCtx>(*model_ptr, kv_size, tinyllm::DataType::F32);
  } else if (device == "cuda") {
    model_cuda_ptr = std::make_unique<tinyllm::ModelCuda>(*model_ptr);
    model_ptr.reset();
    ctx_ptr = std::make_unique<tinyllm::InferenceCtx>(*model_cuda_ptr, kv_size, tinyllm::DataType::F16);
    std::println("[DEBUG] Model loaded on CUDA, memory usage {} MB", model_cuda_ptr->memory_usage() >> 20);
  } else {
    std::println("[ERROR] Unsupported device: {}", device);
    return 1;
  }

  std::println("[DEBUG] Inference memory usage: {} MB", ctx_ptr->memory_usage() >> 20);

  std::string text;
  if (prompt.empty()) {
    if (benchmark_prompt_size > 0) {
      text = std::string("介绍一下杭州的美食");
    } else {
      std::print(">>> ");
      std::getline(std::cin, text);
    }
  } else {
    std::println(">>> {}", prompt);
    text = prompt;
  }

  std::vector<std::int32_t> tokens;
  if (benchmark_prompt_size > 0) {
    tokens = tokenizer.encode_padding(text, benchmark_prompt_size);
    if (tokens.size() != benchmark_prompt_size) {
      std::println("[ERROR] Expected at least {} tokens, got {}.", benchmark_prompt_size, tokens.size());
      return 1;
    }
  } else {
    tokens = tokenizer.encode(text);
  }
  std::println("[DEBUG] Encoded {} tokens: {}", tokens.size(), tokens);
  std::println("[DEBUG] Decoded text: {}", tokenizer._debug_decode(tokens));

  tinyllm::Stopwatch prefill_timer;
  tinyllm::Stopwatch answer_timer;

  for (std::int32_t i = 0; i < tokens.size(); ++i) {
    if (i + 1 < tokens.size()) {
      ctx_ptr->forward_prefill(tokens[i], i);
    } else {
      answer_timer.reset();
      ctx_ptr->forward(tokens[i], i);
    }
  }
  double prefill_time = prefill_timer.elapsed_seconds();
  std::println("[DEBUG] Prefill completed in {:3f} s, throughput {:3f} tok/s.", prefill_time,
               tokens.size() / prefill_time);

  std::string answer_buffer;
  std::vector<double> generate_times;
  generate_times.reserve(max_tokens + 1);

  for (std::int32_t i = 0; i < max_tokens; ++i) {
    std::uint32_t token = ctx_ptr->sample();
    generate_times.emplace_back(answer_timer.elapsed_seconds());
    if (token == config.eos_token_id && benchmark_prompt_size == 0) {
      break;
    }
    if (i + 1 != max_tokens) {
      ctx_ptr->forward(token, tokens.size() + i);
    }
    std::string decoded_token = tokenizer.vocab[token];
    answer_buffer += decoded_token;
    while (std::size_t length = tinyllm::utf8_to_codepoint(answer_buffer).first) {
      std::print("{}", std::string_view(answer_buffer.data(), length));
      std::flush(std::cout);
      answer_buffer.erase(0, length);
    }
  }
  if (!answer_buffer.empty()) {
    std::print("{}", answer_buffer);
  }
  if (generate_times.empty()) {
    std::println("[DEBUG] No tokens generated.");
    return 0;
  }

  double avg_time = generate_times.size() / generate_times.back();

  std::println();
  std::println("[DEBUG] Generated {} tokens in {:3f} s, average {:3f} tok/s.", generate_times.size(),
               generate_times.back(), avg_time);
  if (generate_times.size() >= 32) {
    double avg_first32 = 32 / generate_times[31];
    double avg_last32 = 32 / (generate_times.back() - generate_times[generate_times.size() - 32]);
    std::println("[DEBUG] First 32 with {:3f} tok/s, last 32 with {:3f} tok/s", avg_first32, avg_last32);
  }

  return 0;
}
