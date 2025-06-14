#include "backend_cuda.hpp"
#include "../../core/float.hpp"
#include "infer.hpp"
#include <cstddef>
#include <cstdint>

namespace tinyllm {

InferenceBackendCUDA::InferenceBackendCUDA(ModelCuda &model_, std::size_t kv_size, DataType kv_dtype)
    : model(model_), config(model.config), kv_size(kv_size), kv_dtype(kv_dtype) {
  logits_cpu = Tensor::alloc(alloc, DataType::F32, config.vocab_size);

  ///////////////////

  auto cuda_alloc_fp32 = [this](std::size_t d0, std::size_t d1 = 1, std::size_t d2 = 1) {
    return reinterpret_cast<float *>(cuda_alloc.alloc(d0 * d1 * d2 * sizeof(float)));
  };
  auto cuda_alloc_fp16 = [this](std::size_t d0, std::size_t d1 = 1, std::size_t d2 = 1) {
    return reinterpret_cast<void *>(cuda_alloc.alloc(d0 * d1 * d2 * sizeof(std::uint16_t)));
  };

  x = cuda_alloc_fp32(config.hidden_size);
  xb = cuda_alloc_fp32(config.hidden_size);
  xb2 = cuda_alloc_fp32(config.num_attention_heads, config.head_dim);
  hb = cuda_alloc_fp32(config.intermediate_size);
  hb2 = cuda_alloc_fp32(config.intermediate_size);
  q = cuda_alloc_fp32(config.num_attention_heads, config.head_dim);
  k = cuda_alloc_fp32(config.num_key_value_heads, config.head_dim);
  v = cuda_alloc_fp32(config.num_key_value_heads, config.head_dim);

  k_cache.resize(config.num_hidden_layers);
  v_cache.resize(config.num_hidden_layers);
  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    k_cache[i] = cuda_alloc_fp16(config.num_key_value_heads, kv_size, config.head_dim);
    v_cache[i] = cuda_alloc_fp16(config.num_key_value_heads, kv_size, config.head_dim);
  }

  attn = cuda_alloc_fp32(config.num_attention_heads, kv_size);
  logits = cuda_alloc_fp32(config.vocab_size);

  ////////

  const auto &weight = model.weight;
}

void InferenceBackendCUDA::forward_block(std::size_t block_id, std::int32_t pos, std::int32_t kv_sink,
                                         std::int32_t kv_pos, std::int32_t kv_len) {
  auto &block = model.weight.blocks[block_id];
  auto &kc = k_cache[block_id];
  auto &vc = v_cache[block_id];

  // 1. Layer normalization on input
  cuda::rms_norm_fp32_b_fp16(xb, x, block.input_norm, config.hidden_size, 1, config.rms_norm_eps);

  // 2. Self-attention
  const std::int32_t head_dim = config.head_dim;
  const std::int32_t q_dim = config.num_attention_heads * head_dim;
  const std::int32_t kv_dim = config.num_key_value_heads * head_dim;

  const bool has_qkv_bias = block.attn_k_bias;
  const bool has_qk_norm = block.attn_q_norm;

  if (has_qkv_bias) {
    cuda::matrix_mul_vec_bias_fp32_b_fp16(q, xb, block.attn_q, block.attn_q_bias, config.hidden_size, q_dim);
    cuda::matrix_mul_vec_bias_fp32_b_fp16(k, xb, block.attn_k, block.attn_k_bias, config.hidden_size, kv_dim);
    cuda::matrix_mul_vec_bias_fp32_b_fp16(v, xb, block.attn_v, block.attn_v_bias, config.hidden_size, kv_dim);
  } else {
    cuda::matrix_mul_vec_fp32_b_fp16(q, xb, block.attn_q, config.hidden_size, q_dim);
    cuda::matrix_mul_vec_fp32_b_fp16(k, xb, block.attn_k, config.hidden_size, kv_dim);
    cuda::matrix_mul_vec_fp32_b_fp16(v, xb, block.attn_v, config.hidden_size, kv_dim);
  }

  if (has_qk_norm) {
    cuda::rms_norm_fp32_b_fp16(q, q, block.attn_q_norm, head_dim, config.num_attention_heads, config.rms_norm_eps);
    cuda::rms_norm_fp32_b_fp16(k, k, block.attn_k_norm, head_dim, config.num_key_value_heads, config.rms_norm_eps);
  }
  cuda::rope_inplace_fp32(reinterpret_cast<float *>(q), config.num_attention_heads, head_dim, pos, config.rope_theta);
  cuda::rope_inplace_fp32(reinterpret_cast<float *>(k), config.num_key_value_heads, head_dim, pos, config.rope_theta);

  cuda::copy_fp32_to_fp16_n(k, kv_dim, reinterpret_cast<fp16_t *>(kc) + kv_pos * kv_dim);
  cuda::copy_fp32_to_fp16_n(v, kv_dim, reinterpret_cast<fp16_t *>(vc) + kv_pos * kv_dim);

  for (std::size_t r = 0; r < kv_sink; ++r) {
    cuda::copy_fp16_to_fp32_n(reinterpret_cast<fp16_t *>(kc) + r * kv_dim, kv_dim, k);

    cuda::rope_inplace_fp32(k, config.num_key_value_heads, head_dim, 1, config.rope_theta);

    cuda::copy_fp32_to_fp16_n(k, kv_dim, reinterpret_cast<fp16_t *>(kc) + r * kv_dim);
  }

  // 3. Attention
  cuda::mh_attention_fp32_kv_fp16(xb2, attn, q, kc, vc, config.num_attention_heads, head_dim,
                                  config.num_key_value_heads, kv_len);

  // 4. Combine attention outputs
  cuda::matrix_mul_vec_fp32_b_fp16(xb, xb2, block.attn_o, q_dim, config.hidden_size);

  cuda::vec_add_inplace_fp32(x, xb, config.hidden_size);

  // 5. Layer normalization on output
  cuda::rms_norm_fp32_b_fp16(xb, x, block.post_norm, config.hidden_size, 1, config.rms_norm_eps);

  // 6. MLP
  cuda::matrix_mul_vec_fp32_b_fp16(hb, xb, block.mlp_gate, config.hidden_size, config.intermediate_size);
  cuda::matrix_mul_vec_fp32_b_fp16(hb2, xb, block.mlp_up, config.hidden_size, config.intermediate_size);

  cuda::swiglu_fp32(reinterpret_cast<float *>(hb), reinterpret_cast<const float *>(hb2),
                    reinterpret_cast<const float *>(hb), config.intermediate_size);

  cuda::matrix_mul_vec_fp32_b_fp16(xb2, hb, block.mlp_down, config.intermediate_size, config.hidden_size);

  cuda::vec_add_inplace_fp32(x, xb2, config.hidden_size);
}

void InferenceBackendCUDA::forward(std::int32_t token, std::int32_t pos) {
  // 1. Embed the token
  cuda::copy_fp16_to_fp32_n(reinterpret_cast<std::uint16_t *>(model.weight.embed) + token * config.hidden_size,
                            config.hidden_size, x);

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= kv_size ? 2 : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (kv_size - kv_sink);
  int kv_len = pos >= kv_size ? kv_size : pos + 1;

  // 2. Forward through each block
  for (std::int32_t i = 0; i < config.num_hidden_layers; ++i) {
    forward_block(i, pos, kv_sink, kv_pos, kv_len);
  }

  // 3. Final layer normalization
  cuda::rms_norm_fp32_b_fp16(x, x, model.weight.norm, config.hidden_size, 1, config.rms_norm_eps);

  // 4. Compute logits
  cuda::matrix_mul_vec_fp32_b_fp16(logits, x, model.weight.lm_head, config.hidden_size, config.vocab_size);
  cuda::copy_to_host(logits, logits_cpu.size_bytes(), logits_cpu.as<float>());
  cuda::check_and_sync();
}

void InferenceBackendCUDA::forward_prefill(std::int32_t token, std::int32_t pos) {
  // 1. Embed the token
  cuda::copy_fp16_to_fp32_n(reinterpret_cast<fp16_t *>(model.weight.embed) + token * config.hidden_size,
                            config.hidden_size, x);

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= kv_size ? 2 : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (kv_size - kv_sink);
  int kv_len = pos >= kv_size ? kv_size : pos + 1;

  // 2. Forward through each block
  for (std::int32_t i = 0; i < config.num_hidden_layers; ++i) {
    forward_block(i, pos, kv_sink, kv_pos, kv_len);
  }
  cuda::check_and_sync();
}

std::span<const float> InferenceBackendCUDA::get_logits() const {
  return std::span<const float>(logits_cpu.as<float>(), config.vocab_size);
}

std::size_t InferenceBackendCUDA::memory_usage() const { return cuda_alloc.total_allocated; }

} // namespace tinyllm
