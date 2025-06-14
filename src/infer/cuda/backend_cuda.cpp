#include "backend_cuda.hpp"
#include "../../core/float.hpp"
#include "infer.hpp"
#include <cstddef>

namespace tinyllm {

InferenceBackendCUDA::InferenceBackendCUDA(Model &model_, std::size_t kv_size, DataType kv_dtype)
    : model(model_), config(model.config), kv_size(kv_size), kv_dtype(kv_dtype) {
  x = Tensor::alloc(alloc, DataType::F32, config.hidden_size);
  xb = Tensor::alloc(alloc, DataType::F32, config.hidden_size);
  xb2 = Tensor::alloc(alloc, DataType::F32, config.num_attention_heads, config.head_dim);
  hb = Tensor::alloc(alloc, DataType::F32, config.intermediate_size);
  hb2 = Tensor::alloc(alloc, DataType::F32, config.intermediate_size);
  q = Tensor::alloc(alloc, DataType::F32, config.num_attention_heads, config.head_dim);
  k = Tensor::alloc(alloc, DataType::F32, config.num_key_value_heads, config.head_dim);
  v = Tensor::alloc(alloc, DataType::F32, config.num_key_value_heads, config.head_dim);

  k_cache.resize(config.num_hidden_layers);
  v_cache.resize(config.num_hidden_layers);
  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    k_cache[i] = Tensor::alloc(alloc, kv_dtype, config.num_key_value_heads, kv_size, config.head_dim);
    v_cache[i] = Tensor::alloc(alloc, kv_dtype, config.num_key_value_heads, kv_size, config.head_dim);
  }
  attn = Tensor::alloc(alloc, DataType::F32, config.num_attention_heads, kv_size);

  logits = Tensor::alloc(alloc, DataType::F32, config.vocab_size);

  ///////////////////

  gpu_v.x = reinterpret_cast<float *>(cuda_alloc.alloc(x.size_bytes()));
  gpu_v.xb = reinterpret_cast<float *>(cuda_alloc.alloc(xb.size_bytes()));
  gpu_v.xb2 = reinterpret_cast<float *>(cuda_alloc.alloc(xb2.size_bytes()));
  gpu_v.hb = reinterpret_cast<float *>(cuda_alloc.alloc(hb.size_bytes()));
  gpu_v.hb2 = reinterpret_cast<float *>(cuda_alloc.alloc(hb2.size_bytes()));
  gpu_v.q = reinterpret_cast<float *>(cuda_alloc.alloc(q.size_bytes()));
  gpu_v.k = reinterpret_cast<float *>(cuda_alloc.alloc(k.size_bytes()));
  gpu_v.v = reinterpret_cast<float *>(cuda_alloc.alloc(v.size_bytes()));

  gpu_v.k_cache.resize(config.num_hidden_layers);
  gpu_v.v_cache.resize(config.num_hidden_layers);
  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    gpu_v.k_cache[i] = cuda_alloc.alloc(k_cache[i].size_bytes());
    gpu_v.v_cache[i] = cuda_alloc.alloc(v_cache[i].size_bytes());
  }

  gpu_v.attn = reinterpret_cast<float *>(cuda_alloc.alloc(attn.size_bytes()));
  gpu_v.logits = reinterpret_cast<float *>(cuda_alloc.alloc(logits.size_bytes()));

  ////////

  const auto &weight = model.weight;

  gpu_w.embed = cuda_alloc.upload(weight.embed.data);
  gpu_w.norm = cuda_alloc.upload(weight.norm.data);
  gpu_w.blocks.resize(config.num_hidden_layers);

  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    auto &block = weight.blocks[i];
    auto &gpu_block = gpu_w.blocks[i];
    gpu_block.attn_q = cuda_alloc.upload(block.attn_q.data);
    gpu_block.attn_k = cuda_alloc.upload(block.attn_k.data);
    gpu_block.attn_v = cuda_alloc.upload(block.attn_v.data);
    gpu_block.attn_o = cuda_alloc.upload(block.attn_o.data);
    gpu_block.mlp_down = cuda_alloc.upload(block.mlp_down.data);
    gpu_block.mlp_gate = cuda_alloc.upload(block.mlp_gate.data);
    gpu_block.mlp_up = cuda_alloc.upload(block.mlp_up.data);
    gpu_block.input_norm = cuda_alloc.upload(block.input_norm.data);
    gpu_block.post_norm = cuda_alloc.upload(block.post_norm.data);

    if (config.model_type == "qwen2") {
      gpu_block.attn_k_bias = cuda_alloc.upload(block.attn_k_bias.data);
      gpu_block.attn_q_bias = cuda_alloc.upload(block.attn_q_bias.data);
      gpu_block.attn_v_bias = cuda_alloc.upload(block.attn_v_bias.data);
    }

    if (config.model_type == "qwen3") {
      gpu_block.attn_q_norm = cuda_alloc.upload(block.attn_q_norm.data);
      gpu_block.attn_k_norm = cuda_alloc.upload(block.attn_k_norm.data);
    }
  }
  if (config.tie_word_embeddings) {
    gpu_w.lm_head = gpu_w.embed;
  } else {
    gpu_w.lm_head = cuda_alloc.upload(weight.lm_head.data);
  }
}

void InferenceBackendCUDA::forward_block(std::size_t block_id, std::int32_t pos, std::int32_t kv_sink,
                                         std::int32_t kv_pos, std::int32_t kv_len) {
  const auto &block = model.weight.blocks[block_id];
  auto &gpu_block = gpu_w.blocks[block_id];
  auto &kc = k_cache[block_id];
  auto &vc = v_cache[block_id];
  auto &gpu_kc = gpu_v.k_cache[block_id];
  auto &gpu_vc = gpu_v.v_cache[block_id];

  // 1. Layer normalization on input
  cuda::rms_norm_fp32_b_fp16(gpu_v.xb, gpu_v.x, gpu_block.input_norm, config.hidden_size, 1, config.rms_norm_eps);

  // 2. Self-attention
  const std::int32_t head_dim = config.head_dim;
  const std::int32_t q_dim = config.num_attention_heads * head_dim;
  const std::int32_t kv_dim = config.num_key_value_heads * head_dim;

  const bool has_qkv_bias = !block.attn_k_bias.data.empty();
  const bool has_qk_norm = !block.attn_q_norm.data.empty();

  if (has_qkv_bias) {
    cuda::matrix_mul_vec_bias_fp32_b_fp16(gpu_v.q, gpu_v.xb, gpu_block.attn_q, gpu_block.attn_q_bias,
                                          config.hidden_size, q_dim);
    cuda::matrix_mul_vec_bias_fp32_b_fp16(gpu_v.k, gpu_v.xb, gpu_block.attn_k, gpu_block.attn_k_bias,
                                          config.hidden_size, kv_dim);
    cuda::matrix_mul_vec_bias_fp32_b_fp16(gpu_v.v, gpu_v.xb, gpu_block.attn_v, gpu_block.attn_v_bias,
                                          config.hidden_size, kv_dim);
  } else {
    cuda::matrix_mul_vec_fp32_b_fp16(gpu_v.q, gpu_v.xb, gpu_block.attn_q, config.hidden_size, q_dim);
    cuda::matrix_mul_vec_fp32_b_fp16(gpu_v.k, gpu_v.xb, gpu_block.attn_k, config.hidden_size, kv_dim);
    cuda::matrix_mul_vec_fp32_b_fp16(gpu_v.v, gpu_v.xb, gpu_block.attn_v, config.hidden_size, kv_dim);
  }

  if (has_qk_norm) {
    cuda::rms_norm_fp32_b_fp16(gpu_v.q, gpu_v.q, gpu_block.attn_q_norm, head_dim, config.num_attention_heads,
                               config.rms_norm_eps);
    cuda::rms_norm_fp32_b_fp16(gpu_v.k, gpu_v.k, gpu_block.attn_k_norm, head_dim, config.num_key_value_heads,
                               config.rms_norm_eps);
  }
  cuda::rope_inplace_fp32(reinterpret_cast<float *>(gpu_v.q), config.num_attention_heads, head_dim, pos,
                          config.rope_theta);
  cuda::rope_inplace_fp32(reinterpret_cast<float *>(gpu_v.k), config.num_key_value_heads, head_dim, pos,
                          config.rope_theta);

  cuda::copy_fp32_to_fp16_n(gpu_v.k, kv_dim, reinterpret_cast<fp16_t *>(gpu_kc) + kv_pos * kv_dim);
  cuda::copy_fp32_to_fp16_n(gpu_v.v, kv_dim, reinterpret_cast<fp16_t *>(gpu_vc) + kv_pos * kv_dim);

  for (std::size_t r = 0; r < kv_sink; ++r) {
    cuda::copy_fp16_to_fp32_n(reinterpret_cast<fp16_t *>(gpu_kc) + r * kv_dim, kv_dim, gpu_v.k);

    cuda::rope_inplace_fp32(gpu_v.k, config.num_key_value_heads, head_dim, 1, config.rope_theta);

    cuda::copy_fp32_to_fp16_n(gpu_v.k, kv_dim, reinterpret_cast<fp16_t *>(gpu_kc) + r * kv_dim);
  }

  // 3. Attention
  cuda::mh_attention_fp32_kv_fp16(gpu_v.xb2, gpu_v.attn, gpu_v.q, gpu_kc, gpu_vc, config.num_attention_heads, head_dim,
                                  config.num_key_value_heads, kv_len);

  // 4. Combine attention outputs
  cuda::matrix_mul_vec_fp32_b_fp16(gpu_v.xb, gpu_v.xb2, gpu_block.attn_o, q_dim, config.hidden_size);

  cuda::vec_add_inplace_fp32(gpu_v.x, gpu_v.xb, config.hidden_size);

  // 5. Layer normalization on output
  cuda::rms_norm_fp32_b_fp16(gpu_v.xb, gpu_v.x, gpu_block.post_norm, config.hidden_size, 1, config.rms_norm_eps);

  // 6. MLP
  cuda::matrix_mul_vec_fp32_b_fp16(gpu_v.hb, gpu_v.xb, gpu_block.mlp_gate, config.hidden_size,
                                   config.intermediate_size);
  cuda::matrix_mul_vec_fp32_b_fp16(gpu_v.hb2, gpu_v.xb, gpu_block.mlp_up, config.hidden_size, config.intermediate_size);

  cuda::swiglu_fp32(reinterpret_cast<float *>(gpu_v.hb), reinterpret_cast<const float *>(gpu_v.hb2),
                    reinterpret_cast<const float *>(gpu_v.hb), config.intermediate_size);

  cuda::matrix_mul_vec_fp32_b_fp16(gpu_v.xb2, gpu_v.hb, gpu_block.mlp_down, config.intermediate_size,
                                   config.hidden_size);

  cuda::vec_add_inplace_fp32(gpu_v.x, gpu_v.xb2, config.hidden_size);
}

void InferenceBackendCUDA::forward(std::int32_t token, std::int32_t pos) {
  // 1. Embed the token
  cuda::copy_fp16_to_fp32_n(reinterpret_cast<std::uint16_t *>(gpu_w.embed) + token * config.hidden_size,
                            config.hidden_size, gpu_v.x);

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
  cuda::rms_norm_fp32_b_fp16(gpu_v.x, gpu_v.x, gpu_w.norm, config.hidden_size, 1, config.rms_norm_eps);

  // 4. Compute logits
  cuda::matrix_mul_vec_fp32_b_fp16(gpu_v.logits, gpu_v.x, gpu_w.lm_head, config.hidden_size, config.vocab_size);
  cuda::copy_to_host(gpu_v.logits, logits.size_bytes(), logits.as<float>());
  cuda::check_and_sync();
}

void InferenceBackendCUDA::forward_prefill(std::int32_t token, std::int32_t pos) {
  // 1. Embed the token
  cuda::copy_fp16_to_fp32_n(reinterpret_cast<fp16_t *>(gpu_w.embed) + token * config.hidden_size, config.hidden_size,
                            gpu_v.x);

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

std::uint32_t InferenceBackendCUDA::argmax() const {
  return std::distance(logits.as<float>(),
                       std::max_element(logits.as<float>(), logits.as<float>() + config.vocab_size));
}

std::size_t InferenceBackendCUDA::memory_usage() const { return alloc.total_allocated; }

} // namespace tinyllm
