#include "backend_cpu.hpp"
#include "../../utils/precision.hpp"
#include "infer.hpp"
#include <cstddef>

namespace tinyllm {

InferenceBackendCPU::InferenceBackendCPU(Model &model_, std::size_t kv_size, DataType kv_dtype)
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
}

void InferenceBackendCPU::_rms_norm(float *out, const float *x, const Tensor &weight, std::size_t size, float eps) {
  if (model.dtype == DataType::BF16) {
    rms_norm_fp32_weight_bf16(out, x, weight.as<bf16_t>(), size, eps);
  } else if (model.dtype == DataType::F16) {
    rms_norm_fp32_weight_fp16(out, x, weight.as<fp16_t>(), size, eps);
  } else {
    rms_norm_fp32(out, x, weight.as<float>(), size, eps);
  }
}

void InferenceBackendCPU::_matrix_mul_vec(float *out, const float *a, const Tensor &weight, std::size_t m,
                                          std::size_t n) {
  if (model.dtype == DataType::BF16) {
    matrix_mul_vec_fp32_b_bf16(out, a, weight.as<bf16_t>(), m, n);
  } else if (model.dtype == DataType::F16) {
    matrix_mul_vec_fp32_b_fp16(out, a, weight.as<fp16_t>(), m, n);
  } else {
    matrix_mul_vec_fp32(out, a, weight.as<float>(), m, n);
  }
}

void InferenceBackendCPU::_matrix_mul_vec_bias(float *out, const float *a, const Tensor &weight, const Tensor &bias,
                                               std::size_t m, std::size_t n) {
  if (model.dtype == DataType::BF16) {
    matrix_mul_vec_bias_fp32_b_bf16(out, a, weight.as<bf16_t>(), bias.as<bf16_t>(), m, n);
  } else if (model.dtype == DataType::F16) {
    matrix_mul_vec_bias_fp32_b_fp16(out, a, weight.as<fp16_t>(), bias.as<fp16_t>(), m, n);
  } else {
    matrix_mul_vec_bias_fp32(out, a, weight.as<float>(), bias.as<float>(), m, n);
  }
}

void InferenceBackendCPU::forward_block(const Model::Block &block, Tensor &kc, Tensor &vc, std::int32_t pos,
                                        std::int32_t kv_sink, std::int32_t kv_pos, std::int32_t kv_len) {
  // 1. Layer normalization on input
  _rms_norm(xb.as<float>(), x.as<float>(), block.input_norm, config.hidden_size, config.rms_norm_eps);

  // 2. Self-attention
  const std::int32_t head_dim = config.head_dim;
  const std::int32_t q_dim = config.num_attention_heads * head_dim;
  const std::int32_t kv_dim = config.num_key_value_heads * head_dim;

  const bool has_qkv_bias = !block.attn_k_bias.data.empty();
  const bool has_qk_norm = !block.attn_q_norm.data.empty();

  if (has_qkv_bias) {
    _matrix_mul_vec_bias(q.as<float>(), xb.as<float>(), block.attn_q, block.attn_q_bias, config.hidden_size, q_dim);
    _matrix_mul_vec_bias(k.as<float>(), xb.as<float>(), block.attn_k, block.attn_k_bias, config.hidden_size, kv_dim);
    _matrix_mul_vec_bias(v.as<float>(), xb.as<float>(), block.attn_v, block.attn_v_bias, config.hidden_size, kv_dim);
  } else {
    _matrix_mul_vec(q.as<float>(), xb.as<float>(), block.attn_q, config.hidden_size, q_dim);
    _matrix_mul_vec(k.as<float>(), xb.as<float>(), block.attn_k, config.hidden_size, kv_dim);
    _matrix_mul_vec(v.as<float>(), xb.as<float>(), block.attn_v, config.hidden_size, kv_dim);
  }

  if (has_qk_norm) {
    for (std::size_t h = 0; h < config.num_attention_heads; ++h) {
      float *qh = q.as<float>() + h * head_dim;
      _rms_norm(qh, qh, block.attn_q_norm, head_dim, config.rms_norm_eps);
    }
    for (std::size_t h = 0; h < config.num_key_value_heads; ++h) {
      float *kh = k.as<float>() + h * head_dim;
      _rms_norm(kh, kh, block.attn_k_norm, head_dim, config.rms_norm_eps);
    }
  }

  for (std::size_t h = 0; h < config.num_attention_heads; ++h) {
    rope_inplace_fp32(q.as<float>() + h * head_dim, head_dim, pos, config.rope_theta);
  }
  for (std::size_t h = 0; h < config.num_key_value_heads; ++h) {
    rope_inplace_fp32(k.as<float>() + h * head_dim, head_dim, pos, config.rope_theta);
  }

  if (kv_dtype == DataType::BF16) {
    copy_fp32_to_bf16_n(k.as<float>(), kv_dim, kc.as<bf16_t>() + kv_pos * kv_dim);
    copy_fp32_to_bf16_n(v.as<float>(), kv_dim, vc.as<bf16_t>() + kv_pos * kv_dim);
  } else if (kv_dtype == DataType::F16) {
    copy_fp32_to_fp16_n(k.as<float>(), kv_dim, kc.as<fp16_t>() + kv_pos * kv_dim);
    copy_fp32_to_fp16_n(v.as<float>(), kv_dim, vc.as<fp16_t>() + kv_pos * kv_dim);
  } else {
    std::copy_n(k.as<float>(), kv_dim, kc.as<float>() + kv_pos * kv_dim);
    std::copy_n(v.as<float>(), kv_dim, vc.as<float>() + kv_pos * kv_dim);
  }

  for (std::size_t r = 0; r < kv_sink; ++r) {
    if (kv_dtype == DataType::BF16) {
      copy_bf16_to_fp32_n(kc.as<bf16_t>() + r * kv_dim, kv_dim, k.as<float>());
    } else if (kv_dtype == DataType::F16) {
      copy_fp16_to_fp32_n(kc.as<fp16_t>() + r * kv_dim, kv_dim, k.as<float>());
    } else {
      std::copy_n(kc.as<float>() + r * kv_dim, kv_dim, k.as<float>());
    }

    for (std::size_t h = 0; h < config.num_key_value_heads; ++h) {
      rope_inplace_fp32(k.as<float>() + h * head_dim, head_dim, 1, config.rope_theta);
    }

    if (kv_dtype == DataType::BF16) {
      copy_fp32_to_bf16_n(k.as<float>(), kv_dim, kc.as<bf16_t>() + r * kv_dim);
    } else if (kv_dtype == DataType::F16) {
      copy_fp32_to_fp16_n(k.as<float>(), kv_dim, kc.as<fp16_t>() + r * kv_dim);
    } else {
      std::copy_n(k.as<float>(), kv_dim, kc.as<float>() + r * kv_dim);
    }
  }

  // 3. Attention
  const std::int32_t q_per_head = config.num_attention_heads / config.num_key_value_heads;
  for (std::int32_t h = 0; h < config.num_attention_heads; ++h) {
    float *atth = attn.as<float>() + h * kv_size;
    const float *qh = q.as<float>() + h * head_dim;
    float *xb2h = xb2.as<float>() + h * head_dim;
    std::int32_t kv_offset = (h / q_per_head) * head_dim;
    if (kv_dtype == DataType::BF16) {
      const bf16_t *kh = kc.as<bf16_t>() + kv_offset;
      const bf16_t *vh = vc.as<bf16_t>() + kv_offset;
      attention_softmax_fp32_kv_bf16(xb2h, atth, qh, kh, vh, head_dim, config.num_key_value_heads, kv_len);
    } else if (kv_dtype == DataType::F16) {
      const fp16_t *kh = kc.as<fp16_t>() + kv_offset;
      const fp16_t *vh = vc.as<fp16_t>() + kv_offset;
      attention_softmax_fp32_kv_fp16(xb2h, atth, qh, kh, vh, head_dim, config.num_key_value_heads, kv_len);
    } else {
      const float *kh = kc.as<float>() + kv_offset;
      const float *vh = vc.as<float>() + kv_offset;
      attention_softmax_fp32(xb2h, atth, qh, kh, vh, head_dim, config.num_key_value_heads, kv_len);
    }
  }

  // 4. Combine attention outputs
  _matrix_mul_vec(xb.as<float>(), xb2.as<float>(), block.attn_o, q_dim, config.hidden_size);

  for (std::int32_t i = 0; i < config.hidden_size; ++i) {
    x.as<float>()[i] += xb.as<float>()[i];
  }

  // 5. Layer normalization on output
  _rms_norm(xb.as<float>(), x.as<float>(), block.post_norm, config.hidden_size, config.rms_norm_eps);

  // 6. MLP
  _matrix_mul_vec(hb.as<float>(), xb.as<float>(), block.mlp_gate, config.hidden_size, config.intermediate_size);
  _matrix_mul_vec(hb2.as<float>(), xb.as<float>(), block.mlp_up, config.hidden_size, config.intermediate_size);
  for (std::int32_t i = 0; i < config.intermediate_size; ++i) {
    hb.as<float>()[i] = silu_fp32(hb.as<float>()[i]) * hb2.as<float>()[i];
  }
  _matrix_mul_vec(xb2.as<float>(), hb.as<float>(), block.mlp_down, config.intermediate_size, config.hidden_size);

  for (std::int32_t i = 0; i < config.hidden_size; ++i) {
    x.as<float>()[i] += xb2.as<float>()[i];
  }
}

void InferenceBackendCPU::forward(std::int32_t token, std::int32_t pos) {
  // 1. Embed the token
  if (model.dtype == DataType::BF16) {
    copy_bf16_to_fp32_n(model.weight.embed.as<bf16_t>() + token * config.hidden_size, config.hidden_size,
                        x.as<float>());
  } else if (model.dtype == DataType::F16) {
    copy_fp16_to_fp32_n(model.weight.embed.as<fp16_t>() + token * config.hidden_size, config.hidden_size,
                        x.as<float>());
  } else {
    std::copy_n(model.weight.embed.as<float>() + token * config.hidden_size, config.hidden_size, x.as<float>());
  }

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= kv_size ? 2 : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (kv_size - kv_sink);
  int kv_len = pos >= kv_size ? kv_size : pos + 1;

  // 2. Forward through each block
  for (std::int32_t i = 0; i < config.num_hidden_layers; ++i) {
    const Model::Block &block = model.weight.blocks[i];
    forward_block(block, k_cache[i], v_cache[i], pos, kv_sink, kv_pos, kv_len);
  }

  // 3. Final layer normalization
  _rms_norm(x.as<float>(), x.as<float>(), model.weight.norm, config.hidden_size, config.rms_norm_eps);

  // 4. Compute logits
  _matrix_mul_vec(logits.as<float>(), x.as<float>(), model.weight.lm_head, config.hidden_size, config.vocab_size);
}

void InferenceBackendCPU::forward_prefill(std::int32_t token, std::int32_t pos) {
  // 1. Embed the token
  if (model.dtype == DataType::BF16) {
    copy_bf16_to_fp32_n(model.weight.embed.as<bf16_t>() + token * config.hidden_size, config.hidden_size,
                        x.as<float>());
  } else if (model.dtype == DataType::F16) {
    copy_fp16_to_fp32_n(model.weight.embed.as<fp16_t>() + token * config.hidden_size, config.hidden_size,
                        x.as<float>());
  } else {
    std::copy_n(model.weight.embed.as<float>() + token * config.hidden_size, config.hidden_size, x.as<float>());
  }

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= kv_size ? 2 : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (kv_size - kv_sink);
  int kv_len = pos >= kv_size ? kv_size : pos + 1;

  // 2. Forward through each block
  for (std::int32_t i = 0; i < config.num_hidden_layers; ++i) {
    const Model::Block &block = model.weight.blocks[i];
    forward_block(block, k_cache[i], v_cache[i], pos, kv_sink, kv_pos, kv_len);
  }
}

std::uint32_t InferenceBackendCPU::argmax() const {
  return std::distance(logits.as<float>(),
                       std::max_element(logits.as<float>(), logits.as<float>() + config.vocab_size));
}

std::size_t InferenceBackendCPU::memory_usage() const { return alloc.total_allocated; }

} // namespace tinyllm
