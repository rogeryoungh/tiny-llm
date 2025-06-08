#include "inference_ctx.hpp"
#include "infer.hpp"

namespace tinyllm {

InferenceCtx::InferenceCtx(Config &cfg, std::size_t kv_size_) : config(cfg), kv_size(kv_size_) {
  x = alloc.alloc_fp32(config.hidden_size);
  xb = alloc.alloc_fp32(config.hidden_size);
  xb2 = alloc.alloc_fp32(config.hidden_size);
  hb = alloc.alloc_fp32(config.intermediate_size);
  hb2 = alloc.alloc_fp32(config.intermediate_size);
  const std::int32_t head_dim = config.hidden_size / config.num_attention_heads;
  q = alloc.alloc_fp32(config.num_attention_heads, head_dim);
  k = alloc.alloc_fp32(config.num_key_value_heads, head_dim);
  v = alloc.alloc_fp32(config.num_key_value_heads, head_dim);

  k_cache.resize(config.num_hidden_layers);
  v_cache.resize(config.num_hidden_layers);
  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    k_cache[i] = alloc.alloc_fp32(config.num_key_value_heads, kv_size, head_dim);
    v_cache[i] = alloc.alloc_fp32(config.num_key_value_heads, kv_size, head_dim);
  }
  attn = alloc.alloc_fp32(config.num_attention_heads, kv_size);

  logits = alloc.alloc_fp32(config.vocab_size);
}

InferenceCtx::~InferenceCtx() {
  alloc.dealloc(x.data);
  alloc.dealloc(xb.data);
  alloc.dealloc(xb2.data);
  alloc.dealloc(hb.data);
  alloc.dealloc(hb2.data);
  alloc.dealloc(q.data);
  alloc.dealloc(k.data);
  alloc.dealloc(v.data);
  alloc.dealloc(attn.data);
  alloc.dealloc(logits.data);
}

void InferenceCtx::_embeding(const Model &model, std::int32_t token) {
  const float *ep = model.weight.embed.as<float>();
  float *xp = x.as<float>();

  std::copy_n(ep + token * config.hidden_size, config.hidden_size, xp);
}

void InferenceCtx::forward_block(const Block &block, Tensor &kc, Tensor &vc, std::int32_t pos, std::int32_t kv_sink,
                                 std::int32_t kv_pos, std::int32_t kv_len) {
  // 1. Layer normalization on input
  rms_norm_fp32(xb.as<float>(), x.as<float>(), block.input_norm.as<float>(), config.hidden_size, 1e-5);

  // 2. Self-attention
  const std::int32_t head_dim = config.hidden_size / config.num_attention_heads;
  std::int32_t q_dim = config.num_attention_heads * head_dim;
  std::int32_t kv_dim = config.num_key_value_heads * head_dim;

  matrix_mul_vec_fp32(q.as<float>(), xb.as<float>(), block.attn_q.as<float>(), config.hidden_size, q_dim);
  matrix_mul_vec_fp32(k.as<float>(), xb.as<float>(), block.attn_k.as<float>(), config.hidden_size, kv_dim);
  matrix_mul_vec_fp32(v.as<float>(), xb.as<float>(), block.attn_v.as<float>(), config.hidden_size, kv_dim);

  rope_inplace_fp32(q.as<float>(), q_dim, head_dim, pos, config.rope_theta, head_dim);
  rope_inplace_fp32(k.as<float>(), kv_dim, head_dim, pos, config.rope_theta, head_dim);

  std::copy_n(k.as<float>(), kv_dim, kc.as<float>() + kv_pos * kv_dim);
  std::copy_n(v.as<float>(), kv_dim, vc.as<float>() + kv_pos * kv_dim);

  for (std::size_t r = 0; r < kv_sink; ++r) {
    std::copy_n(kc.as<float>() + r * kv_dim, kv_dim, k.as<float>());

    rope_inplace_fp32(k.as<float>(), kv_dim, head_dim, 1, config.rope_theta, head_dim);

    std::copy_n(k.as<float>(), kv_dim, kc.as<float>() + r * kv_dim);
  }

  // 3. Attention
  const std::int32_t q_per_head = config.num_attention_heads / config.num_key_value_heads;
  for (std::int32_t h = 0; h < config.num_attention_heads; ++h) {
    float *atth = attn.as<float>() + h * kv_size;
    const float *qh = q.as<float>() + h * head_dim;
    float *xb2h = xb2.as<float>() + h * head_dim;
    std::int32_t kv_offset = (h / q_per_head) * head_dim;
    const float *kh = kc.as<float>() + kv_offset;
    const float *vh = vc.as<float>() + kv_offset;
    attention_softmax_fp32(xb2h, atth, qh, kh, vh, head_dim, config.num_key_value_heads, kv_len);
  }

  // 4. Combine attention outputs
  matrix_mul_vec_fp32(hb.as<float>(), xb2.as<float>(), block.attn_o.as<float>(), q_dim, config.hidden_size);

  for (std::int32_t i = 0; i < config.hidden_size; ++i) {
    x.as<float>()[i] += hb.as<float>()[i];
  }

  // 5. Layer normalization on output
  rms_norm_fp32(xb.as<float>(), x.as<float>(), block.post_norm.as<float>(), config.hidden_size, 1e-5);

  // 6. MLP
  matrix_mul_vec_fp32(hb.as<float>(), xb.as<float>(), block.mlp_gate.as<float>(), config.hidden_size,
                      config.intermediate_size);
  matrix_mul_vec_fp32(hb2.as<float>(), xb.as<float>(), block.mlp_up.as<float>(), config.hidden_size,
                      config.intermediate_size);
  for (std::int32_t i = 0; i < config.intermediate_size; ++i) {
    hb.as<float>()[i] = silu_fp32(hb.as<float>()[i]) * hb2.as<float>()[i];
  }
  matrix_mul_vec_fp32(xb2.as<float>(), hb.as<float>(), block.mlp_down.as<float>(), config.intermediate_size,
                      config.hidden_size);

  for (std::int32_t i = 0; i < config.hidden_size; ++i) {
    x.as<float>()[i] += xb2.as<float>()[i];
  }
}

void InferenceCtx::forward(const Model &model, std::int32_t token, std::int32_t pos) {
  // 1. Embed the token
  _embeding(model, token);

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= kv_size ? 2 : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (kv_size - kv_sink);
  int kv_len = pos >= kv_size ? kv_size : pos + 1;

  // 2. Forward through each block
  for (std::int32_t i = 0; i < config.num_hidden_layers; ++i) {
    const Block &block = model.weight.blocks[i];
    forward_block(block, k_cache[i], v_cache[i], pos, kv_sink, kv_pos, kv_len);
  }

  // 3. Final layer normalization
  rms_norm_fp32(x.as<float>(), x.as<float>(), model.weight.norm.as<float>(), config.hidden_size, 1e-5);

  // 4. Compute logits
  matrix_mul_vec_fp32(logits.as<float>(), x.as<float>(), model.weight.lm_head.as<float>(), config.hidden_size,
                      config.vocab_size);
}

void InferenceCtx::forward_prefill(const Model &model, std::int32_t token, std::int32_t pos) {
  // 1. Embed the token
  _embeding(model, token);

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= kv_size ? 2 : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (kv_size - kv_sink);
  int kv_len = pos >= kv_size ? kv_size : pos + 1;

  // 2. Forward through each block
  for (std::int32_t i = 0; i < config.num_hidden_layers; ++i) {
    const Block &block = model.weight.blocks[i];
    forward_block(block, k_cache[i], v_cache[i], pos, kv_sink, kv_pos, kv_len);
  }
}

std::uint32_t InferenceCtx::argmax() const {
  return std::distance(logits.as<float>(),
                       std::max_element(logits.as<float>(), logits.as<float>() + config.vocab_size));
}

} // namespace tinyllm
