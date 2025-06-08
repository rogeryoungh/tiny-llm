#pragma once

#include "../core/config.hpp"
#include "../core/model.hpp"
#include "../core/tensor.hpp"

namespace tinyllm {

struct InferenceCtx {
  Config &config;
  TensorAlloc alloc;
  std::size_t kv_size;

  Tensor x, xb, xb2, hb, hb2, q, k, v, attn, logits;
  std::vector<Tensor> k_cache, v_cache;

  InferenceCtx(Config &cfg, std::size_t kv_size);
  ~InferenceCtx();

  void forward_block(const Block &block, Tensor &kc, Tensor &vc, std::int32_t pos, std::int32_t kv_sink,
                     std::int32_t kv_pos, std::int32_t kv_len);

  void forward(const Model &model, std::int32_t token, std::int32_t pos);

  std::uint32_t argmax() const;

protected:
  void _embeding(const Model &model, std::int32_t token);
};

} // namespace tinyllm
