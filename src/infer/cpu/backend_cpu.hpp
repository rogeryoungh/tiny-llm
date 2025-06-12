#pragma once

#include "../inference_ctx.hpp"
#include "../../core/config.hpp"
#include "../../core/model.hpp"
#include "../../core/tensor.hpp"

namespace tinyllm {

struct InferenceBackendCPU : InferenceBackend {
  const Model &model;
  Config &config;
  ArenaAlloc alloc;
  std::size_t kv_size;

  DataType kv_dtype;

  Tensor x, xb, xb2, hb, hb2, q, k, v, attn, logits;
  std::vector<Tensor> k_cache, v_cache;

  InferenceBackendCPU(Model &model, std::size_t kv_size, DataType kv_dtype = DataType::F32);

  void forward_block(const Block &block, Tensor &kc, Tensor &vc, std::int32_t pos, std::int32_t kv_sink,
                     std::int32_t kv_pos, std::int32_t kv_len);

  void forward(std::int32_t token, std::int32_t pos);

  void forward_prefill(std::int32_t token, std::int32_t pos);

  std::uint32_t argmax() const;

protected:
  void _rms_norm(float *out, const float *x, const Tensor &weight, std::size_t size, float eps);

  void _matrix_mul_vec(float *out, const float *a, const Tensor &weight, std::size_t m, std::size_t n);

  void _matrix_mul_vec_bias(float *out, const float *a, const Tensor &weight, const Tensor &bias, std::size_t m,
                            std::size_t n);
};

} // namespace tinyllm
