#pragma once

#include "../../core/float.hpp"

#include <cmath>

namespace tinyllm {

void rms_norm_fp32(float *out, const float *x, const float *weight, std::size_t size, float eps);

void rms_norm_fp32_weight_bf16(float *out, const float *x, const bf16_t *weight, std::size_t size, float eps);

void rms_norm_fp32_weight_fp16(float *out, const float *x, const fp16_t *weight, std::size_t size, float eps);

void softmax_fp32(float *out, const float *x, std::size_t size);

inline float silu_fp32(float x) { return x / (1.0f + std::exp(-x)); }

// void matrix_mul_fp32(float *out, const float *a, const float *b, std::size_t m, std::size_t n, std::size_t k);

void gemv_fp32(float *out, const float *a, const float *b, std::size_t m, std::size_t n);

void gemv_bias_fp32(float *out, const float *a, const float *b, const float *bias, std::size_t m,
                              std::size_t n);

void gemv_fp32_b_bf16(float *out, const float *a, const bf16_t *b, std::size_t m, std::size_t n);

void gemv_bias_fp32_b_bf16(float *out, const float *a, const bf16_t *b, const bf16_t *bias, std::size_t m,
                                     std::size_t n);

void gemv_fp32_b_fp16(float *out, const float *a, const fp16_t *b, std::size_t m, std::size_t n);

void gemv_bias_fp32_b_fp16(float *out, const float *a, const fp16_t *b, const fp16_t *bias, std::size_t m,
                                     std::size_t n);

void rope_inplace_fp32(float *x, std::size_t head_dim, std::size_t pos, float theta);

void attention_softmax_fp32(float *out, float *atth, const float *qh, const float *kh, const float *vh,
                            std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len);

void attention_softmax_fp32_kv_bf16(float *out, float *atth, const float *qh, const bf16_t *kh, const bf16_t *vh,
                                    std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len);

void attention_softmax_fp32_kv_fp16(float *out, float *atth, const float *qh, const fp16_t *kh, const fp16_t *vh,
                                    std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len);

} // namespace tinyllm
