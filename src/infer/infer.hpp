#pragma once

#include <cmath>
#include <cstdint>

namespace tinyllm {

void rms_norm_fp32(float *out, const float *x, const float *weight, std::size_t size, float eps);

void rms_norm_fp32_weight_bf16(float *out, const float *x, const std::uint16_t *weight, std::size_t size, float eps);

void softmax_fp32(float *out, const float *x, std::size_t size);

inline float silu_fp32(float x) { return x / (1.0f + std::exp(-x)); }

// void matrix_mul_fp32(float *out, const float *a, const float *b, std::size_t m, std::size_t n, std::size_t k);

void matrix_mul_vec_fp32(float *out, const float *a, const float *b, std::size_t m, std::size_t n);

void matrix_mul_vec_bias_fp32(float *out, const float *a, const float *b, const float *bias, std::size_t m,
                              std::size_t n);

void matrix_mul_vec_fp32_b_bf16(float *out, const float *a, const std::uint16_t *b, std::size_t m, std::size_t n);

void matrix_mul_vec_bias_fp32_b_bf16(float *out, const float *a, const std::uint16_t *b, const std::uint16_t *bias, std::size_t m,
                              std::size_t n);

void rope_inplace_fp32(float *x, std::size_t head_dim, std::size_t pos, float theta);

void attention_softmax_fp32(float *out, float *atth, const float *qh, const float *kh, const float *vh,
                            std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len);

void attention_softmax_fp32_kv_bf16(float *out, float *atth, const float *qh, const std::uint16_t *kh,
                                    const std::uint16_t *vh, std::size_t head_dim, std::size_t n_kv_heads,
                                    std::size_t kv_len);

} // namespace tinyllm
