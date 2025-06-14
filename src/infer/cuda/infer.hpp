#pragma once

#include <cstddef>

namespace tinyllm::cuda {

void *cuda_malloc(size_t size);

void copy_to_device(const void *src, size_t size, void *dst);

void copy_to_host(const void *src, size_t size, void *dst);

void cuda_free(void *ptr);

void check_and_sync();

void *upload(const void *src, size_t size);

void rope_inplace_fp32(float *x, int num_heads, int head_dim, int pos, float theta);

// void rope_inplace_fp16(void *x, int num_heads, int head_dim, int pos, float theta);

void rms_norm_fp32_b_fp16(float *out, const float *x, const void *weight, int size, int num_batches, float eps);

void matrix_mul_vec_fp32_b_fp16(float *out, const float *a, const void *b, int m, int n);

void matrix_mul_vec_bias_fp32_b_fp16(float *out, const float *a, const void *b, const void *bias, int m, int n);

void vec_add_inplace_fp32(float *out, const float *a, int size);

void copy_fp16_to_fp32_n(const void *src, int n, float *dst);

void copy_fp32_to_fp16_n(const float *src, int n, void *dst);

void softmax_fp32(float *out, const float *x, int size);

void attention_softmax_fp32_kv_fp16(float *out, float *atth, const float *qh, const void *kh, const void *vh,
                                    int head_dim, int n_kv_heads, int kv_len);

void swiglu_fp32(float *out, const float *x, const float *gate, int size);

void mh_attention_fp32_kv_fp16(float *out, float *att, const float *q, const void *k, const void *v, int num_heads,
                               int head_dim, int n_kv_heads, int kv_len);

} // namespace tinyllm::cuda
