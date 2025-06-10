#include "infer.hpp"
#include "../utils/precision.hpp"

namespace tinyllm {

void rms_norm_fp32(float *out, const float *x, const float *weight, std::size_t size, float eps) {
  float norm = 0.0f;
  for (std::size_t i = 0; i < size; ++i) {
    norm += x[i] * x[i];
  }
  norm = std::sqrt(norm / size + eps);
  float inv_norm = 1.0f / norm;
  for (std::size_t i = 0; i < size; ++i) {
    out[i] = (x[i] * inv_norm) * weight[i];
  }
}

void softmax_fp32(float *out, const float *x, std::size_t size) {
  float max_val = x[0];
  for (std::size_t i = 1; i < size; ++i) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }

  float sum = 0.0f;
  for (std::size_t i = 0; i < size; ++i) {
    out[i] = std::exp(x[i] - max_val);
    sum += out[i];
  }

  for (std::size_t i = 0; i < size; ++i) {
    out[i] /= sum;
  }
}

void matrix_mul_fp32(float *out, const float *a, const float *b, std::size_t m, std::size_t n, std::size_t k) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (std::size_t l = 0; l < k; ++l) {
        sum += a[i * k + l] * b[l * n + j];
      }
      out[i * n + j] = sum;
    }
  }
}

void rope_inplace_fp32(float *x, std::size_t d, std::size_t head_dim, std::size_t pos, float theta,
                       std::size_t rotary_dim) {
  for (std::size_t i = 0; i < d; i += 2) {
    std::size_t j_head = i % head_dim;
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / std::pow(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = std::cos(val);
    float fci = std::sin(val);

    float v0 = x[i];
    float v1 = x[i + 1];
    x[i] = v0 * fcr - v1 * fci;
    x[i + 1] = v0 * fci + v1 * fcr;
  }
}

void attention_softmax_fp32(float *xout, float *atth, const float *qh, const float *kh, const float *vh,
                            std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len) {
  std::size_t kv_stride = n_kv_heads * head_dim;
  for (std::size_t i = 0; i < kv_len; ++i) {
    float sum = 0.0f;
    for (std::size_t j = 0; j < head_dim; ++j) {
      sum += qh[j] * kh[i * kv_stride + j];
    }
    atth[i] = sum / std::sqrt(head_dim);
  }

  softmax_fp32(atth, atth, kv_len);

  for (std::size_t i = 0; i < head_dim; ++i) {
    float sum = 0.0f;
    for (std::size_t j = 0; j < kv_len; ++j) {
      sum += atth[j] * vh[j * kv_stride + i];
    }
    xout[i] = sum;
  }
}

void attention_softmax_fp32(float *xout, float *atth, const float *qh, const std::uint16_t *kh, const std::uint16_t *vh,
                            std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len) {
  std::size_t kv_stride = n_kv_heads * head_dim;
  for (std::size_t i = 0; i < kv_len; ++i) {
    float sum = 0.0f;
    for (std::size_t j = 0; j < head_dim; ++j) {
      sum += qh[j] * bf16_to_fp32(kh[i * kv_stride + j]);
    }
    atth[i] = sum / std::sqrt(head_dim);
  }

  softmax_fp32(atth, atth, kv_len);

  for (std::size_t i = 0; i < head_dim; ++i) {
    float sum = 0.0f;
    for (std::size_t j = 0; j < kv_len; ++j) {
      sum += atth[j] * bf16_to_fp32(vh[j * kv_stride + i]);
    }
    xout[i] = sum;
  }
}

} // namespace tinyllm
