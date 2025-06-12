#include "infer.hpp"
#include "../../utils/precision.hpp"

namespace tinyllm {

template <typename T>
static void _rms_norm_fp32(float *out, const float *x, const T *weight, std::size_t size, float eps) {
  float norm = 0.0f;
  for (std::size_t i = 0; i < size; ++i) {
    norm += x[i] * x[i];
  }
  norm = std::sqrt(norm / size + eps);
  float inv_norm = 1.0f / norm;
  for (std::size_t i = 0; i < size; ++i) {
    out[i] = (x[i] * inv_norm) * _cvt_to_fp32<T>(weight[i]);
  }
}

void rms_norm_fp32(float *out, const float *x, const float *weight, std::size_t size, float eps) {
  _rms_norm_fp32(out, x, weight, size, eps);
}

void rms_norm_fp32_weight_bf16(float *out, const float *x, const std::uint16_t *weight, std::size_t size, float eps) {
  _rms_norm_fp32(out, x, weight, size, eps);
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

void rope_inplace_fp32(float *x, std::size_t head_dim, std::size_t pos, float theta) {
  const std::size_t half_dim = head_dim / 2;
  for (std::size_t i = 0; i < half_dim; ++i) {
    float freq = std::pow(theta, -float(i) / half_dim);
    float val = pos * freq;
    float vc = std::cos(val);
    float vs = std::sin(val);

    float v0 = x[i];
    float v1 = x[i + half_dim];
    x[i] = v0 * vc - v1 * vs;
    x[i + half_dim] = v0 * vs + v1 * vc;
  }
}

template <typename T>
static void _attention_softmax_fp32(float *xout, float *atth, const float *qh, const T *kh, const T *vh,
                                    std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len) {
  std::size_t kv_stride = n_kv_heads * head_dim;
  for (std::size_t i = 0; i < kv_len; ++i) {
    float sum = 0.0f;
    for (std::size_t j = 0; j < head_dim; ++j) {
      sum += qh[j] * _cvt_to_fp32<T>(kh[i * kv_stride + j]);
    }
    atth[i] = sum / std::sqrt(head_dim);
  }

  softmax_fp32(atth, atth, kv_len);

  for (std::size_t i = 0; i < head_dim; ++i) {
    float sum = 0.0f;
    for (std::size_t j = 0; j < kv_len; ++j) {
      sum += atth[j] * _cvt_to_fp32<T>(vh[j * kv_stride + i]);
    }
    xout[i] = sum;
  }
}

void attention_softmax_fp32(float *xout, float *atth, const float *qh, const float *kh, const float *vh,
                            std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len) {
  _attention_softmax_fp32(xout, atth, qh, kh, vh, head_dim, n_kv_heads, kv_len);
}

void attention_softmax_fp32_kv_bf16(float *xout, float *atth, const float *qh, const std::uint16_t *kh,
                                    const std::uint16_t *vh, std::size_t head_dim, std::size_t n_kv_heads,
                                    std::size_t kv_len) {
  _attention_softmax_fp32(xout, atth, qh, kh, vh, head_dim, n_kv_heads, kv_len);
}

} // namespace tinyllm
