#include "infer.hpp"
#include "../../utils/precision.hpp"
#include "parallel_for.hpp"
#include <cfloat>

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

void rms_norm_fp32_weight_bf16(float *out, const float *x, const bf16_t *weight, std::size_t size, float eps) {
  _rms_norm_fp32(out, x, weight, size, eps);
}

void rms_norm_fp32_weight_fp16(float *out, const float *x, const fp16_t *weight, std::size_t size, float eps) {
  _rms_norm_fp32(out, x, weight, size, eps);
}

void softmax_fp32(float *out, const float *x, std::size_t size) {
  float max_val = -FLT_MAX;
  float sum_exp = 0.0f;
  for (std::size_t i = 0; i < size; ++i) {
    if (x[i] > max_val) {
      sum_exp = sum_exp * std::exp(max_val - x[i]) + 1.0f;
      max_val = x[i];
    } else {
      sum_exp += std::exp(x[i] - max_val);
    }
  }

  for (std::size_t i = 0; i < size; ++i) {
    out[i] = std::exp(x[i] - max_val) / sum_exp;
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
static void _attention_softmax_fp32(float *xout, const float *qh, const T *kh, const T *vh, std::size_t head_dim,
                                    std::size_t n_kv_heads, std::size_t kv_len) {
  std::size_t kv_stride = n_kv_heads * head_dim;
  float max_val = -FLT_MAX;
  float sum_exp = 0.0f;
  std::fill_n(xout, head_dim, 0.0f);
  for (std::size_t i = 0; i < kv_len; ++i) {
    float sum = 0.0f;
    for (std::size_t j = 0; j < head_dim; ++j) {
      sum += qh[j] * _cvt_to_fp32<T>(kh[i * kv_stride + j]);
    }
    float ai = sum / std::sqrt(head_dim);
    if (ai > max_val) {
      float exp_max_ai = std::exp(max_val - ai);
      float exp = sum_exp * exp_max_ai + 1.0f;
      float scale1 = exp_max_ai * sum_exp / exp;
      float scale2 = 1.0f / exp;
      for (std::size_t j = 0; j < head_dim; ++j) {
        float hj = _cvt_to_fp32(vh[i * kv_stride + j]);
        xout[j] = xout[j] * scale1 + hj * scale2;
      }
      sum_exp = exp;
      max_val = ai;
    } else {
      float exp_max_ai = std::exp(ai - max_val);
      float exp = sum_exp + exp_max_ai;
      float scale1 = sum_exp / exp;
      float scale2 = exp_max_ai / exp;
      for (std::size_t j = 0; j < head_dim; ++j) {
        float hj = _cvt_to_fp32(vh[i * kv_stride + j]);
        xout[j] = xout[j] * scale1 + hj * scale2;
      }
      sum_exp = exp;
    }
  }
}

template <typename T>
void _mh_attention_fp32(float *out, const float *q, const T *k, const T *v, std::size_t num_heads, std::size_t head_dim,
                        std::size_t n_kv_heads, std::size_t kv_len) {
  const std::int32_t q_per_head = num_heads / n_kv_heads;
  auto attn_f = [&](std::int32_t h) {
    const float *qh = q + h * head_dim;
    float *outh = out + h * head_dim;
    std::int32_t kv_offset = (h / q_per_head) * head_dim;
    const T *kh = k + kv_offset;
    const T *vh = v + kv_offset;
    _attention_softmax_fp32<T>(outh, qh, kh, vh, head_dim, n_kv_heads, kv_len);
  };

  const std::size_t cost = num_heads * head_dim * kv_len;

  if (cost < 2E5) {
    for (std::int32_t h = 0; h < num_heads; ++h) {
      attn_f(h);
    }
  } else {
    parallel_for(0, num_heads, [&](std::size_t beg, std::size_t end) {
      for (std::int32_t h = beg; h < end; ++h) {
        attn_f(h);
      }
    });
  }
}

void attention_softmax_fp32(float *xout, float *atth, const float *qh, const float *kh, const float *vh,
                            std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len) {
  _attention_softmax_fp32(xout, qh, kh, vh, head_dim, n_kv_heads, kv_len);
}

void attention_softmax_fp32_kv_bf16(float *xout, float *atth, const float *qh, const bf16_t *kh, const bf16_t *vh,
                                    std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len) {
  _attention_softmax_fp32(xout, qh, kh, vh, head_dim, n_kv_heads, kv_len);
}

void attention_softmax_fp32_kv_fp16(float *xout, float *atth, const float *qh, const fp16_t *kh, const fp16_t *vh,
                                    std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len) {
  _attention_softmax_fp32(xout, qh, kh, vh, head_dim, n_kv_heads, kv_len);
}

void mh_attention_fp32(float *out, float *atth, const float *q, const float *k, const float *v, std::size_t num_heads,
                       std::size_t head_dim, std::size_t n_kv_heads, std::size_t kv_len) {
  _mh_attention_fp32(out, q, k, v, num_heads, head_dim, n_kv_heads, kv_len);
}

void mh_attention_fp32_kv_bf16(float *out, float *att, const float *q, const bf16_t *k, const bf16_t *v,
                               std::size_t num_heads, std::size_t head_dim, std::size_t n_kv_heads,
                               std::size_t kv_len) {
  _mh_attention_fp32(out, q, k, v, num_heads, head_dim, n_kv_heads, kv_len);
}

void mh_attention_fp32_kv_fp16(float *out, float *att, const float *q, const fp16_t *k, const fp16_t *v,
                               std::size_t num_heads, std::size_t head_dim, std::size_t n_kv_heads,
                               std::size_t kv_len) {
  _mh_attention_fp32(out, q, k, v, num_heads, head_dim, n_kv_heads, kv_len);
}

} // namespace tinyllm
