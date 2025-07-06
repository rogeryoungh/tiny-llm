#include "../../utils/precision.hpp"
#include "infer.hpp"
#include "parallel_for.hpp"

namespace tinyllm {

template <typename T>
static void gemv_fp32_naive(float *out, const float *a, const T *b, std::size_t m, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum = 0.0f;
    const T *b0 = b + i * m;
    for (std::size_t j = 0; j < m; ++j) {
      auto aj = a[j];
      auto bj = _cvt_to_fp32(b0[j]);
      sum += bj * aj;
    }
    out[i] = sum;
  }
}

template <typename T>
static void gemv_bias_fp32_naive(float *out, const float *a, const T *b, const T *bias, std::size_t m, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum = 0;
    const T *b0 = b + i * m;
    for (std::size_t j = 0; j < m; ++j) {
      auto aj = a[j];
      auto bj = _cvt_to_fp32(b0[j]);
      sum += bj * aj;
    }
    out[i] = sum + _cvt_to_fp32(bias[i]);
  }
}

template <typename T>
static void gemv_fp32_threaded(float *out, const float *a, const T *b, std::size_t m, std::size_t n) {
  parallel_for(0, n, [&](size_t beg, size_t end) { gemv_fp32_naive<T>(out + beg, a, b + beg * m, m, end - beg); });
}

template <typename T>
static void gemv_bias_fp32_threaded(float *out, const float *a, const T *b, const T *bias, std::size_t m,
                                    std::size_t n) {
  parallel_for(0, n, [&](size_t beg, size_t end) {
    gemv_bias_fp32_naive<T>(out + beg, a, b + beg * m, bias + beg, m, end - beg);
  });
}

#ifdef TINYLLM_USE_OPENBLAS
#include <cblas.h>
static void gemv_fp32_blas(float *out, const float *a, const float *b, std::size_t m, std::size_t n) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, a, n, b, 1, 0.0f, out, 1);
}
#endif

void gemv_fp32(float *out, const float *a, const float *b, std::size_t m, std::size_t n) {
#ifdef TINYLLM_USE_OPENBLAS
  gemv_fp32_blas(out, a, b, m, n);
#else
  gemv_fp32_threaded(out, a, b, m, n);
#endif
}

void gemv_fp32_b_bf16(float *out, const float *a, const bf16_t *b, std::size_t m, std::size_t n) {
#ifdef TINYLLM_USE_OPENBLAS
  gemv_fp32_blas(out, a, b, m, n);
#else
  gemv_fp32_threaded(out, a, b, m, n);
#endif
}
void gemv_fp32_b_fp16(float *out, const float *a, const fp16_t *b, std::size_t m, std::size_t n) {
#ifdef TINYLLM_USE_OPENBLAS
  gemv_fp32_blas(out, a, b, m, n);
#else
  gemv_fp32_threaded(out, a, b, m, n);
#endif
}

void gemv_bias_fp32(float *out, const float *a, const float *b, const float *bias, std::size_t m, std::size_t n) {
  gemv_bias_fp32_threaded(out, a, b, bias, m, n);
}

void gemv_bias_fp32_b_bf16(float *out, const float *a, const bf16_t *b, const bf16_t *bias, std::size_t m,
                           std::size_t n) {
  gemv_bias_fp32_threaded(out, a, b, bias, m, n);
}

void gemv_bias_fp32_b_fp16(float *out, const float *a, const fp16_t *b, const fp16_t *bias, std::size_t m,
                           std::size_t n) {
  gemv_bias_fp32_threaded(out, a, b, bias, m, n);
}

} // namespace tinyllm