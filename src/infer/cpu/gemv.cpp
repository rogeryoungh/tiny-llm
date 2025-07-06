#include "../../utils/precision.hpp"
#include "infer.hpp"
#include "parallel_for.hpp"

#include <cassert>
#include <immintrin.h>

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

inline float f32x8_reduce_sum(__m256 x) {
  x = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 1));
  x = _mm256_add_ps(x, _mm256_permute_ps(x, 0b01001110));
  x = _mm256_add_ps(x, _mm256_permute_ps(x, 0b10110001));
  return _mm256_cvtss_f32(x);
}

inline float dot_fp32_avx2(const float *a, const float *x, size_t k) {
  __m256 sum8 = _mm256_setzero_ps();
  for (size_t j = 0; j < k; j += 8) {
    __m256 a8 = _mm256_loadu_ps(a + j);
    __m256 x8 = _mm256_loadu_ps(x + j);
    sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(a8, x8));
  }
  return f32x8_reduce_sum(sum8);
}

void gemv_fp32_avx2(float *out, const float *a, const float *x, size_t m, size_t k) {
  assert(k % 8 == 0);
  for (size_t i = 0; i < m; ++i) {
    out[i] = dot_fp32_avx2(a + i * k, x, k);
  }
}

void gemv_bias_fp32_avx2(float *out, const float *a, const float *bias, const float *x, size_t m, size_t k) {
  assert(k % 8 == 0);
  for (size_t i = 0; i < m; ++i) {
    out[i] = dot_fp32_avx2(a + i * k, x, k) + bias[i];
  }
}

inline float dot_bf16_fp32_avx2(const bf16_t *a, const float *x, size_t k) {
  __m256 sum8 = _mm256_setzero_ps();
  __m256i zero = _mm256_setzero_si256();
  for (size_t j = 0; j < k; j += 16) {
    __m256i a8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + j));

    __m256i a8lo_02 = _mm256_unpacklo_epi16(zero, a8);
    __m256i a8hi_13 = _mm256_unpackhi_epi16(zero, a8);

    __m256i a8lo = _mm256_permute2x128_si256(a8lo_02, a8hi_13, 0x20);
    __m256i a8hi = _mm256_permute2x128_si256(a8lo_02, a8hi_13, 0x31);

    __m256 a8lo_f32 = _mm256_castsi256_ps(a8lo);
    __m256 a8hi_f32 = _mm256_castsi256_ps(a8hi);

    __m256 x8lo = _mm256_loadu_ps(x + j);
    __m256 x8hi = _mm256_loadu_ps(x + j + 8);

    sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(a8lo_f32, x8lo));
    sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(a8hi_f32, x8hi));
  }
  return f32x8_reduce_sum(sum8);
}

void gemv_bf16_fp32_avx2(float *out, const bf16_t *a, const float *x, size_t m, size_t k) {
  assert(k % 8 == 0);
  for (size_t i = 0; i < m; ++i) {
    out[i] = dot_bf16_fp32_avx2(a + i * k, x, k);
  }
}

void gemv_bias_bf16_fp32_avx2(float *out, const bf16_t *a, const bf16_t *bias, const float *x, size_t m, size_t k) {
  assert(k % 16 == 0);
  for (size_t i = 0; i < m; ++i) {
    out[i] = dot_bf16_fp32_avx2(a + i * k, x, k) + _cvt_to_fp32(bias[i]);
  }
}

inline float dot_fp16_fp32_f16c(const fp16_t *a, const float *x, size_t k) {
  __m256 sum8 = _mm256_setzero_ps();
  for (size_t j = 0; j < k; j += 16) {
    __m256i a8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + j));

    __m128i a8lo128 = _mm256_castsi256_si128(a8);
    __m128i a8hi128 = _mm256_extractf128_si256(a8, 1);

    __m256 a8lo = _mm256_cvtph_ps(a8lo128);
    __m256 a8hi = _mm256_cvtph_ps(a8hi128);

    __m256 x8lo = _mm256_loadu_ps(x + j);
    __m256 x8hi = _mm256_loadu_ps(x + j + 8);

    sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(a8lo, x8lo));
    sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(a8hi, x8hi));
  }
  return f32x8_reduce_sum(sum8);
}

void gemv_fp16_fp32_f16c(float *out, const fp16_t *a, const float *x, size_t m, size_t k) {
  assert(k % 16 == 0);
  for (size_t i = 0; i < m; ++i) {
    out[i] = dot_fp16_fp32_f16c(a + i * k, x, k);
  }
}

void gemv_bias_fp16_fp32_f16c(float *out, const fp16_t *a, const fp16_t *bias, const float *x, size_t m, size_t k) {
  assert(k % 16 == 0);
  for (size_t i = 0; i < m; ++i) {
    out[i] = dot_fp16_fp32_f16c(a + i * k, x, k) + _cvt_to_fp32(bias[i]);
  }
}

template <typename T>
static void gemv_fp32_threaded(float *out, const float *a, const T *b, std::size_t m, std::size_t n) {
  parallel_for(0, n, [&](size_t beg, size_t end) {
    if constexpr (std::is_same_v<T, float>) {
      gemv_fp32_avx2(out + beg, b + beg * m, a, end - beg, m);
    } else if constexpr (std::is_same_v<T, bf16_t>) {
      gemv_bf16_fp32_avx2(out + beg, b + beg * m, a, end - beg, m);
    } else {
      gemv_fp16_fp32_f16c(out + beg, b + beg * m, a, end - beg, m);
    }
  });
}

template <typename T>
static void gemv_bias_fp32_threaded(float *out, const float *a, const T *b, const T *bias, std::size_t m,
                                    std::size_t n) {
  parallel_for(0, n, [&](size_t beg, size_t end) {
    if constexpr (std::is_same_v<T, float>) {
      gemv_bias_fp32_avx2(out + beg, b + beg * m, bias + beg, a, end - beg, m);
    } else if constexpr (std::is_same_v<T, bf16_t>) {
      gemv_bias_bf16_fp32_avx2(out + beg, b + beg * m, bias + beg, a, end - beg, m);
    } else {
      gemv_bias_fp16_fp32_f16c(out + beg, b + beg * m, bias + beg, a, end - beg, m);
    }
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