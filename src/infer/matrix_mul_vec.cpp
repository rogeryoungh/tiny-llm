#include "../utils/precision.hpp"
#include "infer.hpp"

#include <thread>
#include <vector>

namespace tinyllm {

template <typename T>
static void matrix_mul_vec_fp32_naive(float *out, const float *a, const T *b, std::size_t m, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum = 0.0f;
    const T* b0 = b + i * m;
    for (std::size_t j = 0; j < m; ++j) {
      auto aj = a[j];
      auto bj = _cvt_to_fp32(b0[j]);
      sum += bj * aj;
    }
    out[i] = sum;
  }
}

template <typename T>
static void matrix_mul_vec_bias_fp32_naive(float *out, const float *a, const T *b, const T *bias, std::size_t m,
                                           std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum = 0;
    const T* b0 = b + i * m;
    for (std::size_t j = 0; j < m; ++j) {
      auto aj = a[j];
      auto bj = _cvt_to_fp32(b0[j]);
      sum += bj * aj;
    }
    out[i] = sum + _cvt_to_fp32(bias[i]);
  }
}

template <typename T>
static void matrix_mul_vec_fp32_threaded(float *out, const float *a, const T *b, std::size_t m, std::size_t n) {
  std::uint32_t num_threads = std::max(std::thread::hardware_concurrency() / 2, 1u);
  if (num_threads == 0)
    num_threads = 4;

  std::vector<std::thread> threads;
  std::size_t chunk_size = (n + num_threads - 1) / num_threads;

  for (std::size_t i = 0; i < n; i += chunk_size) {
    std::size_t end = std::min(i + chunk_size, n);
    if (i >= end)
      break;
    threads.emplace_back(matrix_mul_vec_fp32_naive<T>, out + i, a, b + i * m, m, end - i);
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

template <typename T>
static void matrix_mul_vec_bias_fp32_threaded(float *out, const float *a, const T *b, const T *bias, std::size_t m,
                                              std::size_t n) {
  std::uint32_t num_threads = std::max(std::thread::hardware_concurrency() / 2, 1u);
  if (num_threads == 0)
    num_threads = 4;

  std::vector<std::thread> threads;
  std::size_t chunk_size = (n + num_threads - 1) / num_threads;

  for (std::size_t i = 0; i < n; i += chunk_size) {
    std::size_t end = std::min(i + chunk_size, n);
    if (i >= end)
      break;
    threads.emplace_back(matrix_mul_vec_bias_fp32_naive<T>, out + i, a, b + i * m, bias + i, m, end - i);
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

#ifdef TINYLLM_USE_OPENBLAS
#include <cblas.h>
static void matrix_mul_vec_fp32_blas(float *out, const float *a, const float *b, std::size_t m, std::size_t n) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, a, n, b, 1, 0.0f, out, 1);
}
#endif

void matrix_mul_vec_fp32(float *out, const float *a, const float *b, std::size_t m, std::size_t n) {
#ifdef TINYLLM_USE_OPENBLAS
  matrix_mul_vec_fp32_blas(out, a, b, m, n);
#else
  matrix_mul_vec_fp32_threaded(out, a, b, m, n);
#endif
}

void matrix_mul_vec_fp32_b_bf16(float *out, const float *a, const std::uint16_t *b, std::size_t m, std::size_t n) {
#ifdef TINYLLM_USE_OPENBLAS
  matrix_mul_vec_fp32_blas(out, a, b, m, n);
#else
  matrix_mul_vec_fp32_threaded(out, a, b, m, n);
#endif
}

void matrix_mul_vec_bias_fp32(float *out, const float *a, const float *b, const float *bias, std::size_t m,
                              std::size_t n) {
  matrix_mul_vec_bias_fp32_threaded(out, a, b, bias, m, n);
}

void matrix_mul_vec_bias_fp32_b_bf16(float *out, const float *a, const std::uint16_t *b, const std::uint16_t *bias,
                                     std::size_t m, std::size_t n) {
  matrix_mul_vec_bias_fp32_threaded(out, a, b, bias, m, n);
}

} // namespace tinyllm