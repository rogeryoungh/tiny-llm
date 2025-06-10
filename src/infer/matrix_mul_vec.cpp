#include "infer.hpp"

#include <thread>
#include <vector>

namespace tinyllm {

static void matrix_mul_vec_fp32_naive(float *out, const float *a, const float *b, std::size_t m, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (std::size_t j = 0; j < m; ++j) {
      auto aj = a[j];
      auto bj = b[i * m + j];
      sum += bj * aj;
    }
    out[i] = sum;
  }
}

static void matrix_mul_vec_bias_fp32_naive(float *out, const float *a, const float *b, const float *bias, std::size_t m,
                                           std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum = 0;
    for (std::size_t j = 0; j < m; ++j) {
      auto aj = a[j];
      auto bj = b[i * m + j];
      sum += bj * aj;
    }
    out[i] = sum + bias[i];
  }
}

static void matrix_mul_vec_fp32_threaded(float *out, const float *a, const float *b, std::size_t m, std::size_t n) {
  std::uint32_t num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 4;

  std::vector<std::thread> threads;

  auto worker = [&](std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
      float sum = 0.0f;
      for (std::size_t j = 0; j < m; ++j) {
        auto aj = a[j];
        auto bj = b[i * m + j];
        sum += bj * aj;
      }
      out[i] = sum;
    }
  };
  std::size_t chunk_size = (n + num_threads - 1) / num_threads;

  for (std::size_t i = 0; i < n; i += chunk_size) {
    std::size_t end = std::min(i + chunk_size, n);
    threads.emplace_back(worker, i, end);
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

static void matrix_mul_vec_bias_fp32_threaded(float *out, const float *a, const float *b, const float* bias, std::size_t m, std::size_t n) {
  std::uint32_t num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 4;

  std::vector<std::thread> threads;

  auto worker = [&](std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
      float sum = 0.0f;
      for (std::size_t j = 0; j < m; ++j) {
        auto aj = a[j];
        auto bj = b[i * m + j];
        sum += bj * aj;
      }
      out[i] = sum + bias[i];
    }
  };
  std::size_t chunk_size = (n + num_threads - 1) / num_threads;

  for (std::size_t i = 0; i < n; i += chunk_size) {
    std::size_t end = std::min(i + chunk_size, n);
    threads.emplace_back(worker, i, end);
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

void matrix_mul_vec_bias_fp32(float *out, const float *a, const float *b, const float *bias, std::size_t m,
                              std::size_t n) {
  matrix_mul_vec_bias_fp32_threaded(out, a, b, bias, m, n);
}

} // namespace tinyllm