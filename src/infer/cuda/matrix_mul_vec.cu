#include "block_reduce.cuh"
#include "infer.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tinyllm::cuda {

__global__ void matrix_mul_vec_fp32_b_fp16_kernel(const float *a, const half *b, float *out, int m, int n) {
  const int col = blockIdx.x;
  if (col >= n)
    return;

  const int lane = threadIdx.x;
  float sum = 0.0f;
  const half *b_row = b + col * m;

  for (int j = lane; j < m; j += warpSize) {
    float aj = a[j];
    float bj = __half2float(b_row[j]);
    sum += aj * bj;
  }

  sum = warp_reduce_sum(sum);

  if (lane == 0) {
    out[col] = sum;
  }
}

void matrix_mul_vec_fp32_b_fp16(float *out, const float *a, const void *b, int m, int n) {
  constexpr int WARP_SIZE = 32;
  matrix_mul_vec_fp32_b_fp16_kernel<<<n, WARP_SIZE>>>(a, reinterpret_cast<const half *>(b), out, m, n);
}

__global__ void matrix_mul_vec_bias_fp32_b_fp16_kernel(const float *a, const half *b, const half *bias, float *out,
                                                       int m, int n) {
  const int col = blockIdx.x;
  if (col >= n)
    return;

  const int lane = threadIdx.x;
  float sum = 0.0f;
  const half *b_row = b + col * m;

  for (int j = lane; j < m; j += warpSize) {
    float aj = a[j];
    float bj = __half2float(b_row[j]);
    sum += aj * bj;
  }

  sum = warp_reduce_sum(sum);

  if (lane == 0) {
    out[col] = sum + __half2float(bias[col]);
  }
}

void matrix_mul_vec_bias_fp32_b_fp16(float *out, const float *a, const void *b, const void *bias, int m, int n) {
  constexpr int WARP_SIZE = 32;
  matrix_mul_vec_bias_fp32_b_fp16_kernel<<<n, WARP_SIZE>>>(a, reinterpret_cast<const half *>(b),
                                                           reinterpret_cast<const half *>(bias), out, m, n);
}

} // namespace tinyllm::cuda
