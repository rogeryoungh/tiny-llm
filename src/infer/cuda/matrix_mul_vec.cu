#include "infer.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tinyllm::cuda {

__global__ void matrix_mul_vec_fp32_b_fp16_kernel(float *out, const float *a, const half *b, int m, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;

  const half *b_row = b + tid * m;
  float sum = 0.0f;

  for (int j = 0; j < m; ++j) {
    sum += a[j] * __half2float(b_row[j]);
  }
  out[tid] = sum;
}

void matrix_mul_vec_fp32_b_fp16(float *out, const float *a, const void *b, int m, int n) {
  const int block = 16;
  int grid = (n + block - 1) / block;

  matrix_mul_vec_fp32_b_fp16_kernel<<<grid, block>>>(out, a, reinterpret_cast<const half *>(b), m, n);
}

__global__ void matrix_mul_vec_bias_fp32_b_fp16_kernel(float *out, const float *a, const half *b, const half *bias,
                                                       int m, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;

  const half *b_row = b + tid * m;
  float sum = 0.0f;

  for (int j = 0; j < m; ++j) {
    sum += a[j] * __half2float(b_row[j]);
  }
  out[tid] = sum + __half2float(bias[tid]);
}

void matrix_mul_vec_bias_fp32_b_fp16(float *out, const float *a, const void *b, const void *bias, int m, int n) {
  const int block = 16;
  int grid = (n + block - 1) / block;

  matrix_mul_vec_bias_fp32_b_fp16_kernel<<<grid, block>>>(out, a, reinterpret_cast<const half *>(b),
                                                          reinterpret_cast<const half *>(bias), m, n);
}

} // namespace tinyllm::cuda
