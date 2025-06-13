#include "block_reduce.cuh"
#include "infer.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tinyllm::cuda {

__global__ void matvec_fp16_fp32_kernel(const float *A, const half *B, float *out, int m, int n) {
  int col = blockIdx.x;
  if (col >= n)
    return;

  // Local dot‐product accumulation
  float local_sum = 0.0f;
  int base = col * m;

  // Grid‐stride over the m dimension
  for (int j = threadIdx.x; j < m; j += blockDim.x) {
    float a = A[j];
    float b = __half2float(B[base + j]);
    local_sum += a * b;
  }

  // Block‐wide reduction to get the full dot‐product
  float sum = block_reduce_sum(local_sum);

  // Thread 0 writes the result
  if (threadIdx.x == 0) {
    out[col] = sum;
  }
}

void matrix_mul_vec_fp32_b_fp16(float *out, const float *a, const void *b, int m, int n) {
  const int max_blocks = 128;
  int block = std::min(bit_ceil(std::max(m, 32)), max_blocks);

  matvec_fp16_fp32_kernel<<<n, block>>>(a, reinterpret_cast<const half *>(b), out, m, n);
}

__global__ void matvec_bias_fp16_fp32_kernel(const float *A, const half *B, const half *bias, float *out, int m,
                                             int n) {
  int col = blockIdx.x;
  if (col >= n)
    return;

  // Local dot‐product accumulation
  float local_sum = 0.0f;
  int base = col * m;

  // Grid‐stride over the m dimension
  for (int j = threadIdx.x; j < m; j += blockDim.x) {
    float a = A[j];
    float b = __half2float(B[base + j]);
    local_sum += a * b;
  }

  // Block‐wide reduction to get the full dot‐product
  float sum = block_reduce_sum(local_sum);

  // Thread 0 writes the result
  if (threadIdx.x == 0) {
    out[col] = sum + __half2float(bias[col]);
  }
}

void matrix_mul_vec_bias_fp32_b_fp16(float *out, const float *a, const void *b, const void *bias, int m, int n) {
  const int max_blocks = 128;
  int block = std::min(bit_ceil(std::max(m, 32)), max_blocks);

  // 3) Launch
  matvec_bias_fp16_fp32_kernel<<<n, block>>>(a, reinterpret_cast<const half *>(b), reinterpret_cast<const half *>(bias),
                                             out, m, n);
}

} // namespace tinyllm::cuda
