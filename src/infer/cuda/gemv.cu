#include "block_reduce.cuh"
#include "infer.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tinyllm::cuda {

template <int THREADS_PER_BLOCK>
__global__ void gemv_fp32_b_fp16_kernel(const float *a, const half *b, float *out, int m, int n) {
  const int lane = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;

  const int col = blockIdx.x * THREADS_PER_BLOCK + wid;
  if (col >= n)
    return;

  const int m4 = m / 4; // 总是假设 m 是 4 的倍数
  float sum = 0.0f;

  const float4 *a4 = reinterpret_cast<const float4 *>(a);
  const half2 *b2_row = reinterpret_cast<const half2 *>(b + col * m);

  float4 sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  for (int j = lane; j < m4; j += warpSize) {
    float4 a4j = a4[j];
    half2 b2j0 = b2_row[j * 2 + 0];
    half2 b2j1 = b2_row[j * 2 + 1];
    float2 bj0 = __half22float2(b2j0);
    float2 bj1 = __half22float2(b2j1);

    sum4.x += a4j.x * bj0.x;
    sum4.y += a4j.y * bj0.y;
    sum4.z += a4j.z * bj1.x;
    sum4.w += a4j.w * bj1.y;
  }

  sum = sum4.x + sum4.y + sum4.z + sum4.w;

  sum = warp_reduce_sum(sum);

  if (lane == 0) {
    out[col] = sum;
  }
}

void gemv_fp32_b_fp16(float *out, const float *a, const void *b, int m, int n) {
  constexpr int WARP_SIZE = 32, THREADS_PER_BLOCK = 4;
  dim3 grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  dim3 block = WARP_SIZE * THREADS_PER_BLOCK;
  gemv_fp32_b_fp16_kernel<THREADS_PER_BLOCK><<<grid, block>>>(a, reinterpret_cast<const half *>(b), out, m, n);
}

template <int THREADS_PER_BLOCK>
__global__ void gemv_bias_fp32_b_fp16_kernel(const float *a, const half *b, const half *bias, float *out, int m,
                                             int n) {
  const int lane = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;

  const int col = blockIdx.x * THREADS_PER_BLOCK + wid;
  if (col >= n)
    return;

  const int m4 = m / 4; // 总是假设 m 是 4 的倍数
  float sum = 0.0f;
  const float4 *a4 = reinterpret_cast<const float4 *>(a);
  const half2 *b2_row = reinterpret_cast<const half2 *>(b + col * m);

  float4 sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  for (int j = lane; j < m4; j += warpSize) {
    float4 a4j = a4[j];
    half2 b2j0 = b2_row[j * 2 + 0];
    half2 b2j1 = b2_row[j * 2 + 1];
    float2 bj0 = __half22float2(b2j0);
    float2 bj1 = __half22float2(b2j1);

    sum4.x += a4j.x * bj0.x;
    sum4.y += a4j.y * bj0.y;
    sum4.z += a4j.z * bj1.x;
    sum4.w += a4j.w * bj1.y;
  }

  sum = sum4.x + sum4.y + sum4.z + sum4.w;

  sum = warp_reduce_sum(sum);

  if (lane == 0) {
    out[col] = sum + __half2float(bias[col]);
  }
}

void gemv_bias_fp32_b_fp16(float *out, const float *a, const void *b, const void *bias, int m, int n) {
  constexpr int WARP_SIZE = 32, THREADS_PER_BLOCK = 4;
  dim3 grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  dim3 block = WARP_SIZE * THREADS_PER_BLOCK;
  gemv_bias_fp32_b_fp16_kernel<THREADS_PER_BLOCK>
      <<<grid, block>>>(a, reinterpret_cast<const half *>(b), reinterpret_cast<const half *>(bias), out, m, n);
}

} // namespace tinyllm::cuda
