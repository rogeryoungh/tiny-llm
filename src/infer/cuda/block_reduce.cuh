#pragma once

#include <cuda_runtime.h>
#include <cfloat>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      std::fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                  \
      std::exit(-1);                                                                                                   \
    }                                                                                                                  \
  } while (0)

namespace tinyllm::cuda {

template <typename T> inline __device__ T warp_reduce_sum(T val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename T> inline __device__ T warp_reduce_max(T val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

template <typename T> inline __device__ T block_reduce_sum(T val) {
  __shared__ T shared[32];
  const int lane = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;

  val = warp_reduce_sum(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < warp_count) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0) {
      shared[0] = val;
    }
  }
  __syncthreads();

  return shared[0];
}

inline int bit_ceil(int x) {
  if (x <= 0)
    return 1;
  int n = x - 1;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

template <typename T> inline __device__ T block_reduce_max(T val) {
  __shared__ T shared_max[32];
  const int lane = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;

  val = warp_reduce_max(val);

  if (lane == 0) {
    shared_max[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < warp_count) ? shared_max[lane] : -FLT_MAX;
    val = warp_reduce_max(val);
    if (lane == 0) {
      shared_max[0] = val;
    }
  }
  __syncthreads();

  return shared_max[0];
}

} // namespace tinyllm::cuda
