#include "infer.hpp"
#include "block_reduce.cuh"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      std::fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                  \
      std::exit(-1);                                                                                                   \
    }                                                                                                                  \
  } while (0)

namespace tinyllm::cuda {

void *cuda_malloc(std::size_t size) {
  void *device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, size));
  return device;
}

void copy_to_device(const void *src, std::size_t size, void *dst) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void copy_to_host(const void *src, std::size_t size, void *dst) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void cuda_free(void *ptr) { CUDA_CHECK(cudaFree(ptr)); }

void *upload(const void *src, std::size_t size) {
  void *device = cuda_malloc(size);
  copy_to_device(src, size, device);
  return device;
}

void check_and_sync() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    std::exit(-1);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void rope_inplace_fp32_kernel(float *x, int head_dim, int pos, float theta) {
  const int half_dim = head_dim >> 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  int head = blockIdx.y;

  if (tid < half_dim) {
    float freq = powf(theta, -float(tid) / half_dim);
    float val = pos * freq;

    float vs, vc;
    sincosf(val, &vs, &vc);

    float *xh = x + head * head_dim;

    float v0 = xh[tid];
    float v1 = xh[tid + half_dim];

    xh[tid] = v0 * vc - v1 * vs;
    xh[tid + half_dim] = v0 * vs + v1 * vc;
  }
}

void rope_inplace_fp32(float *x, int num_heads, int head_dim, int pos, float theta) {
  const int half_dim = head_dim >> 1;

  const int threads_per_block = 256;

  int blocks_x = (half_dim + threads_per_block - 1) / threads_per_block;
  int blocks_y = int(num_heads);

  dim3 grid(blocks_x, blocks_y);
  dim3 block(threads_per_block);

  rope_inplace_fp32_kernel<<<grid, block>>>(x, head_dim, pos, theta);
}

__global__ void rms_norm_fp32_b_fp16_kernel(const float *x, half const *weight, float *out, int size, float eps) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int batch = blockIdx.y;

  const float *xb = x + batch * size;
  float *outb = out + batch * size;

  float local_sum = 0.0f;
  for (int i = tid; i < size; i += stride) {
    float v = xb[i];
    local_sum += v * v;
  }
  local_sum = block_reduce_sum(local_sum);

  float inv_norm = rsqrtf(local_sum / float(size) + eps);

  for (int i = tid; i < size; i += stride) {
    float w = __half2float(weight[i]);
    outb[i] = xb[i] * inv_norm * w;
  }
}

void rms_norm_fp32_b_fp16(float *out, const float *x, const void *weight, int size, int num_batches, float eps) {
  int blocksize = max(min(1024, bit_ceil(size)), 32);

  dim3 grid(1, num_batches);
  dim3 block(blocksize);

  rms_norm_fp32_b_fp16_kernel<<<grid, block>>>(x, reinterpret_cast<const half *>(weight), out, size, eps);
}

__global__ void vec_add_inplace_fp32_kernel(float *a, const float *b, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    a[tid] += b[tid];
  }
}

void vec_add_inplace_fp32(float *out, const float *a, int n) {
  const int block = 256;

  int grid = (n + block - 1) / block;

  vec_add_inplace_fp32_kernel<<<grid, block>>>(out, a, n);
}

__global__ void copy_fp16_to_fp32_n_kernel(const half *first, int n, float *result) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    result[tid] = __half2float(first[tid]);
  }
}

__global__ void copy_fp32_to_fp16_n_kernel(const float *first, int n, half *result) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    result[tid] = __float2half(first[tid]);
  }
}

void copy_fp16_to_fp32_n(const void *first, int n, float *result) {
  const int block = 256;
  int grid = (n + block - 1) / block;
  copy_fp16_to_fp32_n_kernel<<<grid, block>>>(reinterpret_cast<const half *>(first), n, result);
}

void copy_fp32_to_fp16_n(const float *first, int n, void *result) {
  const int block = 256;
  int grid = (n + block - 1) / block;
  copy_fp32_to_fp16_n_kernel<<<grid, block>>>(first, n, reinterpret_cast<half *>(result));
}

__global__ void swiglu_fp32_kernel(float *out, const float *x, const float *gate, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    float g = gate[tid];
    float sig = 1.0f / (1.0f + expf(-g));
    float swish = g * sig;
    out[tid] = x[tid] * swish;
  }
}

void swiglu_fp32(float *out, const float *x, const float *gate, int size) {
  constexpr int TPB = 256;
  int n_blocks = (size + TPB - 1) / TPB;

  swiglu_fp32_kernel<<<n_blocks, TPB>>>(out, x, gate, size);
}

__global__ void compute_raw_scores(float *atth, const float *qh, const half *kh, int head_dim, int n_kv_heads,
                                   int kv_len) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= kv_len)
    return;

  int kv_stride = n_kv_heads * head_dim;
  const half *kh_row = kh + tid * kv_stride;
  float sum = 0.0f;
  float scale = rsqrtf(float(head_dim));
  for (int j = 0; j < head_dim; ++j) {
    sum += qh[j] * __half2float(kh_row[j]);
  }
  atth[tid] = sum * scale;
}

__global__ void softmax_inplace(float *atth, int kv_len) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  float max = -FLT_MAX;
  for (int i = tid; i < kv_len; i += stride)
    max = fmaxf(max, atth[i]);

  max = block_reduce_max(max);

  float sum = 0.0f;
  for (int i = tid; i < kv_len; i += stride) {
    float e = expf(atth[i] - max);
    atth[i] = e;
    sum += e;
  }
  sum = block_reduce_sum(sum);

  for (int i = tid; i < kv_len; i += stride)
    atth[i] /= sum;
}

__global__ void compute_weighted_sum(float *xout, const float *atth, const half *vh, int head_dim, int n_kv_heads,
                                     int kv_len) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= head_dim)
    return;

  int kv_stride = n_kv_heads * head_dim;
  const half *vh_col = vh + tid;
  float sum = 0.0f;

  for (int i = 0; i < kv_len; ++i) {
    sum += atth[i] * __half2float(vh_col[i * kv_stride]);
  }
  xout[tid] = sum;
}

void attention_softmax_fp32_kv_fp16(float *out, float *atth, const float *qh, const void *kh, const void *vh,
                                    int head_dim, int n_kv_heads, int kv_len) {

  {
    const int block = 16;
    int grid = (kv_len + block - 1) / block;
    compute_raw_scores<<<grid, block>>>(atth, qh, reinterpret_cast<const half *>(kh), head_dim, n_kv_heads, kv_len);
  }

  {
    constexpr int MAX_TPB = 256;
    int tpb = max(32, min(bit_ceil(kv_len), MAX_TPB));

    dim3 block(tpb);
    size_t shm = tpb * sizeof(float);
    softmax_inplace<<<1, block, shm>>>(atth, kv_len);
  }

  {
    const int block = 16;
    int grid = (head_dim + block - 1) / block;
    compute_weighted_sum<<<grid, block>>>(out, atth, reinterpret_cast<const half *>(vh), head_dim, n_kv_heads, kv_len);
  }
}

} // namespace tinyllm::cuda
