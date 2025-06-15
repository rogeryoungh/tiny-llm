#include "inference_ctx.hpp"
#include "cpu/backend_cpu.hpp"
#include "cuda/backend_cuda.hpp"

namespace tinyllm {

InferenceCtx::InferenceCtx(Model &model_, std::size_t kv_size, DataType kv_dtype) {
  backend = std::make_unique<InferenceBackendCPU>(model_, kv_size, kv_dtype);
}

InferenceCtx::InferenceCtx(ModelCuda &model, std::size_t kv_size, DataType kv_dtype) {
  if (model.dtype != DataType::F16 || kv_dtype != DataType::F16) {
    throw std::runtime_error("Unsupported kv_dtype for CUDA backend. Only F16 are supported.");
  }
  backend = std::make_unique<InferenceBackendCUDA>(model, kv_size, kv_dtype);
}

void InferenceCtx::forward(std::int32_t token, std::int32_t pos) { backend->forward(token, pos); }

void InferenceCtx::forward_prefill(std::int32_t token, std::int32_t pos) { backend->forward_prefill(token, pos); }

std::int32_t InferenceCtx::sample_argmax() { return backend->sample_argmax(); }

std::int32_t InferenceCtx::sample() { return backend->sample(); }

std::size_t InferenceCtx::memory_usage() const { return backend->memory_usage(); }

} // namespace tinyllm
