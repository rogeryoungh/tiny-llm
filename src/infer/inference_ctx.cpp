#include "inference_ctx.hpp"
#include "cpu/backend_cpu.hpp"
#include "cuda/backend_cuda.hpp"

namespace tinyllm {

InferenceCtx::InferenceCtx(Model &model_, std::size_t kv_size, DeviceType device, DataType kv_dtype) {

  if (device == DeviceType::GPU) {
    if (model_.dtype != DataType::F16 || kv_dtype != DataType::F16) {
      throw std::runtime_error("Unsupported kv_dtype for GPU backend. Only F16 are supported.");
    }
    backend = std::make_unique<InferenceBackendCUDA>(model_, kv_size, kv_dtype);
  } else if (device == DeviceType::CPU) {
    backend = std::make_unique<InferenceBackendCPU>(model_, kv_size, kv_dtype);
  } else {
    throw std::runtime_error("Unsupported device type.");
  }
}

void InferenceCtx::forward(std::int32_t token, std::int32_t pos) { backend->forward(token, pos); }

void InferenceCtx::forward_prefill(std::int32_t token, std::int32_t pos) { backend->forward_prefill(token, pos); }

std::uint32_t InferenceCtx::argmax() const { return backend->argmax(); }

std::size_t InferenceCtx::memory_usage() const { return backend->memory_usage(); }

} // namespace tinyllm
