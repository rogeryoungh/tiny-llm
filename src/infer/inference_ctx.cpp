#include "inference_ctx.hpp"
#include "cpu/backend_cpu.hpp"

namespace tinyllm {

InferenceCtx::InferenceCtx(Model &model_, std::size_t kv_size, DeviceType device, DataType kv_dtype) {
  if (device != DeviceType::CPU) {
    throw std::runtime_error("Only CPU device is supported in this version.");
  }

  backend = std::make_unique<InferenceBackendCPU>(model_, kv_size, kv_dtype);
}

void InferenceCtx::forward(std::int32_t token, std::int32_t pos) { backend->forward(token, pos); }

void InferenceCtx::forward_prefill(std::int32_t token, std::int32_t pos) { backend->forward_prefill(token, pos); }

std::uint32_t InferenceCtx::argmax() const { return backend->argmax(); }

std::size_t InferenceCtx::memory_usage() const { return backend->memory_usage(); }

} // namespace tinyllm
