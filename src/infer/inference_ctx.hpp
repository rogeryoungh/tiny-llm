#pragma once

#include "../core/model.hpp"
#include "../core/tensor.hpp"

namespace tinyllm {

struct InferenceBackend {
  virtual ~InferenceBackend() = default;

  virtual void forward(std::int32_t token, std::int32_t pos) = 0;

  virtual void forward_prefill(std::int32_t token, std::int32_t pos) = 0;

  virtual std::uint32_t argmax() const = 0;

  virtual std::size_t memory_usage() const = 0;
};

struct InferenceCtx {
  InferenceCtx(Model &model, std::size_t kv_size, DeviceType device = DeviceType::CPU,
               DataType kv_dtype = DataType::F32);

  void forward(std::int32_t token, std::int32_t pos);

  void forward_prefill(std::int32_t token, std::int32_t pos);

  std::uint32_t argmax() const;

  std::size_t memory_usage() const;

protected:
  std::unique_ptr<InferenceBackend> backend;
};

} // namespace tinyllm
