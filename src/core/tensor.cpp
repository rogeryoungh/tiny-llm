#include "tensor.hpp"

#include <cstddef>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace tinyllm {

std::string dtype_to_string(DataType dtype) {
  switch (dtype) {
  case DataType::F32:
    return "F32";
  case DataType::BF16:
    return "BF16";
  case DataType::F16:
    return "F16";
  default:
    throw std::runtime_error("Unknown data type");
  }
}

std::size_t dtype_size(DataType dtype) {
  switch (dtype) {
  case DataType::F32:
    return 4;
  case DataType::BF16:
    return 2;
  case DataType::F16:
    return 2;
  default:
    return 0;
  }
}

DataType string_to_dtype(const std::string &dtype_str) {
  if (dtype_str == "F32") {
    return DataType::F32;
  } else if (dtype_str == "BF16") {
    return DataType::BF16;
  } else if (dtype_str == "F16") {
    return DataType::F16;
  }
  throw std::invalid_argument("Unknown data type: " + dtype_str);
}

std::span<std::byte> TensorAlloc::alloc(std::size_t size) {
  allocated_size += size;
  total_allocated += size;
  std::span<std::byte> span(new std::byte[size], size);
  return span;
}

void TensorAlloc::dealloc(std::span<std::byte> span) {
  allocated_size -= span.size();
  delete[] span.data();
}

Tensor TensorAlloc::alloc_fp32(std::array<std::int32_t, 4> shape) {
  std::size_t size = dtype_size(DataType::F32);
  for (const auto &dim : shape) {
    size *= dim;
  }
  auto data = alloc(size);
  return Tensor{shape, DataType::F32, data};
}

Tensor TensorAlloc::alloc_fp32(std::int32_t s0, std::int32_t s1, std::int32_t s2, std::int32_t s3) {
  return alloc_fp32({s0, s1, s2, s3});
}

} // namespace tinyllm
