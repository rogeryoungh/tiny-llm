#include "tensor.hpp"

#include <nlohmann/json.hpp>

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

Tensor Tensor::alloc(ArenaAlloc &a, DataType dtype, const std::array<std::int32_t, 4> &shape) {
  std::size_t size = 1;
  for (std::int32_t dim : shape) {
    size *= dim;
  }
  size *= dtype_size(dtype);
  auto data = a.alloc(size);
  return Tensor{shape, dtype, data};
}

Tensor Tensor::alloc(ArenaAlloc &a, DataType dtype, std::int32_t d0, std::int32_t d1, std::int32_t d2,
                     std::int32_t d3) {
  return alloc(a, dtype, {d0, d1, d2, d3});
}

} // namespace tinyllm
