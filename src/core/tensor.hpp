#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace tinyllm {

enum class DataType { F32, BF16, F16 };

std::string dtype_to_string(DataType dtype);

std::size_t dtype_size(DataType dtype);

DataType string_to_dtype(const std::string &dtype_str);

struct Tensor {
  std::array<std::int32_t, 4> shape{};
  DataType dtype{};
  std::span<std::byte> data;

  template <typename T> T *as() { return reinterpret_cast<T *>(data.data()); }

  template <typename T> const T *as() const { return reinterpret_cast<const T *>(data.data()); }

  template <typename T> const T *const_as() const { return reinterpret_cast<const T *>(data.data()); }
};

struct TensorAlloc {
  std::size_t total_allocated = 0;
  std::size_t allocated_size = 0;

  std::span<std::byte> alloc(std::size_t size);

  void dealloc(std::span<std::byte> span);
};

} // namespace tinyllm
