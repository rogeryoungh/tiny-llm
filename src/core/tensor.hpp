#pragma once

#include "../utils/arena_alloc.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace tinyllm {

enum class DataType { F32, BF16, F16 };

enum class DeviceType { CPU, CUDA };

std::string dtype_to_string(DataType dtype);

std::size_t dtype_size(DataType dtype);

DataType string_to_dtype(const std::string &dtype_str);

struct Tensor {
  std::array<std::int32_t, 4> shape{};
  DataType dtype{};
  std::span<std::byte> data{};

  std::size_t size_bytes() const { return data.size_bytes(); }

  std::size_t size() const { return data.size() / dtype_size(dtype); }

  template <typename T> T *as() { return reinterpret_cast<T *>(data.data()); }

  template <typename T> const T *as() const { return reinterpret_cast<const T *>(data.data()); }

  template <typename T> const T *const_as() const { return reinterpret_cast<const T *>(data.data()); }

  static Tensor alloc(ArenaAlloc &a, DataType dtype, const std::array<std::int32_t, 4> &shape);

  static Tensor alloc(ArenaAlloc &a, DataType dtype, std::int32_t d0, std::int32_t d1 = 1, std::int32_t d2 = 1,
                      std::int32_t d3 = 1);
};

} // namespace tinyllm
