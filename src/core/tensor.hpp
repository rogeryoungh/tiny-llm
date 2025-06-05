#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace tinyllm {

enum class DataType { F32, BF16, F16 };

std::string dtype_to_string(DataType dtype);

std::size_t dtype_size(DataType dtype);

DataType string_to_dtype(const std::string &dtype_str);

struct Tensor {
  std::array<std::int32_t, 4> shape{};
  DataType dtype{};
  std::span<std::byte> data;
};

struct SafeTensorsData {
  struct Metadata {
    std::string dtype;
    std::vector<std::int32_t> data_offsets, shape;
  };

  std::map<std::string, Metadata> data;
  std::ifstream file;

  SafeTensorsData();

  void load_metadata(const std::filesystem::path &config_path_);
};

struct TensorManager {
  std::map<std::string, Tensor> tensors;

  ~TensorManager();

  void load_tensor(const std::string &name, std::ifstream &is, SafeTensorsData::Metadata metadata);

  Tensor get_tensor(const std::string &name);

  bool has_tensor(const std::string &name);
};

} // namespace tinyllm
