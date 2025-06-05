#include "tensor.hpp"

#include <cstddef>
#include <iostream>
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

SafeTensorsData::SafeTensorsData() = default;

void SafeTensorsData::load_metadata(const std::filesystem::path &config_path) {
  file.open(config_path, std::ios::binary);

  std::uint64_t metadata_size = 0;
  file.read(reinterpret_cast<char *>(&metadata_size), sizeof(metadata_size));
  std::vector<char> v(metadata_size);
  file.read(v.data(), v.size());
  const auto metadata = nlohmann::json::parse(v);

  for (const auto &[name, tensor_info] : metadata.items()) {
    auto dtype = tensor_info.at("dtype").get<std::string>();
    auto data_offsets = tensor_info.at("data_offsets").get<std::vector<std::int32_t>>();
    auto shape = tensor_info.at("shape").get<std::vector<std::int32_t>>();
    data.emplace(name, Metadata{dtype, data_offsets, shape});
  }

  std::cout << "Loaded metadata for tensor: " << config_path << std::endl;
  std::cout << metadata << std::endl;
}

TensorManager::~TensorManager() {
  for (auto &[name, tensor] : tensors) {
    delete[] tensor.data.data();
  }
}

void TensorManager::load_tensor(const std::string &name, std::ifstream &is, SafeTensorsData::Metadata metadata) {
  auto begin = metadata.data_offsets[0];
  auto end = metadata.data_offsets[1];
  is.seekg(begin);
  auto *buffer = new std::byte[end - begin];
  static std::size_t tensor_count = 0;
  tensor_count += end - begin;
  std::cout << "Malloc " << ((end - begin) >> 20) << " MB for tensor: " << name << ", total " << (tensor_count >> 20)
            << " MB" << std::endl;

  is.read(reinterpret_cast<char *>(buffer), end - begin);

  Tensor tensor;
  if (metadata.shape.size() > 4) {
    throw std::runtime_error("Tensor shape exceeds 4 dimensions, which is not supported.");
  } else {
    std::copy(metadata.shape.begin(), metadata.shape.end(), tensor.shape.begin());
  }
  tensor.dtype = string_to_dtype(metadata.dtype);
  tensor.data = std::span<std::byte>(buffer, end - begin);

  tensors[name] = tensor;
}

Tensor TensorManager::get_tensor(const std::string &name) { return tensors.at(name); }

bool TensorManager::has_tensor(const std::string &name) { return tensors.find(name) != tensors.end(); }

} // namespace tinyllm
