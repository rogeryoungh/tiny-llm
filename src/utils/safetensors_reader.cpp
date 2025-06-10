#include "safetensors_reader.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

namespace tinyllm {

void SafeTensorsReader::_load_metadata(const std::string &file_name) {
  std::uint64_t metadata_size = 0;
  std::ifstream is(config_path / file_name, std::ios::binary);
  std::cout << "Loading metadata from: " << (config_path / file_name).string() << ", size "
            << std::filesystem::file_size(config_path / file_name) << " bytes" << std::endl;
  is.read(reinterpret_cast<char *>(&metadata_size), sizeof(metadata_size));
  std::vector<char> v(metadata_size);
  is.read(v.data(), v.size());
  const auto metadata = nlohmann::json::parse(v);
  for (const auto &[name, tensor_info] : metadata.items()) {
    if (name == "__metadata__")
      continue;
    auto dtype = tensor_info.at("dtype").get<std::string>();
    auto data_offsets = tensor_info.at("data_offsets").get<std::vector<std::int64_t>>();
    auto shape = tensor_info.at("shape").get<std::vector<std::int64_t>>();

    for (auto &offset : data_offsets) {
      offset += metadata_size + 8;
    }

    file_names[name] = file_name;
    data[name] = Metadata{dtype, std::move(data_offsets), std::move(shape)};
  }
  files[file_name] = std::move(is);
}

SafeTensorsReader::SafeTensorsReader(const std::filesystem::path &path) : config_path(path) {
  std::ifstream index_file(config_path / "model.safetensors.index.json");

  nlohmann::json index_json;
  index_file >> index_json;

  for (const auto &[tensor_name, tensor_file] : index_json.at("weight_map").items()) {
    const auto file_name = tensor_file.get<std::string>();
    if (files.contains(file_name))
      continue;
    _load_metadata(file_name);
  }
}

std::vector<std::string> SafeTensorsReader::get_tensor_names() const {
  std::vector<std::string> names;
  names.reserve(data.size());
  for (const auto &[name, _] : data) {
    names.push_back(name);
  }
  return names;
}

SafeTensorsReader::Metadata SafeTensorsReader::get_tensor_meta(const std::string &name) const {
  const auto &metadata = data.at(name);
  return metadata;
}

void SafeTensorsReader::load_tensor(const std::string &name, std::span<std::byte> span, DataType type) {
  const auto &metadata = data.at(name);
  auto &file = files.at(file_names.at(name));

  std::size_t begin = metadata.data_offsets[0];
  std::size_t end = metadata.data_offsets[1];

  file.seekg(begin);
  if (!file) {
    throw std::runtime_error("Failed to seek to tensor data for " + name);
  }
  auto meta_dtype = string_to_dtype(metadata.dtype);
  if (type == meta_dtype) {
    file.read(reinterpret_cast<char *>(span.data()), end - begin);
  } else {
    if (type == DataType::F32 && meta_dtype == DataType::BF16) {
      assert(span.size() == (end - begin) * 2);
      auto *bf16_ptr = reinterpret_cast<std::uint16_t *>(span.data() + span.size() / 2);
      auto *fp32_ptr = reinterpret_cast<std::uint32_t *>(span.data());
      file.read(reinterpret_cast<char *>(bf16_ptr), end - begin);
      const std::size_t num_fp32 = span.size() / sizeof(std::uint32_t);
      for (std::size_t i = 0; i < num_fp32; ++i) {
        fp32_ptr[i] = static_cast<std::uint32_t>(bf16_ptr[i]) << 16;
      }
    } else {
      throw std::runtime_error("Unsupported data type conversion from " + metadata.dtype + " to " +
                               dtype_to_string(type));
    }
  }
}

} // namespace tinyllm
