#include "safetensors_reader.hpp"

#include <nlohmann/json.hpp>

namespace tinyllm {

void SafeTensorsReader::_load_metadata(std::ifstream &is) {
  std::uint64_t metadata_size = 0;
  is.read(reinterpret_cast<char *>(&metadata_size), sizeof(metadata_size));
  std::vector<char> v(metadata_size);
  is.read(v.data(), v.size());
  const auto metadata = nlohmann::json::parse(v);
  for (const auto &[name, tensor_info] : metadata.items()) {
    auto dtype = tensor_info.at("dtype").get<std::string>();
    auto data_offsets = tensor_info.at("data_offsets").get<std::vector<std::int32_t>>();
    auto shape = tensor_info.at("shape").get<std::vector<std::int32_t>>();

    data[name] = Metadata{dtype, std::move(data_offsets), std::move(shape)};
  }
}

SafeTensorsReader::SafeTensorsReader(const std::filesystem::path &config_path) {
  std::ifstream index_file(config_path / "model.safetensors.index.json");

  nlohmann::json index_json;
  index_file >> index_json;

  for (const auto &[tensor_name, tensor_file] : index_json.at("weight_map").items()) {
    const auto file_name = tensor_file.get<std::string>();
    if (file_name == "__metadata__")
      continue;
    if (files.contains(file_name))
      continue;
    std::ifstream is(config_path / file_name, std::ios::binary);
    _load_metadata(is);
    files[tensor_name] = std::move(is);
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

void SafeTensorsReader::load_tensor(const std::string &name, std::span<std::byte> buffer) {
  const auto &metadata = data.at(name);
  auto &file = files.at(name);

  std::size_t begin = metadata.data_offsets[0];
  std::size_t end = metadata.data_offsets[1];

  file.seekg(begin);
  file.read(reinterpret_cast<char *>(buffer.data()), end - begin);
}

} // namespace tinyllm
