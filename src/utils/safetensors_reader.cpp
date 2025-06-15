#include "safetensors_reader.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <print>

namespace tinyllm {

namespace fs = std::filesystem;

struct MetadataRaw {
  std::string name, dtype;
  std::int64_t start, end;
  std::array<std::int32_t, 4> shape;
};

void SafeTensorsReader::load_metadata(const std::filesystem::path &path, ArenaAlloc &alloc) {
  std::uint64_t metadata_size = 0;

  std::ifstream is(path, std::ios::binary);
  std::size_t tensor_size = fs::file_size(path);
  is.read(reinterpret_cast<char *>(&metadata_size), sizeof(metadata_size));

  std::vector<char> v(metadata_size);
  is.read(v.data(), v.size());

  std::println("[DEBUG] Loading metadata from: {}, size {} MB", path.string(), (tensor_size >> 20));

  const auto metadata_json = nlohmann::json::parse(v);

  std::vector<MetadataRaw> metadata_raw;

  for (const auto &[name, tensor_info] : metadata_json.items()) {
    if (name == "__metadata__")
      continue;
    auto data_offsets = tensor_info.at("data_offsets").get<std::vector<std::int64_t>>();
    auto shape = tensor_info.at("shape").get<std::vector<std::int32_t>>();

    std::array<std::int32_t, 4> shape_array{1, 1, 1, 1};
    if (shape.size() > 4 || shape.size() < 1) {
      throw std::runtime_error("Invalid tensor shape for " + name);
    } else {
      std::copy(shape.rbegin(), shape.rend(), shape_array.begin());
    }

    MetadataRaw raw;
    raw.name = name;
    raw.dtype = tensor_info.at("dtype").get<std::string>();
    raw.start = data_offsets.at(0);
    raw.end = data_offsets.at(1);
    raw.shape = shape_array;

    metadata_raw.emplace_back(raw);
  }

  std::sort(metadata_raw.begin(), metadata_raw.end(),
            [](const MetadataRaw &a, const MetadataRaw &b) { return a.start < b.start; });

  std::span<std::byte> data = alloc.alloc(tensor_size - metadata_size - 8);
  is.seekg(metadata_size + 8);
  is.read(reinterpret_cast<char *>(data.data()), data.size());
  if (is.eof() || is.fail()) {
    throw std::runtime_error("Failed to read metadata from " + path.string());
  }

  for (const auto &d : metadata_raw) {
    std::span<std::byte> data_span = data.subspan(d.start, d.end - d.start);
    metadata[d.name] = Metadata{d.dtype, d.shape, data_span};
  }
}

SafeTensorsReader::SafeTensorsReader(const std::filesystem::path &path, ArenaAlloc &alloc) {
  fs::path index_path = path / "model.safetensors.index.json";

  std::vector<std::string> files;

  if (fs::exists(index_path)) {

    std::ifstream index_file(index_path);

    nlohmann::json index_json;
    index_file >> index_json;

    for (const auto &[tensor_name, file_name] : index_json.at("weight_map").items()) {
      files.emplace_back(file_name.get<std::string>());
    }
  } else {
    files.emplace_back("model.safetensors");
  }

  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());

  for (const auto &file_name : files) {
    load_metadata(path / file_name, alloc);
  }
}

SafeTensorsReader::Metadata SafeTensorsReader::get_metadata(const std::string &name) const { return metadata.at(name); }

} // namespace tinyllm
