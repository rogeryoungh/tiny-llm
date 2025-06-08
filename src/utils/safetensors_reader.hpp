#pragma once

#include "../core/tensor.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace tinyllm {

struct SafeTensorsReader {
  struct Metadata {
    std::string dtype;
    std::vector<std::int64_t> data_offsets, shape;
  };

  std::filesystem::path config_path;
  std::map<std::string, std::ifstream> files;
  std::map<std::string, Metadata> data;
  std::map<std::string, std::string> file_names;

  SafeTensorsReader(const std::filesystem::path &path);

  Metadata get_tensor_meta(const std::string &name) const;

  std::vector<std::string> get_tensor_names() const;

  void load_tensor(const std::string &name, std::span<std::byte> buffer, DataType type);

protected:
  void _load_metadata(const std::string &file_name);
};

} // namespace tinyllm
