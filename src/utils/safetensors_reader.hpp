#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace tinyllm {

struct SafeTensorsReader {
  struct Metadata {
    std::string dtype;
    std::vector<std::int32_t> data_offsets, shape;
  };

  std::map<std::string, std::ifstream> files;
  std::map<std::string, Metadata> data;

  SafeTensorsReader(const std::filesystem::path &config_path);

  Metadata get_tensor_meta(const std::string &name) const;

  std::vector<std::string> get_tensor_names() const;

  void load_tensor(const std::string &name, std::span<std::byte> buffer);

protected:
  void _load_metadata(std::ifstream &is);
};

} // namespace tinyllm
