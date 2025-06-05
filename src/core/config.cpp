#include "config.hpp"
#include "tensor.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace tinyllm {

void Config::load_tensors(const fs::path path) {
  // Load tensor configurations from a file
  const fs::path config_file = path / "model.safetensors.index.json";
  std::ifstream file(config_file);
  if (!file) {
    throw std::runtime_error("Failed to open config file");
  }

  nlohmann::json config_json;
  file >> config_json;

  std::cout << config_json << std::endl;

  std::map<std::string, SafeTensorsData> tensors;

  for (const auto &[tensor_name, tensor_file] : config_json.at("weight_map").items()) {
    std::cout << "Loading tensor: " << tensor_name << " from " << tensor_file << std::endl;
    const std::string tensor_file_name = tensor_file.get<std::string>();

    if (tensor_file_name == "__metadata__") {
      continue;
    }

    if (!tensors.contains(tensor_file_name)) {
      tensors.emplace(tensor_file_name, SafeTensorsData());
      tensors[tensor_file_name].load_metadata(path / tensor_file_name);
    }

    auto &safetensors_info = tensors.at(tensor_file_name);
    std::cout << tensor_name << std::endl;
    const auto tensor_info = safetensors_info.data.at(tensor_name);

    tensor_manager.load_tensor(tensor_name, safetensors_info.file, tensor_info);
  }
}

} // namespace tinyllm
