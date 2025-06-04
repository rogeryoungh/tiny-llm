#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_safetensors_file>" << std::endl;
    return 1;
  }

  const std::string path = argv[1];
  std::ifstream file(path, std::ios::binary);
  std::int64_t json_size = 0;
  file.read(reinterpret_cast<char *>(&json_size), sizeof(json_size));
  std::cout << "JSON size: " << json_size << std::endl;

  std::vector<char> json_data(json_size);
  file.read(json_data.data(), json_data.size());
  nlohmann::json json = nlohmann::json::parse(json_data);
  std::cout << "JSON content: " << json.dump(4) << std::endl;

  return 0;
}
