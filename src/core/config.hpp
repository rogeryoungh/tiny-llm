#pragma once

#include "tensor.hpp"

#include <filesystem>

namespace tinyllm {

struct Config {
  TensorManager tensor_manager;

  void load_tensors(const std::filesystem::path path);
};

} // namespace tinyllm
