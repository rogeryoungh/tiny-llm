#pragma once

#include "arena_alloc.hpp"

#include <filesystem>
#include <map>
#include <string>

namespace tinyllm {

struct SafeTensorsReader {
  struct Metadata {
    std::string dtype;
    std::array<std::int32_t, 4> shape;
    std::span<std::byte> data;
  };

  std::map<std::string, Metadata> metadata;

  SafeTensorsReader(const std::filesystem::path &path, ArenaAlloc &alloc);

  Metadata get_metadata(const std::string &name) const;

protected:
  void load_metadata(const std::filesystem::path &path, ArenaAlloc &alloc);
};

} // namespace tinyllm
