#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace tinyllm {

struct ArenaAlloc {
  std ::size_t total_allocated = 0;
  std::vector<std::span<std::byte>> arenas;

  std::span<std::byte> alloc(std::size_t size) {
    auto span = std::span<std::byte>(new std::byte[size], size);
    arenas.emplace_back(span);
    total_allocated += size;
    return span;
  }

  ~ArenaAlloc() {
    for (auto &arena : arenas) {
      delete[] arena.data();
    }
  }

  void merge(ArenaAlloc &other) {
    total_allocated += other.total_allocated;
    arenas.insert(arenas.end(), other.arenas.begin(), other.arenas.end());
    other.arenas.clear();
    other.total_allocated = 0;
  }

  void swap(ArenaAlloc &other) {
    std::swap(total_allocated, other.total_allocated);
    arenas.swap(other.arenas);
  }
};

} // namespace tinyllm
