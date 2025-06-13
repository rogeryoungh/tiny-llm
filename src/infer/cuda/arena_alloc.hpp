#pragma once

#include "infer.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace tinyllm::cuda {

struct ArenaAlloc {
  std::size_t total_allocated = 0;
  std::vector<void *> arenas;

  void *alloc(std::size_t size) { return cuda::cuda_malloc(size); }

  void *upload(const void *host, std::size_t size) { return cuda::upload(host, size); }

  void *upload(const std::span<std::byte> span) { return upload(span.data(), span.size()); }

  ~ArenaAlloc() {
    for (void *p : arenas) {
      cuda::cuda_free(p);
    }
  }
};

} // namespace tinyllm::cuda
