#pragma once

#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace tinyllm {

void parallel_for(std::size_t begin, std::size_t end, auto &&func) {
  std::uint32_t num_threads = std::max(std::thread::hardware_concurrency() / 2, 1u);
  std::size_t n = end - begin;
  if (num_threads == 0)
    num_threads = 4;

  std::vector<std::thread> threads;

  if (n < num_threads) {
    for (std::size_t i = begin; i < end; ++i) {
      threads.emplace_back(func, i, i + 1);
    }
  } else {
    std::size_t base = n / num_threads;
    std::size_t rem = n % num_threads;

    std::size_t start = begin;

    for (std::size_t i = 0; i < num_threads; ++i) {
      std::size_t size = base + (i < rem ? 1 : 0);
      threads.emplace_back(func, start, start + size);
      start += size;
    }
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

} // namespace tinyllm
