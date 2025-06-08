#pragma once

#include <chrono>

namespace tinyllm {
class Stopwatch {
public:
  Stopwatch() : start_time(std::chrono::high_resolution_clock::now()) {}

  void reset() { start_time = std::chrono::high_resolution_clock::now(); }

  double elapsed_seconds() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
  }
  double elapsed_ms() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_time;
};

} // namespace tinyllm
