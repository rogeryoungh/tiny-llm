#pragma once

#include <iostream>
#include <ostream>
#include <vector>

std::ostream &operator<<(std::ostream &os, const std::vector<std::int32_t> &v) {
  os << "[";
  if (v.empty()) {
    os << "]";
    return os;
  }
  os << v[0];
  for (size_t i = 1; i < v.size(); ++i) {
    os << ", " << v[i];
  }
  os << "]";
  return os;
}
