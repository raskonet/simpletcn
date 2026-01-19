#ifndef UTILS_HPP
#define UTILS_HPP
#include "tensor.hpp"
#include <string>
Tensor load_timeseries(const std::string& filepath, int max_length = -1);
#endif
