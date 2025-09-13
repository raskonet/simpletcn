#include "tensor.hpp"
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cstring>
#include <utility> 

Tensor::Tensor(int c, int w) : data(nullptr), channels(c), width(w), total_size(0) {
    if (c <= 0 || w <= 0) {
        throw std::invalid_argument("Tensor dimensions must be positive.");
    }
    total_size = static_cast<size_t>(channels) * static_cast<size_t>(width);
    data = static_cast<double*>(malloc(total_size * sizeof(double)));
    if (!data) {
        throw std::bad_alloc();
    }
}

Tensor::~Tensor() {
    free(data);
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data), channels(other.channels), width(other.width), total_size(other.total_size) {
    other.data = nullptr;
    other.channels = 0;
    other.width = 0;
    other.total_size = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free(data); 

        data = other.data;
        channels = other.channels;
        width = other.width;
        total_size = other.total_size;

        other.data = nullptr;
        other.channels = 0;
        other.width = 0;
        other.total_size = 0;
    }
    return *this;
}

double* Tensor::get_data() { return data; }
const double* Tensor::get_data() const { return data; }
int Tensor::get_channels() const { return channels; }
int Tensor::get_width() const { return width; }
size_t Tensor::get_total_size() const { return total_size; }

std::string Tensor::dimensions_str() const {
    return "[" + std::to_string(channels) + ", " + std::to_string(width) + "]";
}

void Tensor::zero() {
    if(data) {
        memset(data, 0, total_size * sizeof(double));
    }
}
