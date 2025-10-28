#include "tensor.hpp"
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cstring>
#include <utility>
#include <iostream>

constexpr size_t ALIGNMENT = 64;

Tensor::Tensor(int c, int w) : data(nullptr), channels(c), width(w), total_size(0) {
    if (c <= 0 || w <= 0) {
        throw std::invalid_argument("Tensor dimensions must be positive.");
    }
    total_size = static_cast<size_t>(channels) * static_cast<size_t>(width);
    
    size_t bytes = total_size * sizeof(double);
    size_t aligned_bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
#ifdef _WIN32
    data = static_cast<double*>(_aligned_malloc(aligned_bytes, ALIGNMENT));
#else
    data = static_cast<double*>(std::aligned_alloc(ALIGNMENT, aligned_bytes));
#endif

    if (!data) throw std::bad_alloc();
}

Tensor::~Tensor() {
    if (data) {
#ifdef _WIN32
        _aligned_free(data);
#else
        free(data);
#endif
    }
}

Tensor Tensor::clone() const {
    Tensor copy(channels, width);
    if (total_size > 0 && data) {
        std::memcpy(copy.data, data, total_size * sizeof(double));
    }
    return copy;
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
#ifdef _WIN32
        _aligned_free(data);
#else
        free(data);
#endif
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
    if(data) std::memset(data, 0, total_size * sizeof(double));
}

void Tensor::save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    out.write(reinterpret_cast<const char*>(&width), sizeof(width));
    out.write(reinterpret_cast<const char*>(data), total_size * sizeof(double));
}

void Tensor::load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&channels), sizeof(channels));
    in.read(reinterpret_cast<char*>(&width), sizeof(width));
    
    size_t new_size = static_cast<size_t>(channels) * static_cast<size_t>(width);
    
    if (new_size != total_size) {
        if (data) {
            #ifdef _WIN32
                _aligned_free(data);
            #else
                free(data);
            #endif
        }
        total_size = new_size;
        constexpr size_t ALIGNMENT = 64;
        size_t bytes = total_size * sizeof(double);
        size_t aligned_bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        #ifdef _WIN32
            data = static_cast<double*>(_aligned_malloc(aligned_bytes, ALIGNMENT));
        #else
            data = static_cast<double*>(std::aligned_alloc(ALIGNMENT, aligned_bytes));
        #endif
    }
    
    in.read(reinterpret_cast<char*>(data), total_size * sizeof(double));
}
