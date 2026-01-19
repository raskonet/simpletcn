#include "tensor.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>

constexpr size_t ALIGNMENT = 64;

Tensor::Tensor(int c, int w) 
    : data(nullptr), device_data(nullptr), channels(c), width(w), total_size(0), current_device(Device::CPU) 
{
    total_size = (size_t)channels * (size_t)width;
    size_t bytes = total_size * sizeof(double);
    size_t aligned_bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
#ifdef _WIN32
    data = (double*)_aligned_malloc(aligned_bytes, ALIGNMENT);
#else
    data = (double*)aligned_alloc(ALIGNMENT, aligned_bytes);
#endif
    if(!data) throw std::bad_alloc();
}

Tensor::~Tensor() {
    if(data) free(data); 
    if(device_data) cudaFree(device_data);
}

Tensor Tensor::clone() const {
    Tensor copy(channels, width);
    if(data) memcpy(copy.data, data, total_size * sizeof(double));
    if(current_device == Device::GPU && device_data) {
        copy.to_device(); 
        cudaMemcpy(copy.device_data, device_data, total_size * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    return copy;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data), device_data(other.device_data),
      channels(other.channels), width(other.width), total_size(other.total_size),
      current_device(other.current_device) {
    other.data = nullptr;
    other.device_data = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if(this != &other) {
        if(data) free(data);
        if(device_data) cudaFree(device_data);
        data = other.data;
        device_data = other.device_data;
        channels = other.channels;
        width = other.width;
        total_size = other.total_size;
        current_device = other.current_device;
        other.data = nullptr;
        other.device_data = nullptr;
    }
    return *this;
}

double* Tensor::get_data() {
    if(current_device == Device::GPU) to_host();
    return data;
}

const double* Tensor::get_data() const {
    const_cast<Tensor*>(this)->to_host();
    return data;
}

double* Tensor::get_device_data() const {
    if(!device_data) {
        // Lazy allocation hack for const correctness in this simplified wrapper
        double** ptr = const_cast<double**>(&device_data);
        cudaMalloc(ptr, total_size * sizeof(double));
        cudaMemcpy(*ptr, data, total_size * sizeof(double), cudaMemcpyHostToDevice);
    }
    return device_data;
}

void Tensor::to_device() {
    if(!device_data) cudaMalloc(&device_data, total_size * sizeof(double));
    cudaMemcpy(device_data, data, total_size * sizeof(double), cudaMemcpyHostToDevice);
    current_device = Device::GPU;
}

void Tensor::to_host() {
    if(device_data && current_device == Device::GPU) {
        cudaMemcpy(data, device_data, total_size * sizeof(double), cudaMemcpyDeviceToHost);
    }
    current_device = Device::CPU;
}

void Tensor::zero() {
    if(current_device == Device::CPU) memset(data, 0, total_size * sizeof(double));
    else if(device_data) cudaMemset(device_data, 0, total_size * sizeof(double));
}

void Tensor::save(std::ofstream& out) const {
    const_cast<Tensor*>(this)->to_host();
    out.write((char*)&channels, sizeof(channels));
    out.write((char*)&width, sizeof(width));
    out.write((char*)data, total_size * sizeof(double));
}

void Tensor::load(std::ifstream& in) {
    in.read((char*)&channels, sizeof(channels));
    in.read((char*)&width, sizeof(width));
    in.read((char*)data, total_size * sizeof(double));
    current_device = Device::CPU;
}
