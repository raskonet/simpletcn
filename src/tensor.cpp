#include "tensor.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <new> // For std::bad_alloc

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

constexpr size_t ALIGNMENT = 64;

Tensor::Tensor() : data(nullptr), device_data(nullptr), channels(0), width(0), total_size(0), current_device(Device::CPU) {}

Tensor::Tensor(int c, int w) : data(nullptr), device_data(nullptr), channels(c), width(w), total_size((size_t)c*w), current_device(Device::CPU) {
    size_t bytes = total_size * sizeof(double);
    size_t aligned = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    #ifdef _WIN32
    data = (double*)_aligned_malloc(aligned, ALIGNMENT);
    #else
    data = (double*)aligned_alloc(ALIGNMENT, aligned);
    #endif
    if (!data && total_size > 0) throw std::bad_alloc();
}

Tensor::~Tensor() {
    if(data) { 
        #ifdef _WIN32 
        _aligned_free(data); 
        #else 
        free(data); 
        #endif 
    }
    #ifdef USE_CUDA
    if(device_data) cudaFree(device_data);
    #endif
}

// FIX APPLIED: Correct logic for CPU/GPU reallocation without forced GPU switch
void Tensor::reallocate(int c, int w) {
    size_t needed = (size_t)c * w;
    
    // 1. Reuse existing capacity if possible
    if (needed <= total_size) {
        channels = c;
        width = w;
        return; 
    }

    // 2. Growth required
    total_size = needed;
    channels = c;
    width = w;
    size_t bytes = total_size * sizeof(double);

    if (current_device == Device::CPU) {
        if (data) {
            #ifdef _WIN32
            _aligned_free(data);
            #else
            free(data);
            #endif
        }
        size_t aligned = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        #ifdef _WIN32
        data = (double*)_aligned_malloc(aligned, ALIGNMENT);
        #else
        data = (double*)aligned_alloc(ALIGNMENT, aligned);
        #endif
        if (!data) throw std::bad_alloc();
    } 
    #ifdef USE_CUDA
    else if (current_device == Device::GPU) {
        if (device_data) cudaFree(device_data);
        cudaMalloc(&device_data, bytes);
    }
    #endif
}

void Tensor::copy_from(const Tensor& other) {
    if (total_size != other.get_total_size()) reallocate(other.get_channels(), other.get_width());
    
    #ifdef USE_CUDA
    if (other.get_device() == Device::GPU) { 
        cudaMemcpy(get_device_data(), other.get_device_data(), total_size*sizeof(double), cudaMemcpyDeviceToDevice); 
        current_device = Device::GPU; 
    } else { 
        cudaMemcpy(get_device_data(), other.get_data(), total_size*sizeof(double), cudaMemcpyHostToDevice); 
        current_device = Device::GPU; 
    }
    #else
    memcpy(get_data(), other.get_data(), total_size*sizeof(double)); 
    current_device = Device::CPU;
    #endif
}

Tensor Tensor::clone() const {
    Tensor copy(channels, width);
    if(data) memcpy(copy.data, data, total_size*sizeof(double));
    #ifdef USE_CUDA
    if(current_device == Device::GPU && device_data) { 
        copy.to_device(); 
        cudaMemcpy(copy.device_data, device_data, total_size*sizeof(double), cudaMemcpyDeviceToDevice); 
    }
    #endif
    return copy;
}

Tensor::Tensor(Tensor&& o) noexcept 
    : data(o.data), device_data(o.device_data), channels(o.channels), width(o.width), total_size(o.total_size), current_device(o.current_device) { 
    o.data=nullptr; o.device_data=nullptr; o.channels=0; o.width=0; o.total_size=0; 
}

Tensor& Tensor::operator=(Tensor&& o) noexcept { 
    if(this!=&o){ 
        if(data) {
            #ifdef _WIN32 
            _aligned_free(data); 
            #else 
            free(data); 
            #endif
        }
        #ifdef USE_CUDA 
        if(device_data) cudaFree(device_data); 
        #endif 
        data=o.data; device_data=o.device_data; channels=o.channels; width=o.width; total_size=o.total_size; current_device=o.current_device; 
        o.data=nullptr; o.device_data=nullptr; 
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
    #ifdef USE_CUDA
    if(!device_data) { 
        double** p = const_cast<double**>(&device_data); 
        cudaMalloc(p, total_size*sizeof(double)); 
        if(data) cudaMemcpy(*p, data, total_size*sizeof(double), cudaMemcpyHostToDevice); 
    } 
    return device_data; 
    #else 
    return nullptr; 
    #endif
}

void Tensor::to_device() { 
    #ifdef USE_CUDA
    // FIX APPLIED: Check for empty tensor
    if (total_size == 0) return;
    
    if(!device_data) cudaMalloc(&device_data, total_size*sizeof(double)); 
    if(data) cudaMemcpy(device_data, data, total_size*sizeof(double), cudaMemcpyHostToDevice); 
    current_device = Device::GPU; 
    #endif
}

void Tensor::to_host() { 
    if(!data) { 
        size_t b = total_size * sizeof(double); 
        size_t a = (b+63)&~63; 
        #ifdef _WIN32 
        data=(double*)_aligned_malloc(a,64); 
        #else 
        data=(double*)aligned_alloc(64,a); 
        #endif 
    }
    #ifdef USE_CUDA
    if(device_data && current_device == Device::GPU) cudaMemcpy(data, device_data, total_size*sizeof(double), cudaMemcpyDeviceToHost); 
    #endif
    current_device = Device::CPU; 
}

void Tensor::zero() { 
    if(current_device==Device::CPU && data) memset(data,0,total_size*sizeof(double)); 
    #ifdef USE_CUDA
    else if(device_data) cudaMemset(device_data,0,total_size*sizeof(double)); 
    #endif
}

void Tensor::save(std::ofstream& out) const { 
    const_cast<Tensor*>(this)->to_host(); 
    out.write((char*)&channels,4); 
    out.write((char*)&width,4); 
    out.write((char*)data,total_size*sizeof(double)); 
}

void Tensor::load(std::ifstream& in) { 
    in.read((char*)&channels,4); 
    in.read((char*)&width,4); 
    reallocate(channels, width); // Ensure memory exists
    in.read((char*)data,total_size*sizeof(double)); 
    current_device=Device::CPU; 
}
