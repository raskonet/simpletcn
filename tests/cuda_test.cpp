#include "conv1d.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

const double TOLERANCE = 1e-5;

bool check_cpu_gpu_consistency() {
    std::cout << "--- Test: CPU vs GPU Numerical Consistency ---" << std::endl;

#ifndef USE_CUDA
    std::cout << "[SKIP] CUDA not compiled. Cannot run GPU consistency check." << std::endl;
    return true; 
#endif

    // 1. Setup Data
    int in_channels = 4;
    int out_channels = 8;
    int kernel_size = 3;
    int dilation = 2;
    int width = 100;

    Conv1D layer(in_channels, out_channels, kernel_size, dilation);
    
    // Create random input
    Tensor input_cpu(in_channels, width);
    for (size_t i = 0; i < input_cpu.get_total_size(); ++i) {
        input_cpu.get_data()[i] = (double)(rand() % 100) / 100.0;
    }

    Tensor out_cpu = layer.forward_ref(input_cpu).clone(); 

    Tensor input_gpu = input_cpu.clone();
    
    input_gpu.to_device(); 
    
    const Tensor& out_gpu_ref = layer.forward_ref(input_gpu);
    
    const double* cpu_ptr = out_cpu.get_data();
    const double* gpu_ptr = out_gpu_ref.get_data();

    if (out_cpu.get_device() != Device::CPU) {
        std::cerr << "FAIL: CPU output should reside on CPU." << std::endl;
        return false;
    }
    
    if (out_gpu_ref.get_device() != Device::GPU) {
         std::cerr << "FAIL: GPU output should reside on GPU. Dispatch failed?" << std::endl;
         return false;
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < out_cpu.get_total_size(); ++i) {
        double diff = std::abs(cpu_ptr[i] - gpu_ptr[i]);
        if (diff > max_diff) max_diff = diff;
        
        if (diff > TOLERANCE) {
            std::cerr << "FAIL: Mismatch at index " << i << "\n"
                      << "  CPU: " << cpu_ptr[i] << "\n"
                      << "  GPU: " << gpu_ptr[i] << "\n"
                      << "  Diff: " << diff << std::endl;
            return false;
        }
    }

    std::cout << "PASS: Max difference between CPU and GPU kernels: " << max_diff << std::endl;
    return true;
}

int main() {
    if (check_cpu_gpu_consistency()) {
        std::cout << "\n[SUCCESS] CUDA consistency checks passed." << std::endl;
        return 0;
    } else {
        std::cerr << "\n[FAILURE] CUDA consistency checks failed." << std::endl;
        return 1;
    }
}
