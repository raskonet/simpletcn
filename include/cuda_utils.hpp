#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include "tensor.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel Launchers
void launch_conv1d(const Tensor& input, const Tensor& weights, const Tensor& bias, Tensor& output, int dilation, int kernel_size);
void launch_relu(Tensor& input);

#endif
