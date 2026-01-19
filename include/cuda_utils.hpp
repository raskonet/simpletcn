#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP
#include "tensor.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#else
#define CUDA_CHECK(call)
#endif

// Forward Pass
void launch_conv1d(const Tensor& input, const Tensor& weights, const Tensor& bias, Tensor& output, int dilation, int kernel_size);
void launch_relu(Tensor& input);
void launch_add(const Tensor& a, const Tensor& b, Tensor& out);
void launch_dropout_apply(const Tensor& input, const Tensor& mask, Tensor& output, double scale);

// Backward Pass & Optimization
void launch_conv1d_backward(const Tensor& grad_out, const Tensor& input, const Tensor& weights, Tensor& grad_in, Tensor& grad_w, Tensor& grad_b, int dilation, int kernel_size);
void launch_relu_backward(const Tensor& grad_out, const Tensor& input_cache, Tensor& grad_in);
void launch_sgd_update(Tensor& params, const Tensor& grads, double lr);

#endif
