#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cstdio>

// --- Conv1D Kernel ---
__global__ void conv1d_kernel(
    const double* __restrict__ input,
    const double* __restrict__ weights,
    const double* __restrict__ bias,
    double* __restrict__ output,
    int in_channels, int out_channels, int width,
    int kernel_size, int dilation) 
{
    int t = blockIdx.x * blockDim.x + threadIdx.x; // Time step
    int o = blockIdx.y * blockDim.y + threadIdx.y; // Output channel

    if (t < width && o < out_channels) {
        double sum = bias[o];
        int padding = (kernel_size - 1) * dilation; 

        for (int i = 0; i < in_channels; ++i) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_t = t - (padding - k * dilation);
                if (input_t >= 0 && input_t < width) {
                    int x_idx = i * width + input_t;
                    int w_idx = (o * in_channels + i) * kernel_size + k;
                    sum += input[x_idx] * weights[w_idx];
                }
            }
        }
        output[o * width + t] = sum;
    }
}

void launch_conv1d(const Tensor& input, const Tensor& weights, const Tensor& bias, Tensor& output, int dilation, int kernel_size) {
    dim3 block(32, 4);
    dim3 grid((output.get_width() + 31) / 32, (output.get_channels() + 3) / 4);

    conv1d_kernel<<<grid, block>>>(
        input.get_device_data(),
        weights.get_device_data(),
        bias.get_device_data(),
        output.get_device_data(),
        input.get_channels(),
        output.get_channels(),
        output.get_width(),
        kernel_size,
        dilation
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// --- ReLU Kernel ---
__global__ void relu_kernel(double* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] < 0.0) data[idx] = 0.0;
    }
}

void launch_relu(Tensor& input) {
    int size = input.get_total_size();
    relu_kernel<<<(size + 255) / 256, 256>>>(input.get_device_data(), size);
    CUDA_CHECK(cudaDeviceSynchronize());
}
