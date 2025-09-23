// File: relu.cpp

#include "relu.hpp"
#include <cmath>
#include <stdexcept>

Tensor ReLU::forward(const Tensor& input) {
    // Cache the input pointer. This is essential for the backward pass.
    input_cache = &input;

    const int channels = input.get_channels();
    const int width = input.get_width();
    
    Tensor output(channels, width);

    const double* input_data = input.get_data();
    double* output_data = output.get_data();

    // The ReLU operation: y = max(0, x)
    for (size_t i = 0; i < input.get_total_size(); ++i) {
        output_data[i] = std::max(0.0, input_data[i]);
    }

    return output;
}

Tensor ReLU::backward(const Tensor& output_gradient) {
    if (!input_cache) {
        throw std::runtime_error("Backward called on ReLU without a forward pass. Input cache is null.");
    }
    
    if (input_cache->get_total_size() != output_gradient.get_total_size()) {
         throw std::runtime_error("Dimension mismatch between cached input and output gradient in ReLU.");
    }

    Tensor grad_input(input_cache->get_channels(), input_cache->get_width());

    const double* cached_input_data = input_cache->get_data();
    const double* grad_output_data = output_gradient.get_data();
    double* grad_input_data = grad_input.get_data();

    // The ReLU gradient logic:
    // dL/dX = dL/dY * dY/dX
    // dY/dX is 1 if X > 0, and 0 otherwise.
    for (size_t i = 0; i < grad_input.get_total_size(); ++i) {
        if (cached_input_data[i] > 0) {
            grad_input_data[i] = grad_output_data[i]; // Pass gradient through
        } else {
            grad_input_data[i] = 0.0; // Block gradient
        }
    }

    // Invalidate the cache after use to prevent accidental reuse.
    input_cache = nullptr;

    return grad_input;
}
