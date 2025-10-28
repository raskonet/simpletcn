#include "relu.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

Tensor ReLU::forward(const Tensor& input) {
    // FIX: Clone input
    input_cache = std::make_unique<Tensor>(input.clone());

    const int channels = input.get_channels();
    const int width = input.get_width();
    Tensor output(channels, width);

    const double* input_data = input.get_data();
    double* output_data = output.get_data();

    for (size_t i = 0; i < input.get_total_size(); ++i) {
        output_data[i] = std::max(0.0, input_data[i]);
    }

    return output;
}

Tensor ReLU::backward(const Tensor& output_gradient) {
    if (!input_cache) {
        throw std::runtime_error("Backward called on ReLU without a forward pass.");
    }
    
    Tensor grad_input(input_cache->get_channels(), input_cache->get_width());

    const double* cached_input_data = input_cache->get_data();
    const double* grad_output_data = output_gradient.get_data();
    double* grad_input_data = grad_input.get_data();

    for (size_t i = 0; i < grad_input.get_total_size(); ++i) {
        if (cached_input_data[i] > 0) {
            grad_input_data[i] = grad_output_data[i];
        } else {
            grad_input_data[i] = 0.0;
        }
    }

    input_cache = nullptr; // Free memory
    return grad_input;
}
