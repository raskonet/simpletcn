#include "dropout.hpp"
#include <random>
#include <stdexcept>
#include <cstring> 

static std::mt19937& get_random_engine() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

Dropout::Dropout(double rate) : dropout_rate(rate), is_training(true) {
    if (rate < 0.0 || rate >= 1.0) {
        throw std::invalid_argument("Dropout rate must be in the range [0, 1).");
    }
}

void Dropout::set_training_mode(bool training) {
    is_training = training;
}

Tensor Dropout::forward(const Tensor& input) {
    if (!is_training || dropout_rate == 0.0) {
        Tensor output(input.get_channels(), input.get_width());
        memcpy(output.get_data(), input.get_data(), input.get_total_size() * sizeof(double));
        return output;
    }

    const int channels = input.get_channels();
    const int width = input.get_width();
    
    Tensor output(channels, width);
    mask = std::make_unique<Tensor>(channels, width);

    const double* input_data = input.get_data();
    double* output_data = output.get_data();
    double* mask_data = mask->get_data();
    
    const double scale = 1.0 / (1.0 - dropout_rate);
    std::bernoulli_distribution dist(1.0 - dropout_rate);

    for (size_t i = 0; i < input.get_total_size(); ++i) {
        if (dist(get_random_engine())) {
            mask_data[i] = 1.0;
            output_data[i] = input_data[i] * scale;
        } else {
            mask_data[i] = 0.0;
            output_data[i] = 0.0;
        }
    }
    
    return output;
}

Tensor Dropout::backward(const Tensor& output_gradient) {
    if (!is_training || dropout_rate == 0.0) {
        Tensor grad_input(output_gradient.get_channels(), output_gradient.get_width());
        memcpy(grad_input.get_data(), output_gradient.get_data(), output_gradient.get_total_size() * sizeof(double));
        return grad_input;
    }
    
    if (!mask || mask->get_total_size() != output_gradient.get_total_size()) {
        throw std::runtime_error("Backward called on Dropout without a corresponding forward pass, or dimension mismatch.");
    }
    
    Tensor grad_input(output_gradient.get_channels(), output_gradient.get_width());
    
    const double* grad_output_data = output_gradient.get_data();
    const double* mask_data = mask->get_data();
    double* grad_input_data = grad_input.get_data();

    const double scale = 1.0 / (1.0 - dropout_rate);
    
    for (size_t i = 0; i < grad_input.get_total_size(); ++i) {
        grad_input_data[i] = grad_output_data[i] * mask_data[i] * scale;
    }

    return grad_input;
}
