#include "residual_block.hpp"
#include <utility>      
#include <cstring>     
#include <stdexcept>  

ResidualBlock::ResidualBlock(int in_channels, int n_filters, int kernel_size, int dilation, double dropout_rate)
    : conv1(in_channels, n_filters, kernel_size, dilation),
      relu1(),
      dropout1(dropout_rate),
      conv2(n_filters, n_filters, kernel_size, dilation),
      relu2(),
      dropout2(dropout_rate),
      downsample(nullptr),
      input_cache(nullptr) 
{
    if (in_channels != n_filters) {
        downsample = std::make_unique<Conv1D>(in_channels, n_filters, 1, 1);
    }
}

void ResidualBlock::set_training_mode(bool training) {
    dropout1.set_training_mode(training);
    dropout2.set_training_mode(training);
}

void ResidualBlock::zero_grad() {
    conv1.zero_grad();
    conv2.zero_grad();
    if (downsample) {
        downsample->zero_grad();
    }
}

void ResidualBlock::update(double learning_rate) {
    conv1.update(learning_rate);
    conv2.update(learning_rate);
    if (downsample) {
        downsample->update(learning_rate);
    }
}

Tensor ResidualBlock::forward(const Tensor& input) {
    this->input_cache = &input;

    Tensor main_path_out = conv1.forward(input);
    main_path_out = relu1.forward(main_path_out);
    main_path_out = dropout1.forward(main_path_out);
    main_path_out = conv2.forward(main_path_out);
    main_path_out = relu2.forward(main_path_out);
    main_path_out = dropout2.forward(main_path_out);

    Tensor residual_out(input.get_channels(), input.get_width());
    if (downsample) {
        residual_out = downsample->forward(input);
    } else {
        memcpy(residual_out.get_data(), input.get_data(), input.get_total_size() * sizeof(double));
    }
    
    double* main_data = main_path_out.get_data();
    const double* res_data = residual_out.get_data();
    for (size_t i = 0; i < main_path_out.get_total_size(); ++i) {
        main_data[i] += res_data[i];
    }

    return main_path_out;
}

Tensor ResidualBlock::backward(const Tensor& output_gradient) {
    if (!this->input_cache) {
        throw std::runtime_error("Backward called on ResidualBlock without a forward pass.");
    }
    
    Tensor grad_main = dropout2.backward(output_gradient);
    grad_main = relu2.backward(grad_main);
    grad_main = conv2.backward(grad_main);
    grad_main = dropout1.backward(grad_main);
    grad_main = relu1.backward(grad_main);
    grad_main = conv1.backward(grad_main);

    if (downsample) {
        Tensor grad_residual = downsample->backward(output_gradient);
        double* g_main_data = grad_main.get_data();
        const double* g_res_data = grad_residual.get_data();
        for (size_t i = 0; i < grad_main.get_total_size(); ++i) {
            g_main_data[i] += g_res_data[i];
        }
    } else {
        double* g_main_data = grad_main.get_data();
        const double* g_res_data = output_gradient.get_data();
        for (size_t i = 0; i < grad_main.get_total_size(); ++i) {
            g_main_data[i] += g_res_data[i];
        }
    }

    this->input_cache = nullptr; 
    return grad_main;
}
